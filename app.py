import os
import sys
import io
import base64
import json
import math
import re
import unicodedata
import subprocess
import time
import tempfile
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import click
import pandas as pd
from rich.console import Console
from rich.progress import track
import shutil

# Import SVO extraction function
try:
    # Add parent directory to path to import convert_to_svo
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TVC-AI"))
    from convert_to_svo import extract_svo
except ImportError:
    # Fallback if import fails
    extract_svo = None

# Optional imports guarded at use

console = Console()

# Import image captioning module
from image_captioning import caption_image_local, caption_image_openai, synthesize_captions

_TEXT_CLASSIFIER_CACHE: Optional[Dict[str, object]] = None
_WHISPER_CACHE: Optional[Dict[str, object]] = None
_FASTER_WHISPER_CACHE: Optional[Dict[str, object]] = None
_YOLO_MODEL_CACHE: Optional[object] = None
_OCR_READER_CACHE: Optional[object] = None
def get_asr_model_id(default_size: str = "small") -> str:
    """Resolve ASR model id. Prefer PhoWhisper env if set, else faster-whisper size."""
    # Examples for PhoWhisper CTranslate2 models on HF:
    #   bmd1905/PhoWhisper-small-ct2, bmd1905/PhoWhisper-medium-ct2, vinai/PhoWhisper-large-ct2 (if available)
    return (
        os.environ.get("PHOWHISPER_MODEL")
        or os.environ.get("FASTER_WHISPER_MODEL")
        or default_size
    )


@dataclass
class CaptionResult:
    second: int
    image_path: str
    caption: str
    caption_original: Optional[str] = None  # Caption gốc trước khi dịch


DEFAULT_OPENAI_CAPTION_PROMPT = (
    "Bạn là chuyên gia mô tả hình ảnh. Hãy mô tả ảnh bằng định dạng SVO (Chủ ngữ - Động từ - Tân ngữ) "
    "với CHỈ MỘT động từ chính, tối đa 20 từ."
)


@dataclass
class OpenAICaptionConfig:
    api_key: str
    model: str
    prompt: str
    temperature: float
    max_tokens: int
    target_language: str = "vi"


def build_openai_caption_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    target_language: Optional[str] = None,
) -> OpenAICaptionConfig:
    """
    Build OpenAI caption config, loading defaults from openai_config.py if available.
    CLI parameters override config file values.
    """
    # Try to load from openai_config.py
    try:
        from openai_config import get_openai_config
        config = get_openai_config()
        # Use config file values as defaults, but allow CLI override
        final_api_key = api_key or config.get("api_key")
        final_model = model or config.get("model", "gpt-4o-mini")
        final_prompt = prompt or config.get("prompt", DEFAULT_OPENAI_CAPTION_PROMPT)
        final_temperature = temperature if temperature is not None else config.get("temperature", 0.3)
        final_max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 150)
        final_target_language = target_language or config.get("target_language", "vi")
    except ImportError:
        # Fallback to environment variables or defaults
        final_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        final_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        final_prompt = prompt or os.environ.get("OPENAI_CAPTION_PROMPT", DEFAULT_OPENAI_CAPTION_PROMPT)
        final_temperature = temperature if temperature is not None else float(os.environ.get("OPENAI_CAPTION_TEMPERATURE", "0.3"))
        final_max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("OPENAI_CAPTION_MAX_TOKENS", "150"))
        final_target_language = target_language or "vi"
    
    if not final_api_key:
        raise ValueError("OpenAI API key is required")
    
    return OpenAICaptionConfig(
        api_key=final_api_key,
        model=final_model,
        prompt=final_prompt,
        temperature=final_temperature,
        max_tokens=final_max_tokens,
        target_language=final_target_language,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_url(path_or_url: str) -> bool:
    """Kiểm tra xem input có phải là URL không."""
    if not path_or_url:
        return False
    parsed = urllib.parse.urlparse(path_or_url)
    return parsed.scheme in ('http', 'https', 'ftp')


def download_video_from_url(url: str, output_path: Optional[str] = None) -> str:
    """Tải video từ URL về file tạm.
    
    Args:
        url: URL của video
        output_path: Đường dẫn file output (optional, nếu không có sẽ tạo file tạm)
    
    Returns:
        Đường dẫn đến file video đã tải về
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError(
            "Thư viện 'requests' chưa được cài đặt. Vui lòng cài đặt bằng: pip install requests"
        )
    
    console.print(f"[cyan]Đang tải video từ URL: {url}[/cyan]")
    
    # Tạo file tạm nếu không có output_path
    if not output_path:
        # Lấy extension từ URL nếu có
        parsed = urllib.parse.urlparse(url)
        ext = os.path.splitext(parsed.path)[1] or '.mp4'
        temp_fd, output_path = tempfile.mkstemp(suffix=ext, prefix='video_')
        os.close(temp_fd)
    
    try:
        # Download với progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        console.print(f"\r[cyan]Đã tải: {downloaded}/{total_size} bytes ({percent:.1f}%)[/cyan]", end="")
        
        console.print(f"\n[green]✓ Đã tải video thành công: {output_path}[/green]")
        return output_path
    except Exception as e:
        # Xóa file tạm nếu download thất bại
        if os.path.exists(output_path) and output_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(output_path)
            except Exception:
                pass
        raise RuntimeError(f"Không thể tải video từ URL: {url}. Lỗi: {e}")


def resolve_video_input(input_path_or_url: str, cleanup_temp: bool = True) -> Tuple[str, bool]:
    """Xử lý input có thể là file path hoặc URL.
    
    Args:
        input_path_or_url: Đường dẫn file hoặc URL
        cleanup_temp: Có xóa file tạm sau khi xử lý không (chỉ áp dụng cho URL)
    
    Returns:
        Tuple (video_path, is_temp_file) - đường dẫn video và có phải file tạm không
    """
    if is_url(input_path_or_url):
        # Tải video từ URL
        temp_path = download_video_from_url(input_path_or_url)
        return temp_path, True
    else:
        # Kiểm tra file có tồn tại không
        if not os.path.exists(input_path_or_url):
            raise FileNotFoundError(f"Video file không tồn tại: {input_path_or_url}")
        return input_path_or_url, False


def seconds_floor(duration_seconds: float) -> int:
    if duration_seconds is None or math.isnan(duration_seconds):
        return 0
    return max(0, int(math.floor(duration_seconds)))


def extract_frames_1fps(video_path: str, output_dir: str) -> List[str]:
    """Extract one frame per second from the video and save as PNG files.

    Returns the list of saved image paths ordered by second (0,1,2,...).
    """
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    total_seconds = seconds_floor(duration)

    saved_paths: List[str] = []

    for sec in track(range(total_seconds), description="Extracting frames (1 fps)"):
        # Position by milliseconds to pick a representative frame at each whole second
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success or frame is None:
            # Fallback: try to set by frame index if ms seek failed
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
                success, frame = cap.read()
        if not success or frame is None:
            console.print(f"[yellow]Warning: could not read frame at {sec}s[/yellow]")
            continue
        # Convert BGR to RGB and save
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        out_path = os.path.join(output_dir, f"frame_{sec:04d}.png")
        img.save(out_path)
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def extract_frames_to_memory(video_path: str) -> List[Tuple[int, Image.Image]]:
    """Extract one frame per second from the video into memory (PIL Images).
    
    Returns the list of (second, PIL Image) tuples ordered by second (0,1,2,...).
    Images are not saved to disk.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    total_seconds = seconds_floor(duration)

    frames: List[Tuple[int, Image.Image]] = []

    for sec in track(range(total_seconds), description="Extracting frames to memory (1 fps)"):
        # Position by milliseconds to pick a representative frame at each whole second
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, frame = cap.read()
        if not success or frame is None:
            # Fallback: try to set by frame index if ms seek failed
            if fps > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
                success, frame = cap.read()
        if not success or frame is None:
            console.print(f"[yellow]Warning: could not read frame at {sec}s[/yellow]")
            continue
        # Convert BGR to RGB and create PIL Image (keep in memory)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        frames.append((sec, img))

    cap.release()
    return frames


# ------------------------ Audio (FFmpeg + Whisper) ------------------------

def _resolve_ffmpeg_cmd() -> str:
    """Find ffmpeg executable from PATH, env var FFMPEG_PATH, or local ./ffmpeg/ffmpeg.exe"""
    # 1) Env var
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2) Local bundled ffmpeg under project dir
    here = os.path.dirname(os.path.abspath(__file__))
    local_ffmpeg = os.path.join(here, "ffmpeg", "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if os.path.isfile(local_ffmpeg):
        return local_ffmpeg
    # 3) System PATH
    which = shutil.which("ffmpeg")
    if which:
        return which
    raise FileNotFoundError("ffmpeg not found. Set FFMPEG_PATH or place ffmpeg in PATH or UI-TVC-2/ffmpeg/")


def extract_audio_wav(video_path: str, output_wav: str, sample_rate: int = 16000, apply_filters: bool = True) -> None:
    """Extract audio track to mono WAV using ffmpeg if available.
    Optionally applies simple filters to improve ASR clarity.
    """
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    ffmpeg_bin = _resolve_ffmpeg_cmd()
    af = []
    if apply_filters:
        af = ["-af", "highpass=f=100,lowpass=f=7500,dynaudnorm"]
    cmd = [
        ffmpeg_bin,
        "-hide_banner", "-nostdin", "-loglevel", "error",
        "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate), *af, "-f", "wav", output_wav,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=180)
    except Exception as e:
        raise RuntimeError(f"ffmpeg failed to extract audio: {e}")


def _get_whisper_model(model_name: str):
    """Load and cache OpenAI Whisper model if installed."""
    global _WHISPER_CACHE
    if _WHISPER_CACHE and _WHISPER_CACHE.get("name") == model_name:
        return _WHISPER_CACHE["model"]
    try:
        import whisper  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Whisper is not installed. Please `pip install openai-whisper`. {e}")
    try:
        model = whisper.load_model(model_name)
        _WHISPER_CACHE = {"name": model_name, "model": model}
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load Whisper model '{model_name}': {e}")


def transcribe_audio_whisper(wav_path: str, model_name: str = "base", language: Optional[str] = None) -> Dict[str, object]:
    """Transcribe audio using Whisper; returns dict with 'text' and 'segments'.
    Avoids Whisper's internal ffmpeg call by passing raw waveform.
    """
    model = _get_whisper_model(model_name)
    # Read mono 16k WAV without external deps
    import wave
    try:
        with wave.open(wav_path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            num_frames = wf.getnframes()
            raw = wf.readframes(num_frames)
        # Expect 16-bit PCM
        if sampwidth != 2:
            raise RuntimeError(f"Unsupported WAV sample width: {sampwidth*8} bit")
        audio_i16 = np.frombuffer(raw, dtype=np.int16)
        if num_channels > 1:
            audio_i16 = audio_i16.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
        # Ensure 16000 Hz; our extractor already enforces 16k
        if sample_rate != 16000:
            raise RuntimeError(f"Unexpected WAV sample rate {sample_rate}, expected 16000")
        audio = (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        result = model.transcribe(audio, language=language)
        return result
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


# ------------------------ OpenAI Whisper API ------------------------

def transcribe_audio_openai_whisper(
    audio_file_path: str,
    api_key: Optional[str] = None,
    language: str = "vi",
    prompt: Optional[str] = None,
    console=None,
) -> Dict[str, object]:
    """Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_file_path: Path to audio file (WAV, MP3, etc.)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env or openai_config.py)
        language: Language code (e.g., "vi", "en")
        prompt: Optional prompt to guide transcription
        console: Console object for logging (optional)
    
    Returns:
        Dict with 'text' and 'segments' (similar to faster-whisper format)
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Package 'openai' is required. Install via `pip install openai`.") from exc
    
    # Get API key from parameter, config, or environment
    key = api_key
    if not key:
        try:
            from openai_config import get_openai_config
            config = get_openai_config()
            key = config.get("api_key")
        except ImportError:
            pass
    
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    
    if not key:
        raise RuntimeError("OpenAI API key is required. Provide api_key parameter, set in openai_config.py, or set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=key)
    
    # Default prompt for Vietnamese TVC ads
    default_prompt = prompt or "Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm"
    
    try:
        if console:
            console.print(f"[cyan]Đang transcribe audio bằng OpenAI Whisper API...[/cyan]")
        
        # Open audio file
        with open(audio_file_path, "rb") as audio_file:
            # Call OpenAI Whisper API
            # Ensure language is valid ISO-639-1 format (not "auto" or None)
            # OpenAI Whisper API requires explicit language code
            valid_language = language if language and language != "auto" and len(language) == 2 else "vi"
            
            # Note: verbose_json format may not be available in all API versions
            # Use default json format and extract text
            try:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=valid_language,  # Must be ISO-639-1 format (e.g., "vi", "en")
                    prompt=default_prompt,
                    response_format="verbose_json",  # Try verbose_json first
                )
            except Exception:
                # Fallback to default json format if verbose_json is not supported
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=valid_language,  # Must be ISO-639-1 format
                    prompt=default_prompt,
                )
        
        # Extract text and segments
        # Handle both dict and object responses
        if isinstance(transcript, dict):
            full_text = transcript.get("text", "").strip()
            transcript_segments = transcript.get("segments", [])
        else:
            full_text = getattr(transcript, "text", "").strip() if hasattr(transcript, 'text') else ""
            transcript_segments = getattr(transcript, "segments", []) if hasattr(transcript, 'segments') else []
        
        # Extract segments if available
        segments = []
        if transcript_segments:
            for seg in transcript_segments:
                if isinstance(seg, dict):
                    segments.append({
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": seg.get("text", "").strip(),
                    })
                else:
                    segments.append({
                        "start": float(getattr(seg, "start", 0.0)),
                        "end": float(getattr(seg, "end", 0.0)),
                        "text": getattr(seg, "text", "").strip(),
                    })
        else:
            # If no segments, create one segment for the entire text
            segments = [{
                "start": 0.0,
                "end": 0.0,
                "text": full_text,
            }]
        
        if console:
            preview = full_text[:80] + "..." if len(full_text) > 80 else full_text
            console.print(f"[green]✓ OpenAI Whisper transcription: {preview}[/green]")
        
        return {
            "text": full_text,
            "segments": segments,
            "model_used": "whisper-1",
        }
    except Exception as e:
        error_msg = f"OpenAI Whisper API error: {e}"
        if console:
            console.print(f"[red]{error_msg}[/red]")
        raise RuntimeError(error_msg) from e


def combine_audio_and_frame_captions(
    audio_text: str,
    frame_captions: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    target_language: str = "vi",
    console=None,
) -> str:
    """Kết hợp audio transcript với frame captions để tạo mô tả hoàn thiện hơn bằng GPT.
    
    Args:
        audio_text: Text từ audio transcription
        frame_captions: Danh sách các caption từ frames (có thể là 1 hoặc nhiều)
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env or openai_config.py)
        model: OpenAI model to use
        target_language: Target language for output
        console: Console object for logging (optional)
    
    Returns:
        Mô tả hoàn thiện đã được kết hợp
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Package 'openai' is required. Install via `pip install openai`.") from exc
    
    # Get API key
    key = api_key
    if not key:
        try:
            from openai_config import get_openai_config
            config = get_openai_config()
            key = config.get("api_key")
        except ImportError:
            pass
    
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    
    if not key:
        raise RuntimeError("OpenAI API key is required for combining audio and frame captions.")
    
    client = OpenAI(api_key=key)
    
    # Chuẩn bị prompt
    frame_captions_text = "\n".join([f"- {i+1}. {cap}" for i, cap in enumerate(frame_captions) if cap and cap.strip()])
    
    if not audio_text and not frame_captions_text:
        return ""
    
    if not audio_text:
        # Chỉ có frame captions, tổng hợp chúng
        if len(frame_captions) == 1:
            return frame_captions[0]
        # Nhiều captions, tổng hợp
        from image_captioning import synthesize_captions
        return synthesize_captions(frame_captions, console=console)
    
    if not frame_captions_text:
        # Chỉ có audio, trả về audio text
        return audio_text
    
    # Có cả audio và frame captions, kết hợp bằng GPT
    prompt = f"""Bạn là chuyên gia phân tích quảng cáo TVC. Hãy kết hợp thông tin từ audio và hình ảnh để tạo mô tả hoàn thiện nhất.

**Nội dung audio:**
{audio_text}

**Mô tả từ hình ảnh (frames):**
{frame_captions_text}

Yêu cầu:
1. Kết hợp thông tin từ audio và hình ảnh một cách tự nhiên
2. Định dạng SVO (Subject - Verb - Object) với CHỈ MỘT động từ chính
3. Tối đa 20-25 từ (ngắn gọn nhưng đầy đủ thông tin)
4. Ưu tiên thông tin từ audio về sản phẩm/thương hiệu, kết hợp với hành động từ hình ảnh
5. Viết bằng tiếng {target_language}

Ví dụ tốt:
- Audio: "Sản phẩm ABC giảm giá 50%"
- Hình ảnh: "người phụ nữ cầm sản phẩm"
- Kết quả: "người phụ nữ cầm sản phẩm ABC đang giảm giá 50%"

Mô tả hoàn thiện:"""
    
    try:
        if console:
            console.print(f"[cyan]Đang kết hợp audio và frame captions bằng GPT...[/cyan]")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"Bạn là chuyên gia kết hợp thông tin audio và hình ảnh để tạo mô tả quảng cáo TVC bằng tiếng {target_language}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=200,  # Tăng từ 150 lên 200 để đảm bảo không bị cắt giữa chừng
        )
        
        combined_caption = response.choices[0].message.content.strip()
        combined_caption = combined_caption.replace("\n", " ").strip()
        
        if console:
            preview = combined_caption[:80] + "..." if len(combined_caption) > 80 else combined_caption
            console.print(f"[green]✓ Mô tả kết hợp: {preview}[/green]")
        
        # Post-processing - nhẹ nhàng hơn để tránh cắt mất phần cuối
        from image_captioning import _remove_repetition, _remove_vague_references
        # Chỉ loại bỏ repetition và vague references, KHÔNG dùng _fix_incomplete_sentence và _ensure_single_verb_svo
        # vì chúng có thể cắt mất phần cuối của caption
        combined_caption = _remove_repetition(combined_caption, max_repeat=2)
        combined_caption = _remove_vague_references(combined_caption)
        
        # Chỉ loại bỏ dấu phẩy/câu không hoàn chỉnh ở cuối nếu thực sự không hoàn chỉnh
        # (không cắt quá mức)
        combined_caption = combined_caption.strip()
        # Loại bỏ dấu phẩy cuối cùng nếu có (thường là dấu hiệu câu bị cắt)
        if combined_caption.endswith(','):
            combined_caption = combined_caption[:-1].strip()
        # Loại bỏ các từ không hoàn chỉnh ở cuối (chỉ 1-2 ký tự)
        words = combined_caption.split()
        if len(words) > 0:
            last_word = words[-1].strip('.,!?;:')
            if len(last_word) <= 2 and last_word not in ['một', 'hai', 'ba', 'em', 'anh', 'chị', 'ông', 'bà']:
                # Có thể là từ không hoàn chỉnh, nhưng chỉ loại bỏ nếu thực sự ngắn và không có nghĩa
                pass  # Giữ nguyên để tránh cắt quá mức
        
        return combined_caption
    except Exception as e:
        error_msg = f"Error combining audio and frame captions: {e}"
        if console:
            console.print(f"[yellow]{error_msg}[/yellow]")
        # Fallback: return synthesized frame captions or audio text
        if frame_captions:
            from image_captioning import synthesize_captions
            return synthesize_captions(frame_captions, console=console)
        return audio_text


# ------------------------ Faster-Whisper (preferred for Vietnamese) ------------------------

def _get_faster_whisper_model(model_name: str = "small", device: Optional[str] = None, compute_type: Optional[str] = None):
    global _FASTER_WHISPER_CACHE
    # Include device index in cache key if specified
    env_idx = os.environ.get("FASTER_WHISPER_DEVICE_INDEX") or os.environ.get("GPU_INDEX") or os.environ.get("CUDA_DEVICE")
    try:
        dev_index = int(env_idx) if env_idx is not None else 0
    except Exception:
        dev_index = 0
    key = f"{model_name}:{device or ''}:{compute_type}:{dev_index}"
    if _FASTER_WHISPER_CACHE and _FASTER_WHISPER_CACHE.get("key") == key:
        return _FASTER_WHISPER_CACHE["model"]
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as e:
        raise RuntimeError(f"faster-whisper is not installed. Please `pip install faster-whisper`. {e}")
    try:
        # Auto device + compute type selection
        if device is None:
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        # Honor device index for multi-GPU; for single GPU RTX 3050, index=0
        if device == "cuda":
            model = WhisperModel(model_name, device=device, compute_type=compute_type, device_index=dev_index)
        else:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _FASTER_WHISPER_CACHE = {"key": key, "model": model}
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load faster-whisper model '{model_name}': {e}")


def transcribe_audio_faster_whisper(wav_path: str, model_name: str = "small", language: str = "vi", initial_prompt: Optional[str] = None) -> Dict[str, object]:
    """Transcribe audio using faster-whisper; returns dict with 'text' and 'segments'."""
    model = _get_faster_whisper_model(model_name=model_name)
    try:
        # Enable VAD only if onnxruntime is available
        use_vad = True
        try:
            import onnxruntime  # type: ignore  # noqa: F401
        except Exception:
            use_vad = False
        segments, info = model.transcribe(
            wav_path,
            language=language,
            vad_filter=use_vad,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.5],
            initial_prompt=(initial_prompt or "Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm"),
        )
        out_segments = []
        full_text_parts = []
        for seg in segments:
            text_seg = (seg.text or "").strip()
            out_segments.append({
                "start": float(seg.start) if seg.start is not None else 0.0,
                "end": float(seg.end) if seg.end is not None else 0.0,
                "text": text_seg,
            })
            if text_seg:
                full_text_parts.append(text_seg)
        return {"text": " ".join(full_text_parts).strip(), "segments": out_segments}
    except Exception as e:
        raise RuntimeError(f"Faster-Whisper transcription failed: {e}")


# ------------------------ Object Detection (YOLO) ------------------------

def _get_yolo_model(model_name: str = "yolov8n.pt"):
    """Load and cache YOLO model for object detection."""
    global _YOLO_MODEL_CACHE
    if _YOLO_MODEL_CACHE is not None:
        return _YOLO_MODEL_CACHE
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics is not installed. Please `pip install ultralytics`.")
    try:
        model = YOLO(model_name)
        _YOLO_MODEL_CACHE = model
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load YOLO model '{model_name}': {e}")


def detect_objects(image_path: str, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25) -> List[Dict[str, object]]:
    """Detect objects in an image using YOLO.
    
    Returns:
        List of dicts with keys: 'class', 'confidence', 'bbox' (x1, y1, x2, y2)
    """
    try:
        model = _get_yolo_model(model_name)
        results = model(image_path, conf=conf_threshold, verbose=False)
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    })
        return detections
    except Exception as e:
        console.print(f"[yellow]Object detection failed: {e}[/yellow]")
        return []


# ------------------------ OCR (EasyOCR) ------------------------

def _get_ocr_reader(languages: List[str] = ["vi", "en"]):
    """Load and cache EasyOCR reader."""
    global _OCR_READER_CACHE
    cache_key = ",".join(sorted(languages))
    if _OCR_READER_CACHE and _OCR_READER_CACHE.get("key") == cache_key:
        return _OCR_READER_CACHE["reader"]
    try:
        import easyocr
    except ImportError:
        raise RuntimeError("easyocr is not installed. Please `pip install easyocr`.")
    try:
        # Check if GPU is available for EasyOCR
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
        reader = easyocr.Reader(languages, gpu=use_gpu)
        _OCR_READER_CACHE = {"key": cache_key, "reader": reader}
        return reader
    except Exception as e:
        raise RuntimeError(f"Cannot load EasyOCR reader: {e}")


def extract_text_from_image(image_path: str, languages: List[str] = ["vi", "en"]) -> List[Dict[str, object]]:
    """Extract text from image using OCR.
    
    Returns:
        List of dicts with keys: 'text', 'confidence', 'bbox' (x1, y1, x2, y2)
    """
    try:
        reader = _get_ocr_reader(languages)
        results = reader.readtext(image_path)
        ocr_texts = []
        for (bbox, text, conf) in results:
            # bbox is list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Convert to [x1, y1, x2, y2] format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            ocr_texts.append({
                "text": text.strip(),
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            })
        return ocr_texts
    except Exception as e:
        console.print(f"[yellow]OCR failed: {e}[/yellow]")
        return []


# ------------------------ Local HF captioning ------------------------
# Logic captioning đã được chuyển sang image_captioning.py


# ------------------------ Text classifier (HF) ------------------------

def _get_text_classifier(model_dir: str):
    global _TEXT_CLASSIFIER_CACHE
    
    # Validate model directory exists
    if not os.path.isdir(model_dir):
        # Clear cache if path is invalid
        if _TEXT_CLASSIFIER_CACHE:
            _TEXT_CLASSIFIER_CACHE = None
        raise RuntimeError(
            f"Classifier model directory does not exist: {model_dir}\n"
            f"Please check the path and ensure it points to a valid model directory."
        )
    
    # Check for required model files
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        # Clear cache if model files are missing
        if _TEXT_CLASSIFIER_CACHE:
            _TEXT_CLASSIFIER_CACHE = None
        raise RuntimeError(
            f"Classifier model directory is invalid: {model_dir}\n"
            f"Missing required file: config.json\n"
            f"Please ensure the directory contains a valid Hugging Face model."
        )
    
    # Check cache - only use if path matches exactly
    if _TEXT_CLASSIFIER_CACHE and _TEXT_CLASSIFIER_CACHE.get("dir") == model_dir:
        # Verify cached model directory still exists
        cached_dir = _TEXT_CLASSIFIER_CACHE.get("dir")
        if cached_dir and os.path.isdir(cached_dir) and os.path.exists(os.path.join(cached_dir, "config.json")):
            return _TEXT_CLASSIFIER_CACHE["tokenizer"], _TEXT_CLASSIFIER_CACHE["model"], _TEXT_CLASSIFIER_CACHE["device"]
        else:
            # Cache is stale, clear it
            _TEXT_CLASSIFIER_CACHE = None
    
    # Load model
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")
        console.print(f"[cyan]Loading classifier from: {model_dir}[/cyan]")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        _TEXT_CLASSIFIER_CACHE = {"dir": model_dir, "tokenizer": tokenizer, "model": model, "device": device}
        console.print(f"[green]✓ Classifier loaded successfully[/green]")
        return tokenizer, model, device
    except Exception as e:
        # Clear cache on error
        _TEXT_CLASSIFIER_CACHE = None
        raise RuntimeError(f"Cannot load classifier from: {model_dir}. {e}")


def _convert_to_svo_format(text: str) -> str:
    """Convert caption text to SVO format for classification.
    
    Args:
        text: Full caption text
        
    Returns:
        Formatted string in "subject verb object" format for model input
    """
    if not text or not extract_svo:
        return text
    
    try:
        s, v, o = extract_svo(text)
        # Format as "subject verb object" (same format as model training)
        # If any part is empty, use original text as fallback
        if s or v or o:
            # Join non-empty parts with spaces
            parts = [p for p in [s, v, o] if p and p.strip()]
            if parts:
                return " ".join(parts)
        return text
    except Exception:
        # If SVO extraction fails, return original text
        return text


def classify_text(text: str, model_dir: str, min_score: Optional[float] = None) -> Dict[str, object]:
    """Classify text and return label. Converts text to SVO format before classification.
    If score is below min_score, label_id will be None.
    
    Args:
        text: Input text to classify (will be converted to SVO format)
        model_dir: Directory containing the classifier model
        min_score: Minimum confidence score threshold (0.0-1.0). If None, uses env var CLASSIFIER_MIN_SCORE or 0.7.
    
    Returns:
        Dict with label_id (None if score < min_score), score, and probs
    """
    if not text:
        return {"label_id": None, "score": 0.0, "probs": []}
    
    # Convert caption to SVO format for model input
    svo_text = _convert_to_svo_format(text)
    
    # Get threshold from parameter or environment variable
    # Default 0.7 means model must be at least 70% confident to assign a label
    if min_score is None:
        min_score = float(os.environ.get("CLASSIFIER_MIN_SCORE", "0.6"))
    
    tokenizer, model, device = _get_text_classifier(model_dir)
    import torch
    with torch.no_grad():
        enc = tokenizer(svo_text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits.detach().cpu().numpy()[0]
    # softmax
    ex = np.exp(logits - np.max(logits))
    probs = (ex / ex.sum()).tolist()
    label_id = int(np.argmax(probs))
    score = float(probs[label_id])
    
    # If score is below threshold, mark as unknown (None)
    if score < min_score:
        label_id = None
    
    return {"label_id": label_id, "score": score, "probs": probs, "svo_text": svo_text}


# ------------------------ Optional translation ------------------------

def translate_text(text: str, target_lang: str) -> str:
    if not text or not target_lang:
        return text
    # Use deep-translator (Google translate) as a simple online translation helper
    try:
        from deep_translator import GoogleTranslator
        dest = target_lang
        # Normalize some common aliases
        if dest.lower() in {"vi", "vn", "vie", "vietnamese", "tiếng việt"}:
            dest = "vi"
        translated = GoogleTranslator(source="auto", target=dest).translate(text)
        return translated or text
    except Exception:
        # If translation fails (no internet / quota), return original
        return text


# ------------------------ CLI ------------------------

@click.command()
@click.option("--input", "input_video", required=False, type=str, help="Path to input video file or URL (http/https)")
@click.option("--out_dir", default="output", type=click.Path(dir_okay=True, file_okay=False), help="Directory to write frames and captions")
@click.option("--hf_model", default=None, help="Path to fine-tuned LoRA adapter (optional). If not provided, uses hardcoded path in image_captioning.py")
@click.option(
    "--caption_backend",
    type=click.Choice(["local", "openai"]),
    default="local",
    help="Backend used to generate frame captions (local Hugging Face or OpenAI).",
)
@click.option("--openai_key", default=None, help="OpenAI API key (fallback to openai_config.py or OPENAI_API_KEY env).")
@click.option("--openai_model", default=None, help="OpenAI vision model (overrides openai_config.py).")
@click.option("--openai_prompt", default=None, help="Custom prompt (overrides openai_config.py).")
@click.option("--openai_temperature", default=None, type=float, help="Temperature (overrides openai_config.py).")
@click.option("--openai_max_tokens", default=None, type=int, help="Max tokens (overrides openai_config.py).")
@click.option("--clf_dir", default=r"D:\Study\TVC-AI\output_moderation", help="Directory of trained text classifier to label captions. Use --list-models to see available models.")
@click.option("--list-models", is_flag=True, default=False, help="List all available classifier models and exit")
@click.option("--language", default="vi", help="Preferred caption language (e.g., vi, en)")
@click.option("--translate", is_flag=True, default=False, help="Translate captions to the target language if backend returns other language")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing frame files")
@click.option("--serve/--no-serve", "serve_web", default=True, help="Also start the web UI server (default: on)")
@click.option("--audio/--no-audio", "process_audio", default=True, help="Also transcribe audio to text and label")
@click.option("--asr_model", default="vinai/phowhisper-large", help="Faster-Whisper ASR model (e.g., vinai/phowhisper-large, small, medium, large)")
@click.option("--detect-objects/--no-detect-objects", "use_object_detection", default=False, help="Use YOLO object detection to enhance captions (default: off)")
@click.option("--yolo-model", default="yolov8n.pt", help="YOLO model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
@click.option("--auto-caption/--no-auto-caption", "auto_caption", default=False, help="Automatically generate captions in CLI mode (default: off)")
def main(
    input_video: str,
    out_dir: str,
    hf_model: str,
    caption_backend: str,
    openai_key: Optional[str],
    openai_model: Optional[str],
    openai_prompt: Optional[str],
    openai_temperature: Optional[float],
    openai_max_tokens: Optional[int],
    language: str,
    translate: bool,
    overwrite: bool,
    serve_web: bool,
    clf_dir: str,
    process_audio: bool,
    asr_model: str,
    use_object_detection: bool,
    yolo_model: str,
    auto_caption: bool,
    list_models: bool,
) -> None:
    # Handle --list-models flag
    if list_models:
        tvc_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI"))
        console.print(f"[cyan]Available classifier models in {tvc_ai_dir}:[/cyan]")
        models_found = False
        if os.path.isdir(tvc_ai_dir):
            for item in sorted(os.listdir(tvc_ai_dir)):
                if item.startswith("output_moderation"):
                    model_dir = os.path.join(tvc_ai_dir, item)
                    config_file = os.path.join(model_dir, "config.json")
                    if os.path.isdir(model_dir) and os.path.exists(config_file):
                        models_found = True
                        console.print(f"  [green]✓[/green] {item}")
                        console.print(f"    Path: {model_dir}")
        if not models_found:
            console.print(f"[yellow]No classifier models found in {tvc_ai_dir}[/yellow]")
        sys.exit(0)
    
    console.rule("Video to Step Images + Captions (1 fps)")

    # Only validate classifier directory if auto_caption is enabled (it will be used)
    # If not needed, just warn but don't exit
    if auto_caption:
        if not os.path.isdir(clf_dir):
            console.print(f"[yellow]Warning: Classifier directory does not exist: {clf_dir}[/yellow]")
            console.print(f"[yellow]Classification will be skipped. To enable, set --clf_dir to a valid model directory.[/yellow]")
            clf_dir = None  # Mark as unavailable
        elif not os.path.exists(os.path.join(clf_dir, "config.json")):
            console.print(f"[yellow]Warning: Invalid classifier directory: {clf_dir}[/yellow]")
            console.print(f"[yellow]Classification will be skipped. Missing required file: config.json[/yellow]")
            clf_dir = None  # Mark as unavailable
        else:
            # Clear cache if path changed from previous run
            global _TEXT_CLASSIFIER_CACHE
            if _TEXT_CLASSIFIER_CACHE and _TEXT_CLASSIFIER_CACHE.get("dir") != clf_dir:
                _TEXT_CLASSIFIER_CACHE = None
                console.print(f"[yellow]Classifier path changed, clearing cache[/yellow]")
            console.print(f"[green]Using classifier from: {clf_dir}[/green]")
    else:
        console.print(f"[blue]Classifier not needed (auto_caption is off). Classification will be skipped.[/blue]")

    caption_backend = (caption_backend or "local").lower()
    openai_config: Optional[OpenAICaptionConfig] = None
    if caption_backend == "openai":
        try:
            openai_config = build_openai_caption_config(
                api_key=openai_key,  # Can be None, will load from config file or env
                model=openai_model,  # Can be None, will load from config file
                prompt=openai_prompt,  # Can be None, will load from config file
                temperature=openai_temperature,  # None means use config file default
                max_tokens=openai_max_tokens,  # None means use config file default
                target_language=language or None,  # Can be None, will load from config file
            )
            console.print(f"[cyan]Loaded OpenAI config from openai_config.py[/cyan]")
            if openai_config.prompt:
                prompt_preview = openai_config.prompt[:60] + "..." if len(openai_config.prompt) > 60 else openai_config.prompt
                console.print(f"[dim]Prompt: {prompt_preview}[/dim]")
        except ValueError as e:
            console.print(f"[red]OpenAI backend selected but configuration failed: {e}[/red]")
            console.print("[yellow]Set OPENAI_API_KEY in openai_config.py, environment variable, or pass --openai_key[/yellow]")
            sys.exit(1)

    web_proc = None
    if serve_web:
        try:
            web_proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "web_app.py")])
            console.print("[cyan]Web UI đang chạy tại[/cyan] http://localhost:5000")
            time.sleep(1.0)
        except Exception as e:
            console.print(f"[yellow]Không thể khởi động web UI:[/yellow] {e}")
            web_proc = None

    # Nếu có video đầu vào thì chạy xử lý CLI, ngược lại chỉ chạy web và giữ tiến trình
    if not input_video and serve_web:
        console.print("[cyan]Chạy ở chế độ Web UI. Để xử lý video, vui lòng chỉ định --input <video_path_or_url>[/cyan]")
        console.print("[yellow]Ví dụ: python app.py --input video.mp4[/yellow]")
        console.print("[yellow]Hoặc: python app.py --input https://example.com/video.mp4[/yellow]")
    
    video_path = None
    is_temp_file = False
    temp_video_path = None
    
    if input_video:
        try:
            # Xử lý input có thể là URL hoặc file path
            video_path, is_temp_file = resolve_video_input(input_video)
            temp_video_path = video_path if is_temp_file else None
        except Exception as e:
            console.print(f"[red]Lỗi khi xử lý video input: {e}[/red]")
            if web_proc:
                web_proc.terminate()
            sys.exit(1)
        
        frames_dir = os.path.join(out_dir, "frames")
        ensure_dir(out_dir)
        if overwrite and os.path.isdir(frames_dir):
            for name in os.listdir(frames_dir):
                try:
                    os.remove(os.path.join(frames_dir, name))
                except Exception:
                    pass
        ensure_dir(frames_dir)
        # 1) Extract frames at 1 fps
        frame_paths = extract_frames_1fps(video_path, frames_dir)
        console.print(f"[green]Extracted {len(frame_paths)} frame(s) into {frames_dir}[/green]")
        # 2) Caption each frame (Local HF + Object Detection + OCR) - optional
        results: List[CaptionResult] = []
        labels: List[Dict[str, object]] = []
        if auto_caption:
            # Set YOLO model in environment for caption_image_local to use
            if use_object_detection:
                os.environ["YOLO_MODEL"] = yolo_model
            
            # Nhóm frames thành nhóm 3 và tổng hợp caption
            frame_groups = []
            for i in range(0, len(frame_paths), 3):
                group = frame_paths[i:i+3]
                frame_groups.append(group)
            
            track_desc = f"Captioning frame groups ({caption_backend})"
            for group_idx, group in track(enumerate(frame_groups), description=track_desc):
                # Tạo caption cho từng frame trong nhóm
                group_captions = []
                group_seconds = []
                group_image_paths = []
                
                for image_path in group:
                    base = os.path.basename(image_path)
                    try:
                        sec = int(os.path.splitext(base)[0].split("_")[-1])
                    except Exception:
                        sec = len(results)
                    
                    # Step 1: Caption từng frame bằng backend tương ứng
                    if caption_backend == "openai" and openai_config:
                        caption_text = caption_image_openai(
                            image_path,
                            model_name=openai_config.model,
                            api_key=openai_config.api_key,
                            target_language=openai_config.target_language,
                            prompt=openai_config.prompt,
                            temperature=openai_config.temperature,
                            max_tokens=openai_config.max_tokens,
                            console=console,
                        )
                    else:
                        caption_text = caption_image_local(
                            image_path, 
                            hf_model, 
                            use_object_detection=use_object_detection,
                            detect_objects_func=detect_objects,
                            console=console
                        )
                    
                    group_captions.append(caption_text if caption_text else "")
                    group_seconds.append(sec)
                    group_image_paths.append(image_path)
                
                # Tổng hợp 3 captions thành 1 caption
                synthesized_caption = synthesize_captions(group_captions, console=console) if group_captions else ""
                
                # Nếu dùng OpenAI backend và có audio, kết hợp audio với frame captions
                if caption_backend == "openai" and openai_config and process_audio:
                    # Audio sẽ được xử lý sau khi extract, nhưng ta cần lưu synthesized_caption để kết hợp sau
                    # Tạm thời giữ synthesized_caption, sẽ được cập nhật sau khi có audio transcript
                    final_caption = synthesized_caption
                else:
                    # Tự động dịch sang tiếng Việt
                    if synthesized_caption:
                        if translate and caption_backend != "openai":
                            try:
                                final_caption = translate_text(synthesized_caption, "vi")
                            except Exception:
                                final_caption = synthesized_caption
                        else:
                            final_caption = synthesized_caption
                    else:
                        final_caption = ""
                
                # Lưu caption tổng hợp cho tất cả frames trong nhóm
                for sec, image_path in zip(group_seconds, group_image_paths):
                    results.append(CaptionResult(
                        second=sec, 
                        image_path=image_path, 
                        caption=final_caption,
                        caption_original=synthesized_caption if synthesized_caption else None
                    ))
                    
                    # Step 2: BERT classify (only if clf_dir is available)
                    if clf_dir:
                        try:
                            pred = classify_text(final_caption, clf_dir)
                        except Exception as e:
                            pred = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
                        labels.append(pred)
                    else:
                        labels.append({"label_id": None, "score": 0.0, "probs": []})
        else:
            # If not auto-captioning, just prepare results with empty captions for export/markdown later
            for image_path in frame_paths:
                base = os.path.basename(image_path)
                try:
                    sec = int(os.path.splitext(base)[0].split("_")[-1])
                except Exception:
                    sec = len(results)
                results.append(CaptionResult(second=sec, image_path=image_path, caption=""))
        # 3) (Optional) Transcribe audio and combine with frame captions (if OpenAI backend)
        audio_summary = None
        audio_transcript = None
        audio_wav_path = os.path.join(out_dir, "audio.wav")
        audio_json_path = os.path.join(out_dir, "audio_transcript.json")
        audio_text_for_combination = None  # Lưu audio text để kết hợp với frame captions
        if process_audio:
            try:
                extract_audio_wav(video_path, audio_wav_path, sample_rate=16000)
                
                # Nếu dùng OpenAI backend, dùng OpenAI Whisper API
                if caption_backend == "openai" and openai_config:
                    asr_lang = language if language else "vi"
                    # Lấy audio prompt từ config nếu có
                    audio_prompt = openai_config.__dict__.get("audio_prompt") if hasattr(openai_config, "__dict__") else None
                    if not audio_prompt:
                        try:
                            from openai_config import get_openai_config
                            config_dict = get_openai_config()
                            audio_prompt = config_dict.get("audio_prompt", "Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm")
                        except ImportError:
                            audio_prompt = "Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm"
                    asr_result = transcribe_audio_openai_whisper(
                        audio_wav_path,
                        api_key=openai_config.api_key,
                        language=asr_lang,
                        prompt=audio_prompt,
                        console=console,
                    )
                    asr_model_used = "whisper-1"
                else:
                    # Use faster-whisper with PhoWhisper Large for better Vietnamese support
                    asr_model_used = asr_model if asr_model else "vinai/phowhisper-large"
                    asr_lang = language if language else "vi"
                    asr_result = transcribe_audio_faster_whisper(
                        audio_wav_path, 
                        model_name=asr_model_used, 
                        language=asr_lang,
                        initial_prompt="Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm"
                    )
                
                # Get full text as single line (no segments)
                full_text = asr_result.get("text", "").strip()
                audio_text_for_combination = full_text  # Lưu để kết hợp với frame captions
                
                # Classify the entire audio text as one unit (only if clf_dir is available)
                label_id = None
                label_score = 0.0
                if full_text and clf_dir:
                    try:
                        pred = classify_text(full_text, clf_dir)
                        label_id = pred.get("label_id")
                        label_score = pred.get("score", 0.0)
                    except Exception as e:
                        console.print(f"[yellow]Audio classification failed: {e}[/yellow]")
                
                # Create summary with single classification result
                num_ok = 1 if label_id == 0 else 0
                num_violate = 1 if label_id == 1 else 0
                num_unknown = 1 if label_id is None else 0
                
                audio_summary = {
                    "total_segments": 1,
                    "num_compliant": num_ok,
                    "num_violations": num_violate,
                    "num_unknown": num_unknown,
                    "percent_compliant": 100.0 if num_ok else 0.0,
                    "percent_violations": 100.0 if num_violate else 0.0,
                }
                audio_transcript = {
                    "text": full_text,
                    "label_id": label_id,
                    "label_score": label_score,
                    "model_used": asr_model_used,
                }
                
                # Nếu dùng OpenAI backend và có audio text, kết hợp với frame captions
                if caption_backend == "openai" and openai_config and audio_text_for_combination and auto_caption:
                    console.print(f"[cyan]Đang kết hợp audio transcript với frame captions...[/cyan]")
                    # Cập nhật captions trong results bằng cách kết hợp với audio
                    for result in results:
                        if result.caption:  # Chỉ cập nhật nếu đã có caption
                            # Lấy frame caption
                            frame_captions_to_combine = [result.caption]
                            
                            # Kết hợp audio với frame caption
                            try:
                                combined_caption = combine_audio_and_frame_captions(
                                    audio_text=audio_text_for_combination,
                                    frame_captions=frame_captions_to_combine,
                                    api_key=openai_config.api_key,
                                    model=openai_config.model,
                                    target_language=openai_config.target_language,
                                    console=console,
                                )
                                if combined_caption:
                                    # Cập nhật caption trong result
                                    result.caption = combined_caption
                            except Exception as e:
                                console.print(f"[yellow]Không thể kết hợp audio với frame caption: {e}[/yellow]")
                                # Giữ nguyên caption gốc
                    console.print(f"[green]✓ Đã kết hợp audio với tất cả frame captions[/green]")
            except Exception as e:
                console.print(f"[yellow]Audio processing skipped:[/yellow] {e}")
        
        # Compute SVO per caption for display/export
        svos: List[Tuple[str, str, str]] = []
        for r in results:
            if extract_svo:
                try:
                    s, v, o = extract_svo(r.caption)
                    s = (s or '').strip()
                    v = (v or '').strip()
                    o = (o or '').strip()
                    svos.append((s, v, o))
                except Exception:
                    svos.append((r.caption, '', ''))
            else:
                svos.append((r.caption, '', ''))
        # Xuất kết quả ra CSV, Markdown
        import pandas as pd
        df = pd.DataFrame([
            {
                "second": r.second,
                "image_path": os.path.relpath(r.image_path, out_dir),
                "caption": r.caption,
                "caption_original": r.caption_original if hasattr(r, 'caption_original') else None,
                "s": (svos[i][0] if i < len(svos) else ''),
                "v": (svos[i][1] if i < len(svos) else ''),
                "o": (svos[i][2] if i < len(svos) else ''),
                "svo_format": (labels[i].get("svo_text", r.caption) if i < len(labels) else r.caption),
                "label_id": (labels[i].get("label_id") if i < len(labels) else None),
                "label_score": (labels[i].get("score") if i < len(labels) else None),
            }
            for i, r in enumerate(results)
        ]).sort_values("second")
        # Summary: assume label_id 0 = compliant, 1 = violating; others/None = unknown
        total_frames = len(df)
        num_violate = int((df["label_id"] == 1).sum()) if total_frames > 0 and "label_id" in df.columns else 0
        num_ok = int((df["label_id"] == 0).sum()) if total_frames > 0 and "label_id" in df.columns else 0
        num_unknown = total_frames - num_violate - num_ok
        pct_violate = (num_violate / total_frames * 100.0) if total_frames > 0 else 0.0
        pct_ok = (num_ok / total_frames * 100.0) if total_frames > 0 else 0.0
        summary = {
            "total_frames": total_frames,
            "num_compliant": num_ok,
            "num_violations": num_violate,
            "num_unknown": num_unknown,
            "percent_compliant": round(pct_ok, 2),
            "percent_violations": round(pct_violate, 2),
        }
        csv_path = os.path.join(out_dir, "captions.csv")
        md_path = os.path.join(out_dir, "captions.md")
        summary_path = os.path.join(out_dir, "summary.json")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        # Write summary JSON
        try:
            merged_summary = {"frames": summary}
            if audio_summary is not None:
                merged_summary["audio"] = audio_summary
            with open(summary_path, "w", encoding="utf-8") as fs:
                json.dump(merged_summary, fs, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # Write audio transcript if available
        if audio_transcript is not None:
            try:
                with open(audio_json_path, "w", encoding="utf-8") as fa:
                    json.dump(audio_transcript, fa, ensure_ascii=False, indent=2)
            except Exception:
                pass
        with open(md_path, "w", encoding="utf-8") as fmd:
            # Summary header
            fmd.write(f"# Kết quả tổng quát\n\n")
            fmd.write(f"## Hình ảnh\n\n")
            fmd.write(f"- Tổng khung hình: {total_frames}\n\n")
            fmd.write(f"- Khung hình đạt chuẩn: {num_ok} ({pct_ok:.2f}%)\n\n")
            fmd.write(f"- Khung hình vi phạm: {num_violate} ({pct_violate:.2f}%)\n\n")
            if num_unknown > 0:
                fmd.write(f"- Không xác định: {num_unknown}\n\n")
            if audio_summary is not None:
                fmd.write(f"## Âm thanh\n\n")
                if audio_transcript and audio_transcript.get("text"):
                    fmd.write(f"**Nội dung:** {audio_transcript.get('text')}\n\n")
                fmd.write(f"- Nhãn: {audio_transcript.get('label_id') if audio_transcript else 'N/A'} (score: {audio_transcript.get('label_score', 0.0):.4f if audio_transcript else 0.0})\n\n")
                fmd.write(f"- Đạt chuẩn: {audio_summary['num_compliant']} ({audio_summary['percent_compliant']:.2f}%)\n\n")
                fmd.write(f"- Vi phạm: {audio_summary['num_violations']} ({audio_summary['percent_violations']:.2f}%)\n\n")
                if audio_summary.get("num_unknown", 0) > 0:
                    fmd.write(f"- Không xác định: {audio_summary['num_unknown']}\n\n")
            fmd.write(f"# Mô tả từng bước (1 giây/ảnh)\n\n")
            for _, row in df.iterrows():
                fmd.write(f"## Giây {int(row.second)}\n\n")
                path_md = row.image_path.replace("\\", "/")
                fmd.write(f"![frame](./{path_md})\n\n")
                # Show SVO as the description
                s_disp = (row.s if isinstance(row.s, str) else str(row.s))
                v_disp = (row.v if isinstance(row.v, str) else str(row.v))
                o_disp = (row.o if isinstance(row.o, str) else str(row.o))
                fmd.write(f"- SVO: {s_disp}, {v_disp}, {o_disp}\n")
                # Include caption (đã dịch sang tiếng Việt)
                fmd.write(f"- Mô tả (tiếng Việt): {row.caption}\n")
                # Optionally include original caption if different
                if 'caption_original' in df.columns:
                    caption_orig = row['caption_original']
                    if pd.notna(caption_orig) and str(caption_orig).strip() and caption_orig != row.caption:
                        fmd.write(f"- Mô tả gốc: {caption_orig}\n")
                fmd.write(f"- Nhãn: {row.label_id} (score: {row.label_score})\n\n")
        console.print(
            "[bold green]Done[/bold green]. Results written to: "
            f"\n- {csv_path}\n- {md_path}\n- {summary_path}\n"
            + (f"- {audio_json_path}\n" if audio_transcript is not None else "")
        )
        
        # Xóa file video tạm nếu là file tải từ URL
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                console.print(f"[green]✓ Đã xóa file video tạm[/green]")
            except Exception as e:
                console.print(f"[yellow]Không thể xóa file tạm {temp_video_path}: {e}[/yellow]")
        
        if web_proc is not None:
            try:
                web_proc.wait()
            except KeyboardInterrupt:
                pass
    else:
        if web_proc is not None:
            try:
                web_proc.wait()
            except KeyboardInterrupt:
                pass
        else:
            console.print("[yellow]Không có --input và không khởi động được web. Thoát.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
