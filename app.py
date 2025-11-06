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
from image_captioning import caption_image_local

# Import CLIP moderation module (optional, will fail gracefully if not installed)
try:
    from clip_moderation import pre_filter_image, verify_caption_quality, classify_image_zeroshot
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    pre_filter_image = None
    verify_caption_quality = None
    classify_image_zeroshot = None
finally:
    try:
        if CLIP_AVAILABLE:
            console.print("[green]CLIP moderation: enabled[/green]")
        else:
            console.print("[yellow]CLIP moderation: not available (skipping CLIP steps)[/yellow]")
    except Exception:
        pass
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    if _TEXT_CLASSIFIER_CACHE and _TEXT_CLASSIFIER_CACHE.get("dir") == model_dir:
        return _TEXT_CLASSIFIER_CACHE["tokenizer"], _TEXT_CLASSIFIER_CACHE["model"], _TEXT_CLASSIFIER_CACHE["device"]
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        has_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if has_cuda else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        _TEXT_CLASSIFIER_CACHE = {"dir": model_dir, "tokenizer": tokenizer, "model": model, "device": device}
        return tokenizer, model, device
    except Exception as e:
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
        min_score = float(os.environ.get("CLASSIFIER_MIN_SCORE", "0.7"))
    
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
@click.option("--input", "input_video", required=False, type=click.Path(exists=True, dir_okay=False), help="Path to input video")
@click.option("--out_dir", default="output", type=click.Path(dir_okay=True, file_okay=False), help="Directory to write frames and captions")
@click.option("--hf_model", default="Salesforce/blip-image-captioning-base", help="Hugging Face model id for local backend")
@click.option("--clf_dir", default=r"C:\Users\Kris\TVC-AI\output_moderation", help="Directory of trained text classifier to label captions")
@click.option("--language", default="en", help="Preferred caption language (e.g., vi, en)")
@click.option("--translate", is_flag=True, default=False, help="Translate captions to the target language if backend returns other language")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing frame files")
@click.option("--serve/--no-serve", "serve_web", default=True, help="Also start the web UI server (default: on)")
@click.option("--audio/--no-audio", "process_audio", default=True, help="Also transcribe audio to text and label")
@click.option("--asr_model", default="base", help="Whisper ASR model size (tiny, base, small, medium, large)")
@click.option("--detect-objects/--no-detect-objects", "use_object_detection", default=False, help="Use YOLO object detection to enhance captions (default: off)")
@click.option("--yolo-model", default="yolov8n.pt", help="YOLO model name (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
@click.option("--auto-caption/--no-auto-caption", "auto_caption", default=False, help="Automatically generate captions in CLI mode (default: off)")
def main(input_video: str, out_dir: str, hf_model: str, language: str, translate: bool, overwrite: bool, serve_web: bool, clf_dir: str, process_audio: bool, asr_model: str, use_object_detection: bool, yolo_model: str, auto_caption: bool) -> None:
    console.rule("Video to Step Images + Captions (1 fps)")

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
        console.print("[cyan]Chạy ở chế độ Web UI. Để xử lý video, vui lòng chỉ định --input <video_path>[/cyan]")
        console.print("[yellow]Ví dụ: python app.py --input video.mp4[/yellow]")
    if input_video:
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
        frame_paths = extract_frames_1fps(input_video, frames_dir)
        console.print(f"[green]Extracted {len(frame_paths)} frame(s) into {frames_dir}[/green]")
        # 2) Caption each frame (Local HF + Object Detection + OCR) - optional
        results: List[CaptionResult] = []
        labels: List[Dict[str, object]] = []
        if auto_caption:
            # Set YOLO model in environment for caption_image_local to use
            if use_object_detection:
                os.environ["YOLO_MODEL"] = yolo_model
            for image_path in track(frame_paths, description="Captioning frames (with object detection)"):
                base = os.path.basename(image_path)
                try:
                    sec = int(os.path.splitext(base)[0].split("_")[-1])
                except Exception:
                    sec = len(results)
                
                # Step 1: CLIP pre-filter (nhanh, zero-shot)
                clip_result = None
                if CLIP_AVAILABLE and pre_filter_image:
                    try:
                        clip_result = pre_filter_image(
                            image_path,
                            violation_threshold=float(os.environ.get("CLIP_VIOLATION_THRESHOLD", "0.3")),
                            skip_threshold=float(os.environ.get("CLIP_SKIP_THRESHOLD", "0.7"))
                        )
                        # Log CLIP prefilter scores
                        try:
                            console.print(
                                f"[blue]Frame {sec}s: CLIP prefilter → violation={clip_result.get('violation_score', 0):.2f}, "
                                f"healthy={clip_result.get('healthy_score', 0):.2f}[/blue]"
                            )
                        except Exception:
                            pass
                        
                        # Nếu vi phạm rõ ràng, skip BLIP + BERT
                        if clip_result.get("skip_processing", False):
                            console.print(f"[yellow]Frame {sec}s: CLIP detected clear violation, skipping captioning[/yellow]")
                            # Mark as violation
                            pred = {
                                "label_id": 1,  # Assuming 1 = violation
                                "score": clip_result.get("violation_score", 0.0),
                                "probs": [],
                                "method": "clip_prefilter",
                                "clip_result": clip_result
                            }
                            labels.append(pred)
                            # Add placeholder caption
                            results.append(CaptionResult(second=sec, image_path=image_path, caption="[CLIP: Vi phạm rõ ràng]"))
                            continue
                    except Exception as e:
                        console.print(f"[yellow]CLIP pre-filter failed: {e}, continuing with normal flow[/yellow]")
                
                # Step 2: BLIP caption (chỉ khi không skip)
                caption_en = caption_image_local(
                    image_path, 
                    hf_model, 
                    use_object_detection=use_object_detection,
                    detect_objects_func=detect_objects,
                    console=console
                )
                final_caption = caption_en
                if translate and language and language.lower() not in {"en", "english"}:
                    final_caption = translate_text(caption_en, language)
                
                # Step 3: CLIP verify caption quality
                caption_verified = True
                if CLIP_AVAILABLE and verify_caption_quality and final_caption:
                    try:
                        verify_result = verify_caption_quality(
                            image_path,
                            final_caption,
                            threshold=float(os.environ.get("CLIP_VERIFY_THRESHOLD", "0.7"))
                        )
                        caption_verified = verify_result.get("is_valid", True)
                        # Log CLIP verify similarity
                        try:
                            error_msg = verify_result.get('error', '')
                            error_str = f", error={error_msg}" if error_msg else ""
                            console.print(
                                f"[blue]Frame {sec}s: CLIP verify → similarity={verify_result.get('similarity', 0):.2f}, "
                                f"valid={caption_verified}{error_str}[/blue]"
                            )
                        except Exception:
                            pass
                        if not caption_verified:
                            console.print(f"[yellow]Frame {sec}s: Caption quality low (similarity: {verify_result.get('similarity', 0):.2f})[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]CLIP verify failed: {e}[/yellow]")
                
                results.append(CaptionResult(second=sec, image_path=image_path, caption=final_caption))
                
                # Step 4: BERT classify (chỉ khi caption verified hoặc không có CLIP)
                if caption_verified or not CLIP_AVAILABLE:
                    try:
                        pred = classify_text(final_caption, clf_dir)
                        # Add CLIP info if available
                        if clip_result:
                            pred["clip_prefilter"] = clip_result
                        # Optional: CLIP zero-shot frame labels via env CLIP_FRAME_LABELS (pipe-separated)
                        clip_frame_labels = os.environ.get("CLIP_FRAME_LABELS", "").strip()
                        # Fallback to frame_labels.FRAME_LABELS_VI if env not set
                        if not clip_frame_labels:
                            try:
                                from frame_labels import FRAME_LABELS_VI, labels_to_pipe  # type: ignore
                                clip_frame_labels = labels_to_pipe(FRAME_LABELS_VI)
                            except Exception:
                                clip_frame_labels = ""
                        if CLIP_AVAILABLE and classify_image_zeroshot and clip_frame_labels:
                            labels_list = [s.strip() for s in clip_frame_labels.split("|") if s.strip()]
                            if labels_list:
                                try:
                                    cls_res = classify_image_zeroshot(image_path, labels_list)
                                    pred["clip_label"] = cls_res.get("top_label")
                                    pred["clip_label_score"] = cls_res.get("top_score")
                                except Exception:
                                    pass
                    except Exception as e:
                        pred = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
                else:
                    # Caption quality low, mark as unknown
                    pred = {
                        "label_id": None,
                        "score": 0.0,
                        "probs": [],
                        "method": "clip_verify_failed",
                        "note": "Caption quality too low"
                    }
                labels.append(pred)
        else:
            # If not auto-captioning, just prepare results with empty captions for export/markdown later
            for image_path in frame_paths:
                base = os.path.basename(image_path)
                try:
                    sec = int(os.path.splitext(base)[0].split("_")[-1])
                except Exception:
                    sec = len(results)
                results.append(CaptionResult(second=sec, image_path=image_path, caption=""))
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
                "s": (svos[i][0] if i < len(svos) else ''),
                "v": (svos[i][1] if i < len(svos) else ''),
                "o": (svos[i][2] if i < len(svos) else ''),
                "svo_format": (labels[i].get("svo_text", r.caption) if i < len(labels) else r.caption),
                "label_id": (labels[i].get("label_id") if i < len(labels) else None),
                "label_score": (labels[i].get("score") if i < len(labels) else None),
                "clip_label": (labels[i].get("clip_label") if i < len(labels) else None),
                "clip_label_score": (labels[i].get("clip_label_score") if i < len(labels) else None),
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
        # 3) (Optional) Transcribe audio and label segments
        audio_summary = None
        audio_transcript = None
        audio_wav_path = os.path.join(out_dir, "audio.wav")
        audio_json_path = os.path.join(out_dir, "audio_transcript.json")
        if process_audio:
            try:
                extract_audio_wav(input_video, audio_wav_path, sample_rate=16000)
                asr_result = transcribe_audio_whisper(audio_wav_path, model_name=asr_model, language=(language if language else None))
                # asr_result['segments'] with start/end/text
                segments = asr_result.get("segments", []) or []
                labeled_segments = []
                num_seg_ok = 0
                num_seg_violate = 0
                num_seg_unknown = 0
                for seg in segments:
                    text_seg = (seg.get("text") or "").strip()
                    if not text_seg:
                        labeled_segments.append({**seg, "label_id": None, "label_score": 0.0})
                        num_seg_unknown += 1
                        continue
                    try:
                        pred_seg = classify_text(text_seg, clf_dir)
                    except Exception as e:
                        pred_seg = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
                    lid = pred_seg.get("label_id")
                    if lid == 0:
                        num_seg_ok += 1
                    elif lid == 1:
                        num_seg_violate += 1
                    else:
                        num_seg_unknown += 1
                    labeled_segments.append({**seg, "label_id": lid, "label_score": pred_seg.get("score")})
                total_segs = len(labeled_segments)
                pct_seg_ok = round((num_seg_ok / total_segs * 100.0), 2) if total_segs else 0.0
                pct_seg_violate = round((num_seg_violate / total_segs * 100.0), 2) if total_segs else 0.0
                audio_summary = {
                    "total_segments": total_segs,
                    "num_compliant": num_seg_ok,
                    "num_violations": num_seg_violate,
                    "num_unknown": num_seg_unknown,
                    "percent_compliant": pct_seg_ok,
                    "percent_violations": pct_seg_violate,
                }
                audio_transcript = {
                    "text": asr_result.get("text", ""),
                    "segments": labeled_segments,
                }
            except Exception as e:
                console.print(f"[yellow]Audio processing skipped:[/yellow] {e}")
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
                fmd.write(f"- Tổng đoạn thoại: {audio_summary['total_segments']}\n\n")
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
                # Optionally include original caption
                fmd.write(f"- Mô tả gốc: {row.caption}\n")
                fmd.write(f"- Nhãn: {row.label_id} (score: {row.label_score})\n\n")
        console.print(
            "[bold green]Done[/bold green]. Results written to: "
            f"\n- {csv_path}\n- {md_path}\n- {summary_path}\n"
            + (f"- {audio_json_path}\n" if audio_transcript is not None else "")
        )
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
