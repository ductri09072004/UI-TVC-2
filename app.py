import os
import sys
import io
import base64
import json
import math
import unicodedata
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import numpy as np
from PIL import Image
import click
import pandas as pd
from rich.console import Console
from rich.progress import track
import shutil

# Optional imports guarded at use

console = Console()

# Cache for local HF captioning pipelines to avoid reloading per frame
_CAPTION_PIPELINE_CACHE: Dict[str, object] = {}
_TEXT_CLASSIFIER_CACHE: Optional[Dict[str, object]] = None
_WHISPER_CACHE: Optional[Dict[str, object]] = None
_FASTER_WHISPER_CACHE: Optional[Dict[str, object]] = None
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


# ------------------------ Local HF captioning ------------------------

def _get_local_captioner(model_name: str):
    """Create or fetch a cached HF pipeline for image captioning."""
    from transformers import pipeline
    try:
        import torch  # prefer GPU if available
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    default_model = "Salesforce/instructblip-flan-t5-xl"
    use_model = model_name or default_model

    if use_model in _CAPTION_PIPELINE_CACHE:
        return _CAPTION_PIPELINE_CACHE[use_model]

    device = 0 if has_cuda else -1
    # Try safetensors first; if unavailable, fallback to default loader
    try:
        captioner = pipeline(
            "image-to-text",
            model=use_model,
            device=device,
            model_kwargs={"use_safetensors": True},
        )
    except Exception:
        captioner = pipeline(
            "image-to-text",
            model=use_model,
            device=device,
        )
    _CAPTION_PIPELINE_CACHE[use_model] = captioner
    return captioner


def caption_image_local(image_path: str, model_name: str) -> str:
    """Caption an image using a local Hugging Face pipeline model (cached)."""
    captioner = _get_local_captioner(model_name)
    # Instruction prompt to steer model (Vietnamese)
    prompt = os.environ.get(
        "CAPTION_PROMPT_VI",
        "Hãy mô tả chính xác, ngắn gọn nội dung bức ảnh (nêu vật thể/nhãn hiệu nếu rõ).",
    )
    # Decoding params
    num_beams = int(os.environ.get("CAPTION_NUM_BEAMS", "10"))
    temperature = float(os.environ.get("CAPTION_TEMPERATURE", "0.3"))
    max_new_tokens = int(os.environ.get("CAPTION_MAX_TOKENS", "100"))
    try:
        result = captioner(
            image_path,
            prompt=prompt,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    except TypeError:
        # Fallback if backend ignores prompt/params
        result = captioner(image_path, max_new_tokens=max_new_tokens)
    # result like: [{"generated_text": "..."}]
    if isinstance(result, list) and result:
        item = result[0]
        text = item.get("generated_text", "").strip()
        return text
    return ""


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


def classify_text(text: str, model_dir: str) -> Dict[str, object]:
    if not text:
        return {"label_id": None, "score": 0.0, "probs": []}
    tokenizer, model, device = _get_text_classifier(model_dir)
    import torch
    with torch.no_grad():
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits.detach().cpu().numpy()[0]
    # softmax
    ex = np.exp(logits - np.max(logits))
    probs = (ex / ex.sum()).tolist()
    label_id = int(np.argmax(probs))
    score = float(probs[label_id])
    return {"label_id": label_id, "score": score, "probs": probs}


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
@click.option("--hf_model", default="Salesforce/blip2-flan-t5-xl", help="Hugging Face model id for local backend")
@click.option("--clf_dir", default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation")), help="Directory of trained text classifier to label captions")
@click.option("--language", default="vi", help="Preferred caption language (e.g., vi, en)")
@click.option("--translate", is_flag=True, default=True, help="Translate captions to the target language if backend returns other language")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing frame files")
@click.option("--serve/--no-serve", "serve_web", default=True, help="Also start the web UI server (default: on)")
@click.option("--audio/--no-audio", "process_audio", default=True, help="Also transcribe audio to text and label")
@click.option("--asr_model", default="base", help="Whisper ASR model size (tiny, base, small, medium, large)")
def main(input_video: str, out_dir: str, hf_model: str, language: str, translate: bool, overwrite: bool, serve_web: bool, clf_dir: str, process_audio: bool, asr_model: str) -> None:
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
        # 2) Caption each frame (Local HF only)
        results: List[CaptionResult] = []
        labels: List[Dict[str, object]] = []
        for image_path in track(frame_paths, description="Captioning frames"):
            base = os.path.basename(image_path)
            try:
                sec = int(os.path.splitext(base)[0].split("_")[-1])
            except Exception:
                sec = len(results)
            caption_en = caption_image_local(image_path, hf_model)
            final_caption = caption_en
            if translate and language and language.lower() not in {"en", "english"}:
                final_caption = translate_text(caption_en, language)
            results.append(CaptionResult(second=sec, image_path=image_path, caption=final_caption))
            try:
                pred = classify_text(final_caption, clf_dir)
            except Exception as e:
                pred = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
            labels.append(pred)
        # Xuất kết quả ra CSV, Markdown
        import pandas as pd
        df = pd.DataFrame([
            {
                "second": r.second,
                "image_path": os.path.relpath(r.image_path, out_dir),
                "caption": r.caption,
                "label_id": (labels[i].get("label_id") if i < len(labels) else None),
                "label_score": (labels[i].get("score") if i < len(labels) else None),
            }
            for i, r in enumerate(results)
        ]).sort_values("second")
        # Summary: assume label_id 0 = compliant, 1 = violating; others/None = unknown
        total_frames = len(df)
        num_violate = int((df["label_id"] == 1).sum()) if total_frames > 0 else 0
        num_ok = int((df["label_id"] == 0).sum()) if total_frames > 0 else 0
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
                fmd.write(f"- Mô tả: {row.caption}\n")
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
