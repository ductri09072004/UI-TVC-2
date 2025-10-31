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

# Optional imports guarded at use

console = Console()

# Cache for local HF captioning pipelines to avoid reloading per frame
_CAPTION_PIPELINE_CACHE: Dict[str, object] = {}
_TEXT_CLASSIFIER_CACHE: Optional[Dict[str, object]] = None


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


# ------------------------ Local HF captioning ------------------------

def _get_local_captioner(model_name: str):
    """Create or fetch a cached HF pipeline for image captioning."""
    from transformers import pipeline
    try:
        import torch  # prefer GPU if available
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    default_model = "Salesforce/blip-image-captioning-base"
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
    result = captioner(image_path, max_new_tokens=64)
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
@click.option("--hf_model", default="Salesforce/blip-image-captioning-base", help="Hugging Face model id for local backend")
@click.option("--clf_dir", default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation")), help="Directory of trained text classifier to label captions")
@click.option("--language", default="vi", help="Preferred caption language (e.g., vi, en)")
@click.option("--translate", is_flag=True, default=True, help="Translate captions to the target language if backend returns other language")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing frame files")
@click.option("--serve/--no-serve", "serve_web", default=True, help="Also start the web UI server (default: on)")
def main(input_video: str, out_dir: str, hf_model: str, language: str, translate: bool, overwrite: bool, serve_web: bool, clf_dir: str) -> None:
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
        csv_path = os.path.join(out_dir, "captions.csv")
        md_path = os.path.join(out_dir, "captions.md")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        with open(md_path, "w", encoding="utf-8") as fmd:
            fmd.write(f"# Mô tả từng bước (1 giây/ảnh)\n\n")
            for _, row in df.iterrows():
                fmd.write(f"## Giây {int(row.second)}\n\n")
                path_md = row.image_path.replace("\\", "/")
                fmd.write(f"![frame](./{path_md})\n\n")
                fmd.write(f"- Mô tả: {row.caption}\n")
                fmd.write(f"- Nhãn: {row.label_id} (score: {row.label_score})\n\n")
        console.print(f"[bold green]Done[/bold green]. Results written to: \n- {csv_path}\n- {md_path}")
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
