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
_ZS_CLASSIFIER_CACHE: Optional[object] = None


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
    captioner = pipeline("image-to-text", model=use_model, device=device)
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


# ------------------------ Zero-shot moderation (local, free) ------------------------
def _get_zero_shot_classifier():
    global _ZS_CLASSIFIER_CACHE
    if _ZS_CLASSIFIER_CACHE is not None:
        return _ZS_CLASSIFIER_CACHE
    from transformers import pipeline
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    model_id = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    device = 0 if has_cuda else -1
    _ZS_CLASSIFIER_CACHE = pipeline("zero-shot-classification", model=model_id, device=device)
    return _ZS_CLASSIFIER_CACHE


_MOD_LABELS = [
    "alcohol",
    "nudity",
    "violence",
    "hate",
    "drugs",
    "weapons",
    "self-harm",
    "minors",
]

_LABEL_TO_CATEGORY = {
    "alcohol": "restricted-alcohol",
    "nudity": "sensitive-nudity",
    "violence": "restricted-violence",
    "hate": "unsafe-hate",
    "drugs": "restricted-drugs",
    "weapons": "restricted-weapons",
    "self-harm": "unsafe-self-harm",
    "minors": "protected-minors",
}

# Sentence labels and thresholds (tunable)
_LABEL_SPECS = {
    "alcohol": {"sentence": "involves alcohol.", "warn": 0.70, "block": 0.95},
    "nudity": {"sentence": "contains nudity.", "warn": 0.78, "block": 0.92},
    "violence": {"sentence": "depicts violence.", "warn": 0.80, "block": 0.95},
    "hate": {"sentence": "contains hate speech.", "warn": 0.80, "block": 0.95},
    "drugs": {"sentence": "involves illegal drugs.", "warn": 0.75, "block": 0.92},
    "weapons": {"sentence": "involves weapons.", "warn": 0.80, "block": 0.95},
    "self-harm": {"sentence": "promotes or depicts self-harm.", "warn": 0.80, "block": 0.95},
    "minors": {"sentence": "involves minors.", "warn": 0.75, "block": 0.92},
}

_SAFE_SENTENCE = "is safe and contains no sensitive content."

# Lightweight keyword support to reduce false positives (EN + VI common terms)
_KEYWORDS = {
    "alcohol": ["beer", "wine", "alcohol", "bia", "rượu"],
    "violence": ["fight", "blood", "gunshot", "đánh nhau", "bạo lực"],
    "weapons": ["gun", "knife", "rifle", "dao", "súng"],
    "self-harm": ["suicide", "self-harm", "tự sát", "tự hại"],
    "drugs": ["cocaine", "heroin", "meth", "ma túy", "thuốc phiện"],
    "nudity": ["nude", "naked", "khoả thân", "khỏa thân"],
}

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def moderate_caption(caption: str) -> Dict:
    if not caption:
        return {
            "labels": [],
            "severity": "allow",
            "confidence": 0.0,
            "evidence": [],
            "categories": [],
            "primary_category": None,
            "verdict": "allow",
        }
    clf = _get_zero_shot_classifier()
    # Build sentence-style labels and use hypothesis template
    candidate_sentences = [_LABEL_SPECS[k]["sentence"] for k in _MOD_LABELS]
    out = clf(
        caption,
        candidate_labels=candidate_sentences,
        multi_label=True,
        hypothesis_template="This text {}",
    )
    # Map back to base labels
    label_by_sentence = {v["sentence"]: k for k, v in _LABEL_SPECS.items()}
    preds = []
    for sent, score in zip(out["labels"], out["scores"]):
        base = label_by_sentence.get(sent, sent)
        preds.append((base, float(score)))
    # sort by score desc
    preds.sort(key=lambda x: x[1], reverse=True)
    # Thresholding per label
    violations = []
    for l, s in preds:
        th = _LABEL_SPECS.get(l, {}).get("warn", 0.70)
        if s >= th:
            violations.append((l, s))

    # Map labels to higher-level categories
    categories = [_LABEL_TO_CATEGORY.get(l, l) for l, _ in violations]
    primary_label, primary_score = (preds[0] if preds else (None, 0.0))
    primary_category = _LABEL_TO_CATEGORY.get(primary_label, None) if primary_label else None

    # Keyword check to reduce FP for generic scenes
    txt_norm = _normalize_text(caption)
    matched_keywords = set()
    for l, words in _KEYWORDS.items():
        for w in words:
            if w in txt_norm:
                matched_keywords.add(l)
                break

    # Compute severity with stricter thresholds
    def over_block(l, s):
        return s >= _LABEL_SPECS.get(l, {}).get("block", 0.9)

    def over_warn(l, s):
        return s >= _LABEL_SPECS.get(l, {}).get("warn", 0.7)

    # For hate/violence/weapons/self-harm, require keyword evidence unless score is extremely high
    filtered = []
    for l, s in violations:
        if l in {"hate", "violence", "weapons", "self-harm"}:
            if s >= 0.92 or l in matched_keywords:
                filtered.append((l, s))
            else:
                # drop weak, non-evidenced hits
                continue
        else:
            filtered.append((l, s))

    violations = filtered

    has_block = any(over_block(l, s) for l, s in violations)
    if has_block:
        severity = "block"
        verdict = "block"
    elif violations:
        only_soft = all(l in {"violence", "weapons", "self-harm"} and s < _LABEL_SPECS[l]["block"] for l, s in violations)
        # If only soft labels and no keywords, downgrade to allow
        if only_soft and not any(l in matched_keywords for l, _ in violations):
            severity = "allow"
            verdict = "allow"
        else:
            severity = "warn"
            verdict = "restricted"
    else:
        severity = "allow"
        verdict = "allow"

    return {
        "labels": [l for l, _ in violations],
        "severity": severity,
        "confidence": float(primary_score or 0.0),
        "evidence": [f"{l}:{s:.2f}" for l, s in preds[:3]],
        "categories": categories,
        "primary_category": primary_category,
        "verdict": verdict,
    }


# ------------------------ OpenAI vision captioning ------------------------

def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def caption_image_openai(image_path: str, model: str, language: str) -> str:
    """Caption using OpenAI image understanding (e.g., gpt-4o-mini)."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. Install from requirements.txt") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")

    client = OpenAI(api_key=api_key)
    b64 = _encode_image_to_base64(image_path)

    # Ask for a concise one-sentence caption in target language
    prompt = (
        f"Mô tả ngắn gọn một câu cho hình ảnh này bằng tiếng {('Việt' if language.lower().startswith('vi') else language)}. "
        "Tập trung vào nội dung chính."
    )

    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64}},
                ],
            }
        ],
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()
    return text


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
@click.option("--backend", type=click.Choice(["local", "openai"], case_sensitive=False), default="local", help="Captioning backend")
@click.option("--hf_model", default="Salesforce/blip-image-captioning-base", help="Hugging Face model id for local backend")
@click.option("--openai_model", default="gpt-4o-mini", help="OpenAI model for vision captioning")
@click.option("--language", default="vi", help="Preferred caption language (e.g., vi, en)")
@click.option("--translate", is_flag=True, default=True, help="Translate captions to the target language if backend returns other language")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing frame files")
@click.option("--serve/--no-serve", "serve_web", default=True, help="Also start the web UI server (default: on)")
def main(input_video: str, out_dir: str, backend: str, hf_model: str, openai_model: str, language: str, translate: bool, overwrite: bool, serve_web: bool) -> None:
    console.rule("Video to Step Images + Captions (1 fps)")

    web_proc = None
    if serve_web:
        try:
            web_proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "web_app.py")])
            console.print("[cyan]Web UI đang chạy tại[/cyan] http://localhost:5000")
            # Chờ một chút cho web server khởi động
            time.sleep(1.0)
        except Exception as e:
            console.print(f"[yellow]Không thể khởi động web UI:[/yellow] {e}")
            web_proc = None

    # Nếu có video đầu vào thì chạy xử lý CLI, ngược lại chỉ chạy web và giữ tiến trình
    if input_video:
        frames_dir = os.path.join(out_dir, "frames")
        ensure_dir(out_dir)

        if overwrite and os.path.isdir(frames_dir):
            # Remove existing frames to ensure a clean run
            for name in os.listdir(frames_dir):
                try:
                    os.remove(os.path.join(frames_dir, name))
                except Exception:
                    pass

        ensure_dir(frames_dir)

        # 1) Extract frames at 1 fps
        frame_paths = extract_frames_1fps(input_video, frames_dir)
        console.print(f"[green]Extracted {len(frame_paths)} frame(s) into {frames_dir}[/green]")

        # 2) Caption each frame
        results: List[CaptionResult] = []

        for image_path in track(frame_paths, description="Captioning frames"):
            # Determine second from filename pattern
            base = os.path.basename(image_path)
            try:
                sec = int(os.path.splitext(base)[0].split("_")[-1])
            except Exception:
                sec = len(results)

            if backend.lower() == "local":
                caption_en = caption_image_local(image_path, hf_model)
                final_caption = caption_en
                if translate and language and language.lower() not in {"en", "english"}:
                    final_caption = translate_text(caption_en, language)
            else:
                # Directly ask OpenAI for caption in target language
                final_caption = caption_image_openai(image_path, openai_model, language)

            # moderation
            mod = moderate_caption(final_caption)

            results.append(CaptionResult(second=sec, image_path=image_path, caption=final_caption))

        # 3) Write outputs: CSV, JSONL, Markdown
        df = pd.DataFrame([
            {"second": r.second, "image_path": os.path.relpath(r.image_path, out_dir), "caption": r.caption}
            for r in results
        ]).sort_values("second")

        csv_path = os.path.join(out_dir, "captions.csv")
        jsonl_path = os.path.join(out_dir, "captions.jsonl")
        md_path = os.path.join(out_dir, "captions.md")

        df.to_csv(csv_path, index=False, encoding="utf-8")

        with open(jsonl_path, "w", encoding="utf-8") as fjson:
            # Rebuild with moderation info by recomputing per row or mapping
            sec_to_caption = {r.second: r.caption for r in results}
            for _, row in df.iterrows():
                sec = int(row.second)
                cap = sec_to_caption.get(sec, row.caption)
                mod = moderate_caption(cap)
                fjson.write(json.dumps({
                    "second": sec,
                    "image_path": row.image_path,
                    "caption": cap,
                    "moderation": mod,
                }, ensure_ascii=False) + "\n")

        with open(md_path, "w", encoding="utf-8") as fmd:
            fmd.write(f"# Mô tả từng bước (1 giây/ảnh)\n\n")
            for _, row in df.iterrows():
                fmd.write(f"## Giây {int(row.second)}\n\n")
                fmd.write(f"![frame](./{row.image_path.replace('\\\\', '/')})\n\n")
                fmd.write(f"- Mô tả: {row.caption}\n\n")

        console.print(f"[bold green]Done[/bold green]. Results written to: \n- {csv_path}\n- {jsonl_path}\n- {md_path}")

        # Nếu web server đang chạy, tiếp tục giữ tiến trình mở để dùng song song
        if web_proc is not None:
            try:
                web_proc.wait()
            except KeyboardInterrupt:
                pass
    else:
        # Không có input_video -> chỉ chạy web và chờ
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
