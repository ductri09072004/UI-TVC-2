import os
import uuid
import json
from typing import List, Dict

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort

# Reuse processing utilities from app.py
from app import (
    ensure_dir,
    extract_frames_1fps,
    caption_image_local,
    caption_image_openai,
    translate_text,
    CaptionResult,
    moderate_caption,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")

ensure_dir(UPLOAD_DIR)
ensure_dir(OUTPUT_DIR)

app = Flask(__name__)


def _sanitize_filename(name: str) -> str:
    """Keep basename only and strip unsafe chars; preserve spaces, dots, dashes, underscores."""
    base = os.path.basename(name)
    allowed = []
    for ch in base:
        if ch.isalnum() or ch in " .-_()[]{}":
            allowed.append(ch)
        else:
            allowed.append("_")
    cleaned = "".join(allowed).strip()
    # avoid empty name
    return cleaned or "uploaded_video.mp4"


def _unique_path(directory: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        cand = os.path.join(directory, f"{base}-{idx}{ext}")
        if not os.path.exists(cand):
            return cand
        idx += 1


def process_video(job_id: str, video_path: str, backend: str, language: str, hf_model: str, openai_model: str, translate: bool) -> Dict:
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    frames_dir = os.path.join(out_dir, "frames")
    ensure_dir(out_dir)
    ensure_dir(frames_dir)

    frame_paths = extract_frames_1fps(video_path, frames_dir)

    # Step 1: Save frames only; no captions yet
    data = {
        "job_id": job_id,
        "out_dir": out_dir,
        "frames": [
            {
                "second": int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]),
                "image_rel": os.path.relpath(p, out_dir).replace("\\", "/"),
            }
            for p in frame_paths
        ],
    }
    data["frames"] = sorted(data["frames"], key=lambda x: x["second"])

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def caption_existing(job_id: str, language: str, hf_model: str, translate: bool) -> Dict:
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    frames_dir = os.path.join(out_dir, "frames")
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        raise RuntimeError("result.json not found for this job")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Caption all frames listed in result.json
    updated_frames = []
    for item in data.get("frames", []):
        image_path = os.path.join(out_dir, item["image_rel"]).replace("/", os.sep)
        if not os.path.exists(image_path):
            continue
        if True:  # local backend only
            cap_en = caption_image_local(image_path, hf_model)
            final = cap_en
            if translate and language and language.lower() not in {"en", "english"}:
                final = translate_text(cap_en, language)
        # moderation
        mod = moderate_caption(final)
        updated_frames.append({
            "second": item["second"],
            "image_rel": item["image_rel"],
            "caption": final,
            "moderation": mod,
        })

    data["frames"] = sorted(updated_frames, key=lambda x: x["second"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


@app.route("/")
def index():
    # List existing uploaded videos
    uploads = []
    try:
        if os.path.isdir(UPLOAD_DIR):
            for name in os.listdir(UPLOAD_DIR):
                if not name.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                    continue
                full = os.path.join(UPLOAD_DIR, name)
                try:
                    stat = os.stat(full)
                    uploads.append({
                        "name": name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except Exception:
                    continue
        # Sort by most recent
        uploads.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        uploads = []

    return render_template("index.html", uploads=uploads)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    # Force local backend; remove OpenAI from UI
    backend = "local"
    language = request.form.get("language", "vi")
    hf_model = request.form.get("hf_model", "Salesforce/blip-image-captioning-base")
    openai_model = ""
    translate = request.form.get("translate", "on") == "on"

    job_id = str(uuid.uuid4())
    ensure_dir(UPLOAD_DIR)
    # Preserve original filename safely
    original = file.filename or "uploaded_video.mp4"
    safe_name = _sanitize_filename(original)
    # enforce allowed extensions
    _, ext = os.path.splitext(safe_name)
    if ext.lower() not in [".mp4", ".mov", ".mkv", ".webm"]:
        safe_name = (os.path.splitext(safe_name)[0] or "uploaded_video") + ".mp4"
    video_path = _unique_path(UPLOAD_DIR, safe_name)
    file.save(video_path)

    # Step 1: extract frames only
    try:
        process_video(job_id, video_path, backend, language, hf_model, openai_model, translate)
    except Exception as e:
        # On error, clean up and show a simple error page
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir="")

    return redirect(url_for("result", job_id=job_id))


@app.route("/use_uploaded", methods=["POST"])
def use_uploaded():
    name = request.form.get("existing")
    if not name:
        return redirect(url_for("index"))

    language = request.form.get("language", "vi")
    hf_model = request.form.get("hf_model", "Salesforce/blip-image-captioning-base")
    translate = request.form.get("translate", "on") == "on"

    video_path = os.path.join(UPLOAD_DIR, name)
    if not os.path.isfile(video_path):
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    try:
        process_video(job_id, video_path, "local", language, hf_model, "", translate)
    except Exception as e:
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir="")

    return redirect(url_for("result", job_id=job_id))


@app.route("/result/<job_id>")
def result(job_id: str):
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        abort(404)
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return render_template("result.html", error=None, job_id=job_id, frames=data.get("frames", []), out_dir=out_dir)


@app.route("/caption/<job_id>", methods=["POST"])
def caption_job(job_id: str):
    language = request.form.get("language", "vi")
    hf_model = request.form.get("hf_model", "Salesforce/blip-image-captioning-base")
    translate = request.form.get("translate", "on") == "on"
    try:
        caption_existing(job_id, language, hf_model, translate)
    except Exception as e:
        out_dir = os.path.join(OUTPUT_DIR, job_id)
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir=out_dir)
    return redirect(url_for("result", job_id=job_id))


@app.route("/outputs/<job_id>/frames/<path:filename>")
def serve_frame(job_id: str, filename: str):
    directory = os.path.join(OUTPUT_DIR, job_id, "frames")
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    # Start Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
