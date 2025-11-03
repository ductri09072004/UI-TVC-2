import os
import uuid
import json
from typing import List, Dict
import threading
import unicodedata

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort

# Reuse processing utilities from app.py
from app import (
    ensure_dir,
    extract_frames_1fps,
    caption_image_local,
    translate_text,
    CaptionResult,
    classify_text,
    extract_audio_wav,
    transcribe_audio_whisper,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")
# Default classifier directory (same as CLI default in app.py)
DEFAULT_CLF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation"))
def _split_sentences(text: str) -> list:
    """Naive sentence splitter for vi/en by punctuation. Returns non-empty trimmed sentences."""
    if not text:
        return []
    import re
    # Split on period, question, exclamation, ellipsis, also handle multiple spaces/newlines
    parts = re.split(r"(?<=[\.\!\?…])\s+|\n+", text.strip())
    sentences = [s.strip() for s in parts if s and s.strip()]
    return sentences

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


# ------------------------ Vietnamese text correction ------------------------
_LT_TOOL = None
_SYMSPELL = None
_BRAND_MAP = None


def _init_languagetool():
    global _LT_TOOL
    if _LT_TOOL is not None:
        return _LT_TOOL
    try:
        import language_tool_python  # type: ignore
        _LT_TOOL = language_tool_python.LanguageToolPublicAPI("vi")
    except Exception:
        _LT_TOOL = False
    return _LT_TOOL


def _apply_languagetool(text: str) -> str:
    tool = _init_languagetool()
    if not tool:
        return text
    try:
        matches = tool.check(text)
        # simple apply: replace by first suggestion where safe
        corrected = language_tool_python.utils.correct(text, matches)  # type: ignore
        return corrected or text
    except Exception:
        return text


def _init_symspell():
    global _SYMSPELL
    if _SYMSPELL is not None:
        return _SYMSPELL
    dict_path = os.environ.get("SYMSPELL_DICTIONARY")
    if not dict_path or not os.path.isfile(dict_path):
        _SYMSPELL = False
        return _SYMSPELL
    try:
        from symspellpy import SymSpell  # type: ignore
        _SYMSPELL = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Expect a TSV word\tfrequency
        _SYMSPELL.load_dictionary(dict_path, term_index=0, count_index=1, separator="\t")
    except Exception:
        _SYMSPELL = False
    return _SYMSPELL


def _apply_symspell(sentence: str) -> str:
    sym = _init_symspell()
    if not sym:
        return sentence
    try:
        suggestions = sym.lookup_compound(sentence, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        return sentence
    except Exception:
        return sentence


def _init_brand_map():
    global _BRAND_MAP
    if _BRAND_MAP is not None:
        return _BRAND_MAP
    path = os.environ.get("BRAND_MAP_JSON")
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _BRAND_MAP = json.load(f)
        except Exception:
            _BRAND_MAP = {}
    else:
        _BRAND_MAP = {}
    return _BRAND_MAP


def _apply_brand_map(text: str) -> str:
    mapping = _init_brand_map() or {}
    if not mapping:
        return text
    out = text
    try:
        for k, v in mapping.items():
            if not k:
                continue
            out = out.replace(k, v)
        return out
    except Exception:
        return text


def normalize_and_correct(text: str) -> str:
    if not text:
        return text
    # 1) Unicode NFC
    s = unicodedata.normalize("NFC", text)
    # 2) LanguageTool
    s = _apply_languagetool(s)
    # 3) SymSpell compound correction
    s = _apply_symspell(s)
    # 4) Brand map
    s = _apply_brand_map(s)
    # Cleanup spaces
    s = " ".join(s.split())
    return s


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

    # Step 2: Extract + transcribe audio immediately, and label segments
    audio_path = os.path.join(out_dir, "audio.wav")
    try:
        extract_audio_wav(video_path, audio_path, sample_rate=16000, apply_filters=True)
        # Use faster-whisper with Vietnamese settings and contextual prompt
        asr = transcribe_audio_whisper(audio_path, model_name="small", language="vi")
        # Full text then sentence-level labeling
        full_text = asr.get("text", "") or ""
        sentences = _split_sentences(full_text)
        labeled_sentences = []
        s_ok = s_violate = s_unknown = 0
        # Heuristics to reduce false violations from noisy ASR
        VIOLATION_THRESHOLD = float(os.environ.get("VIOLATION_THRESHOLD", "0.85"))
        MIN_WORDS = int(os.environ.get("ASR_MIN_WORDS", "3"))
        for idx, sent in enumerate(sentences):
            sent = normalize_and_correct(sent)
            try:
                pred = classify_text(sent, DEFAULT_CLF_DIR) if sent else {"label_id": None, "score": 0.0}
            except Exception as e:
                pred = {"label_id": None, "score": 0.0, "error": str(e)}
            lid = pred.get("label_id")
            score = float(pred.get("score") or 0.0)
            # Apply safeguards: require enough words and high confidence to flag violation
            num_words = len((sent or "").split())
            if lid == 1 and (score < VIOLATION_THRESHOLD or num_words < MIN_WORDS):
                lid = None  # abstain instead of false-positive violation
            if lid == 0:
                s_ok += 1
            elif lid == 1:
                s_violate += 1
            else:
                s_unknown += 1
            labeled_sentences.append({"index": idx + 1, "text": sent, "label_id": lid, "label_score": score})
        total_sent = len(labeled_sentences)
        data["audio"] = {
            "full_text": full_text,
            "sentences": labeled_sentences,
            "sentence_summary": {
                "total_sentences": total_sent,
                "num_compliant": s_ok,
                "num_violations": s_violate,
                "num_unknown": s_unknown,
                "percent_compliant": round((s_ok / total_sent * 100.0), 2) if total_sent else 0.0,
                "percent_violations": round((s_violate / total_sent * 100.0), 2) if total_sent else 0.0,
            },
        }
    except Exception:
        # If audio step fails (no ffmpeg/whisper), continue with frames only
        pass

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data
def process_audio_only(job_id: str, video_path: str, language: str) -> Dict:
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    ensure_dir(out_dir)
    audio_path = os.path.join(out_dir, "audio.wav")
    extract_audio_wav(video_path, audio_path, sample_rate=16000, apply_filters=True)
    # Use faster-whisper with Vietnamese settings and contextual prompt
    asr = transcribe_audio_whisper(audio_path, model_name="small", language="vi")
    full_text = asr.get("text", "") or ""
    sentences = _split_sentences(full_text)
    labeled_sentences = []
    s_ok = s_violate = s_unknown = 0
    VIOLATION_THRESHOLD = float(os.environ.get("VIOLATION_THRESHOLD", "0.85"))
    MIN_WORDS = int(os.environ.get("ASR_MIN_WORDS", "3"))
    for idx, sent in enumerate(sentences):
        sent = normalize_and_correct(sent)
        try:
            pred = classify_text(sent, DEFAULT_CLF_DIR) if sent else {"label_id": None, "score": 0.0}
        except Exception as e:
            pred = {"label_id": None, "score": 0.0, "error": str(e)}
        lid = pred.get("label_id")
        score = float(pred.get("score") or 0.0)
        num_words = len((sent or "").split())
        if lid == 1 and (score < VIOLATION_THRESHOLD or num_words < MIN_WORDS):
            lid = None
        if lid == 0:
            s_ok += 1
        elif lid == 1:
            s_violate += 1
        else:
            s_unknown += 1
        labeled_sentences.append({"index": idx + 1, "text": sent, "label_id": lid, "label_score": score})
    total_sent = len(labeled_sentences)
    data = {
        "job_id": job_id,
        "out_dir": out_dir,
        "audio": {
            "full_text": full_text,
            "sentences": labeled_sentences,
            "sentence_summary": {
                "total_sentences": total_sent,
                "num_compliant": s_ok,
                "num_violations": s_violate,
                "num_unknown": s_unknown,
                "percent_compliant": round((s_ok / total_sent * 100.0), 2) if total_sent else 0.0,
                "percent_violations": round((s_violate / total_sent * 100.0), 2) if total_sent else 0.0,
            },
        },
    }
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
        updated_frames.append({
            "second": item["second"],
            "image_rel": item["image_rel"],
            "caption": final,
        })

    data["frames"] = sorted(updated_frames, key=lambda x: x["second"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


@app.route("/")
def index():
    # List existing uploaded videos
    uploads = []
    jobs = []
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
        # List existing processed jobs (web_outputs/<job_id>/result.json)
        if os.path.isdir(OUTPUT_DIR):
            for job_id in os.listdir(OUTPUT_DIR):
                job_dir = os.path.join(OUTPUT_DIR, job_id)
                meta = os.path.join(job_dir, "result.json")
                if not os.path.isfile(meta):
                    continue
                try:
                    stat = os.stat(meta)
                    with open(meta, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    frames = data.get("frames", [])
                    jobs.append({
                        "job_id": job_id,
                        "count": len(frames),
                        "mtime": stat.st_mtime,
                    })
                except Exception:
                    continue
            jobs.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        uploads = []
        jobs = []

    return render_template("index.html", uploads=uploads, jobs=jobs)
@app.route("/audio")
def audio_index():
    uploads = []
    try:
        if os.path.isdir(UPLOAD_DIR):
            for name in os.listdir(UPLOAD_DIR):
                if not name.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
                    continue
                full = os.path.join(UPLOAD_DIR, name)
                try:
                    stat = os.stat(full)
                    uploads.append({"name": name, "size": stat.st_size, "mtime": stat.st_mtime})
                except Exception:
                    continue
        uploads.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        uploads = []
    return render_template("audio.html", uploads=uploads)


@app.route("/audio_process", methods=["POST"])
def audio_process():
    source = request.form.get("source", "uploaded")
    language = request.form.get("language", "vi")
    if source == "upload":
        file = request.files.get("video")
        if not file or file.filename == "":
            return redirect(url_for("audio_index"))
        job_id = str(uuid.uuid4())
        safe_name = _sanitize_filename(file.filename or "uploaded_audio.mp4")
        _, ext = os.path.splitext(safe_name)
        if ext.lower() not in [".mp4", ".mov", ".mkv", ".webm"]:
            safe_name = (os.path.splitext(safe_name)[0] or "uploaded_audio") + ".mp4"
        video_path = _unique_path(UPLOAD_DIR, safe_name)
        file.save(video_path)
    else:
        name = request.form.get("existing")
        if not name:
            return redirect(url_for("audio_index"))
        video_path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(video_path):
            return redirect(url_for("audio_index"))
        job_id = str(uuid.uuid4())
    def _run_audio_job():
        out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(out_dir, exist_ok=True)
        status_path = os.path.join(out_dir, "status.json")
        log_path = os.path.join(out_dir, "audio_job.log")
        try:
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"state": "processing"}, sf)
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("[start] audio job started\n")
                lf.flush()
                lf.write("- extracting audio...\n")
                lf.flush()
            process_audio_only(job_id, video_path, language)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("- transcription done\n")
                lf.flush()
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"state": "done"}, sf)
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write("[error] " + err + "\n")
                    lf.flush()
            except Exception:
                pass
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"state": "error", "message": str(e)}, sf)

    threading.Thread(target=_run_audio_job, daemon=True).start()
    return redirect(url_for("audio_wait", job_id=job_id))


@app.route("/audio_wait/<job_id>")
def audio_wait(job_id: str):
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    status_path = os.path.join(out_dir, "status.json")
    meta_path = os.path.join(out_dir, "result.json")
    state = "processing"
    message = ""
    if os.path.isfile(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                st = json.load(sf)
            state = st.get("state", "processing")
            message = st.get("message", "")
        except Exception:
            state = "processing"
    if state == "done" and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return render_template(
            "audio_result.html",
            error=None,
            job_id=job_id,
            out_dir=data.get("out_dir", ""),
            audio_summary=data.get("audio", {}).get("sentence_summary"),
            audio_sentences=data.get("audio", {}).get("sentences"),
        )
    # read last logs if present
    log_text = ""
    log_path = os.path.join(out_dir, "audio_job.log")
    if os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as lf:
                log_text = lf.read()[-4000:]
        except Exception:
            log_text = ""
    if state == "error":
        return render_template("audio_result.html", error=message or log_text, job_id=job_id, out_dir="", audio_summary=None, audio_sentences=None)
    return render_template("audio_wait.html", job_id=job_id, logs=log_text)



@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    # Force local backend; remove OpenAI from UI
    backend = "local"
    language = request.form.get("language", "vi")
    # Force backend default model (ignore UI selection)
    hf_model = ""
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
    # Force backend default model (ignore UI selection)
    hf_model = ""
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
    frames = data.get("frames", [])
    # Compute summary if labels exist
    summary = None
    try:
        if frames and any("label_id" in fr for fr in frames):
            total = len(frames)
            num_ok = sum(1 for fr in frames if fr.get("label_id") == 0)
            num_violate = sum(1 for fr in frames if fr.get("label_id") == 1)
            num_unknown = total - num_ok - num_violate
            pct_ok = round((num_ok / total * 100.0), 2) if total else 0.0
            pct_violate = round((num_violate / total * 100.0), 2) if total else 0.0
            summary = {
                "total_frames": total,
                "num_compliant": num_ok,
                "num_violations": num_violate,
                "num_unknown": num_unknown,
                "percent_compliant": pct_ok,
                "percent_violations": pct_violate,
            }
    except Exception:
        summary = None
    audio = data.get("audio")
    audio_summary = audio.get("sentence_summary") if isinstance(audio, dict) else None
    audio_sentences = audio.get("sentences") if isinstance(audio, dict) else None
    return render_template(
        "result.html",
        error=None,
        job_id=job_id,
        frames=frames,
        out_dir=out_dir,
        summary=summary,
        audio_summary=audio_summary,
        audio_sentences=audio_sentences,
    )


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


@app.route("/label/<job_id>", methods=["POST"])
def label_job(job_id: str):
    # Label existing captions in result.json without re-captioning
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        abort(404)
    # classifier directory (reuse default from app.py CLI default)
    clf_dir = request.form.get(
        "clf_dir",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation")),
    )
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        updated = []
        for item in frames:
            cap = item.get("caption", "")
            if cap:
                try:
                    pred = classify_text(cap, clf_dir)
                except Exception as e:
                    pred = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
            else:
                pred = {"label_id": None, "score": 0.0, "probs": []}
            new_item = dict(item)
            new_item["label_id"] = pred.get("label_id")
            new_item["label_score"] = pred.get("score")
            updated.append(new_item)
        data["frames"] = updated
        # Compute and persist summary to result.json
        try:
            total = len(updated)
            num_ok = sum(1 for fr in updated if fr.get("label_id") == 0)
            num_violate = sum(1 for fr in updated if fr.get("label_id") == 1)
            num_unknown = total - num_ok - num_violate
            pct_ok = round((num_ok / total * 100.0), 2) if total else 0.0
            pct_violate = round((num_violate / total * 100.0), 2) if total else 0.0
            data["summary"] = {
                "total_frames": total,
                "num_compliant": num_ok,
                "num_violations": num_violate,
                "num_unknown": num_unknown,
                "percent_compliant": pct_ok,
                "percent_violations": pct_violate,
            }
        except Exception:
            pass
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return redirect(url_for("result", job_id=job_id))
    except Exception as e:
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir=out_dir)


@app.route("/outputs/<job_id>/frames/<path:filename>")
def serve_frame(job_id: str, filename: str):
    directory = os.path.join(OUTPUT_DIR, job_id, "frames")
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    # Start Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
