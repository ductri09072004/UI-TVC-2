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
    extract_frames_to_memory,
    translate_text,
    CaptionResult,
    classify_text,
    extract_audio_wav,
    transcribe_audio_whisper,
    detect_objects,
    console,
    is_url,
    download_video_from_url,
    resolve_video_input,
)
from rich.progress import track
# Import captioning from dedicated module
from image_captioning import caption_image_local, caption_image_llava, caption_images_openai_batch, _extract_ocr_text_simple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")
# Default classifier directory (same as CLI default in app.py)
DEFAULT_CLF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation"))

def get_available_classifier_models():
    """Tự động phát hiện các model classifier có sẵn trong TVC-AI."""
    tvc_ai_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI"))
    models = []
    if os.path.isdir(tvc_ai_dir):
        for item in os.listdir(tvc_ai_dir):
            if item.startswith("output_moderation"):
                model_dir = os.path.join(tvc_ai_dir, item)
                config_file = os.path.join(model_dir, "config.json")
                if os.path.isdir(model_dir) and os.path.exists(config_file):
                    # Tạo tên hiển thị đẹp hơn
                    display_name = item.replace("output_moderation", "").strip("_")
                    if display_name:
                        display_name = f"Model {display_name.replace('_', ' ').title()}"
                    else:
                        display_name = "Model Default"
                    
                    models.append({
                        "name": item,
                        "path": model_dir,
                        "display_name": display_name
                    })
    # Sort by name
    models.sort(key=lambda x: x["name"])
    return models
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


def process_video(job_id: str, video_path: str, backend: str, language: str, hf_model: str, openai_model: str, translate: bool, use_object_detection: bool = False, use_llava: bool = False, use_ocr_only: bool = False) -> Dict:
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    ensure_dir(out_dir)

    # Extract frames to memory (don't save to disk)
    frames_in_memory = extract_frames_to_memory(video_path)

    # Use default model if not specified
    if not hf_model:
        hf_model = "Salesforce/blip-image-captioning-base"

    # Step 1: Create data structure and caption frames
    # For OpenAI backend: group 3 frames and create 1 caption per group
    # For other backends: process each frame individually
    frames_data = []
    
    if backend == "openai":
        console.print(f"[cyan]Extracting and captioning {len(frames_in_memory)} frames (OpenAI: 3 frames = 1 caption)...[/cyan]")
        
        # Load OpenAI config
        try:
            from app import build_openai_caption_config
            openai_config_obj = build_openai_caption_config(
                api_key=None,  # Will load from config or env
                model=openai_model or None,
                prompt=None,
                temperature=None,
                max_tokens=None,
                target_language=language or None,
            )
            api_key = openai_config_obj.api_key
            model_name = openai_config_obj.model
            prompt_text = openai_config_obj.prompt
            temperature_val = openai_config_obj.temperature
            max_tokens_val = openai_config_obj.max_tokens
        except Exception as e:
            console.print(f"[red]OpenAI config error: {e}[/red]")
            raise RuntimeError(f"Cannot load OpenAI config: {e}")
        
        # Process each frame individually with detailed captions
        # This uses more tokens but generates detailed descriptions for each frame
        import time
        delay_between_frames = float(os.environ.get("OPENAI_REQUEST_DELAY", "0.3"))  # Default 0.3s delay per frame
        
        # Get all frames as individual images (not batched)
        all_frames_list = [(sec, img) for sec, img in frames_in_memory]
        
        try:
            # Generate detailed caption for each frame separately
            # detailed=True returns a list of captions (one per image)
            captions_list = caption_images_openai_batch(
                [img for _, img in all_frames_list],  # List of images only
                model_name=model_name,
                api_key=api_key,
                target_language=language or "vi",
                prompt=prompt_text,
                temperature=temperature_val,
                max_tokens=max_tokens_val,
                console=console,
                detailed=True  # Enable detailed mode - one caption per frame
            )
            
            # Process each frame with its corresponding caption
            for idx, (sec, img) in enumerate(all_frames_list):
                cap_en = captions_list[idx] if idx < len(captions_list) else ""
                caption_original = cap_en if cap_en else None
                
                # Auto translate to Vietnamese (usually already in Vietnamese)
                final = cap_en if cap_en else ""
                if translate and cap_en and language != "vi":
                    try:
                        final = translate_text(cap_en, "vi")
                    except Exception:
                        final = cap_en
                
                # Store each frame individually (no grouping)
                frames_data.append({
                    "second": sec,
                    "caption": final,
                    "caption_original": caption_original,
                })
                
                # Add delay between frames (except last one)
                if idx < len(all_frames_list) - 1:
                    time.sleep(delay_between_frames)
                    
        except Exception as e:
            # If captioning fails, log error and store empty captions
            console.print(f"[yellow]Warning: Failed to caption frames: {e}[/yellow]")
            for sec, img in all_frames_list:
                frames_data.append({
                    "second": sec,
                    "caption": "",
                    "caption_original": None,
                })
    else:
        # Local backend: process each frame individually
        console.print(f"[cyan]Extracting and captioning {len(frames_in_memory)} frames (mỗi frame có mô tả riêng)...[/cyan]")
        
        for sec, pil_image in track(frames_in_memory, description="Captioning frames (1 caption per frame)"):
            try:
                # Generate caption from PIL Image
                if use_ocr_only:
                    # OCR-only mode: bypass visual captioning models
                    try:
                        cap_en = _extract_ocr_text_simple(pil_image, languages=["vi", "en"], min_confidence=0.3)
                    except Exception as _:
                        cap_en = ""
                elif use_llava:
                    cap_en = caption_image_llava(pil_image, console=console)
                else:
                    cap_en = caption_image_local(
                        pil_image,
                        hf_model,
                        use_object_detection=use_object_detection,
                        detect_objects_func=detect_objects,
                        console=console
                    )
                caption_original = cap_en if cap_en else None
                
                # Auto translate to Vietnamese
                final = ""
                if cap_en:
                    if translate:
                        try:
                            final = translate_text(cap_en, "vi")
                        except Exception:
                            # If translation fails, keep original caption
                            final = cap_en
                    else:
                        final = cap_en
                
                # Lưu caption riêng cho frame này
                frames_data.append({
                    "second": sec,
                    "caption": final,
                    "caption_original": caption_original,
                })
            except Exception as e:
                # If captioning fails, log error but continue
                console.print(f"[yellow]Warning: Failed to caption frame at {sec}s: {e}[/yellow]")
                frames_data.append({
                    "second": sec,
                    "caption": "",
                    "caption_original": None,
                })
    
    data = {
        "job_id": job_id,
        "out_dir": out_dir,
        "video_path": video_path,  # Save video path for later captioning
        "frames": sorted(frames_data, key=lambda x: x["second"]),
    }

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



def caption_existing(job_id: str, language: str, hf_model: str, translate: bool, use_object_detection: bool = True, video_path: str = None) -> Dict:
    """Caption frames from video. Extracts frames from video since frames are not saved to disk."""
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        raise RuntimeError("result.json not found for this job")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get video_path from result.json or use provided one
    video_path = video_path or data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("Cannot caption: video_path not found in result.json and not provided. Frames are not saved to disk, so video is required.")
    
    # Extract frames from video (frames are not saved to disk)
    frames_in_memory = extract_frames_to_memory(video_path)
    # Create a dict for quick lookup by second
    frames_dict = {sec: img for sec, img in frames_in_memory}

    # Tạo caption cho từng frame riêng biệt (không gộp)
    frames_list = data.get("frames", [])
    updated_frames = []
    
    for item in frames_list:
        sec = item["second"]
        if sec not in frames_dict:
            # Frame not found, giữ nguyên item
            updated_frames.append(item)
            continue
        
        pil_image = frames_dict[sec]
        try:
            cap_en = caption_image_local(
                pil_image,  # Pass PIL Image instead of path
                hf_model, 
                use_object_detection=use_object_detection,
                detect_objects_func=detect_objects,
                console=console
            )
            caption_original = cap_en if cap_en else None
            
            # Tự động dịch sang tiếng Việt
            final = ""
            if cap_en:
                try:
                    final = translate_text(cap_en, "vi")
                except Exception:
                    # Nếu dịch lỗi, giữ nguyên caption gốc
                    final = cap_en
        except RuntimeError as e:
            # Nếu là lỗi model không load được, raise để dừng lại
            raise
        except Exception as e:
            # Lỗi khác, giữ caption cũ
            import traceback
            print(f"Error in caption_existing for frame {sec}: {e}")
            print(traceback.format_exc())
            final = item.get("caption", "")
            caption_original = item.get("caption_original", None)
        
        # Lưu caption riêng cho frame này
        updated_item = {
            "second": sec,
            # No image_rel - images are not saved
            "caption": final,
            "caption_original": caption_original,
        }
        # Giữ lại các thông tin khác nếu có (như label_id, label_score)
        if "label_id" in item:
            updated_item["label_id"] = item["label_id"]
        if "label_score" in item:
            updated_item["label_score"] = item["label_score"]
        updated_frames.append(updated_item)

    data["frames"] = sorted(updated_frames, key=lambda x: x["second"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def regenerate_captions(job_id: str, language: str, hf_model: str, translate: bool, use_object_detection: bool = True) -> Dict:
    """Tạo lại mô tả cho tất cả các frame trong job đã có mô tả cũ.
    Tương tự caption_existing nhưng có thể dùng object detection.
    Khi tạo lại mô tả, xóa label cũ vì mô tả mới có thể khác.
    """
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        raise RuntimeError("result.json not found for this job")
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get video_path from result.json
    video_path = data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("Cannot regenerate captions: video_path not found in result.json. Frames are not saved to disk, so video is required.")
    
    # Extract frames from video (frames are not saved to disk)
    frames_in_memory = extract_frames_to_memory(video_path)
    # Create a dict for quick lookup by second
    frames_dict = {sec: img for sec, img in frames_in_memory}

    # Tạo caption cho từng frame riêng biệt (không gộp)
    frames_list = data.get("frames", [])
    updated_frames = []
    
    for item in frames_list:
        sec = item["second"]
        if sec not in frames_dict:
            # Frame not found, keep old data
            updated_frames.append(item)
            continue
        
        pil_image = frames_dict[sec]
        try:
            # local backend only
            cap_en = caption_image_local(pil_image, hf_model, use_object_detection=use_object_detection, detect_objects_func=detect_objects, console=console)
            caption_original = cap_en if cap_en else None
            
            # Tự động dịch sang tiếng Việt
            final = ""
            if cap_en:
                try:
                    final = translate_text(cap_en, "vi")
                except Exception as e:
                    # Nếu dịch lỗi, giữ nguyên caption gốc
                    final = cap_en
        except RuntimeError as e:
            # RuntimeError thường là lỗi nghiêm trọng (OOM, model không load được)
            # Dừng lại và báo lỗi rõ ràng
            import traceback
            error_msg = f"Lỗi nghiêm trọng khi tạo mô tả: {e}"
            print(error_msg)
            print(traceback.format_exc())
            # Nếu là lỗi OOM, giữ caption cũ và log lỗi
            if "out of memory" in str(e).lower() or "oom" in str(e).lower() or "quá lớn" in str(e).lower():
                final = item.get("caption", "")  # Giữ caption cũ
                caption_original = item.get("caption_original", None)
                print(f"Model quá lớn, giữ caption cũ cho frame {sec}")
            else:
                # Các lỗi khác, raise để dừng lại
                raise
        except Exception as e:
            # Nếu caption lỗi, giữ lại caption cũ hoặc để trống
            import traceback
            print(f"Error captioning frame {sec}: {e}")
            print(traceback.format_exc())
            final = item.get("caption", "")  # Giữ caption cũ nếu có
            caption_original = item.get("caption_original", None)
        
        # Lưu caption riêng cho frame này
        # Tạo lại mô tả, không giữ label cũ vì mô tả mới có thể khác
        updated_item = {
            "second": sec,
            # No image_rel - images are not saved
            "caption": final,
            "caption_original": caption_original,
        }
        updated_frames.append(updated_item)

    data["frames"] = sorted(updated_frames, key=lambda x: x["second"])
    # Xóa summary cũ nếu có vì label đã thay đổi
    if "summary" in data:
        del data["summary"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


@app.route("/classify", methods=["POST"])
def classify_description():
    """Xử lý form gán nhãn mô tả."""
    description = request.form.get("description", "").strip()
    clf_dir = request.form.get("clf_dir", DEFAULT_CLF_DIR)
    
    if not description:
        return redirect(url_for("index"))
    
    classification_result = None
    try:
        # Sử dụng model được chọn để phân loại
        pred = classify_text(description, clf_dir)
        classification_result = {
            "description": description,
            "label_id": pred.get("label_id"),
            "score": pred.get("score", 0.0),
            "probs": pred.get("probs", {})
        }
    except Exception as e:
        classification_result = {
            "description": description,
            "label_id": None,
            "score": 0.0,
            "error": str(e)
        }
    
    # Lấy danh sách uploads và jobs như trong index()
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
        uploads.sort(key=lambda x: x["mtime"], reverse=True)
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
                    jobs.append({
                        "job_id": job_id,
                        "count": len(data.get("frames", [])),
                        "mtime": stat.st_mtime,
                    })
                except Exception:
                    continue
        jobs.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        pass
    
    # Lưu kết quả vào session và redirect với anchor để scroll đến form
    import json as json_module
    result_json = json_module.dumps(classification_result)
    return redirect(url_for("index", classification_result=result_json) + "#classify-section")


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

    # Kiểm tra xem có classification_result trong session không (từ POST redirect)
    classification_result = None
    if request.args.get("classification_result"):
        try:
            import json as json_module
            classification_result = json_module.loads(request.args.get("classification_result"))
        except Exception:
            pass
    
    # Lấy danh sách model có sẵn
    available_models = get_available_classifier_models()
    
    return render_template("index.html", 
                         uploads=uploads, 
                         jobs=jobs,
                         classification_result=classification_result,
                         available_models=available_models)
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
    # Kiểm tra xem có URL hay file upload
    video_url = request.form.get("video_url", "").strip()
    file = request.files.get("video")
    
    video_path = None
    is_temp_file = False
    temp_video_path = None
    
    # Tạo job_id trước
    job_id = str(uuid.uuid4())
    
    if video_url:
        # Xử lý URL
        try:
            video_path, is_temp_file = resolve_video_input(video_url)
            temp_video_path = video_path if is_temp_file else None
        except Exception as e:
            return render_template("result.html", error=f"Lỗi khi tải video từ URL: {e}", job_id=job_id, frames=[], out_dir="")
    elif file and file.filename:
        # Xử lý file upload
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
    else:
        return redirect(url_for("index"))

    # Get backend from form (default to local)
    backend = request.form.get("caption_backend", "local")
    language = request.form.get("language", "vi")
    # Get OpenAI config if using OpenAI backend
    openai_key_from_form = request.form.get("openai_key", "").strip()
    openai_model_from_form = request.form.get("openai_model", "").strip()
    hf_model = ""  # Use default from image_captioning.py
    openai_model = openai_model_from_form if openai_model_from_form else ""
    translate = True  # Always translate to Vietnamese
    
    # Step 1: extract frames and caption immediately
    try:
        # Get use_object_detection from form (default False)
        use_object_detection = request.form.get("extract_scene") == "on"  # If "Scene description" is checked
        use_llava = request.form.get("use_llava") == "on"
        use_ocr_only = request.form.get("use_ocr_only") == "on"
        process_video(job_id, video_path, backend, language, hf_model, openai_model, translate, use_object_detection=use_object_detection, use_llava=use_llava, use_ocr_only=use_ocr_only)
    except Exception as e:
        # On error, clean up and show a simple error page
        # Xóa file tạm nếu có
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir="")
    finally:
        # Xóa file video tạm nếu là file tải từ URL
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass

    return redirect(url_for("result", job_id=job_id))


@app.route("/use_uploaded", methods=["POST"])
def use_uploaded():
    name = request.form.get("existing")
    if not name:
        return redirect(url_for("index"))

    language = "vi"  # Auto translate to Vietnamese
    # Force backend default model (ignore UI selection)
    hf_model = ""
    translate = True  # Always translate to Vietnamese

    video_path = os.path.join(UPLOAD_DIR, name)
    if not os.path.isfile(video_path):
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    try:
        # Get use_object_detection from form (default False)
        use_object_detection = request.form.get("extract_scene") == "on"  # If "Scene description" is checked
        use_llava = request.form.get("use_llava") == "on"
        use_ocr_only = request.form.get("use_ocr_only") == "on"
        process_video(job_id, video_path, "local", language, hf_model, "", translate, use_object_detection=use_object_detection, use_llava=use_llava, use_ocr_only=use_ocr_only)
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
    
    # Check regenerate status
    regenerate_status = None
    status_path = os.path.join(out_dir, "regenerate_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                regenerate_status = json.load(sf)
        except Exception:
            pass
    
    # Lấy danh sách model có sẵn
    available_models = get_available_classifier_models()
    
    return render_template(
        "result.html",
        error=None,
        job_id=job_id,
        frames=frames,
        out_dir=out_dir,
        summary=summary,
        audio_summary=audio_summary,
        audio_sentences=audio_sentences,
        regenerate_status=regenerate_status,
        available_models=available_models,
    )


@app.route("/caption/<job_id>", methods=["POST"])
def caption_job(job_id: str):
    language = "vi"  # Auto translate to Vietnamese
    hf_model = request.form.get("hf_model", "Salesforce/blip-image-captioning-base")
    translate = True  # Always translate to Vietnamese
    use_object_detection = request.form.get("use_object_detection", "on") == "on"
    try:
        caption_existing(job_id, language, hf_model, translate, use_object_detection=use_object_detection)
    except Exception as e:
        out_dir = os.path.join(OUTPUT_DIR, job_id)
        return render_template("result.html", error=str(e), job_id=job_id, frames=[], out_dir=out_dir)
    return redirect(url_for("result", job_id=job_id))


@app.route("/api/regenerate-status/<job_id>")
def regenerate_status_api(job_id: str):
    """API endpoint để check regenerate status."""
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    status_path = os.path.join(out_dir, "regenerate_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as sf:
                status = json.load(sf)
                return json.dumps(status), 200, {"Content-Type": "application/json"}
        except Exception:
            pass
    return json.dumps({"state": "none"}), 200, {"Content-Type": "application/json"}


@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate_job(job_id: str):
    """Tạo lại mô tả cho các frame đã có mô tả cũ. Chạy trong background thread."""
    language = "vi"  # Auto translate to Vietnamese
    hf_model = request.form.get("hf_model", "Salesforce/blip-image-captioning-base")
    translate = True  # Always translate to Vietnamese
    use_object_detection = request.form.get("use_object_detection", "on") == "on"
    
    # Chạy trong background thread để không block UI và tránh timeout
    def _run_regenerate():
        out_dir = os.path.join(OUTPUT_DIR, job_id)
        status_path = os.path.join(out_dir, "regenerate_status.json")
        try:
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"state": "processing", "message": "Đang tạo lại mô tả..."}, sf)
            regenerate_captions(job_id, language, hf_model, translate, use_object_detection)
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"state": "done"}, sf)
        except Exception as e:
            import traceback
            err_msg = f"{e}\n{traceback.format_exc()}"
            print(f"Error in regenerate_captions: {err_msg}")
            try:
                with open(status_path, "w", encoding="utf-8") as sf:
                    json.dump({"state": "error", "message": str(e)}, sf)
            except Exception:
                pass
    
    threading.Thread(target=_run_regenerate, daemon=True).start()
    # Redirect ngay, người dùng sẽ thấy kết quả sau khi xong (có thể refresh trang)
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
