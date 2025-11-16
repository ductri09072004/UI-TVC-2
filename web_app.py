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
    transcribe_audio_openai_whisper,
    transcribe_audio_faster_whisper,
    combine_audio_and_frame_captions,
    detect_objects,
    console,
    is_url,
    download_video_from_url,
    resolve_video_input,
    build_openai_caption_config,
    normalize_caption_for_classification,
)
from rich.progress import track
# Import captioning from dedicated module
from image_captioning import caption_image_local, caption_image_llava, caption_images_openai_batch, _extract_ocr_text_simple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "web_outputs")
# Default classifier directory (same as CLI default in app.py)
DEFAULT_CLF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation"))
# Path to banned words file
BANNED_WORDS_FILE = os.path.join(BASE_DIR, "data_ban.txt")

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

def load_banned_words(banned_words_file: str = BANNED_WORDS_FILE) -> set:
    """Đọc danh sách từ cấm từ file và trả về set các từ (lowercase, đã normalize)."""
    banned_words = set()
    if not os.path.exists(banned_words_file):
        console.print(f"[yellow]Cảnh báo: File từ cấm không tồn tại: {banned_words_file}[/yellow]")
        return banned_words
    
    try:
        with open(banned_words_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    # Normalize: lowercase và loại bỏ khoảng trắng thừa
                    normalized = ' '.join(word.lower().split())
                    banned_words.add(normalized)
        console.print(f"[green]Đã tải {len(banned_words)} từ cấm từ {banned_words_file}[/green]")
    except Exception as e:
        console.print(f"[red]Lỗi khi đọc file từ cấm: {e}[/red]")
    
    return banned_words

def check_banned_words(text: str, banned_words: set) -> Dict:
    """Kiểm tra xem text có chứa từ cấm không.
    Sử dụng word boundary để tránh false positive (ví dụ: "phòng" không bị match với "phò").
    
    Returns:
        Dict với keys: 'has_violation' (bool), 'matched_words' (list), 'label_id' (1 nếu vi phạm, 0 nếu không), 'score' (1.0 nếu vi phạm, 0.0 nếu không)
    """
    if not text or not banned_words:
        return {"has_violation": False, "matched_words": [], "label_id": 0, "score": 0.0}
    
    import re
    
    # Normalize text: lowercase và loại bỏ khoảng trắng thừa
    normalized_text = ' '.join(text.lower().split())
    
    matched_words = []
    
    # Sắp xếp từ cấm theo độ dài giảm dần để ưu tiên match từ dài hơn trước
    # (tránh trường hợp "phò" match trong "phòng")
    sorted_banned_words = sorted(banned_words, key=len, reverse=True)
    
    # Tạo set để track các vị trí đã match (tránh match trùng)
    matched_positions = set()
    
    # Ký tự tiếng Việt (bao gồm cả dấu)
    vi_chars = r'a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    
    for banned_word in sorted_banned_words:
        # Escape special regex characters trong từ cấm
        escaped_word = re.escape(banned_word)
        
        # Nếu từ cấm có khoảng trắng (cụm từ), match trực tiếp
        if ' ' in banned_word:
            # Cụm từ: match chính xác cụm từ đó
            pattern = escaped_word
        else:
            # Từ đơn: sử dụng word boundary để chỉ match từ hoàn chỉnh
            # Sử dụng \b (word boundary) hoặc kiểm tra ký tự trước/sau
            # Pattern: từ cấm phải đứng độc lập (không phải substring của từ khác)
            # Sử dụng word boundary hoặc kiểm tra ký tự không phải chữ cái/số ở đầu/cuối
            # \b không hoạt động tốt với tiếng Việt, nên dùng negative lookbehind/lookahead
            pattern = r'(?<![{}0-9])'.format(vi_chars) + escaped_word + r'(?![{}0-9])'.format(vi_chars)
        
        matches = list(re.finditer(pattern, normalized_text))
        
        # Chỉ thêm vào nếu có match và chưa bị overlap với match trước đó
        if matches:
            # Kiểm tra xem có overlap với các match trước không
            has_overlap = False
            for match in matches:
                start, end = match.span()
                # Kiểm tra xem vị trí này có overlap với match trước không
                if any(start < pos_end and end > pos_start for pos_start, pos_end in matched_positions):
                    has_overlap = True
                    break
            
            if not has_overlap:
                matched_words.append(banned_word)
                # Lưu các vị trí đã match
                for match in matches:
                    matched_positions.add(match.span())
    
    has_violation = len(matched_words) > 0
    
    return {
        "has_violation": has_violation,
        "matched_words": matched_words,
        "label_id": 1 if has_violation else 0,
        "score": 1.0 if has_violation else 0.0
    }
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
                
                # Chuẩn hóa caption cho classification (chỉ khi dùng OpenAI backend)
                caption_for_clf = None
                if backend == "openai" and final:
                    try:
                        caption_for_clf = normalize_caption_for_classification(
                            final,
                            api_key=api_key,
                            model=model_name,
                            target_language=language or "vi",
                            console=console,
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Không thể chuẩn hóa caption: {e}[/yellow]")
                        caption_for_clf = None
                
                # Store each frame individually (no grouping)
                frames_data.append({
                    "second": sec,
                    "caption": final,
                    "caption_original": caption_original,
                    "caption_for_classification": caption_for_clf,
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
    audio_text_for_combination = None  # Lưu để kết hợp với frame captions
    try:
        extract_audio_wav(video_path, audio_path, sample_rate=16000, apply_filters=True)
        
        # Nếu dùng OpenAI backend, dùng OpenAI Whisper API (chỉ dùng OpenAI, không fallback)
        if backend == "openai":
            openai_config_obj = build_openai_caption_config(
                api_key=None,  # Will load from config or env
                model=openai_model or None,
                prompt=None,
                temperature=None,
                max_tokens=None,
                target_language=language or None,
            )
            # Đảm bảo language là ISO-639-1 format (vi, en, etc.), không phải "auto"
            asr_language = "vi" if not language or language == "auto" else language
            # Chỉ lấy 2 ký tự đầu nếu language dài hơn (ví dụ: "vietnamese" -> "vi")
            if len(asr_language) > 2:
                asr_language = asr_language[:2]
            
            asr = transcribe_audio_openai_whisper(
                audio_path,
                api_key=openai_config_obj.api_key,
                language=asr_language,
                prompt="Quảng cáo TVC bằng tiếng Việt, thương hiệu, khuyến mãi, sản phẩm",
                console=console,
            )
        else:
            # Use local Whisper for non-OpenAI backends
            asr = transcribe_audio_whisper(audio_path, model_name="small", language="vi")
        
        # Full text then sentence-level labeling
        full_text = asr.get("text", "") or ""
        audio_text_for_combination = full_text  # Lưu để kết hợp với frame captions
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
    except Exception as e:
        # If audio step fails (no ffmpeg/whisper), continue with frames only
        console.print(f"[yellow]Audio processing skipped: {e}[/yellow]")
    
    # Step 3: Nếu dùng OpenAI backend và có audio, kết hợp audio với frame captions
    if backend == "openai" and audio_text_for_combination and frames_data:
        try:
            console.print(f"[cyan]Đang kết hợp audio transcript với frame captions...[/cyan]")
            # Load OpenAI config
            openai_config_obj = build_openai_caption_config(
                api_key=None,
                model=openai_model or None,
                prompt=None,
                temperature=None,
                max_tokens=None,
                target_language=language or None,
            )
            
            # Kết hợp audio với từng frame caption
            for frame_data in frames_data:
                if frame_data.get("caption"):  # Chỉ cập nhật nếu đã có caption
                    frame_captions_to_combine = [frame_data["caption"]]
                    
                    try:
                        combined_caption = combine_audio_and_frame_captions(
                            audio_text=audio_text_for_combination,
                            frame_captions=frame_captions_to_combine,
                            api_key=openai_config_obj.api_key,
                            model=openai_config_obj.model,
                            target_language=openai_config_obj.target_language,
                            console=console,
                        )
                        if combined_caption:
                            # Cập nhật caption trong frame_data
                            frame_data["caption"] = combined_caption
                            
                            # Chuẩn hóa caption mới cho classification
                            try:
                                frame_data["caption_for_classification"] = normalize_caption_for_classification(
                                    combined_caption,
                                    api_key=openai_config_obj.api_key,
                                    model=openai_config_obj.model,
                                    target_language=openai_config_obj.target_language,
                                    console=console,
                                )
                            except Exception as e:
                                console.print(f"[yellow]Warning: Không thể chuẩn hóa caption sau khi kết hợp audio: {e}[/yellow]")
                    except Exception as e:
                        console.print(f"[yellow]Không thể kết hợp audio với frame caption: {e}[/yellow]")
                        # Giữ nguyên caption gốc
            
            # Cập nhật lại data["frames"] với captions đã được kết hợp
            data["frames"] = sorted(frames_data, key=lambda x: x["second"])
            console.print(f"[green]✓ Đã kết hợp audio với tất cả frame captions[/green]")
        except Exception as e:
            console.print(f"[yellow]Error combining audio with frame captions: {e}[/yellow]")
            # Continue without combination

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

    # Get backend from form (default to openai)
    backend = request.form.get("caption_backend", "openai")
    language = "vi"  # Mặc định tiếng Việt
    # Get OpenAI config if using OpenAI backend
    openai_key_from_form = request.form.get("openai_key", "").strip()
    openai_model_from_form = request.form.get("openai_model", "").strip()
    hf_model = ""  # Use default from image_captioning.py
    openai_model = openai_model_from_form if openai_model_from_form else ""
    translate = True  # Always translate to Vietnamese
    
    # Nếu có OpenAI key từ form, set vào environment variable tạm thời
    if openai_key_from_form:
        os.environ["OPENAI_API_KEY"] = openai_key_from_form
    
    # Step 1: extract frames and caption immediately
    try:
        # Mặc định không dùng object detection, llava, ocr_only (đã bỏ UI options)
        use_object_detection = False
        use_llava = False
        use_ocr_only = False
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

    language = "vi"  # Mặc định tiếng Việt
    # Force backend to openai (default)
    backend = "openai"
    hf_model = ""
    openai_model = ""
    translate = True  # Always translate to Vietnamese

    video_path = os.path.join(UPLOAD_DIR, name)
    if not os.path.isfile(video_path):
        return redirect(url_for("index"))

    job_id = str(uuid.uuid4())
    try:
        # Mặc định không dùng object detection, llava, ocr_only
        use_object_detection = False
        use_llava = False
        use_ocr_only = False
        process_video(job_id, video_path, backend, language, hf_model, openai_model, translate, use_object_detection=use_object_detection, use_llava=use_llava, use_ocr_only=use_ocr_only)
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
            # Bước 1: Kiểm tra banned words trước (ưu tiên dùng caption_for_classification)
            cap_for_clf = item.get("caption_for_classification")
            cap_for_banned = cap_for_clf if cap_for_clf else item.get("caption", "")
            banned_result = None
            if cap_for_banned:
                banned_result = check_banned_words(cap_for_banned, load_banned_words())
            
            # Bước 2: Gán nhãn ML (ưu tiên sử dụng caption_for_classification nếu có)
            cap = cap_for_clf if cap_for_clf else item.get("caption", "")
            if cap:
                try:
                    pred = classify_text(cap, clf_dir)
                except Exception as e:
                    pred = {"label_id": None, "score": 0.0, "probs": [], "error": str(e)}
            else:
                pred = {"label_id": None, "score": 0.0, "probs": []}
            
            new_item = dict(item)
            
            # Ưu tiên nhãn từ banned words nếu có vi phạm
            if banned_result and banned_result.get("label_id") == 1:
                new_item["label_id"] = banned_result.get("label_id")
                new_item["label_score"] = banned_result.get("score")
                if banned_result.get("matched_words"):
                    new_item["matched_banned_words"] = banned_result.get("matched_words")
            else:
                # Nếu không có vi phạm từ cấm, dùng nhãn từ ML classifier
                new_item["label_id"] = pred.get("label_id")
                new_item["label_score"] = pred.get("score")
                # Xóa matched_banned_words nếu nhãn mới không phải vi phạm (label_id != 1)
                if pred.get("label_id") != 1:
                    # Xóa matched_banned_words nếu có (dùng pop để tránh lỗi nếu key không tồn tại)
                    if "matched_banned_words" in new_item:
                        del new_item["matched_banned_words"]
            
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


@app.route("/moderate/<job_id>", methods=["POST"])
def moderate_job(job_id: str):
    """Kiểm duyệt mô tả: Gán nhãn từ cấm trước, sau đó chỉ gán nhãn ML cho những mô tả đạt chuẩn."""
    from flask import jsonify
    
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        return jsonify({"success": False, "message": "Không tìm thấy file result.json"}), 404
    
    # classifier directory
    clf_dir = request.form.get(
        "clf_dir",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "TVC-AI", "output_moderation")),
    )
    
    try:
        # Load banned words
        banned_words = load_banned_words()
        if not banned_words:
            return jsonify({
                "success": False,
                "message": "Không thể tải danh sách từ cấm. Vui lòng kiểm tra file data_ban.txt."
            }), 400
        
        # Load data
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        audio_sentences = data.get("audio", {}).get("sentences", [])
        
        # Bước 1: Gán nhãn từ cấm cho tất cả frames
        updated_frames = []
        banned_violations_count = 0
        for item in frames:
            # Ưu tiên sử dụng caption_for_classification nếu có (caption rút gọn cho kiểm duyệt)
            # Nếu không có thì mới dùng caption đầy đủ
            cap_for_clf = item.get("caption_for_classification")
            cap = cap_for_clf if cap_for_clf else item.get("caption", "")
            if cap:
                result = check_banned_words(cap, banned_words)
                new_item = dict(item)
                new_item["label_id"] = result.get("label_id")
                new_item["label_score"] = result.get("score")
                if result.get("matched_words"):
                    new_item["matched_banned_words"] = result.get("matched_words")
                    banned_violations_count += 1
                else:
                    # Nếu không có từ cấm, xóa matched_banned_words cũ (nếu có)
                    new_item.pop("matched_banned_words", None)
                updated_frames.append(new_item)
            else:
                updated_frames.append(item)
        
        # Bước 2: Chỉ gán nhãn ML cho những frame đạt chuẩn (label_id != 1)
        # Nếu đã bị từ cấm đánh là vi phạm, giữ nguyên nhãn đó
        ml_labeled_count = 0
        for item in updated_frames:
            # Chỉ gán nhãn ML nếu chưa bị từ cấm đánh là vi phạm
            if item.get("label_id") != 1:
                # Ưu tiên sử dụng caption_for_classification nếu có (caption rút gọn cho model gán nhãn)
                # Nếu không có thì mới dùng caption đầy đủ
                cap_for_clf = item.get("caption_for_classification")
                cap = cap_for_clf if cap_for_clf else item.get("caption", "")
                if cap and clf_dir and os.path.isdir(clf_dir):
                    try:
                        pred = classify_text(cap, clf_dir)
                        # Cập nhật nhãn ML (chỉ khi chưa bị từ cấm đánh dấu)
                        item["label_id"] = pred.get("label_id")
                        item["label_score"] = pred.get("score")
                        ml_labeled_count += 1
                        # Xóa matched_banned_words nếu ML classifier đánh là đạt chuẩn
                        if pred.get("label_id") != 1:
                            item.pop("matched_banned_words", None)
                    except Exception as e:
                        # Nếu ML classifier lỗi, giữ nguyên nhãn từ cấm
                        pass
        
        # Xử lý audio sentences tương tự
        audio_banned_violations_count = 0
        audio_ml_labeled_count = 0
        updated_audio_sentences = []
        if audio_sentences:
            # Bước 1: Gán nhãn từ cấm
            for sent in audio_sentences:
                text = sent.get("text", "")
                if text:
                    result = check_banned_words(text, banned_words)
                    new_sent = dict(sent)
                    new_sent["label_id"] = result.get("label_id")
                    new_sent["label_score"] = result.get("score")
                    if result.get("matched_words"):
                        new_sent["matched_banned_words"] = result.get("matched_words")
                        audio_banned_violations_count += 1
                    else:
                        new_sent.pop("matched_banned_words", None)
                    updated_audio_sentences.append(new_sent)
                else:
                    updated_audio_sentences.append(sent)
            
            # Bước 2: Chỉ gán nhãn ML cho những câu đạt chuẩn
            for sent in updated_audio_sentences:
                if sent.get("label_id") != 1:
                    text = sent.get("text", "")
                    if text and clf_dir and os.path.isdir(clf_dir):
                        try:
                            pred = classify_text(text, clf_dir)
                            sent["label_id"] = pred.get("label_id")
                            sent["label_score"] = pred.get("score")
                            audio_ml_labeled_count += 1
                            if pred.get("label_id") != 1:
                                sent.pop("matched_banned_words", None)
                        except Exception:
                            pass
            
            if "audio" in data:
                data["audio"]["sentences"] = updated_audio_sentences
                # Recalculate audio summary
                total_sent = len(updated_audio_sentences)
                s_ok = sum(1 for s in updated_audio_sentences if s.get("label_id") == 0)
                s_violate = sum(1 for s in updated_audio_sentences if s.get("label_id") == 1)
                s_unknown = total_sent - s_ok - s_violate
                data["audio"]["sentence_summary"] = {
                    "total_sentences": total_sent,
                    "num_compliant": s_ok,
                    "num_violations": s_violate,
                    "num_unknown": s_unknown,
                    "percent_compliant": round((s_ok / total_sent * 100.0), 2) if total_sent else 0.0,
                    "percent_violations": round((s_violate / total_sent * 100.0), 2) if total_sent else 0.0,
                }
        
        data["frames"] = updated_frames
        
        # Compute and persist summary to result.json
        try:
            total = len(updated_frames)
            num_ok = sum(1 for fr in updated_frames if fr.get("label_id") == 0)
            num_violate = sum(1 for fr in updated_frames if fr.get("label_id") == 1)
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
        
        message = f"✓ Kiểm duyệt hoàn tất! Phát hiện {banned_violations_count} frame và {audio_banned_violations_count} câu audio vi phạm từ cấm. Đã gán nhãn ML cho {ml_labeled_count} frame và {audio_ml_labeled_count} câu audio đạt chuẩn."
        return jsonify({
            "success": True,
            "message": message,
            "stats": {
                "banned_frame_violations": banned_violations_count,
                "banned_audio_violations": audio_banned_violations_count,
                "ml_labeled_frames": ml_labeled_count,
                "ml_labeled_audio": audio_ml_labeled_count,
                "total_frames": len(updated_frames),
                "total_audio_sentences": len(updated_audio_sentences) if audio_sentences else 0
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Lỗi khi kiểm duyệt: {str(e)}"
        }), 500


@app.route("/label-banned-words/<job_id>", methods=["POST"])
def label_banned_words(job_id: str):
    """Gán nhãn vi phạm dựa trên danh sách từ cấm trong data_ban.txt."""
    from flask import jsonify
    
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    meta_path = os.path.join(out_dir, "result.json")
    if not os.path.exists(meta_path):
        return jsonify({"success": False, "message": "Không tìm thấy file result.json"}), 404
    
    try:
        # Load banned words
        banned_words = load_banned_words()
        if not banned_words:
            return jsonify({
                "success": False,
                "message": "Không thể tải danh sách từ cấm. Vui lòng kiểm tra file data_ban.txt."
            }), 400
        
        # Load data
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        audio_sentences = data.get("audio", {}).get("sentences", [])
        
        updated_frames = []
        violations_count = 0
        for item in frames:
            # Ưu tiên sử dụng caption_for_classification nếu có (caption rút gọn cho kiểm duyệt)
            # Nếu không có thì mới dùng caption đầy đủ
            cap_for_clf = item.get("caption_for_classification")
            cap = cap_for_clf if cap_for_clf else item.get("caption", "")
            if cap:
                result = check_banned_words(cap, banned_words)
                new_item = dict(item)
                new_item["label_id"] = result.get("label_id")
                new_item["label_score"] = result.get("score")
                if result.get("matched_words"):
                    new_item["matched_banned_words"] = result.get("matched_words")
                    violations_count += 1
                else:
                    # Nếu không có từ cấm, xóa matched_banned_words cũ (nếu có)
                    new_item.pop("matched_banned_words", None)
                updated_frames.append(new_item)
            else:
                updated_frames.append(item)
        
        # Also label audio sentences if available
        audio_violations_count = 0
        updated_audio_sentences = []
        if audio_sentences:
            for sent in audio_sentences:
                text = sent.get("text", "")
                if text:
                    result = check_banned_words(text, banned_words)
                    new_sent = dict(sent)
                    new_sent["label_id"] = result.get("label_id")
                    new_sent["label_score"] = result.get("score")
                    if result.get("matched_words"):
                        new_sent["matched_banned_words"] = result.get("matched_words")
                        audio_violations_count += 1
                    updated_audio_sentences.append(new_sent)
                else:
                    updated_audio_sentences.append(sent)
            
            if "audio" in data:
                data["audio"]["sentences"] = updated_audio_sentences
                # Recalculate audio summary
                total_sent = len(updated_audio_sentences)
                s_ok = sum(1 for s in updated_audio_sentences if s.get("label_id") == 0)
                s_violate = sum(1 for s in updated_audio_sentences if s.get("label_id") == 1)
                s_unknown = total_sent - s_ok - s_violate
                data["audio"]["sentence_summary"] = {
                    "total_sentences": total_sent,
                    "num_compliant": s_ok,
                    "num_violations": s_violate,
                    "num_unknown": s_unknown,
                    "percent_compliant": round((s_ok / total_sent * 100.0), 2) if total_sent else 0.0,
                    "percent_violations": round((s_violate / total_sent * 100.0), 2) if total_sent else 0.0,
                }
        
        data["frames"] = updated_frames
        
        # Compute and persist summary to result.json
        try:
            total = len(updated_frames)
            num_ok = sum(1 for fr in updated_frames if fr.get("label_id") == 0)
            num_violate = sum(1 for fr in updated_frames if fr.get("label_id") == 1)
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
        
        message = f"✓ Đã gán nhãn thành công! Phát hiện {violations_count} frame và {audio_violations_count} câu audio vi phạm."
        return jsonify({
            "success": True,
            "message": message,
            "stats": {
                "frame_violations": violations_count,
                "audio_violations": audio_violations_count,
                "total_frames": len(updated_frames),
                "total_audio_sentences": len(updated_audio_sentences) if audio_sentences else 0
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Lỗi khi gán nhãn: {str(e)}"
        }), 500


@app.route("/outputs/<job_id>/frames/<path:filename>")
def serve_frame(job_id: str, filename: str):
    directory = os.path.join(OUTPUT_DIR, job_id, "frames")
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    # Start Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
