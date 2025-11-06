"""
Module xử lý tạo mô tả hình ảnh (Image Captioning).
Bao gồm logic captioning, post-processing và formatting.
"""

import os
import re
from typing import Dict, Optional

# Model mặc định cho captioning (thống nhất cho tất cả)
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"

# Import SVO extraction function
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TVC-AI"))
    from convert_to_svo import extract_svo
except ImportError:
    extract_svo = None

# Optional: CLIP verify for reranking captions
try:
    from clip_moderation import verify_caption_quality  # returns {similarity, is_valid, ...}
    _CLIP_VERIFY_AVAILABLE = True
except Exception:
    verify_caption_quality = None
    _CLIP_VERIFY_AVAILABLE = False

# Cache for local HF captioning pipelines to avoid reloading per frame
_CAPTION_PIPELINE_CACHE: Dict[str, object] = {}


def _get_local_captioner(model_name: str):
    """Create or fetch a cached HF pipeline for image captioning."""
    import warnings
    from transformers import pipeline
    try:
        import torch  # prefer GPU if available
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    
    # Suppress pipeline batch efficiency warning
    warnings.filterwarnings("ignore", message=".*using the pipelines sequentially on GPU.*")

    use_model = model_name or DEFAULT_CAPTION_MODEL

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


def _remove_repetition(text: str, max_repeat: int = 2) -> str:
    """Remove excessive repetition in text (e.g., 'mang mang mang' -> 'mang').
    
    Args:
        text: Input text that may contain repetition
        max_repeat: Maximum number of times a word can repeat consecutively
    
    Returns:
        Cleaned text with repetition removed
    """
    if not text:
        return text
    
    # Split into words while preserving punctuation
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Step 1: Remove consecutive duplicate words
    cleaned = []
    prev_word_normalized = None
    repeat_count = 0
    
    for word in words:
        # Normalize word for comparison (lowercase, remove punctuation)
        word_normalized = re.sub(r'[^\w]', '', word.lower())
        if word_normalized and word_normalized == prev_word_normalized:
            repeat_count += 1
            if repeat_count < max_repeat:
                cleaned.append(word)
        else:
            prev_word_normalized = word_normalized if word_normalized else None
            repeat_count = 0
            cleaned.append(word)
    
    result = " ".join(cleaned)
    
    # Step 2: Remove patterns like "word word word" (3+ consecutive) using regex
    # Pattern matches word boundary followed by same word 2+ more times
    pattern = r'\b(\w+)(?:\s+\1){2,}\b'
    result = re.sub(pattern, r'\1', result)
    
    # Step 3: Remove repeated phrases (e.g., "mang & vincent mang & vincent")
    # Split and check for 2-3 word phrase repetition
    phrases = result.split()
    if len(phrases) >= 6:
        # Check for 2-word phrase repetition
        i = 0
        while i < len(phrases) - 4:
            phrase2 = " ".join(phrases[i:i+2]).lower()
            # Count how many times this phrase appears in the next 10 words
            count = 0
            for j in range(i, min(i + 10, len(phrases) - 1)):
                check_phrase = " ".join(phrases[j:j+2]).lower()
                if check_phrase == phrase2:
                    count += 1
            if count >= 3:
                # Found excessive repetition, remove duplicates
                new_phrases = phrases[:i+2]
                j = i + 2
                while j < len(phrases) - 1:
                    check_phrase = " ".join(phrases[j:j+2]).lower()
                    if check_phrase != phrase2:
                        new_phrases.append(phrases[j])
                        j += 1
                    else:
                        j += 2  # Skip the duplicate phrase
                if j < len(phrases):
                    new_phrases.extend(phrases[j:])
                result = " ".join(new_phrases)
                phrases = result.split()
                break
            i += 1
    
    return result.strip()


def _remove_vague_references(text: str) -> str:
    """Loại bỏ các cụm từ mơ hồ như 'trên đó', 'ở đó', 'này', 'kia' 
    vì chúng khiến model không thể gán nhãn đúng.
    
    Ví dụ: 
    - 'một chai màu xanh lá cây có logo màu trắng trên đó' 
      -> 'một chai màu xanh lá cây có logo màu trắng'
    """
    if not text:
        return text
    
    # Danh sách các cụm từ mơ hồ cần loại bỏ (sắp xếp theo độ dài giảm dần)
    vague_phrases = [
        'trên đó', 'trên này', 'trên kia',
        'ở đó', 'ở này', 'ở kia',
        'trong đó', 'trong này', 'trong kia',
        'với đó', 'với này', 'với kia',
        'từ đó', 'từ này', 'từ kia',
        'vào đó', 'vào này', 'vào kia',
        'phía trên', 'phía dưới', 'phía trong', 'phía ngoài',
        'bên trên', 'bên dưới', 'bên trong', 'bên ngoài',
        'ở trên', 'ở dưới', 'ở trong', 'ở ngoài',
        'có trên', 'có ở', 'có trong', 'có với',
        'đó', 'này', 'kia', 'đây', 'ấy',
    ]
    
    words = text.split()
    if len(words) <= 2:
        return text
    
    # Loại bỏ các cụm từ mơ hồ ở cuối câu
    result = text
    removed = True
    
    while removed:
        removed = False
        words = result.split()
        
        # Kiểm tra cụm từ dài trước (2 từ)
        for phrase in vague_phrases:
            phrase_words = phrase.split()
            if len(phrase_words) == 2 and len(words) >= 2:
                # Kiểm tra match ở cuối
                end_words = [w.lower().strip('.,!?;:') for w in words[-2:]]
                if end_words == phrase_words:
                    words = words[:-2]
                    result = " ".join(words).strip()
                    result = re.sub(r'\s*[.,!?;:]+\s*$', '', result).strip()
                    removed = True
                    break
        
        # Nếu không match cụm 2 từ, kiểm tra từ đơn
        if not removed:
            if words:
                last_word = words[-1].lower().strip('.,!?;:')
                if last_word in ['đó', 'này', 'kia', 'đây', 'ấy']:
                    words = words[:-1]
                    result = " ".join(words).strip()
                    result = re.sub(r'\s*[.,!?;:]+\s*$', '', result).strip()
                    removed = True
    
    return result if result else text


def _ensure_single_verb_svo(text: str) -> str:
    """Đảm bảo mô tả chỉ có một động từ trong định dạng SVO.
    
    Xử lý đặc biệt: Nếu có động từ về trang phục/phụ kiện (mặc, đeo) và động từ hành động khác,
    chuyển động từ trang phục thành giới từ "với".
    Ví dụ: 
    - 'hai người phụ nữ mặc trang phục công sở nói chuyện với nhau' 
      -> 'hai người phụ nữ mặc trang phục công sở'
    - 'người phụ nữ mặc váy màu xanh lá cây cầm một cái chai'
      -> 'người phụ nữ với váy màu xanh cầm một cái chai'
    """
    if not text:
        return text
    
    # Động từ về trang phục/phụ kiện (sẽ được chuyển thành "với")
    clothing_verbs = ['mặc', 'đeo', 'mang', 'đội', 'quàng']
    
    # Động từ hành động chính (ưu tiên giữ lại)
    action_verbs = [
        'cầm', 'nắm', 'vẽ', 'ăn', 'uống', 'nói', 'nghe', 'nhìn', 
        'đứng', 'ngồi', 'đi', 'chạy', 'nhảy', 'quay', 'mở', 'đóng', 'đọc', 'viết',
        'làm', 'chơi', 'học', 'dạy', 'mua', 'bán', 'cho', 'nhận', 'đưa', 'lấy',
        'đặt', 'để', 'giữ', 'thả', 'ném', 'bắt', 'đánh', 'chạm', 'sờ', 'ôm',
        'hôn', 'cười', 'khóc', 'hét', 'gọi', 'trả lời', 'hỏi', 'giải thích',
        'xem', 'chiếu', 'hiển thị', 'mô tả', 'thể hiện', 'biểu diễn', 'trình bày',
        'nói chuyện', 'trò chuyện', 'gặp gỡ', 'gặp nhau'
    ]
    
    # Tất cả động từ (bao gồm cả động từ ghép)
    all_verbs = clothing_verbs + action_verbs
    
    # Thử dùng extract_svo nếu có (ưu tiên)
    if extract_svo:
        try:
            s, v, o = extract_svo(text)
            if s and v:
                # Nếu có đầy đủ SVO, ghép lại
                if o:
                    return f"{s} {v} {o}".strip()
                else:
                    return f"{s} {v}".strip()
        except Exception:
            pass
    
    words = text.split()
    if len(words) <= 3:
        return text  # Quá ngắn, không cần xử lý
    
    # Tìm tất cả động từ trong câu
    verb_positions = []  # [(index, verb, is_clothing, is_phrase)]
    
    for i in range(len(words)):
        word_clean = words[i].lower().strip('.,!?;:')
        
        # Kiểm tra động từ đơn
        if word_clean in all_verbs:
            is_clothing = word_clean in clothing_verbs
            verb_positions.append((i, word_clean, is_clothing, False))
        # Kiểm tra động từ ghép (như "nói chuyện")
        elif i + 1 < len(words):
            next_word = words[i + 1].lower().strip('.,!?;:')
            verb_phrase = f"{word_clean} {next_word}"
            if verb_phrase in action_verbs:
                verb_positions.append((i, verb_phrase, False, True))
    
    if len(verb_positions) == 0:
        return text  # Không tìm thấy động từ
    
    # Nếu có cả động từ trang phục và động từ hành động
    clothing_verb = None
    action_verb = None
    
    for idx, verb, is_clothing, is_phrase in verb_positions:
        if is_clothing and clothing_verb is None:
            clothing_verb = (idx, verb, is_phrase)
        elif not is_clothing and action_verb is None:
            action_verb = (idx, verb, is_phrase)
    
    # Nếu có cả hai loại động từ, chuyển động từ trang phục thành "với"
    if clothing_verb and action_verb:
        clothing_idx, clothing_v, clothing_is_phrase = clothing_verb
        action_idx, action_v, action_is_phrase = action_verb
        
        # Tìm phần object của động từ trang phục (từ sau động từ đến trước động từ hành động)
        clothing_obj_start = clothing_idx + (2 if clothing_is_phrase else 1)
        clothing_obj_end = action_idx
        clothing_object = " ".join(words[clothing_obj_start:clothing_obj_end])
        
        # Rút gọn object (bỏ "màu xanh lá cây" -> "màu xanh", "lá cây" -> "")
        if "màu xanh lá cây" in clothing_object.lower():
            clothing_object = re.sub(r'\b(màu\s+)?xanh\s+lá\s+cây\b', 'màu xanh', clothing_object, flags=re.IGNORECASE)
        clothing_object = re.sub(r'\s+', ' ', clothing_object).strip()
        
        # Tìm phần subject (từ đầu đến trước động từ trang phục)
        subject = " ".join(words[:clothing_idx]).strip()
        
        # Tìm phần object của động từ hành động (từ sau động từ đến cuối)
        action_obj_start = action_idx + (2 if action_is_phrase else 1)
        action_object = " ".join(words[action_obj_start:])
        
        # Loại bỏ từ nối ở cuối object
        action_object = re.sub(r'\s+(và|với|về|cho|từ|vào|trong|trên|dưới|sau|trước|với nhau|với người)\s*$', '', action_object)
        
        # Ghép lại: subject + "với" + object_trang_phục + động_từ_hành_động + object_hành_động
        if clothing_object:
            result = f"{subject} với {clothing_object} {action_v} {action_object}".strip()
        else:
            result = f"{subject} {action_v} {action_object}".strip()
        
        return re.sub(r'\s+', ' ', result).strip()
    
    # Nếu chỉ có một động từ, xử lý như cũ
    verb_found_index, verb, verb_is_phrase = verb_positions[0][0], verb_positions[0][1], verb_positions[0][3]
    
    # Tìm vị trí kết thúc hợp lý (trước động từ thứ 2 nếu có)
    end_index = len(words)
    if len(verb_positions) > 1:
        end_index = verb_positions[1][0]
    
    # Lấy phần từ đầu đến end_index
    result = " ".join(words[:end_index]).strip()
    # Loại bỏ các từ nối ở cuối nếu có
    result = re.sub(r'\s+(và|với|về|cho|từ|vào|trong|trên|dưới|sau|trước|với nhau|với người)\s*$', '', result)
    return result


def caption_image_local(image_path: str, model_name: str, use_object_detection: bool = False, detect_objects_func=None, console=None) -> str:
    """Caption an image using a local Hugging Face pipeline model (cached).
    Optionally integrates object detection (YOLO) for richer descriptions.
    Returns caption in SVO format (Subject Verb Object), e.g., 'người đàn ông vẽ tranh'.
    
    Args:
        image_path: Path to image file
        model_name: Hugging Face model name for captioning
        use_object_detection: Whether to use YOLO object detection
        detect_objects_func: Function to detect objects (optional, for dependency injection)
        console: Console object for logging (optional)
    """
    # 1. Generate base caption using image-to-text model
    captioner = _get_local_captioner(model_name)
    # Instruction prompt to steer model to SVO format (Vietnamese)
    prompt = os.environ.get(
        "CAPTION_PROMPT_VI",
        "Describe the image concisely in SVO: SUBJECT - ONE VERB - OBJECT. Use ONLY ONE verb. Examples: 'a man paints', 'a girl eats ice cream', 'two women talk'. Do NOT use multiple verbs like 'wears and holds'. Focus on the single main action.",
    )
    # Decoding params
    num_beams = int(os.environ.get("CAPTION_NUM_BEAMS", "10"))
    temperature = float(os.environ.get("CAPTION_TEMPERATURE", "0.3"))
    max_new_tokens = int(os.environ.get("CAPTION_MAX_TOKENS", "100"))
    repetition_penalty = float(os.environ.get("CAPTION_REPETITION_PENALTY", "1.2"))
    no_repeat_ngram_size = int(os.environ.get("CAPTION_NO_REPEAT_NGRAM", "3"))

    # Support generating multiple candidates and reranking with CLIP
    num_candidates = int(os.environ.get("CAPTION_NUM_CANDIDATES", "1"))
    candidates: list[str] = []
    try:
        # Try to request multiple sequences directly
        result = captioner(
            image_path,
            prompt=prompt,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_candidates,
        )
        if isinstance(result, list):
            candidates = [it.get("generated_text", "").strip() for it in result]
    except TypeError:
        # Fallback pathways
        try:
            result = captioner(
                image_path,
                prompt=prompt,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            if isinstance(result, list) and result:
                candidates = [result[0].get("generated_text", "").strip()]
        except TypeError:
            result = captioner(image_path, max_new_tokens=max_new_tokens)
            if isinstance(result, list) and result:
                candidates = [result[0].get("generated_text", "").strip()]

    # If we still need more candidates, attempt to sample with varied temperature
    if num_candidates > 1 and len(candidates) < num_candidates:
        extra_needed = num_candidates - len(candidates)
        for i in range(extra_needed):
            try:
                res_i = captioner(
                    image_path,
                    prompt=prompt,
                    num_beams=max(1, num_beams // 2),
                    temperature=min(1.0, temperature + 0.2 * (i + 1)),
                    max_new_tokens=max_new_tokens,
                )
                if isinstance(res_i, list) and res_i:
                    candidates.append(res_i[0].get("generated_text", "").strip())
            except Exception:
                break

    # Clean up candidates and rerank
    processed_candidates: list[str] = []
    for text in candidates:
        if not text:
            continue
        t = _remove_repetition(text, max_repeat=2)
        t = _remove_vague_references(t)
        t = _ensure_single_verb_svo(t)
        processed_candidates.append(t)

    base_caption = ""
    if processed_candidates:
        if _CLIP_VERIFY_AVAILABLE:
            # Rerank by CLIP similarity
            scored: list[tuple[float, str]] = []
            for t in processed_candidates:
                try:
                    vres = verify_caption_quality(
                        image_path,
                        t,
                        threshold=float(os.environ.get("CLIP_VERIFY_THRESHOLD", "0.7")),
                    )
                    scored.append((float(vres.get("similarity", 0.0)), t))
                except Exception:
                    scored.append((0.0, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            base_caption = scored[0][1]
        else:
            base_caption = processed_candidates[0]
    
    # 2. Enhance with Object Detection (optional)
    detected_objects = []
    if use_object_detection and detect_objects_func:
        try:
            yolo_model = os.environ.get("YOLO_MODEL", "yolov8n.pt")
            conf_threshold = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
            detected_objects = detect_objects_func(image_path, model_name=yolo_model, conf_threshold=conf_threshold)
        except Exception as e:
            if console:
                console.print(f"[yellow]Object detection skipped: {e}[/yellow]")
    
    # 3. Combine information into enhanced description
    enhancement_parts = []
    
    # 4. Build final enhanced caption
    if enhancement_parts:
        enhanced_context = " | ".join(enhancement_parts)
        # Combine base caption with enhancements
        if base_caption:
            enhanced_caption = f"{base_caption} ({enhanced_context})"
        else:
            enhanced_caption = enhanced_context
    else:
        enhanced_caption = base_caption
    
    # 6. Convert to SVO format using extract_svo if available
    if extract_svo and enhanced_caption:
        try:
            s, v, o = extract_svo(enhanced_caption)
            # Format as "subject verb object"
            if s or v or o:
                parts = [p for p in [s, v, o] if p and p.strip()]
                if parts:
                    svo_text = " ".join(parts)
                    return svo_text
        except Exception:
            # If SVO extraction fails, return enhanced caption
            pass
    
    return enhanced_caption if enhanced_caption else ""

