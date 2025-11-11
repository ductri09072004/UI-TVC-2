"""
Module xử lý tạo mô tả hình ảnh (Image Captioning).
Bao gồm logic captioning, post-processing và formatting.
"""

import os
import re
from typing import Dict, Optional

# Path đến model đã fine-tune (bắt buộc)
FINETUNED_MODEL_PATH = r"C:\Users\Kris\blip-image-LLM\output\blip-flickr30k-local"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"  # Base model (chỉ dùng để load LoRA adapter)

# Import SVO extraction function
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TVC-AI"))
    from convert_to_svo import extract_svo
except ImportError:
    extract_svo = None


# Cache for local HF captioning pipelines to avoid reloading per frame
_CAPTION_PIPELINE_CACHE: Dict[str, object] = {}
# Cache for OCR readers
_OCR_READER_CACHE: Dict[str, object] = {}


def _get_local_captioner(model_name: str):
    """Create or fetch a cached HF pipeline for image captioning.
    Supports both base models and fine-tuned LoRA adapters.
    """
    import warnings
    from transformers import pipeline, BlipForConditionalGeneration, BlipProcessor
    from peft import PeftModel
    import os
    try:
        import torch  # prefer GPU if available
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    
    # Suppress pipeline batch efficiency warning
    warnings.filterwarnings("ignore", message=".*using the pipelines sequentially on GPU.*")

    # Luôn dùng model đã fine-tune
    # Nếu model_name được chỉ định, dùng nó; nếu không, dùng FINETUNED_MODEL_PATH
    if model_name and os.path.isdir(model_name):
        use_model = model_name
    else:
        # Dùng path mặc định đến model đã fine-tune
        use_model = FINETUNED_MODEL_PATH
        print(f"Using fine-tuned model from: {FINETUNED_MODEL_PATH}")
    
    if use_model in _CAPTION_PIPELINE_CACHE:
        return _CAPTION_PIPELINE_CACHE[use_model]

    device = 0 if has_cuda else -1
    
    # Check if model_name is a path to a fine-tuned LoRA adapter
    # Look for adapter_config.json or adapter_model.safetensors in the path
    is_lora_adapter = False
    adapter_path = None
    
    if os.path.isdir(use_model):
        # Check if it's a LoRA adapter directory
        adapter_config = os.path.join(use_model, "adapter_config.json")
        adapter_model = os.path.join(use_model, "adapter_model.safetensors")
        if os.path.exists(adapter_config) or os.path.exists(adapter_model):
            is_lora_adapter = True
            adapter_path = use_model
    
    # If using LoRA adapter, load base model + adapter manually
    if is_lora_adapter:
        try:
            base_model_name = DEFAULT_CAPTION_MODEL
            print(f"Loading fine-tuned BLIP model from: {adapter_path}")
            print(f"Base model: {base_model_name}")
            
            # Load base model
            base_model = BlipForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if has_cuda else torch.float32,
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, adapter_path)
            if has_cuda:
                model = model.to("cuda")
            model.eval()
            
            # Load processor
            processor = BlipProcessor.from_pretrained(base_model_name)
            
            # Create a custom pipeline-like wrapper
            class LoRACaptioner:
                def __init__(self, model, processor, device_idx):
                    self.model = model
                    self.processor = processor
                    self.device_idx = device_idx
                
                def __call__(self, image_path, **kwargs):
                    from PIL import Image
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    
                    if has_cuda:
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    # Extract generation parameters
                    max_new_tokens = kwargs.get("max_new_tokens", 150)  # Tăng từ 100 lên 150 để tránh câu bị ngắt
                    num_beams = kwargs.get("num_beams", 5)
                    temperature = kwargs.get("temperature", 0.7)
                    repetition_penalty = kwargs.get("repetition_penalty", 1.2)
                    num_return_sequences = kwargs.get("num_return_sequences", 1)
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_length=max_new_tokens,
                            num_beams=num_beams,
                            temperature=temperature,
                            repetition_penalty=repetition_penalty,
                            num_return_sequences=num_return_sequences,
                        )
                    
                    generated_texts = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    
                    if num_return_sequences > 1:
                        return [{"generated_text": text.strip()} for text in generated_texts]
                    else:
                        return [{"generated_text": generated_texts[0].strip()}]
            
            captioner = LoRACaptioner(model, processor, device)
            _CAPTION_PIPELINE_CACHE[use_model] = captioner
            print(f"✓ Fine-tuned model loaded successfully!")
            return captioner
        except Exception as e:
            raise RuntimeError(
                f"Failed to load fine-tuned LoRA adapter from {adapter_path}: {e}\n"
                f"Please ensure the fine-tuned model exists at: {adapter_path}\n"
                f"Model path is hardcoded in image_captioning.py: FINETUNED_MODEL_PATH"
            )
    
    # Nếu không phải LoRA adapter, raise error (không dùng base model nữa)
    raise RuntimeError(
        f"Model path '{use_model}' is not a fine-tuned LoRA adapter.\n"
        f"Expected directory containing 'adapter_config.json' or 'adapter_model.safetensors'.\n"
        f"Current path: {FINETUNED_MODEL_PATH}\n"
        f"Please update FINETUNED_MODEL_PATH in image_captioning.py if needed."
    )


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
    
    # Step 3: Remove repeated phrases and full sentence repetition
    phrases = result.split()
    
    # Step 3a: Detect and remove full sentence repetition (e.g., entire caption repeated)
    # Check if the text is essentially the same phrase/sentence repeated
    if len(phrases) >= 4:
        # Try to find if first half matches second half
        mid_point = len(phrases) // 2
        first_half = " ".join(phrases[:mid_point]).lower().strip()
        second_half = " ".join(phrases[mid_point:]).lower().strip()
        
        # Normalize for comparison (remove punctuation, extra spaces)
        first_norm = re.sub(r'[^\w\s]', '', first_half)
        second_norm = re.sub(r'[^\w\s]', '', second_half)
        first_norm = re.sub(r'\s+', ' ', first_norm).strip()
        second_norm = re.sub(r'\s+', ' ', second_norm).strip()
        
        # If halves are very similar (>80% overlap), remove duplicate
        if first_norm and second_norm:
            # Calculate similarity (simple word overlap)
            words1 = set(first_norm.split())
            words2 = set(second_norm.split())
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap > 0.8:
                    # Remove second half, keep first half
                    result = " ".join(phrases[:mid_point]).strip()
                    phrases = result.split()
    
    # Step 3b: Remove repeated phrases (2-5 word phrases)
    if len(phrases) >= 6:
        # Check for phrase repetition (2-5 words)
        for phrase_len in range(5, 1, -1):  # Check longer phrases first
            i = 0
            changed = False
            while i < len(phrases) - phrase_len * 2:
                phrase = " ".join(phrases[i:i+phrase_len]).lower()
                # Check if this phrase appears again within next 20 words
                for j in range(i + phrase_len, min(i + phrase_len + 20, len(phrases) - phrase_len + 1)):
                    check_phrase = " ".join(phrases[j:j+phrase_len]).lower()
                    # Normalize for comparison
                    phrase_norm = re.sub(r'[^\w\s]', '', phrase)
                    check_norm = re.sub(r'[^\w\s]', '', check_phrase)
                    phrase_norm = re.sub(r'\s+', ' ', phrase_norm).strip()
                    check_norm = re.sub(r'\s+', ' ', check_norm).strip()
                    
                    if phrase_norm == check_norm and len(phrase_norm) > 10:  # Only for meaningful phrases
                        # Remove duplicate phrase
                        new_phrases = phrases[:j] + phrases[j+phrase_len:]
                        result = " ".join(new_phrases).strip()
                        phrases = result.split()
                        changed = True
                        break
                if changed:
                    break
                i += 1
            if changed:
                break
    
    # Step 4: Remove repeated words at the end (e.g., "cơ thể, và cơ thể, và cơ thể")
    # Remove trailing repeated words/phrases
    if len(phrases) > 3:
        # Check last 3-5 words for repetition
        for check_len in range(min(5, len(phrases) // 2), 1, -1):
            if len(phrases) < check_len * 2:
                continue
            last_phrase = " ".join(phrases[-check_len:]).lower()
            second_last_phrase = " ".join(phrases[-check_len*2:-check_len]).lower()
            if last_phrase == second_last_phrase:
                # Remove last repeated phrase
                result = " ".join(phrases[:-check_len]).strip()
                phrases = result.split()
                break
    
    return result.strip()


def _fix_incomplete_sentence(text: str) -> str:
    """Phát hiện và sửa câu bị ngắt giữa chừng.
    
    Dấu hiệu câu bị ngắt:
    - Kết thúc bằng từ không hoàn chỉnh (1-2 ký tự, không phải từ có nghĩa)
    - Kết thúc bằng từ không có dấu câu nhưng không phải từ kết thúc tự nhiên
    - Kết thúc bằng các từ không hoàn chỉnh như "là", "và", "của", "với" (nếu đứng cuối)
    - Kết thúc bằng từ ngắn không có nghĩa (1-2 ký tự)
    
    Args:
        text: Input text có thể bị ngắt
        
    Returns:
        Text đã được sửa (loại bỏ phần cuối không hoàn chỉnh)
    """
    if not text or len(text.strip()) < 5:
        return text
    
    # Danh sách các từ kết thúc không tự nhiên (thường là dấu hiệu câu bị ngắt)
    incomplete_endings = [
        'là', 'và', 'của', 'với', 'từ', 'vào', 'trong', 'trên', 'dưới', 
        'sau', 'trước', 'ngoài', 'giữa', 'bên', 'phía', 'theo', 'cho',
        'đến', 'từ', 'về', 'bằng', 'do', 'bởi', 'như', 'nhưng', 'mà',
        'nếu', 'khi', 'để', 'được', 'bị', 'phải', 'sẽ', 'đã', 'đang',
        'có', 'không', 'chưa', 'chẳng', 'chả', 'mà', 'rằng', 'làm'
    ]
    
    words = text.strip().split()
    if len(words) <= 2:
        return text
    
    # Kiểm tra từ cuối cùng
    last_word = words[-1].lower().strip('.,!?;:')
    
    # Nếu từ cuối quá ngắn (1-2 ký tự) và không phải từ có nghĩa
    if len(last_word) <= 2 and last_word not in ['một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'em', 'anh', 'chị', 'ông', 'bà', 'cô', 'chú', 'bác']:
        # Loại bỏ từ cuối
        words = words[:-1]
        if len(words) > 0:
            return " ".join(words).strip()
        return text
    
    # Nếu từ cuối là từ kết thúc không tự nhiên (không có dấu câu)
    if last_word in incomplete_endings and not words[-1].endswith(('.', '!', '?', ',', ';', ':')):
        # Kiểm tra xem có phải là từ cuối thực sự không (không phải phần của cụm từ)
        # Nếu từ trước đó cũng là từ kết thúc không tự nhiên, có thể là câu bị ngắt
        if len(words) >= 2:
            second_last = words[-2].lower().strip('.,!?;:')
            if second_last in incomplete_endings:
                # Có 2 từ kết thúc không tự nhiên liên tiếp -> câu bị ngắt
                # Loại bỏ từ cuối
                words = words[:-1]
                if len(words) > 0:
                    return " ".join(words).strip()
    
    # Kiểm tra nếu từ cuối là từ không hoàn chỉnh (chỉ có 1-2 ký tự và không phải từ có nghĩa)
    # Ví dụ: "ng", "là", "và" ở cuối câu mà không có dấu câu
    if len(last_word) <= 3 and last_word in ['ng', 'là', 'và', 'của', 'với'] and not words[-1].endswith(('.', '!', '?', ',', ';', ':')):
        # Loại bỏ từ cuối
        words = words[:-1]
        if len(words) > 0:
            return " ".join(words).strip()
    
    # Kiểm tra nếu câu kết thúc bằng từ không hoàn chỉnh (có dấu nháy đơn hoặc dấu phẩy nhưng không hoàn chỉnh)
    # Ví dụ: "với dòng chữ 'ng" -> loại bỏ phần "'ng" hoặc toàn bộ cụm từ
    if "'" in words[-1] or '"' in words[-1]:
        # Kiểm tra xem có dấu nháy mở nhưng không đóng không
        quote_count = text.count("'") + text.count('"')
        if quote_count % 2 != 0:  # Số dấu nháy lẻ -> có dấu nháy chưa đóng
            # Tìm từ cuối cùng có dấu nháy mở
            last_word_with_quote = words[-1]
            # Nếu từ cuối có dấu nháy mở nhưng không đóng, loại bỏ từ đó
            if ("'" in last_word_with_quote and last_word_with_quote.count("'") == 1) or \
               ('"' in last_word_with_quote and last_word_with_quote.count('"') == 1):
                # Có dấu nháy mở nhưng không đóng -> loại bỏ từ cuối
                # Kiểm tra xem từ trước đó có phải là từ liên quan đến text không
                if len(words) >= 2:
                    second_last = words[-2].lower().strip('.,!?;:')
                    if second_last in ['chữ', 'text', 'dòng', 'nội', 'dung']:
                        # Từ trước là từ liên quan đến text -> có thể là cụm "với dòng chữ '..." bị ngắt
                        # Loại bỏ cả từ cuối và từ trước đó
                        words = words[:-2]
                    else:
                        # Chỉ loại bỏ từ cuối
                        words = words[:-1]
                else:
                    words = words[:-1]
                
                if len(words) > 0:
                    return " ".join(words).strip()
                return text
    
    # Kiểm tra nếu từ cuối là từ rất ngắn (1-3 ký tự) và không có dấu câu
    # Đặc biệt là các từ như "ng", "là", "và" ở cuối câu
    if len(last_word) <= 3 and last_word in ['ng', 'là', 'và', 'của', 'với', 'từ', 'vào'] and \
       not words[-1].endswith(('.', '!', '?', ',', ';', ':')):
        # Loại bỏ từ cuối
        words = words[:-1]
        if len(words) > 0:
            return " ".join(words).strip()
    
    return text.strip()


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


def _is_bad_caption(caption: str) -> bool:
    """Phát hiện caption 'lạ' khi BLIP gặp chữ tiếng Việt.
    
    Dấu hiệu của caption lạ:
    - Nhiều dấu phẩy liên tiếp (',,,,,')
    - Nhiều dấu nháy đơn ("'i', ', ', ' và '")
    - Chỉ có ký tự đơn lẻ và dấu câu
    - Tỷ lệ dấu câu quá cao (>30%)
    - Có dấu nháy đơn bao quanh text (dấu hiệu BLIP đang cố đọc chữ)
    """
    if not caption or len(caption.strip()) < 5:
        return False
    
    # Đếm dấu phẩy và dấu nháy đơn
    comma_count = caption.count(',')
    quote_count = caption.count("'")
    total_chars = len(caption)
    
    # Nếu có quá nhiều dấu phẩy (>30% ký tự)
    if comma_count > total_chars * 0.3:
        return True
    
    # Nếu có nhiều dấu nháy đơn và dấu phẩy (>20% ký tự)
    if (comma_count + quote_count) > total_chars * 0.2:
        return True
    
    # Kiểm tra pattern như "'i', ', ', ' và '"
    if "', '" in caption or "','," in caption:
        return True
    
    # Kiểm tra nhiều dấu phẩy liên tiếp
    if ",,," in caption or ",,," in caption.replace(" ", ""):
        return True
    
    # Phát hiện khi có dấu nháy đơn bao quanh text (dấu hiệu BLIP đang cố đọc chữ)
    # Pattern: "với dòng chữ 'text'"
    import re
    # Tìm pattern có dấu nháy đơn bao quanh text (ít nhất 2 ký tự)
    quoted_text_pattern = r"'[^']{2,}'"
    if re.search(quoted_text_pattern, caption):
        # Nếu có text trong dấu nháy, có thể là BLIP đang cố đọc chữ
        # Kiểm tra xem text trong dấu nháy có vẻ như sai chính tả không
        quoted_matches = re.findall(quoted_text_pattern, caption)
        for match in quoted_matches:
            text_inside = match.strip("'")
            # Nếu text ngắn (< 10 ký tự) và không phải là từ tiếng Anh/Việt phổ biến
            # thì có thể là sai chính tả
            if len(text_inside) < 10:
                # Kiểm tra xem có vẻ như là text bị sai chính tả
                # (ví dụ: "ngsai" thay vì "ngủ say")
                return True
    
    return False


def _has_text_in_caption(caption: str) -> bool:
    """Kiểm tra xem caption có chứa text (chữ) được đề cập không.
    Dấu hiệu: có dấu nháy đơn bao quanh text hoặc từ khóa 'chữ', 'text', 'dòng chữ'.
    """
    if not caption:
        return False
    
    import re
    # Tìm pattern có dấu nháy đơn bao quanh text
    quoted_text_pattern = r"'[^']{1,}'"
    if re.search(quoted_text_pattern, caption):
        return True
    
    # Kiểm tra từ khóa liên quan đến text
    text_keywords = ['chữ', 'text', 'dòng chữ', 'có chữ', 'ghi chữ', 'viết']
    caption_lower = caption.lower()
    for keyword in text_keywords:
        if keyword in caption_lower:
            return True
    
    return False


def _get_paddleocr_reader():
    """Lấy hoặc tạo PaddleOCR reader với cache."""
    global _OCR_READER_CACHE
    cache_key = "paddleocr_vie"
    
    if cache_key in _OCR_READER_CACHE:
        return _OCR_READER_CACHE[cache_key]
    
    try:
        from paddleocr import PaddleOCR
        # Khởi tạo PaddleOCR với tiếng Việt
        # use_angle_cls=True để xử lý text xoay
        # use_gpu=False để tránh conflict với BLIP
        ocr = PaddleOCR(use_angle_cls=True, lang='vie', use_gpu=False, show_log=False)
        _OCR_READER_CACHE[cache_key] = ocr
        return ocr
    except ImportError:
        return None
    except Exception:
        return None


def _get_easyocr_reader(languages: list = ["vi", "en"]):
    """Lấy hoặc tạo EasyOCR reader với cache."""
    global _OCR_READER_CACHE
    cache_key = f"easyocr_{','.join(sorted(languages))}"
    
    if cache_key in _OCR_READER_CACHE:
        return _OCR_READER_CACHE[cache_key]
    
    try:
        import easyocr
        reader = easyocr.Reader(languages, gpu=False)  # Dùng CPU để tránh conflict với BLIP
        _OCR_READER_CACHE[cache_key] = reader
        return reader
    except ImportError:
        return None
    except Exception:
        return None


def _clean_ocr_candidate(text: str, min_letters_ratio: float = 0.45) -> Optional[str]:
    """Làm sạch chuỗi text do OCR trả về và loại bỏ các mẫu 'dị'.

    Loại bỏ các ký tự không thuộc bảng chữ cái, bỏ các token giống mã hoặc noise
    (ví dụ: '1ays)', 'l|\/'), đồng thời chuẩn hóa khoảng trắng.

    Args:
        text: Chuỗi raw do OCR trả về.
        min_letters_ratio: Ngưỡng tối thiểu giữa số ký tự chữ và tổng ký tự (không tính khoảng trắng).

    Returns:
        Chuỗi đã làm sạch hoặc None nếu text bị coi là noise.
    """
    if not text:
        return None

    candidate = text.strip()
    if len(candidate) < 2:
        return None

    # Loại bỏ dấu câu dư thừa ở đầu/cuối và chuẩn hóa khoảng trắng
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = candidate.strip("\"'`.,:;!?()[]{}<>|\/\\")
    if len(candidate) < 2:
        return None

    # Nếu tất cả ký tự đều giống nhau (ví dụ '----' hoặc 'iiii'), coi như noise
    unique_chars = set(candidate.lower())
    if len(unique_chars) <= 2 and len(candidate) > 4:
        return None

    # Thay thế các ký tự không phải chữ/số (ngoại trừ dấu tiếng Việt) bằng khoảng trắng
    candidate = re.sub(r"[^0-9A-Za-zÀ-ỹ\s]", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    if len(candidate) < 2:
        return None

    no_space = candidate.replace(" ", "")
    if not no_space:
        return None

    letters = len(re.findall(r"[A-Za-zÀ-ỹ]", no_space))
    digits = len(re.findall(r"\d", no_space))
    total_chars = len(no_space)

    if letters == 0 and digits == 0:
        return None

    if letters / max(total_chars, 1) < min_letters_ratio and letters < 3:
        return None

    # Loại bỏ token dạng '123abc' hoặc 'abc123abc' thường là noise
    if re.search(r"\d+[A-Za-zÀ-ỹ]+", candidate):
        candidate = re.sub(r"\b\d+[A-Za-zÀ-ỹ]+\b", "", candidate).strip()

    if not candidate or len(candidate) < 2:
        return None

    # Bỏ các token đơn lẻ 1-2 ký tự (trừ khi là số có ý nghĩa)
    tokens = [tok for tok in candidate.split() if len(tok) > 2 or tok.isdigit()]
    if not tokens:
        return None

    cleaned = " ".join(tokens)
    if not cleaned or len(cleaned) < 2:
        return None

    return cleaned


def synthesize_captions(captions: list[str], console=None) -> str:
    """Tổng hợp nhiều caption thành 1 mô tả tổng quát.
    
    Args:
        captions: Danh sách các caption cần tổng hợp (thường là 3)
        console: Console object cho logging (optional)
    
    Returns:
        Caption tổng quát đã được tổng hợp
    """
    if not captions or len(captions) == 0:
        return ""
    if len(captions) == 1:
        return captions[0]
    
    # Loại bỏ caption trống
    valid_captions = [c.strip() for c in captions if c and c.strip()]
    if not valid_captions:
        return ""
    if len(valid_captions) == 1:
        return valid_captions[0]
    
    # Kiểm tra xem có API key không, nếu không có thì dùng rule-based luôn
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    # Phương pháp 1: Dùng LLM để tổng hợp (nếu có API key)
    if has_openai_key or has_anthropic_key:
        synthesized = _synthesize_with_llm(valid_captions, console=console)
        if synthesized:
            return synthesized
    
    # Phương pháp 2: Rule-based synthesis (fallback hoặc mặc định)
    return _synthesize_rule_based(valid_captions, console=console)


def _synthesize_with_llm(captions: list[str], console=None) -> str:
    """Tổng hợp caption bằng LLM (OpenAI, Anthropic, hoặc local LLM).
    
    Ưu tiên:
    1. OpenAI API (nếu có OPENAI_API_KEY)
    2. Anthropic API (nếu có ANTHROPIC_API_KEY)
    3. Local LLM (nếu có)
    """
    # Thử OpenAI trước
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            
            # Tạo prompt để tổng hợp
            captions_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
            prompt = f"""Bạn là một chuyên gia tóm tắt mô tả hình ảnh. Hãy tổng hợp các mô tả sau thành một mô tả tổng quát, ngắn gọn và chính xác nhất (định dạng SVO: Subject - Verb - Object):

{captions_text}

Yêu cầu:
- Chỉ dùng MỘT động từ chính
- Giữ lại thông tin quan trọng nhất từ tất cả các mô tả
- Loại bỏ thông tin trùng lặp
- Định dạng: tiếng Việt, SVO format
- Độ dài: ngắn gọn (tối đa 20 từ)

Mô tả tổng quát:"""
            
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia tóm tắt mô tả hình ảnh bằng tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            synthesized = response.choices[0].message.content.strip()
            if synthesized and len(synthesized) > 10:
                if console:
                    console.print(f"[green]✓ Tổng hợp {len(captions)} caption bằng OpenAI[/green]")
                return synthesized
        except Exception as e:
            if console:
                console.print(f"[yellow]OpenAI synthesis failed: {e}, using fallback[/yellow]")
    
    # Thử Anthropic (Claude)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            
            captions_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(captions)])
            prompt = f"""Tổng hợp các mô tả sau thành một mô tả tổng quát, ngắn gọn (SVO format, tiếng Việt):

{captions_text}

Mô tả tổng quát:"""
            
            message = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                max_tokens=150,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            synthesized = message.content[0].text.strip()
            if synthesized and len(synthesized) > 10:
                if console:
                    console.print(f"[green]✓ Tổng hợp {len(captions)} caption bằng Claude[/green]")
                return synthesized
        except Exception as e:
            if console:
                console.print(f"[yellow]Anthropic synthesis failed: {e}, using fallback[/yellow]")
    
    return ""


def _synthesize_rule_based(captions: list[str], console=None) -> str:
    """Tổng hợp caption bằng rule-based (không cần API).
    
    Phương pháp cải tiến:
    1. Tách SVO từ mỗi caption (nếu có extract_svo)
    2. Tìm các từ/cụm từ chung xuất hiện trong >= 2 caption
    3. Kết hợp Subject, Verb, Object từ các caption
    4. Ưu tiên thông tin xuất hiện nhiều lần
    5. Loại bỏ thông tin trùng lặp
    """
    if not captions:
        return ""
    if len(captions) == 1:
        return captions[0]
    
    import re
    
    # Bước 1: Thử extract SVO từ mỗi caption (nếu có hàm extract_svo)
    svos = []
    if extract_svo:
        for caption in captions:
            try:
                s, v, o = extract_svo(caption)
                if s or v or o:
                    svos.append({
                        'subject': s or '',
                        'verb': v or '',
                        'object': o or '',
                        'original': caption
                    })
            except Exception:
                svos.append({
                    'subject': '',
                    'verb': '',
                    'object': '',
                    'original': caption
                })
    
    # Nếu có đủ SVO từ các caption, tổng hợp SVO
    if len(svos) >= 2 and any(svo['subject'] or svo['verb'] or svo['object'] for svo in svos):
        # Tổng hợp Subject: lấy subject xuất hiện nhiều nhất hoặc dài nhất
        subjects = [svo['subject'] for svo in svos if svo['subject']]
        verbs = [svo['verb'] for svo in svos if svo['verb']]
        objects = [svo['object'] for svo in svos if svo['object']]
        
        # Chọn subject phổ biến nhất hoặc dài nhất
        if subjects:
            subject_counts = {}
            for s in subjects:
                s_lower = s.lower().strip()
                if s_lower:
                    subject_counts[s_lower] = subject_counts.get(s_lower, 0) + 1
            if subject_counts:
                # Ưu tiên subject xuất hiện nhiều lần, nếu bằng nhau thì chọn dài nhất
                best_subject = max(subject_counts.items(), key=lambda x: (x[1], len(x[0])))[0]
                # Tìm subject gốc (giữ nguyên chữ hoa/thường)
                for s in subjects:
                    if s.lower().strip() == best_subject:
                        final_subject = s
                        break
                else:
                    final_subject = best_subject
            else:
                final_subject = max(subjects, key=len)
        else:
            final_subject = ""
        
        # Chọn verb phổ biến nhất
        if verbs:
            verb_counts = {}
            for v in verbs:
                v_lower = v.lower().strip()
                if v_lower:
                    verb_counts[v_lower] = verb_counts.get(v_lower, 0) + 1
            if verb_counts:
                best_verb = max(verb_counts.items(), key=lambda x: (x[1], len(x[0])))[0]
                for v in verbs:
                    if v.lower().strip() == best_verb:
                        final_verb = v
                        break
                else:
                    final_verb = best_verb
            else:
                final_verb = max(verbs, key=len)
        else:
            final_verb = ""
        
        # Tổng hợp object: kết hợp các object không trùng lặp
        if objects:
            object_parts = []
            seen_objects = set()
            for o in objects:
                o_lower = o.lower().strip()
                if o_lower and o_lower not in seen_objects:
                    # Kiểm tra xem object này có chứa object khác không
                    is_subset = False
                    for seen in seen_objects:
                        if o_lower in seen or seen in o_lower:
                            is_subset = True
                            if len(o_lower) > len(seen):
                                # Object mới chi tiết hơn, thay thế
                                object_parts = [obj for obj in object_parts if obj.lower().strip() != seen]
                                seen_objects.remove(seen)
                                object_parts.append(o)
                                seen_objects.add(o_lower)
                            break
                    if not is_subset:
                        object_parts.append(o)
                        seen_objects.add(o_lower)
            
            # Kết hợp các object parts
            if len(object_parts) == 1:
                final_object = object_parts[0]
            elif len(object_parts) > 1:
                # Kết hợp các object, loại bỏ phần trùng lặp
                final_object = " ".join(object_parts)
            else:
                final_object = ""
        else:
            final_object = ""
        
        # Ghép lại thành SVO
        parts = [p for p in [final_subject, final_verb, final_object] if p and p.strip()]
        if parts:
            synthesized = " ".join(parts)
            if console:
                console.print(f"[green]✓ Tổng hợp {len(captions)} caption bằng SVO extraction[/green]")
            # Áp dụng post-processing
            synthesized = _remove_repetition(synthesized, max_repeat=2)
            synthesized = _remove_vague_references(synthesized)
            return synthesized
    
    # Bước 2: Nếu không có SVO, dùng phương pháp từ vựng
    # Tách từ và tìm từ/cụm từ chung
    all_words = []
    word_freq = {}
    phrase_freq = {}  # Cụm từ 2-3 từ
    
    for caption in captions:
        # Tách từ
        words = re.findall(r'\b\w+\b', caption)
        all_words.append(words)
        
        # Đếm tần suất từ
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 2:  # Bỏ qua từ quá ngắn
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Tìm cụm từ 2-3 từ
        if len(words) >= 2:
            for i in range(len(words) - 1):
                # Cụm 2 từ
                phrase2 = " ".join(words[i:i+2]).lower()
                if len(phrase2) > 5:  # Cụm từ có ý nghĩa
                    phrase_freq[phrase2] = phrase_freq.get(phrase2, 0) + 1
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    # Cụm 3 từ
                    phrase3 = " ".join(words[i:i+3]).lower()
                    if len(phrase3) > 8:
                        phrase_freq[phrase3] = phrase_freq.get(phrase3, 0) + 1
    
    # Tìm từ/cụm từ xuất hiện trong >= 2 caption
    common_words = {w: freq for w, freq in word_freq.items() if freq >= 2}
    common_phrases = {p: freq for p, freq in phrase_freq.items() if freq >= 2}
    
    # Bước 3: Xây dựng caption tổng hợp
    # Ưu tiên: cụm từ chung > từ chung > caption dài nhất
    
    if common_phrases:
        # Có cụm từ chung, ưu tiên sử dụng
        # Sắp xếp theo tần suất và độ dài
        sorted_phrases = sorted(common_phrases.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        
        # Tìm caption chứa nhiều cụm từ chung nhất
        best_caption = None
        max_phrase_score = 0
        
        for caption in captions:
            caption_lower = caption.lower()
            score = sum(freq for phrase, freq in common_phrases.items() if phrase in caption_lower)
            if score > max_phrase_score:
                max_phrase_score = score
                best_caption = caption
        
        if best_caption:
            synthesized = best_caption
        else:
            # Fallback: chọn caption dài nhất
            synthesized = max(captions, key=len)
    elif common_words:
        # Có từ chung, chọn caption chứa nhiều từ chung nhất
        best_caption = None
        max_word_score = 0
        
        for caption in captions:
            words_in_caption = set(re.findall(r'\b\w+\b', caption.lower()))
            score = len(words_in_caption & set(common_words.keys()))
            if score > max_word_score:
                max_word_score = score
                best_caption = caption
        
        if best_caption:
            synthesized = best_caption
        else:
            synthesized = max(captions, key=len)
    else:
        # Không có từ/cụm từ chung, chọn caption dài nhất và chi tiết nhất
        synthesized = max(captions, key=len)
    
    # Bước 4: Bổ sung thông tin từ các caption khác (nếu thiếu)
    # Tìm các từ quan trọng từ caption khác chưa có trong synthesized
    synthesized_words = set(re.findall(r'\b\w+\b', synthesized.lower()))
    
    for caption in captions:
        if caption == synthesized:
            continue
        caption_words = set(re.findall(r'\b\w+\b', caption.lower()))
        # Tìm từ quan trọng (danh từ, tính từ) chưa có
        new_words = caption_words - synthesized_words
        # Chỉ thêm từ quan trọng (dài > 3 ký tự, xuất hiện trong common_words)
        important_new_words = [w for w in new_words if len(w) > 3 and w in common_words]
        if important_new_words and len(important_new_words) <= 3:
            # Thêm một vài từ quan trọng vào cuối (nếu không làm caption quá dài)
            if len(synthesized) < 100:
                # Tìm từ gốc trong caption để giữ nguyên chữ hoa/thường
                for word in important_new_words:
                    for orig_word in re.findall(r'\b\w+\b', caption):
                        if orig_word.lower() == word:
                            synthesized += " " + orig_word
                            break
    
    # Bước 5: Post-processing
    synthesized = _remove_repetition(synthesized, max_repeat=2)
    synthesized = _remove_vague_references(synthesized)
    synthesized = _ensure_single_verb_svo(synthesized)
    
    if console:
        console.print(f"[blue]ℹ Tổng hợp {len(captions)} caption bằng rule-based (từ vựng + SVO)[/blue]")
    
    return synthesized.strip()


def _extract_ocr_text_simple(image_input, languages: list = ["vi", "en"], min_confidence: float = 0.3) -> str:
    """Trích xuất text từ ảnh bằng OCR và trả về chuỗi text đơn giản.
    
    Ưu tiên PaddleOCR cho tiếng Việt (chính xác hơn), fallback về EasyOCR nếu không có.
    
    Args:
        image_input: Đường dẫn đến ảnh (str) hoặc PIL Image
        languages: Danh sách ngôn ngữ (mặc định: ["vi", "en"])
        min_confidence: Ngưỡng confidence tối thiểu (mặc định: 0.3, thấp hơn để bắt được text mờ)
    """
    from PIL import Image
    import numpy as np
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image_input, Image.Image):
        image_array = np.array(image_input)
    else:
        image_array = image_input  # Assume it's a path string
    
    # Thử PaddleOCR trước (tốt hơn cho tiếng Việt)
    ocr = _get_paddleocr_reader()
    if ocr is not None:
        try:
            # Đọc text từ ảnh (PaddleOCR accepts both path and numpy array)
            if isinstance(image_input, Image.Image):
                result = ocr.ocr(image_array, cls=True)
            else:
                result = ocr.ocr(image_input, cls=True)
            
            if result and result[0]:
                texts = []
                seen_candidates = set()
                for line in result[0]:
                    if line and len(line) >= 2:
                        # line[1] là tuple (text, confidence)
                        text_info = line[1]
                        if isinstance(text_info, tuple) and len(text_info) >= 2:
                            text = text_info[0]
                            conf = text_info[1]
                            # Lọc text có confidence >= min_confidence
                            if conf >= min_confidence:
                                cleaned = _clean_ocr_candidate(text)
                                if cleaned:
                                    normalized = cleaned.lower()
                                    if normalized not in seen_candidates:
                                        texts.append(cleaned)
                                        seen_candidates.add(normalized)
                
                if texts:
                    return " ".join(texts)
        except Exception:
            # Lỗi khi dùng PaddleOCR, thử EasyOCR
            pass
    
    # Fallback: Sử dụng EasyOCR
    reader = _get_easyocr_reader(languages)
    if reader is not None:
        try:
            # Đọc text từ ảnh với tham số tối ưu cho tiếng Việt
            # Thử với threshold thấp hơn để bắt được text mờ
            # EasyOCR accepts both path and numpy array
            if isinstance(image_input, Image.Image):
                results = reader.readtext(
                    image_array,
                    paragraph=False,  # Không nhóm thành đoạn để giữ thứ tự
                    width_ths=0.5,    # Giảm threshold để bắt được text nhỏ hơn
                    height_ths=0.5,   # Giảm threshold để bắt được text nhỏ hơn
                )
            else:
                results = reader.readtext(
                    image_input,
                    paragraph=False,  # Không nhóm thành đoạn để giữ thứ tự
                    width_ths=0.5,    # Giảm threshold để bắt được text nhỏ hơn
                    height_ths=0.5,   # Giảm threshold để bắt được text nhỏ hơn
                )
            
            if not results:
                return ""

            # Lọc text có confidence >= min_confidence và ghép lại
            texts = []
            seen_candidates = set()
            for (bbox, text, conf) in results:
                if conf >= min_confidence:
                    cleaned = _clean_ocr_candidate(text)
                    if cleaned:
                        normalized = cleaned.lower()
                        if normalized not in seen_candidates:
                            texts.append(cleaned)
                            seen_candidates.add(normalized)

            if texts:
                return " ".join(texts)
        except Exception:
            pass
    
    return ""


def caption_image_local(image_input, model_name: str, use_object_detection: bool = False, detect_objects_func=None, console=None) -> str:
    """Caption an image using a local Hugging Face pipeline model (cached).
    Optionally integrates object detection (YOLO) for richer descriptions.
    Returns caption in SVO format (Subject Verb Object), e.g., 'người đàn ông vẽ tranh'.
    
    Args:
        image_input: Path to image file (str) or PIL Image
        model_name: Hugging Face model name for captioning
        use_object_detection: Whether to use YOLO object detection
        detect_objects_func: Function to detect objects (optional, for dependency injection)
        console: Console object for logging (optional)
    """
    from PIL import Image
    
    # Convert PIL Image to path if needed (save to temp file for captioner)
    import tempfile
    import os
    temp_path = None
    if isinstance(image_input, Image.Image):
        # Save PIL Image to temporary file for captioner
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        image_input.save(temp_path)
        image_path_for_captioner = temp_path
    else:
        image_path_for_captioner = image_input
    
    try:
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
        max_new_tokens = int(os.environ.get("CAPTION_MAX_TOKENS", "150"))  # Tăng từ 100 lên 150 để tránh câu bị ngắt
        repetition_penalty = float(os.environ.get("CAPTION_REPETITION_PENALTY", "1.2"))
        no_repeat_ngram_size = int(os.environ.get("CAPTION_NO_REPEAT_NGRAM", "3"))

        # Support generating multiple candidates and reranking with CLIP
        # Nếu bật synthesis (tổng hợp), tự động tạo 3 candidates
        use_synthesis = os.environ.get("CAPTION_USE_SYNTHESIS", "false").lower() in ("true", "1", "yes")
        num_candidates = int(os.environ.get("CAPTION_NUM_CANDIDATES", "3" if use_synthesis else "1"))
        candidates: list[str] = []
        try:
            # Try to request multiple sequences directly
            result = captioner(
                image_path_for_captioner,
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
                    image_path_for_captioner,
                    prompt=prompt,
                    num_beams=num_beams,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )
                if isinstance(result, list) and result:
                    candidates = [result[0].get("generated_text", "").strip()]
            except TypeError:
                result = captioner(image_path_for_captioner, max_new_tokens=max_new_tokens)
                if isinstance(result, list) and result:
                    candidates = [result[0].get("generated_text", "").strip()]

        # If we still need more candidates, attempt to sample with varied temperature
        if num_candidates > 1 and len(candidates) < num_candidates:
            extra_needed = num_candidates - len(candidates)
            for i in range(extra_needed):
                try:
                    res_i = captioner(
                        image_path_for_captioner,
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
            t = _fix_incomplete_sentence(t)  # Sửa câu bị ngắt trước khi xử lý khác
            t = _remove_vague_references(t)
            t = _ensure_single_verb_svo(t)
            processed_candidates.append(t)

        base_caption = ""
        if processed_candidates:
            # Nếu có nhiều candidates (>= 3), tổng hợp thành 1 mô tả tổng quát
            if len(processed_candidates) >= 3:
                base_caption = synthesize_captions(processed_candidates[:3], console=console)
            else:
                base_caption = processed_candidates[0]
        
        # 1.5. Xử lý trường hợp BLIP trả về caption "lạ" hoặc có chữ nhưng sai chính tả
        use_ocr = False
        ocr_text = ""
        if base_caption:
            # Kiểm tra nếu caption có vấn đề hoặc có chứa text (có thể sai chính tả)
            is_bad = _is_bad_caption(base_caption)
            has_text = _has_text_in_caption(base_caption)
            
            if is_bad or has_text:
                if console:
                    if is_bad:
                        console.print(f"[yellow]Detected bad caption (likely Vietnamese text), using OCR...[/yellow]")
                    elif has_text:
                        console.print(f"[yellow]Detected text in caption (may have spelling errors), using OCR to verify...[/yellow]")
                
                use_ocr = True
                # Thử OCR với confidence threshold thấp hơn (0.3) để bắt được text mờ
                # Use original image_input (PIL Image or path)
                ocr_text = _extract_ocr_text_simple(image_input if isinstance(image_input, Image.Image) else image_path_for_captioner, languages=["vi", "en"], min_confidence=0.3)
                
                # Nếu OCR tìm thấy text
                if ocr_text:
                    if console:
                        console.print(f"[green]✓ OCR found text: {ocr_text[:50]}...[/green]" if len(ocr_text) > 50 else f"[green]✓ OCR found text: {ocr_text}[/green]")
                    
                    # Nếu caption có chứa text (như "với dòng chữ 'ngsai'")
                    # Thay thế text sai bằng text đúng từ OCR
                    if has_text:
                        import re
                        # Tìm và thay thế text trong dấu nháy đơn
                        quoted_text_pattern = r"'[^']{1,}'"
                        def replace_quoted_text(match):
                            # Thay thế bằng text từ OCR (giới hạn độ dài)
                            safe_ocr_text = ocr_text.replace("'", " ")
                            ocr_words = safe_ocr_text.split()
                            if len(ocr_words) <= 10:
                                return f"'{safe_ocr_text}'"
                            else:
                                short_text = " ".join(ocr_words[:5])
                                return f"'{short_text}...'"
                        
                        base_caption = re.sub(quoted_text_pattern, replace_quoted_text, base_caption)
                    else:
                        # Caption lạ nhưng không có pattern text rõ ràng
                        # Tạo caption mới từ OCR
                        safe_ocr_text = ocr_text.replace("'", " ")
                        ocr_words = safe_ocr_text.split()
                        if len(ocr_words) <= 10:
                            base_caption = f"hình ảnh có chữ '{safe_ocr_text}'"
                        else:
                            short_text = " ".join(ocr_words[:5])
                            base_caption = f"hình ảnh có chữ '{short_text}...'"
                else:
                    # OCR không tìm thấy text
                    if console:
                        console.print(f"[yellow]⚠ OCR không tìm thấy text trong ảnh (có thể text quá nhỏ, mờ, hoặc không rõ)[/yellow]")
                    
                    if is_bad:
                        # Caption lạ và OCR không tìm thấy text
                        # Thử giữ lại một phần caption gốc nếu có thể
                        # Ví dụ: "hình ảnh một chiếc hộp có nền xanh" -> "hình ảnh một chiếc hộp có nền xanh có nội dung văn bản"
                        if len(base_caption) > 30:
                            # Nếu caption dài, thử tách lấy phần đầu (trước khi có dấu phẩy nhiều)
                            parts = base_caption.split(',')
                            if len(parts) > 0:
                                first_part = parts[0].strip()
                                if len(first_part) > 10:
                                    base_caption = f"{first_part} có nội dung văn bản"
                                else:
                                    base_caption = "hình ảnh có nội dung văn bản"
                        else:
                            base_caption = "hình ảnh có nội dung văn bản"
                    # Nếu has_text nhưng OCR không tìm thấy, giữ nguyên caption gốc
                    # (có thể text quá nhỏ hoặc không rõ)
        
        # 2. Enhance with Object Detection (optional)
        detected_objects = []
        if use_object_detection and detect_objects_func:
            try:
                yolo_model = os.environ.get("YOLO_MODEL", "yolov8n.pt")
                conf_threshold = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
                # Use temp path or original path for object detection
                detected_objects = detect_objects_func(image_path_for_captioner, model_name=yolo_model, conf_threshold=conf_threshold)
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
    finally:
        # Clean up temp file if we created one
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

