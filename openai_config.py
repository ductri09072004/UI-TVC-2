"""
Cấu hình OpenAI API cho captioning.
Chỉnh sửa file này để thay đổi prompt, model, và các tham số khác.
"""

# OpenAI API Key
# Có thể đặt ở đây hoặc trong biến môi trường OPENAI_API_KEY
OPENAI_API_KEY = None  # Nếu None, sẽ lấy từ biến môi trường

# Model mặc định
DEFAULT_MODEL = "gpt-4o-mini"

# Prompt template cho image captioning
# Dùng cho việc mô tả frames từ TVC quảng cáo
CAPTION_PROMPT = """Bạn là chuyên gia phân tích và mô tả frames từ video quảng cáo TVC.

Nhiệm vụ: Mô tả frame này một cách ngắn gọn, chính xác theo định dạng SVO (Subject - Verb - Object) bằng tiếng Việt.

Yêu cầu bắt buộc:
1. Chỉ sử dụng MỘT động từ chính duy nhất (ví dụ: cầm, uống, mặc, nhìn, nói)
2. Mô tả ngắn gọn: tối đa 15-20 từ
3. Định dạng SVO: [Chủ ngữ] [Động từ] [Tân ngữ]
4. Tập trung vào:
   - Hành động chính trong frame
   - Đối tượng/sản phẩm nổi bật
   - Số lượng người (nếu có)
   - Trạng thái/hành động cụ thể

Quan trọng:
- KHÔNG dùng nhiều động từ như "mặc và cầm", "đứng và nói"
  (Lý do: Định dạng SVO yêu cầu một động từ chính, giúp caption ngắn gọn và dễ xử lý.
   Nếu có cả động từ trang phục và hành động, ưu tiên hành động chính)
- KHÔNG mô tả chi tiết nền hoặc vật dụng phụ
- KHÔNG dùng cụm từ mơ hồ như "trên đó", "ở đó", "này", "kia"
- Ưu tiên hành động và sản phẩm chính của quảng cáo

Ví dụ tốt:
- "một người đàn ông cầm chai rượu"
- "hai người phụ nữ uống cà phê"
- "cô gái mặc váy màu đỏ"
- "người đàn ông nói chuyện với khách hàng"

Mô tả frame này:"""

# Các prompt khác (có thể mở rộng)
PROMPTS = {
    "default": CAPTION_PROMPT,
    
    "fashion": """Bạn là chuyên gia phân tích frames quảng cáo thời trang từ TVC.

Nhiệm vụ: Mô tả frame này ngắn gọn theo định dạng SVO (Subject - Verb - Object) bằng tiếng Việt.

Yêu cầu:
- Chỉ MỘT động từ chính (mặc, đeo, cầm, nhìn, đi, đứng...)
- Tối đa 15-20 từ
- Tập trung vào: trang phục/phụ kiện, màu sắc nổi bật, hành động của người mẫu
- Định dạng: [Người] [Động từ] [Trang phục/Phụ kiện]

Ví dụ:
- "cô gái mặc váy màu đỏ"
- "người mẫu đeo túi xách màu đen"
- "người đàn ông mặc áo sơ mi trắng"

Mô tả frame:""",
    
    "food": """Bạn là chuyên gia phân tích frames quảng cáo đồ ăn/đồ uống từ TVC.

Nhiệm vụ: Mô tả frame này ngắn gọn theo định dạng SVO (Subject - Verb - Object) bằng tiếng Việt.

Yêu cầu:
- Chỉ MỘT động từ chính (ăn, uống, cầm, nhìn, nếm, mở...)
- Tối đa 15-20 từ
- Tập trung vào: món ăn/đồ uống, hành động liên quan, bao bì/sản phẩm
- Định dạng: [Người] [Động từ] [Món ăn/Đồ uống]

Ví dụ:
- "người phụ nữ uống cà phê"
- "người đàn ông ăn bánh mì"
- "cô gái cầm chai nước ngọt"

Mô tả frame:""",
    
    "product": """Bạn là chuyên gia phân tích frames quảng cáo sản phẩm từ TVC.

Nhiệm vụ: Mô tả frame này ngắn gọn theo định dạng SVO (Subject - Verb - Object) bằng tiếng Việt.

Yêu cầu:
- Chỉ MỘT động từ chính (cầm, sử dụng, mở, nhìn, chỉ vào, giới thiệu...)
- Tối đa 15-20 từ
- Tập trung vào: sản phẩm chính, hành động sử dụng, đặc điểm nổi bật của sản phẩm
- Định dạng: [Người] [Động từ] [Sản phẩm]

Ví dụ:
- "người đàn ông cầm chai dầu gội"
- "cô gái sử dụng máy tính"
- "người phụ nữ mở hộp sản phẩm"

Mô tả frame:""",
    
    "people": """Bạn là chuyên gia phân tích frames quảng cáo TVC tập trung vào người.

Nhiệm vụ: Mô tả frame này ngắn gọn theo định dạng SVO (Subject - Verb - Object) bằng tiếng Việt.

Yêu cầu:
- Chỉ MỘT động từ chính (nói, cười, nhìn, đi, đứng, ngồi, hát...)
- Tối đa 15-20 từ
- Tập trung vào: số lượng người, hành động chính, cảm xúc/biểu cảm
- Định dạng: [Số lượng + Người] [Động từ] [Tân ngữ/Nội dung hành động]

Ví dụ:
- "hai người phụ nữ nói chuyện"
- "người đàn ông cười"
- "nhóm người nhìn vào sản phẩm"

Mô tả frame:""",
}

# Tham số mặc định cho API calls
DEFAULT_TEMPERATURE = 0.3  # Độ sáng tạo (0.0-2.0)
DEFAULT_MAX_TOKENS = 120  # Số token tối đa cho response (vừa đủ cho mô tả chi tiết vừa phải 15-25 từ)
DEFAULT_TARGET_LANGUAGE = "vi"  # Ngôn ngữ mục tiêu
# Kích thước tối đa cho images (pixels) - resize để giảm token input
# OpenAI Vision tính token:
#   Low-res: 85 base + (width * height) / (512 * 512) * 170 tokens/image
#   High-res: 85 base + (width * height) / (512 * 512) * 170 * 2 tokens/image
# Ví dụ: 768x432 in low-res ≈ 85 + (768*432)/(512*512)*170 ≈ 304 tokens/image
#       3 images ≈ 912 tokens + prompt (~100-150 tokens) = ~1100 tokens/request
DEFAULT_IMAGE_MAX_DIMENSION = 768  # Max width hoặc height (giảm từ 1024 xuống 768 để tiết kiệm hơn)
# Chi tiết image mode: "low" (tiết kiệm ~50% tokens) hoặc "high" (chất lượng cao hơn)
# Low-res mode: 85 + (w*h)/(512*512)*170 tokens/image
# High-res mode: 85 + (w*h)/(512*512)*170*2 tokens/image
DEFAULT_IMAGE_DETAIL = "low"  # Dùng low-res mode để tiết kiệm tokens
# Thời gian delay giữa các API calls (giây) - để tránh rate limit
# Free tier: 200k tokens/phút = ~3.3k tokens/giây
# Nên set delay 0.5-1s giữa các batch để tránh vượt quá rate limit
DEFAULT_REQUEST_DELAY = 0.5  # Giây

# Prompt style để chọn
# Có thể thay đổi giá trị này để dùng prompt khác
SELECTED_PROMPT_STYLE = "default"  # "default", "fashion", "food", "product", "people"


def get_openai_config():
    """
    Trả về cấu hình OpenAI hoàn chỉnh.
    
    Returns:
        dict: Dictionary chứa api_key, model, prompt, temperature, max_tokens, target_language
    """
    import os
    
    # Lấy API key từ config hoặc environment
    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    
    # Lấy prompt theo style đã chọn
    prompt = PROMPTS.get(SELECTED_PROMPT_STYLE, PROMPTS["default"])
    
    return {
        "api_key": api_key,
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "target_language": DEFAULT_TARGET_LANGUAGE,
        "prompt_style": SELECTED_PROMPT_STYLE,
    }


def get_prompt(style: str = None) -> str:
    """
    Lấy prompt theo style được chỉ định.
    
    Args:
        style: Tên style prompt ("default", "fashion", "food", "product", "people")
              Nếu None, dùng SELECTED_PROMPT_STYLE
    
    Returns:
        str: Prompt template
    """
    if style is None:
        style = SELECTED_PROMPT_STYLE
    return PROMPTS.get(style, PROMPTS["default"])


def add_custom_prompt(name: str, prompt: str):
    """
    Thêm prompt tùy chỉnh mới.
    
    Args:
        name: Tên của prompt style
        prompt: Nội dung prompt template
    """
    PROMPTS[name] = prompt

