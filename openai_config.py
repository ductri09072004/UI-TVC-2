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
# {image} sẽ được thay thế bằng base64 encoded image
CAPTION_PROMPT = """Bạn là chuyên gia mô tả hình ảnh quảng cáo TVC bằng tiếng Việt.

Hãy mô tả hình ảnh này một cách ngắn gọn và chính xác theo định dạng SVO (Subject - Verb - Object).

Yêu cầu:
- Chỉ dùng MỘT động từ chính
- Mô tả ngắn gọn (tối đa 20 từ)
- Định dạng: tiếng Việt, SVO format
- Tập trung vào hành động chính và đối tượng trong hình
- Ví dụ: "người đàn ông cầm chai rượu", "cô gái uống nước cam"

Mô tả hình ảnh:"""

# Các prompt khác (có thể mở rộng)
PROMPTS = {
    "default": CAPTION_PROMPT,
    "fashion": """Bạn là chuyên gia thời trang. Mô tả hình ảnh quảng cáo thời trang bằng tiếng Việt theo định dạng SVO.

Yêu cầu:
- Mô tả trang phục, phụ kiện, phong cách
- Định dạng SVO, tối đa 20 từ
- Tiếng Việt

Mô tả:""",
    
    "food": """Bạn là chuyên gia ẩm thực. Mô tả hình ảnh quảng cáo đồ ăn/đồ uống bằng tiếng Việt theo định dạng SVO.

Yêu cầu:
- Mô tả món ăn, đồ uống, hành động liên quan
- Định dạng SVO, tối đa 20 từ
- Tiếng Việt

Mô tả:""",
    
    "product": """Bạn là chuyên gia marketing. Mô tả hình ảnh quảng cáo sản phẩm bằng tiếng Việt theo định dạng SVO.

Yêu cầu:
- Mô tả sản phẩm, hành động sử dụng, đặc điểm nổi bật
- Định dạng SVO, tối đa 20 từ
- Tiếng Việt

Mô tả:""",
}

# Tham số mặc định cho API calls
DEFAULT_TEMPERATURE = 0.3  # Độ sáng tạo (0.0-2.0)
DEFAULT_MAX_TOKENS = 150  # Số token tối đa cho response
DEFAULT_TARGET_LANGUAGE = "vi"  # Ngôn ngữ mục tiêu

# Prompt style để chọn
# Có thể thay đổi giá trị này để dùng prompt khác
SELECTED_PROMPT_STYLE = "default"  # "default", "fashion", "food", "product"


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
        style: Tên style prompt ("default", "fashion", "food", "product")
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

