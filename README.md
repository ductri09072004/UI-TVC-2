# Tool tách video thành ảnh mỗi giây và mô tả từng ảnh

Công cụ Python: nhập một video ngắn, tự động tách thành ảnh theo tốc độ 1 ảnh/giây, và sinh mô tả cho từng ảnh.

## Yêu cầu
- Python 3.10+
- Windows (đã test) hoặc các hệ điều hành khác có thể chạy Python và OpenCV

## Cài đặt
```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# Cài dependency
pip install -r requirements.txt
```

Ghi chú:
- Mặc định dùng mô hình local Hugging Face `Salesforce/blip-image-captioning-base` (chạy được CPU, ~1GB).
- Có thể dùng OpenAI (GPT-4o-mini) để mô tả chính xác hơn (cần biến môi trường `OPENAI_API_KEY`).
- Có tùy chọn dịch sang tiếng Việt (dùng `deep-translator`; cần Internet).

## Sử dụng (CLI)
```bash
# Local (Hugging Face), xuất ảnh và mô tả tiếng Việt
python app.py --input path/to/video.mp4 --out_dir output --backend local --language vi

# Chỉ thay đổi mô hình HF (tùy chọn)
python app.py --input path/to/video.mp4 --backend local --hf_model Salesforce/blip-image-captioning-base

# Dùng OpenAI để mô tả trực tiếp bằng tiếng Việt
$env:OPENAI_API_KEY="YOUR_KEY"  # PowerShell
python app.py --input path/to/video.mp4 --backend openai --openai_model gpt-4o-mini --language vi
```

## Chạy giao diện web (Flask)
```bash
# Bật server web (http://localhost:5000)
python web_app.py
```
- Truy cập trang: `http://localhost:5000`
- Tải lên video, chọn backend (Local/OpenAI), ngôn ngữ, model; bấm xử lý.
- Kết quả hiển thị dưới dạng lưới ảnh theo từng giây kèm mô tả.
- Ảnh và dữ liệu đầu ra lưu tại thư mục `web_outputs/<job_id>/`.

## Kết quả đầu ra (CLI)
Trong thư mục `out_dir`:
- Thư mục `frames/`: ảnh PNG trích mỗi giây (1 ảnh/giây).
- `captions.csv`: Bảng `second, image_path, caption`.
- `captions.jsonl`: Một dòng/ảnh, gồm `second, image_path, caption`.
- `captions.md`: File Markdown hiển thị ảnh và mô tả theo từng giây.

## Sự cố thường gặp
- Cài `torch` chậm hoặc lỗi: thử cài bản phù hợp phần cứng từ trang PyTorch chính thức.
- Dịch không hoạt động: có thể do không có Internet; mô tả sẽ giữ nguyên tiếng Anh.
- OpenAI không chạy: cần đặt `OPENAI_API_KEY` trong biến môi trường.
