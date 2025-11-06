"""
Danh sách nhãn dán (labels) cho phân loại frame bằng CLIP (zero-shot).
Bạn có thể chỉnh sửa file này để cố định bộ nhãn trong dự án.

Cách sử dụng (mặc định trong app.py sẽ đọc biến môi trường; bạn có thể
set biến từ danh sách này):

PowerShell:
    $env:CLIP_FRAME_LABELS = get-content -raw frame_labels_pipe.txt

Hoặc trong code (ví dụ tuỳ biến):
    from frame_labels import FRAME_LABELS_VI, labels_to_pipe
    os.environ["CLIP_FRAME_LABELS"] = labels_to_pipe(FRAME_LABELS_VI)
"""

from __future__ import annotations

from typing import List

# Nhãn mặc định tiếng Việt (có thể sửa tuỳ domain)
FRAME_LABELS_VI: List[str] = [
    # Healthy (EN + VI)
    "safe content", "no violation", "clean",
    "lành mạnh", "nội dung lành mạnh", "không vi phạm",

    # Violations (EN + VI) — keep visual-first concepts
    # Alcohol
    "alcohol", "beer", "wine", "rượu", "bia",
    # Tobacco
    "tobacco", "cigarette", "vape", "thuốc lá", "thuốc lá điện tử",
    # Weapons
    "weapons", "gun", "knife", "firearms", "bladed weapons", "vũ khí", "súng", "dao",
    # Violence
    "violence", "fighting", "domestic violence", "bạo lực", "đánh nhau",
    # Nudity / Sexual
    "nudity", "revealing outfit", "sexual suggestive", "provocative pose",
    "khiêu dâm", "hở hang", "gợi dục",
    # Sensitive content
    "sensitive content", "nội dung nhạy cảm",
    # Blood / gore (visual)
    "blood", "máu me",
    # Gambling (visual cues)
    "gambling", "betting", "casino games", "cờ bạc", "cá cược",
    # Drugs (visual cues)
    "illegal drugs", "ma túy",
    # Counterfeit (visual)
    "counterfeit", "knockoff", "hàng giả", "hàng nhái",
]


def labels_to_pipe(labels: List[str]) -> str:
    """Chuyển list nhãn thành chuỗi pipe-separated để set vào env.

    Ví dụ: ["lành mạnh", "bạo lực"] -> "lành mạnh|bạo lực"
    """
    return "|".join(lbl.strip() for lbl in labels if lbl and lbl.strip())


# Lưu sẵn file pipe để dễ dùng từ shell (không chạy tự động):
# - Mở file `frame_labels_pipe.txt` (nếu tự tạo) và đặt nội dung từ labels_to_pipe(FRAME_LABELS_VI)
# - Sau đó: $env:CLIP_FRAME_LABELS = get-content -raw frame_labels_pipe.txt


