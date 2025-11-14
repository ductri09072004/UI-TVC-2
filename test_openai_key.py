"""
Utility script to verify an OpenAI API key works with the Responses API.

Usage:
    python test_openai_key.py --key sk-... --model gpt-4o-mini --prompt "xin chao"

If --key is omitted, the script will fall back to the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI API key connectivity.")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY environment variable if omitted.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model to call (default: gpt-4o-mini or OPENAI_MODEL env).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Xin chào, vui lòng xác nhận bạn đang hoạt động?",
        help="Prompt text to send for verification.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=30,
        help="Maximum output tokens for the verification request (default: 30).",
    )
    return parser.parse_args()


def load_dotenv_file(dotenv_path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file if it exists.

    Simple parser that reads KEY=VALUE pairs (ignoring comments and quotes).
    """
    path = Path(dotenv_path or ".env")
    if not path.exists():
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"[yellow]Cảnh báo: Không thể đọc file .env ({exc}). Bỏ qua...[/yellow]")


def test_openai_key(
    api_key: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 30,
) -> bool:
    """
    Attempt to call the OpenAI Responses API.

    Returns True if the call succeeds; False otherwise.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        print("✗ Thư viện 'openai' chưa được cài đặt. Vui lòng `pip install openai`.", file=sys.stderr)
        raise SystemExit(1) from exc

    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as exc:  # Broad catch to show user-friendly message
        print("✗ Gọi API thất bại. Vui lòng kiểm tra API key hoặc quyền truy cập của model.", file=sys.stderr)
        print(f"  Lỗi: {exc}", file=sys.stderr)
        return False

    # Try to extract the response text
    output_text = ""
    output = getattr(resp, "output", None)
    if output:
        for item in output:
            contents = getattr(item, "content", None) or item.get("content", [])  # type: ignore[arg-type]
            for content in contents or []:
                content_type = getattr(content, "type", None) or content.get("type")  # type: ignore[union-attr]
                if content_type == "output_text":
                    text_value = getattr(content, "text", None) or content.get("text")  # type: ignore[union-attr]
                    if text_value:
                        output_text += text_value + " "
    if not output_text and getattr(resp, "choices", None):
        for choice in resp.choices:
            message = getattr(choice, "message", None) or choice.get("message")  # type: ignore[union-attr]
            if not message:
                continue
            content = getattr(message, "content", None) or message.get("content")  # type: ignore[union-attr]
            if isinstance(content, str):
                output_text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        output_text += item.get("text", "")

    output_text = output_text.strip()

    print("✓ API key hoạt động bình thường!")
    print(f"  Model: {model_name}")
    print(f"  Prompt: {prompt}")
    if output_text:
        print(f"  Phản hồi: {output_text}")
    else:
        print("  (Không lấy được nội dung phản hồi, nhưng cuộc gọi thành công.)")
    return True


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    api_key: Optional[str] = args.key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("✗ Không tìm thấy API key. Dùng --key hoặc đặt biến môi trường OPENAI_API_KEY.", file=sys.stderr)
        raise SystemExit(1)

    success = test_openai_key(api_key, args.model, args.prompt, args.max_tokens)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

