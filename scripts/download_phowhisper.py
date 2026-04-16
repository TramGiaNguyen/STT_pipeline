#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_phowhisper.py - Script tai PhoWhisper-large-ctranslate2

PhoWhisper là mô hình Whisper fine-tuned đặc biệt cho tiếng Việt bởi VinAI,
phiên bản CTranslate2 tương thích với faster-whisper.

Nguồn: Clicia/PhoWhisper-large-ctranslate2 (HuggingFace)
Cơ sở: vinai/PhoWhisper-large (fine-tuned từ whisper-large-v2 trên dữ liệu tiếng Việt)

Cách sử dụng:
    python scripts/download_phowhisper.py

Sau khi tải xong, dùng với example_usage.py:
    python example_usage.py --input audio.mp4 --model models/PhoWhisper-large-ct2 --language vi
"""

import io
import os
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding (tranh loi UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from huggingface_hub import snapshot_download

# -------------------------------------------------------------------
# Cấu hình mô hình
# -------------------------------------------------------------------

# Repo CT2 của PhoWhisper-large trên HuggingFace (community conversion)
MODEL_ID = "Clicia/PhoWhisper-large-ctranslate2"

# Thư mục lưu cục bộ
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "PhoWhisper-large-ct2"

# HuggingFace Token (tùy chọn, để tăng tốc download và tránh rate limit)
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def download_phowhisper():
    """Tải PhoWhisper-large-ctranslate2 về thư mục models/"""
    local_dir_str = str(LOCAL_MODEL_PATH)

    print("=" * 65)
    print("Tải PhoWhisper-large-ctranslate2 (Fine-tuned tiếng Việt)")
    print("=" * 65)
    print(f"  Nguồn HuggingFace : {MODEL_ID}")
    print(f"  Lưu tại           : {local_dir_str}")
    print(f"  Kích thước ước tính: ~3GB")
    print("=" * 65)

    if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
        print(f"\n[Thông báo] Thư mục đã tồn tại và không trống: {local_dir_str}")
        print("Nếu cần tải lại, hãy xóa thư mục này trước.")
        response = input("Tiếp tục tải (sẽ bỏ qua các file đã có)? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Đã hủy.")
            sys.exit(0)

    try:
        print("\n[Bắt đầu tải]...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=local_dir_str,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=HF_TOKEN,
        )
        print(f"\n[Thành công] Tải PhoWhisper-large-ct2 hoàn tất!")
        print(f"Đường dẫn mô hình: {local_dir_str}")
        print("\nCách dùng với example_usage.py:")
        print(f"  python example_usage.py --input audio.mp4 \\")
        print(f"    --model models/PhoWhisper-large-ct2 \\")
        print(f"    --language vi --no-enhance --no-condition-prev-text \\")
        print(f"    --beam-size 8 --prompt \"Hội thoại tiếng Việt miền Nam\"")

    except Exception as e:
        print(f"\n[Lỗi] Tải mô hình thất bại: {e}")
        print("\nGợi ý khắc phục:")
        print("  1. Kiểm tra kết nối mạng")
        print("  2. Thêm HF_TOKEN: $env:HF_TOKEN='hf_xxx...' (PowerShell)")
        print("  3. Thử lại lệnh sau: python scripts/download_phowhisper.py")
        sys.exit(1)


if __name__ == "__main__":
    download_phowhisper()
