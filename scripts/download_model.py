#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_model.py - Script tải mô hình

Script này sử dụng hàm snapshot_download từ huggingface_hub để tải mô hình chỉ định
xuống thư mục models/ cục bộ của dự án, hoàn toàn bỏ qua cơ chế cache mặc định của HuggingFace.

Sau khi tải xong, mô hình có thể được load trực tiếp từ đường dẫn cục bộ mà không cần
phụ thuộc vào mạng hay HF cache.

Cách sử dụng:
    python scripts/download_model.py

Lưu ý:
    - Lần đầu tải cần kết nối mạng ổn định, kích thước mô hình thường khá lớn (vài GB)
    - Quá trình tải sẽ hiển thị thanh tiến trình
    - Nếu cần tải lại, hãy xóa thư mục models/EraX-WoW-Turbo-V1.1-CT2 rồi chạy lại
"""

import os
import sys
from pathlib import Path

# Import công cụ tải từ huggingface_hub
from huggingface_hub import snapshot_download

# -------------------------------------------------------------------
# Cấu hình: sửa các biến bên dưới theo mô hình thực tế đang dùng
# -------------------------------------------------------------------

# ID kho lưu trữ mô hình trên HuggingFace Hub (định dạng: tên_user/tên_mô_hình)
MODEL_ID = "erax-ai/EraX-WoW-Turbo-V1.1-CT2"

# Đường dẫn lưu trữ cục bộ của mô hình: tương đối so với thư mục gốc dự án
# snapshot_download sẽ tải toàn bộ nội dung kho về thư mục này
LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "EraX-WoW-Turbo-V1.1-CT2"

# HuggingFace Hub API Token (tùy chọn)
# Nếu mô hình là private hoặc có hạn chế truy cập, hãy điền HF Token của bạn vào đây
# Lấy token tại: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def download_model():
    """
    Hàm chính: thực hiện logic tải mô hình

    Các tham số quan trọng:
        - local_dir: đường dẫn lưu trữ cục bộ của file mô hình
        - local_dir_use_symlinks: đặt là False để force copy file thay vì tạo symlink
          Điều này đảm bảo file mô hình hoàn toàn độc lập với HF cache,
          tránh mất mô hình khi cache bị dọn dẹp
        - resume_download: đặt là True để hỗ trợ tải tiếp khi bị gián đoạn mạng
    """
    # Lấy chuỗi đường dẫn cục bộ (Path object có thể bị một số thư viện hiểu sai)
    local_dir_str = str(LOCAL_MODEL_PATH)

    print(f"=" * 60)
    print(f"Bắt đầu tải mô hình: {MODEL_ID}")
    print(f"Đường dẫn lưu: {local_dir_str}")
    print(f"=" * 60)

    # Nếu thư mục mô hình đã tồn tại, thông báo cho người dùng có thể bỏ qua
    if LOCAL_MODEL_PATH.exists() and any(LOCAL_MODEL_PATH.iterdir()):
        print(f"\n[Thông báo] Thư mục mô hình đã tồn tại và không trống: {local_dir_str}")
        print("Nếu cần tải lại, hãy xóa thư mục này trước.")
        response = input("Tiếp tục tải (sẽ bỏ qua các file đã có)? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Đã hủy tải.")
            sys.exit(0)

    try:
        # Thực hiện tải mô hình
        snapshot_download(
            repo_id=MODEL_ID,                        # ID kho mô hình trên HuggingFace
            local_dir=local_dir_str,                 # Đường dẫn lưu cục bộ
            local_dir_use_symlinks=False,            # Tắt symlink, copy trực tiếp file
            resume_download=True,                    # Bật tải tiếp khi bị gián đoạn
            token=HF_TOKEN,                          # HuggingFace API Token (tùy chọn)
        )
        print(f"\n[Thành công] Tải mô hình hoàn tất!")
        print(f"Đường dẫn mô hình: {local_dir_str}")

    except Exception as e:
        print(f"\n[Lỗi] Tải mô hình thất bại: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
