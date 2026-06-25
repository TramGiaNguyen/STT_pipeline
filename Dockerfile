FROM python:3.10-slim

WORKDIR /app

# ── Chọn biến thể PyTorch ────────────────────────────────────────────────────
# cpu   (mặc định): nhẹ ~200MB, build nhanh, chạy được mọi máy (KHÔNG cần GPU). Đủ để test.
# cu121 (GPU)     : kéo ~5-6GB thư viện CUDA (nvidia-*). Chỉ build khi đẩy lên máy CÓ GPU
#                   + NVIDIA Container Toolkit. Build GPU bằng:
#                     docker build --build-arg TORCH_VARIANT=cu121 -t stt .
#                   (hoặc trong docker-compose: build.args TORCH_VARIANT=cu121)
ARG TORCH_VARIANT=cpu

# System deps tối thiểu để xử lý âm thanh (toàn bộ thư viện python dùng wheel sẵn nên
# KHÔNG cần gcc/python3-dev/git → apt nhẹ & nhanh hơn nhiều).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy danh sách thư viện trước để cache layer pip (không đổi lib thì không cài lại).
COPY requirements.lock .

# 1) Cài torch/torchaudio/torchvision theo biến thể đã chọn (mặc định CPU → nhẹ & nhanh).
#    Lấy đúng từ index của PyTorch; +${TORCH_VARIANT} đảm bảo bản cpu/cu121 chính xác.
RUN pip install --no-cache-dir \
        "torch==2.4.1+${TORCH_VARIANT}" \
        "torchaudio==2.4.1+${TORCH_VARIANT}" \
        "torchvision==0.19.1+${TORCH_VARIANT}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
        --extra-index-url https://pypi.org/simple

# 2) Cài phần còn lại từ lock, BỎ 3 dòng torch (đã cài ở bước 1 theo biến thể cpu/cu121).
#    Dùng --no-deps: lock VỐN ĐÃ là toàn bộ cây phụ thuộc (đầy đủ closure), nên cài thẳng
#    đúng từng version mà KHÔNG cho pip resolve lại. Tránh lỗi ResolutionImpossible do
#    bộ freeze (cài dồn qua thời gian) không "chứng minh" được tính nhất quán với resolver mới.
RUN grep -ivE '^torch(audio|vision)?==' requirements.lock > /tmp/req.rest.txt \
    && pip install --no-cache-dir --no-deps -r /tmp/req.rest.txt

# Để faster-whisper/ctranslate2 TÌM THẤY cuDNN & cuBLAS khi build GPU (cu121): các file .so này
# nằm trong site-packages/nvidia/*/lib (do torch cu121 kéo về), KHÔNG nằm trên path mặc định.
# Thiếu dòng này, bản GPU sẽ lỗi "libcudnn_ops... not found". Với bản CPU, thư mục này không tồn
# tại nên dòng này vô hại (loader tự bỏ qua path không có).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# Copy code vào container (models/ và file test ĐÃ bị .dockerignore loại — xem .dockerignore).
COPY . .

# Mở cổng cho fastapi
EXPOSE 8000

# Lệnh chạy server
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
