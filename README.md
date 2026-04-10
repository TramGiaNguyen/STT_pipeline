# 🎙️ Tối Đa Hóa Độ Chính Xác: Streaming STT Module 

Hệ thống Speech-to-Text (STT) hoạt động trên nền tảng **FastAPI**, **WebSockets** kết hợp cùng **Faster-Whisper (PhoWhisper)** và **Silero VAD**. Giải pháp ưu tiên tuyệt đối vào **độ nguyên bản tín hiệu và độ chính xác của ngữ cảnh** (Max Accuracy Pipeline) hơn là tốc độ trễ thấp thông thường.

Dự án cung cấp giao diện Web trực quan để:
- Thu âm và dịch giọng nói trực tiếp qua Micro (thời gian thực một phần).
- Tải file âm thanh/video có sẵn lên để trích xuất văn bản chữ.
- Hỗ trợ mớm từ vựng (Initial Prompt) để tối ưu các thuật ngữ, tên riêng chuyên ngành.

---

## ⚡ Các Thay Đổi & Pipeline Tính Năng Cốt Lõi

1. **Âm thanh nguyên bản (Raw Audio Layer)**
   - Phía Client vô hiệu hoá hoàn toàn bộ lọc WebRTC (`echoCancellation`, `noiseSuppression`, `AGC`) để giữ mô hình tần số giọng nói gốc hoàn hảo nhất trước khi tới server.
2. **Bộ Lọc Cứng Đa Tầng (Multi-layered VAD Gate)**
   - Tầng 1: Loại bỏ nhiễu tĩnh với `MIN_RAW_RMS = 0.0005` (~-66 dBFS).
   - Tầng 2: Xác nhận giọng người chuyên sâu qua lõi **Silero VAD v4/v5** với độ chắc chắn cao (`Threshold = 0.35`).
3. **Mở rộng dải ngữ cảnh nhận dạng**
   - Accumulation Buffer gom tín hiệu chờ lên tới **15s liên tiếp** hoặc đợi tới khi thực sự ngắt câu **>= 1.0s** mới thực hiện dịch ➔ cung cấp cho Whisper một lượng từ vựng Context dồi dào để đoán trước câu sau thay vì dịch từng chữ rời rạc.
4. **Anti-Hallucination Độc Quyền**
   - Kỹ thuật chặn ảo giác mạnh mẽ: Cắt im lặng cấu hình cao (`no_speech_threshold = 0.5`) + Tự động bóp vứt các cụm chữ dư thừa lặp lại không có nghĩa sinh ra bởi Micro nhiễu (`compression_ratio_threshold = 2.2`).

---

## 📁 Kiến Trúc Thư Mục

```text
STT_module/
│
├── models/                   # Lưu trữ các file tải về của mô hình CTranslate2 (PhoWhisper)
├── scripts/                  # Chứa script hỗ trợ: e.g. download_model.py
├── server/                   # (Backend) Logic máy chủ và pipeline AI của bạn
│   ├── main.py               # Lõi điều phối FastAPI (khai báo các endpoints)
│   ├── stt_ws.py             # Bộ não xử lý khối lệnh WebSocket Audio, VAD và Chunking
├── src/                      
│   ├── stt_engine.py         # Module Wrapper bao bọc thư viện Faster-Whisper 
│   ├── vad_processor.py      # Module Wrapper bao bọc thao tác tính toán Silero VAD
├── www/                      # (Frontend) Giao diện UI Web Client
│   ├── index.html            # Giao diện HTML
│   ├── style.css             # Theme màu sắc, thiết kế hiển thị
│   ├── app.js                # Xử lý luồng thu âm WebRTC, Fetch API, WebSocket Handler
└── README.md                 # Tài liệu này
```

---

## 🛠️ Cài đặt & Khởi động Dự Án

### B1: Thiết Lập Môi Trường
Đảm bảo bạn đang sử dụng Python từ phiên bản **3.10** trở lên. Cài đặt các thư viện lõi thông qua môi trường ảo (Virtual Environment):
```bash
python -m venv venv

# Chạy môi trường trên Windows:
.\venv\Scripts\activate

# Chạy môi trường trên macOS / Linux:
source venv/bin/activate
```

Cài đặt package bắt buộc:
```bash
pip install fastapi uvicorn websockets python-multipart faster-whisper numpy torch torchaudio
```

### B2: Tải Mô hình (Nếu chưa có)
Mô hình Faster-Whisper có thể được tải cục bộ bằng các tệp lệnh trong folder `scripts/` (ví dụ `download_model.py`) hoặc module sẽ tự động down bằng `huggingface_hub` qua cờ `download_if_missing=True` nếu lập trình.

### B3: Bật Máy Chủ
Từ thư mục gốc dự án (STT_module), tiến hành nổ máy chủ uvicorn:
```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### B4: Sử dụng
Mở trình duyệt lên và truy cập [http://localhost:8000](http://localhost:8000)
- Thiết lập lại tên cấu hình Model tuỳ ý.
- Sử dụng Tab 1 để thử nói trực tiếp với độ chính xác cao.
- Sử dụng Tab 2 để vứt các file `mp3, wav...` nhờ STT bóc băng thành chữ (Hỗ trợ cả File Video được giải mã qua chuẩn backend của Whisper).

---
