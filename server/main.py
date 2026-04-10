#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - FastAPI server cho streaming STT real-time

Server này cung cấp WebSocket endpoint để client (JavaScript/browser) gửi
audio chunk từ micro, server xử lý VAD hybrid và transcribe bằng faster-whisper.

Chạy:
    uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
    python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /           → Giao diện web client (www/index.html)
    GET  /health     → Health check, kiểm tra server alive
    WS   /ws/stt     → WebSocket endpoint cho streaming STT

Mở trình duyệt tại http://localhost:8000 để sử dụng.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import tempfile

from server.stt_ws import STTWebSocketHandler

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==== Lifespan: khởi tạo STT singleton khi server start ====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: khởi tạo khi start, cleanup khi shutdown"""
    logger.info("=" * 50)
    logger.info("Server khởi động, load STT models...")

    # Khởi tạo singleton STT và VAD
    model_path = os.environ.get("STT_MODEL_PATH")
    device = os.environ.get("STT_DEVICE")

    try:
        STTWebSocketHandler.initialize(
            model_path=model_path,
            device=device,
        )
        logger.info("STT models loaded thành công")
    except FileNotFoundError as e:
        logger.error(f"Không tìm thấy mô hình STT: {e}")
        logger.error("Chạy 'python scripts/download_model.py' để tải mô hình")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo STT: {e}")

    logger.info("=" * 50)

    yield  # Server đang chạy

    # Cleanup khi shutdown
    logger.info("Server shutdown, cleanup...")
    if STTWebSocketHandler._stt_engine:
        del STTWebSocketHandler._stt_engine
        STTWebSocketHandler._stt_engine = None
    if STTWebSocketHandler._vad:
        del STTWebSocketHandler._vad
        STTWebSocketHandler._vad = None
    STTWebSocketHandler._initialized = False
    logger.info("Cleanup hoàn tất")


# ==== FastAPI App ====
app = FastAPI(
    title="Streaming STT API",
    description="Server nhận âm thanh từ micro qua WebSocket, trả về văn bản nhận dạng real-time",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files (www/) — dùng /static/ cho CSS/JS
www_dir = Path(__file__).parent.parent / "www"
if www_dir.exists():
    app.mount("/static", StaticFiles(directory=str(www_dir)), name="static")


# ==== Routes ====

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Trả về giao diện web client
    """
    index_path = www_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(
        content="""
        <html><body>
            <h1>Streaming STT Server</h1>
            <p>File www/index.html không tồn tại. Chạy server với thư mục www/ đầy đủ.</p>
        </body></html>
        """,
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint — kiểm tra server alive và trạng thái models
    """
    stt_ready = (
        STTWebSocketHandler._initialized
        and STTWebSocketHandler._stt_engine is not None
    )
    vad_ready = (
        STTWebSocketHandler._initialized
        and STTWebSocketHandler._vad is not None
    )

    return JSONResponse({
        "status": "ok" if (stt_ready and vad_ready) else "degraded",
        "stt_ready": stt_ready,
        "vad_ready": vad_ready,
        "model_device": (
            STTWebSocketHandler._stt_engine.device
            if stt_ready else None
        ),
        "model_path": (
            STTWebSocketHandler._stt_engine.model_path
            if stt_ready else None
        ),
    })

@app.post("/api/transcribe-file")
async def transcribe_file_endpoint(
    file: UploadFile = File(...),
    language: str = Form("vi")
):
    """
    Endpoint upload file âm thanh để nhận dạng STT.
    Trả về toàn bộ văn bản sau khi mô hình nhận dạng xong.
    """
    if not STTWebSocketHandler._initialized or STTWebSocketHandler._stt_engine is None:
        raise HTTPException(status_code=503, detail="Máy chủ STT chưa sẵn sàng")

    # Lưu file tạm thời để xử lý
    fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix)
    try:
        # Đóng fd vì ta sẽ dùng open() bên dưới (hoặc dùng fd luôn, nhưng close cho chắc)
        os.close(fd)
        
        # Ghi nội dung upload vào file tạm
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Gọi hàm transcribe trong thread pool để khỏi block event loop
        # faster-whisper xử lý input đường dẫn file native
        engine = STTWebSocketHandler.get_engine()
        results = await run_in_threadpool(
            engine.transcribe,
            temp_path,
            language=language,
            vad_filter=True, # BẬT CHẾ ĐỘ NÀY (BẮT BUỘC KHÁC VỚI MICRO) vì file không đi qua Silero VAD
            temperature=0.0,
            beam_size=8,
            condition_on_previous_text=True
        )

        full_text = " ".join([seg["text"] for seg in results])
        return JSONResponse({"text": full_text, "segments": results})
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Dọn dẹp
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.websocket("/ws/stt")
async def websocket_stt(
    websocket: WebSocket,
    language: str = Query(default="vi", description="Mã ngôn ngữ cho STT"),
    vad_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    """
    WebSocket endpoint cho streaming STT

    Protocol:
        Client → Server: JSON message (xem server/stt_ws.py)
        Server → Client: JSON message (xem server/stt_ws.py)

    Query params:
        language: Mã ngôn ngữ (mặc định 'vi')
        vad_threshold: Ngưỡng VAD [0.0-1.0] (mặc định 0.3)

    Ví dụ kết nối từ JavaScript:
        const ws = new WebSocket("ws://localhost:8000/ws/stt?language=vi&vad_threshold=0.3");
    """
    # Kiểm tra server đã initialize chưa
    if not STTWebSocketHandler._initialized:
        await websocket.close(code=1011, reason="Server STT chưa được khởi tạo (model chưa load)")
        return

    # Chấp nhận kết nối
    await websocket.accept()

    # Tạo handler và xử lý
    handler = STTWebSocketHandler(
        websocket=websocket,
        language=language,
        vad_threshold=vad_threshold,
    )

    try:
        await handler.handle()
    except WebSocketDisconnect:
        logger.info("Client ngắt kết nối WebSocket")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


# ==== Entry point ====
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Khởi động server tại http://{host}:{port}")
    logger.info(f"Mở http://localhost:{port} để sử dụng giao diện web")

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=False,  # Bật True khi dev để auto-reload khi sửa code
        log_level="info",
    )
