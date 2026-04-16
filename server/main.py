import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import tempfile

from server.stt_ws import STTWebSocketHandler

# ThreadPoolExecutor riêng cho file transcription (tránh block event loop)
_file_transcribe_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="file_transcribe")

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

# Thêm CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.post("/api/transcribe-stream")
async def transcribe_stream_endpoint(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    diarize: str = Form("false"),
):
    """
    Streaming STT endpoint qua Server-Sent Events (SSE).
    Upload file -> server convert -> stream từng segment về ngay khi xong.
    Hỗ trợ file từ 4 phút đến 3 tiếng mà không bị timeout.

    SSE Events (JSON):
        {"type": "info",    "total_duration": 176.3, "language": "vi"}
        {"type": "segment", "text": "...",  "start": 0.5, "end": 2.1, "progress": 12, "speaker": "Người nói 1"}
        {"type": "done",    "total_segments": 42}
        {"type": "error",   "message": "..."}
    """
    if not STTWebSocketHandler._initialized or STTWebSocketHandler._stt_engine is None:
        raise HTTPException(status_code=503, detail="Máy chủ STT chưa sẵn sàng")

    from src.audio_utils import convert_to_16k_mono
    from src.audio_enhancer import enhance_audio_file

    # 1. Lưu file upload vào temp
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    fd, temp_original = tempfile.mkstemp(suffix=suffix)
    temp_enhanced = None
    temp_converted = None
    diarization_timeline = []

    try:
        os.close(fd)
        content = await file.read()
        with open(temp_original, "wb") as f:
            f.write(content)
        logger.info(f"Đã lưu file upload: {temp_original} ({len(content)//1024}KB)")

        # 2a. Tiền xử lý âm thanh (Lọc nhiễu & Chuẩn hóa) - Chạy tự động
        temp_enhanced = temp_original + "_enhanced.wav"
        logger.info(f"Đang tiến hành tiền xử lý (Denoise & Normalize)...")
        # Run_in_threadpool để tránh block event loop
        await run_in_threadpool(
            enhance_audio_file,
            input_path=temp_original,
            output_path=temp_enhanced,
            sample_rate=16000,
            target_db=-20.0,
            apply_denoising=True
        )

        # 2b. Luôn convert sang 16kHz mono WAV (Dù enhance_audio_file cũng có thể trả về, nhưng làm thế này đảm bảo format chuẩn nhất)
        temp_converted = temp_enhanced + "_16k.wav"
        logger.info(f"Đang convert qua định dạng nhận dạng chuẩn...")
        await run_in_threadpool(convert_to_16k_mono, temp_enhanced, temp_converted)
        audio_path = temp_converted

        logger.info(f"Audio đã sẵn sàng (đã Enhanced & Normalized) để stream: {audio_path}")
        
    except Exception as e:
        # Dọn dẹp nếu lỗi trước khi stream
        for p in [temp_original, temp_enhanced, temp_converted]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        raise HTTPException(status_code=500, detail=f"Lỗi chuẩn bị file: {e}")

    # 3. Stream SSE
    engine = STTWebSocketHandler.get_engine()
    _lang = language if language != "auto" else None

    async def generate_sse():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        def sync_worker():
            """Chạy trong thread riêng để không block event loop."""
            try:
                # 3.1: Nếu có yêu cầu Diarize thì chạy model Pyannote trước
                timeline = []
                if diarize == "true":
                    loop.call_soon_threadsafe(
                        queue.put_nowait, 
                        {"type": "status", "message": "Đang phân tách người nói..."}
                    )
                    from src.diarization import DiarizationProcessor
                    # Chạy process offline blocking function trong cái executor thread này luôn
                    timeline = DiarizationProcessor.get_instance().process_audio(audio_path)
                
                # Hàm hỗ trợ map người nói
                def get_speaker_for_segment(start, end, t_line):
                    if not t_line: return None
                    midpoint = (start + end) / 2.0
                    for turn in t_line:
                        if turn["start"] <= midpoint <= turn["end"]:
                            return turn["speaker"]
                    # Nếu midpoint không nằm trong đoạn nào, lấy người nói có độ giao lớn nhất
                    max_overlap = 0
                    best_speaker = None
                    for turn in t_line:
                        overlap_start = max(start, turn["start"])
                        overlap_end = min(end, turn["end"])
                        overlap = max(0, overlap_end - overlap_start)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_speaker = turn["speaker"]
                    return best_speaker
                
                # 3.2: Chạy transcribe
                for event_dict in engine.transcribe_stream(
                    audio_path,
                    language=_lang,
                    beam_size=8,
                    condition_on_previous_text=False,
                ):
                    if event_dict.get("type") == "segment" and diarize == "true":
                        words = event_dict.get("words", [])
                        if words and len(words) > 0:
                            current_speaker = None
                            current_words = []
                            
                            for w in words:
                                spk = get_speaker_for_segment(w["start"], w["end"], timeline) or "Không rõ"
                                if current_speaker is None:
                                    current_speaker = spk
                                
                                # Nếu người nói thay đổi giữa chừng trong 1 segment dài
                                if spk != current_speaker:
                                    # Yield ngắt đoạn
                                    sub_event = {
                                        "type": "segment",
                                        "start": current_words[0]["start"],
                                        "end": current_words[-1]["end"],
                                        "text": " ".join([xw["word"] for xw in current_words]).strip(),
                                        "words": current_words,
                                        "speaker": current_speaker,
                                        "progress": event_dict.get("progress", 0)
                                    }
                                    loop.call_soon_threadsafe(queue.put_nowait, sub_event)
                                    
                                    # Tạo đoạn mới
                                    current_speaker = spk
                                    current_words = [w]
                                else:
                                    current_words.append(w)
                                    
                            # Yield phần còn lại cuối cùng
                            if current_words:
                                sub_event = {
                                    "type": "segment",
                                    "start": current_words[0]["start"],
                                    "end": current_words[-1]["end"],
                                    "text": " ".join([xw["word"] for xw in current_words]).strip(),
                                    "words": current_words,
                                    "speaker": current_speaker,
                                    "progress": event_dict.get("progress", 0)
                                }
                                loop.call_soon_threadsafe(queue.put_nowait, sub_event)
                        else:
                            # Fallback cho đoạn không hỗ trợ word level timestamp
                            event_dict["speaker"] = get_speaker_for_segment(
                                event_dict.get("start", 0), 
                                event_dict.get("end", 0), 
                                timeline
                            ) or "Không rõ"
                            loop.call_soon_threadsafe(queue.put_nowait, event_dict)
                    else:
                        loop.call_soon_threadsafe(queue.put_nowait, event_dict)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "message": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        try:
            _file_transcribe_executor.submit(sync_worker)

            segment_count = 0
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event.get("type") == "segment":
                    segment_count += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # yield cho event loop, tránh block

            # Gửi event done
            yield f'data: {{"type":"done","total_segments":{segment_count}}}\n\n'

        finally:
            # Dọn dẹp temp files sau khi stream xong
            for p in [temp_original, temp_converted]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        logger.info(f"Đã xóa temp: {p}")
                    except Exception:
                        pass

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Tắt nginx buffering (nếu có)
            "Connection": "keep-alive",
        },
    )


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
            condition_on_previous_text=False # Tắt để tránh lỗi hallucination cascading
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
