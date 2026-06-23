import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

# Nạp biến môi trường từ file .env (nếu có) – phải chạy trước mọi import khác
from dotenv import load_dotenv
load_dotenv()
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
from src.subtitle_formatter import process_text_glossary

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

    from src.audio_utils import convert_to_16k_mono, convert_to_wav_mono, enhance_audio_file

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
            normalize_db=-20.0,
            apply_denoising=False,   # Tắt: denoiser xoá giọng điện thoại (8kHz codec) của người gọi
            apply_vad_filtering=False # Không được cắt khoảng lặng để tránh lệch timestamp
        )

        # 2b. Luôn convert sang 16kHz mono WAV (Dù enhance_audio_file cũng có thể trả về, nhưng làm thế này đảm bảo format chuẩn nhất)
        temp_converted = temp_enhanced + "_16k.wav"
        logger.info(f"Đang convert qua định dạng nhận dạng chuẩn...")
        await run_in_threadpool(convert_to_16k_mono, temp_enhanced, temp_converted)
        audio_path = temp_converted

        # 2c. Tạo bản 16kHz mono NGUYÊN GỐC (không denoised) cho Diarization
        # Phải resample lên 16kHz: Pyannote train trên 16kHz, nhận 8kHz sẽ thiếu
        # spectral features → misidentify speaker trong 30s đầu (cold-start problem)
        temp_diarization = temp_original + "_diarize.wav"
        await run_in_threadpool(convert_to_16k_mono, temp_original, temp_diarization)

        logger.info(f"Audio đã sẵn sàng (đã Enhanced & Normalized) để stream: {audio_path}")
        
    except Exception as e:
        # Dọn dẹp nếu lỗi trước khi stream
        for p in [temp_original, temp_enhanced, temp_converted, temp_diarization if 'temp_diarization' in locals() else None]:
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
                    timeline = DiarizationProcessor.get_instance().process_audio(temp_diarization)
                    
                    # --- DEBUG DUMP ---
                    try:
                        with open("timeline_debug.json", "w", encoding="utf-8") as f:
                            import json
                            json.dump(timeline, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    # ------------------
                
                def get_speaker_for_segment(start, end, t_line):
                    if not t_line: return None
                    candidates = []
                    
                    # Tìm tất cả các đoạn có giao nhau với [start, end] của từ
                    for turn in t_line:
                        overlap_start = max(start, turn["start"])
                        overlap_end = min(end, turn["end"])
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > 0:
                            seg_duration = turn["end"] - turn["start"]
                            candidates.append({
                                "speaker": turn["speaker"],
                                "overlap": overlap,
                                "seg_duration": seg_duration
                            })
                            
                    if candidates:
                        # Ưu tiên overlap lớn nhất, nếu bằng nhau thì ưu tiên segment NGẮN NHẤT 
                        # (để bắt được các câu ngắt lời/chuyển thoại thay vì bị đè bởi người nói dài)
                        candidates.sort(key=lambda x: (-x["overlap"], x["seg_duration"]))
                        return candidates[0]["speaker"]
                        
                    # Nếu không có giao thoa nào (từ nằm trong khoảng lặng của Pyannote)
                    # thì dùng khoảng cách gần nhất
                    midpoint = (start + end) / 2.0
                    min_dist = float('inf')
                    best_speaker = "Không rõ"
                    for turn in t_line:
                        dist = min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"]))
                        if dist < min_dist:
                            min_dist = dist
                            best_speaker = turn["speaker"]
                            
                    return best_speaker
                
                # 3.2: Chạy transcribe
                for event_dict in engine.transcribe_stream(
                    audio_path,
                    language=_lang,
                    beam_size=5,
                    condition_on_previous_text=False,
                ):
                    if event_dict.get("type") == "segment" and diarize == "true":
                        words = event_dict.get("words", [])
                        if words and len(words) > 0:
                            # Gán nhãn người nói cho từng chữ
                            for w in words:
                                w["speaker"] = get_speaker_for_segment(w["start"], w["end"], timeline)
                            
                            # === SMOOTHING: Loại bỏ backchannel cắt ngang ===
                            # CHỈ hấp thụ nếu: đúng 1 từ, thời lượng < 0.8s, 
                            # VÀ cả trước+sau đều là cùng 1 speaker với run >= 3 từ
                            speakers = [w.get("speaker", "Không rõ") for w in words]
                            smoothed = list(speakers)
                            
                            i = 0
                            while i < len(smoothed):
                                if i > 0 and smoothed[i] != smoothed[i - 1]:
                                    # Đếm run hiện tại
                                    run_start = i
                                    run_speaker = smoothed[i]
                                    j = i
                                    while j < len(smoothed) and smoothed[j] == run_speaker:
                                        j += 1
                                    run_len = j - run_start
                                    
                                    # Chỉ smooth nếu: đúng 1 từ, sau nó quay lại speaker trước
                                    if run_len == 1 and j < len(smoothed) and smoothed[j] == smoothed[i - 1]:
                                        # Kiểm tra thời lượng từ đó < 0.8s (backchannel thường rất ngắn)
                                        word_duration = words[i]["end"] - words[i]["start"]
                                        if word_duration < 0.8:
                                            # Kiểm tra run trước >= 3 từ (speaker đang nói liên tục)
                                            prev_run_len = 0
                                            k = i - 1
                                            while k >= 0 and smoothed[k] == smoothed[i - 1]:
                                                prev_run_len += 1
                                                k -= 1
                                            if prev_run_len >= 3:
                                                smoothed[i] = smoothed[i - 1]
                                                i = j
                                                continue
                                i += 1
                            
                            # Ghi lại speaker đã smooth vào words
                            for idx, w in enumerate(words):
                                w["speaker"] = smoothed[idx]
                                    
                            # === GOM CÁC TỪ THEO NGƯỜI NÓI VÀ YIELD SEGMENT ===
                            current_speaker = None
                            current_words = []
                            
                            for w in words:
                                spk = w.get("speaker", "Không rõ")
                                if current_speaker is None:
                                    current_speaker = spk
                                    
                                needs_cut = False
                                if spk != current_speaker:
                                    needs_cut = True
                                elif current_words:
                                    last_w = current_words[-1]
                                    # Ngắt nếu câu đã dài và kết thúc bằng dấu chấm/hỏi/chấm than
                                    if len(current_words) > 20 and any(last_w["word"].strip().endswith(p) for p in [".", "?", "!"]):
                                        needs_cut = True
                                    # Ngắt nếu khoảng lặng > 1.5s
                                    elif w["start"] - last_w["end"] > 1.5:
                                        needs_cut = True
                                        
                                if needs_cut:
                                    if current_words:
                                        sub_event = {
                                            "type": "segment",
                                            "start": current_words[0]["start"],
                                            "end": current_words[-1]["end"],
                                            "text": process_text_glossary(" ".join([xw["word"] for xw in current_words]).strip()),
                                            "words": current_words,
                                            "speaker": current_speaker,
                                            "progress": event_dict.get("progress", 0)
                                        }
                                        loop.call_soon_threadsafe(queue.put_nowait, sub_event)
                                    current_speaker = spk
                                    current_words = [w]
                                else:
                                    current_words.append(w)
                                    
                            if current_words:
                                sub_event = {
                                    "type": "segment",
                                    "start": current_words[0]["start"],
                                    "end": current_words[-1]["end"],
                                    "text": process_text_glossary(" ".join([xw["word"] for xw in current_words]).strip()),
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
            for p in [temp_original, temp_converted, temp_diarization]:
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
            beam_size=5,
            condition_on_previous_text=False # Tắt để tránh lỗi hallucination cascading
        )

        full_text = " ".join([seg["text"] for seg in results])
        return JSONResponse({"text": full_text, "segments": results})
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/stt/subtitle_export")
async def subtitle_export_endpoint(
    file: UploadFile = File(...),
    language: str = Form("vi")
):
    """
    API Upload âm thanh và trả về trực tiếp file phụ đề (.srt)
    Đã được cắt khối thời gian chuẩn và sửa lỗi các thuật ngữ tiếng Anh.
    """
    if not STTWebSocketHandler._initialized or STTWebSocketHandler._stt_engine is None:
        raise HTTPException(status_code=503, detail="Máy chủ STT chưa sẵn sàng")

    fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix if file.filename else ".tmp")
    try:
        os.close(fd)
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 1. Chạy Diarization trước để lấy mốc thời gian của từng người nói
        from src.diarization import DiarizationProcessor
        from src.audio_utils import convert_to_16k_mono
        logger.info("[Subtitle Export] Đang convert định dạng và chạy Diarization...")
        temp_diarization = temp_path + "_diarize.wav"
        await run_in_threadpool(convert_to_16k_mono, temp_path, temp_diarization)
        
        try:
            timeline = await run_in_threadpool(DiarizationProcessor.get_instance().process_audio, temp_diarization)
        finally:
            if os.path.exists(temp_diarization):
                try:
                    os.remove(temp_diarization)
                except Exception:
                    pass

        # 2. Chạy STT Engine (Bật word_timestamps để Format SRT chia block)
        engine = STTWebSocketHandler.get_engine()
        logger.info("[Subtitle Export] Đang chạy Nhận dạng STT (có timestamps từ)...")
        results = await run_in_threadpool(
            engine.transcribe,
            temp_path,
            language=language,
            vad_filter=True,
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
            word_timestamps=True
        )

        # 3. Gắn nhãn và sinh file SRT chuẩn hóa bằng module mới
        from src.subtitle_formatter import generate_srt
        logger.info("[Subtitle Export] Đang định dạng và sinh phụ đề SRT...")
        srt_content = await run_in_threadpool(
            generate_srt, 
            segments=results, 
            diarization=timeline, 
            max_words=12, 
            max_duration=5.0
        )

        # Tạo Readable Streaming Response gửi file txt
        import io
        from fastapi.responses import StreamingResponse
        stream = io.StringIO(srt_content)
        original_name = Path(file.filename).stem if file.filename else "audio"
        file_name = f"{original_name}_subtitle.srt"

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={file_name}"}
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý xuất phụ đề: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/generate-bienban")
async def generate_bienban_endpoint(
    transcript: str = Form(...),
    filename: str = Form("transcript"),
):
    """
    Endpoint tạo biên bản họp từ transcript.
    
    Nhận nội dung transcript (plain text đã parse từ .srt/.txt),
    gọi Gemini AI phân tích, fill vào template DOCX, trả về file .docx.
    """
    from server.bienban_generator import analyze_transcript_with_gemini, fill_docx_template
    from starlette.concurrency import run_in_threadpool

    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Nội dung transcript trống")

    try:
        # 1. Gọi Gemini phân tích transcript
        logger.info(f"[Biên bản] Bắt đầu phân tích transcript ({len(transcript)} ký tự)...")
        analysis = await run_in_threadpool(analyze_transcript_with_gemini, transcript.strip())
        logger.info(f"[Biên bản] Gemini phân tích xong, đang tạo DOCX...")

        # 2. Fill vào template DOCX
        docx_bytes = await run_in_threadpool(fill_docx_template, analysis)
        logger.info(f"[Biên bản] Đã tạo file DOCX ({len(docx_bytes)} bytes)")

        # 3. Trả về file .docx
        import io
        output_filename = f"Bien_ban_hop_{filename}.docx"
        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "Content-Length": str(len(docx_bytes)),
            },
        )

    except ValueError as e:
        logger.error(f"[Biên bản] Lỗi: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"[Biên bản] Không tìm thấy template: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[Biên bản] Lỗi không xác định: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo biên bản: {e}")


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
