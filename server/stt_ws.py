#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stt_ws.py - WebSocket handler cho streaming STT real-time

Module này xử lý kết nối WebSocket từ client, nhận audio chunk đã được
WebRTC VAD (phía client) phát hiện có lời nói, sau đó:
    1. Silero VAD xác nhận lại (hybrid VAD - secondary check)
    2. faster-whisper transcribe chunk đã xác nhận
    3. Trả kết quả text về cho client qua WebSocket

Kiến trúc VAD Hybrid:
    - Client (JavaScript/WebRTC): phát hiện lời nói NhanH (ưu tiên latency)
    - Server (Silero VAD): xác nhận lại Chính xác (ưu tiên accuracy)

Protocol WebSocket:
    Client → Server (JSON message):
        {
            "type": "audio",
            "audio": "<base64 encoded PCM 16kHz mono 16-bit>",
            "sample_rate": 16000
        }

    Server → Client (JSON message):
        {
            "type": "result",
            "text": "<văn bản nhận dạng>",
            "confidence": <xác suất>,           # tùy chọn
            "timestamp": "<ISO timestamp>"      # tùy chọn
        }

        {
            "type": "status",
            "status": "listening"|"processing"|"silence"|"error",
            "message": "<mô tả>"
        }
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Dict, Optional

import numpy as np
from starlette.concurrency import run_in_threadpool

from src.stt_engine import STTEngine
from src.vad_processor import SileroVAD

logger = logging.getLogger(__name__)

# Noisereduce cho giảm nhiễu real-time mic
try:
    import noisereduce as nr
    _noisereduce_available = True
except ImportError:
    _noisereduce_available = False
    logger.warning("noisereduce không khả dụng, bỏ qua giảm nhiễu real-time")

# Hằng số cấu hình
SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SEC = 0.25  # Tối thiểu 0.25s (chuẩn Silero)
MAX_AUDIO_DURATION_SEC = 120.0 # Clip dài hơn 120s bỏ qua (tránh tràn bộ nhớ)
VAD_THRESHOLD = 0.35

# Ngưỡng năng lượng RAW tối thiểu — dưới mức này coi là tuyệt đối im lặng,
# bỏ qua TRƯỚC VAD để tránh noise bị boost và gây ảo giác cho Whisper.
# Tăng lên 0.0005 (~-66 dBFS) để lấy MAX ACCURACY.
MIN_RAW_RMS = 0.0005  # ~ -66 dBFS

# Ngưỡng năng lượng buffer tối thiểu trước khi transcribe (sau normalize)
MIN_AUDIO_ENERGY = 0.008

# DeepFilterNet DISABLED - không tương thích với torchaudio hiện tại
# Dùng RNNoise + noisereduce thay thế
_df_available = False  # Force disable

# RNNoise singleton (lazy load)
_rnnoise_denoiser = None
_rnnoise_available = None

def get_df_model():
    """DeepFilterNet disabled - không tương thích với torchaudio version hiện tại"""
    return None, None

def get_rnnoise_denoiser():
    """Lấy RNNoise denoiser singleton (lazy load)"""
    global _rnnoise_denoiser, _rnnoise_available
    
    if _rnnoise_available is False:
        return None
    
    if _rnnoise_denoiser is None:
        try:
            from rnnoise import RNNoise
            _rnnoise_denoiser = RNNoise()
            _rnnoise_available = True
            logger.info("RNNoise denoiser loaded successfully")
        except ImportError as e:
            logger.warning(f"RNNoise not available: {e}")
            _rnnoise_available = False
            return None
        except Exception as e:
            logger.warning(f"RNNoise load failed: {e}")
            _rnnoise_available = False
            return None
    
    return _rnnoise_denoiser

def denoise_with_logmmse(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Giảm nhiễu bằng Log-MMSE (alternative cho RNNoise)
    
    Log-MMSE (Log Minimum Mean Square Error) là thuật toán speech enhancement
    cổ điển nhưng hiệu quả, không cần deep learning.
    
    Tham số:
        audio: numpy array float32 [-1, 1]
        sample_rate: tần số mẫu
    
    Trả về:
        numpy array float32 [-1, 1]
    """
    try:
        from logmmse import logmmse_from_float
        # logmmse_from_float nhận float32 [-1, 1] và trả về cùng format
        denoised = logmmse_from_float(audio, sample_rate)
        return denoised
    except Exception as e:
        logger.debug(f"LogMMSE failed: {e}")
        return audio

def denoise_with_rnnoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Giảm nhiễu bằng RNNoise
    
    RNNoise yêu cầu:
    - Sample rate: 48kHz
    - Format: int16
    - Frame size: 480 samples (10ms @ 48kHz)
    
    Tham số:
        audio: numpy array float32 [-1, 1], sample_rate bất kỳ
        sample_rate: tần số mẫu của audio input
    
    Trả về:
        numpy array float32 [-1, 1], cùng sample_rate với input
    """
    denoiser = get_rnnoise_denoiser()
    if denoiser is None:
        return audio
    
    try:
        # Bước 1: Resample lên 48kHz nếu cần
        if sample_rate != 48000:
            from scipy.signal import resample
            num_samples_48k = int(len(audio) * 48000 / sample_rate)
            audio_48k = resample(audio, num_samples_48k)
        else:
            audio_48k = audio
        
        # Bước 2: Chuyển sang int16
        audio_int16 = (audio_48k * 32767).astype(np.int16)
        
        # Bước 3: Xử lý theo frame 480 samples
        frame_size = 480  # 10ms @ 48kHz
        num_frames = len(audio_int16) // frame_size
        denoised_frames = []
        
        for i in range(num_frames):
            frame = audio_int16[i * frame_size:(i + 1) * frame_size]
            # RNNoise.process() trả về frame đã denoise
            denoised_frame = denoiser.process_frame(frame)
            denoised_frames.append(denoised_frame)
        
        # Xử lý phần dư (nếu có)
        remainder = len(audio_int16) % frame_size
        if remainder > 0:
            # Pad frame cuối với zeros
            last_frame = np.zeros(frame_size, dtype=np.int16)
            last_frame[:remainder] = audio_int16[-remainder:]
            denoised_last = denoiser.process_frame(last_frame)
            denoised_frames.append(denoised_last[:remainder])
        
        # Bước 4: Ghép lại
        denoised_int16 = np.concatenate(denoised_frames)
        
        # Bước 5: Chuyển về float32
        denoised_48k = denoised_int16.astype(np.float32) / 32767.0
        
        # Bước 6: Resample về sample_rate gốc nếu cần
        if sample_rate != 48000:
            from scipy.signal import resample
            num_samples_orig = int(len(denoised_48k) * sample_rate / 48000)
            denoised = resample(denoised_48k, num_samples_orig)
        else:
            denoised = denoised_48k
        
        return denoised
        
    except Exception as e:
        logger.warning(f"RNNoise processing failed: {e}, return original audio")
        return audio


class STTWebSocketHandler:
    """
    Handler xử lý WebSocket cho streaming STT

    Mỗi kết nối WebSocket tạo một instance mới của handler này.
    STTEngine và SileroVAD được reuse từ singleton đã load sẵn.

    Thuộc tính:
        websocket: Connection WebSocket của client
        stt_engine: Instance STTEngine (singleton)
        vad: Instance SileroVAD (singleton)
        language: Ngôn ngữ cho STT
        audio_buffer: Buffer tích lũy audio chunk cho một câu
        is_speaking: Cờ đánh dấu đang trong đoạn lời nói
    """

    # Class-level singleton: load 1 lần, reuse cho mọi connection
    _stt_engine: Optional[STTEngine] = None
    _vad: Optional[SileroVAD] = None
    _initialized: bool = False

    def __init__(
        self,
        websocket,
        language: str = "vi",
        vad_threshold: float = VAD_THRESHOLD,
    ):
        """
        Khởi tạo STT WebSocket Handler

        Tham số:
            websocket: Instance WebSocket từ FastAPI/Starlette
            language: Mã ngôn ngữ cho STT
            vad_threshold: Ngưỡng Silero VAD
        """
        self.websocket = websocket
        self.language = language
        self.vad_threshold = vad_threshold
        self.audio_buffer: list = []
        self.is_speaking = False
        self.last_speech_time: float = 0.0
        self.silence_duration: float = 0.0
        self.min_audio_length: float = 0.3   # 0.3s tối thiểu (đủ cho 1-2 từ ngắn)
        self.max_buffer_duration: float = 15.0  # Tăng lên 15s buffer

        # Adaptive silence timeout: đợi lâu hơn khi buffer còn ngắn
        self.silence_timeout_short: float = 1.8  # Buffer < 3s: đợi 1.8s im lặng (tránh cắt giữa câu)
        self.silence_timeout_long: float = 1.2   # Buffer >= 3s: đợi 1.2s (đã đủ context)

    @classmethod
    def initialize(cls, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Khởi tạo singleton STTEngine và SileroVAD (gọi 1 lần khi server start)

        Tham số:
            model_path: Đường dẫn mô hình STT (mặc định: PhoWhisper-large-ct2)
            device: Thiết bị suy luận ('cuda' hoặc 'cpu')
        """
        if cls._initialized:
            logger.warning("STTWebSocketHandler đã được khởi tạo, bỏ qua...")
            return

        logger.info("Khởi tạo STT singleton...")

        # Load Silero VAD
        cls._vad = SileroVAD(threshold=VAD_THRESHOLD)
        logger.info("Silero VAD loaded")

        # Ưu tiên load PhoWhisper-large-ct2 nếu tồn tại, nếu không mới dùng EraX
        if model_path is None:
            phowhisper_path = "models/PhoWhisper-large-ct2"
            erax_path = "models/EraX-WoW-Turbo-V1.1-CT2"
            
            if os.path.exists(phowhisper_path):
                model_path = phowhisper_path
                logger.info("Phát hiện PhoWhisper-large-ct2, sẽ sử dụng làm mô hình mặc định")
            else:
                model_path = erax_path
                logger.info("Không tìm thấy PhoWhisper, sử dụng EraX làm mặc định")
        
        cls._stt_engine = STTEngine(model_path=model_path, device=device)
        logger.info(f"STT Engine loaded, device: {cls._stt_engine.device}, model: {model_path}")

        cls._initialized = True

    @classmethod
    def get_engine(cls) -> STTEngine:
        """Lấy instance STTEngine singleton"""
        if not cls._initialized or cls._stt_engine is None:
            raise RuntimeError("STTWebSocketHandler chưa được initialize()")
        return cls._stt_engine

    @classmethod
    def get_vad(cls) -> SileroVAD:
        """Lấy instance SileroVAD singleton"""
        if not cls._initialized or cls._vad is None:
            raise RuntimeError("STTWebSocketHandler chưa được initialize()")
        return cls._vad

    def _decode_audio(self, audio_b64: str) -> np.ndarray:
        """
        Giải mã base64 audio thành numpy array

        Client gửi PCM 16-bit signed integer (little-endian), cần chuyển về float32 [-1, 1].

        Tham số:
            audio_b64: Chuỗi base64 của raw PCM 16-bit

        Trả về:
            Numpy array 1D float32, tần số 16kHz
        """
        try:
            raw_bytes = base64.b64decode(audio_b64)
            # Chuyển bytes → numpy int16 → float32 [-1, 1]
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            logger.error(f"Lỗi giải mã audio base64: {e}")
            raise ValueError(f"Lỗi giải mã audio: {e}")

    def _encode_audio(self, audio: np.ndarray) -> str:
        """
        Mã hóa numpy array thành chuỗi base64

        Dùng cho trường hợp gửi lại audio đã xử lý (nếu cần).

        Tham số:
            audio: Numpy array float32 [-1, 1]

        Trả về:
            Chuỗi base64 của PCM 16-bit
        """
        audio_int16 = (audio * 32768.0).astype(np.int16)
        raw_bytes = audio_int16.tobytes()
        return base64.b64encode(raw_bytes).decode("ascii")

    async def _send_status(self, status: str, message: str = ""):
        """Gửi trạng thái về cho client"""
        try:
            # Kiểm tra trạng thái kết nối trước khi gửi
            if self.websocket.client_state.value == 1: # 1 là CONNECTED
                await self.websocket.send_json({
                    "type": "status",
                    "status": str(status),
                    "message": str(message),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
        except Exception as e:
            # Chỉ log nếu không phải lỗi do đóng kết nối
            if "close message" not in str(e).lower():
                logger.error(f"Lỗi gửi status: {e}")

    async def _send_result(self, text: str, confidence: float = 0.0):
        """Gửi kết quả nhận dạng về cho client"""
        try:
            if self.websocket.client_state.value == 1:
                await self.websocket.send_json({
                    "type": "result",
                    "text": text.strip(),
                    "confidence": round(float(confidence), 4),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
        except Exception as e:
            if "close message" not in str(e).lower():
                logger.error(f"Lỗi gửi kết quả: {e}")

    async def _send_error(self, message: str):
        """Gửi thông báo lỗi về cho client"""
        try:
            await self.websocket.send_json({
                "type": "error",
                "message": str(message),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except Exception:
            pass

    def _silero_confirm(self, audio: np.ndarray) -> bool:
        """
        Xác nhận lại audio bằng Silero VAD (secondary VAD check)

        Đây là bước xác nhận thứ 2 trong kiến trúc VAD hybrid:
        WebRTC VAD (client) đã phát hiện lời nói → gửi chunk
        → Silero VAD (server) xác nhận lại → mới transcribe.

        Phương thức này dùng get_speech_probabilities() để hỗ trợ
        audio chunk có ĐỘ DÀI TÙY Ý (ví dụ: 4096 samples từ client).

        Tham số:
            audio: Numpy array 1D float32, tần số 16kHz

        Trả về:
            True nếu Silero VAD confirm có lời nói, False ngược lại
        """
        # Kiểm tra độ dài audio hợp lệ
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_DURATION_SEC:
            logger.debug(f"Audio quá ngắn ({duration:.2f}s < {MIN_AUDIO_DURATION_SEC}s), bỏ qua")
            return False
        if duration > MAX_AUDIO_DURATION_SEC:
            logger.warning(f"Audio quá dài ({duration:.2f}s > {MAX_AUDIO_DURATION_SEC}s), cắt")
            audio = audio[: int(MAX_AUDIO_DURATION_SEC * SAMPLE_RATE)]

        # Dùng is_speech_from_probs hỗ trợ chunk dài (4096 samples)
        return self.get_vad().is_speech_from_probs(audio)

    async def _transcribe_and_respond(self, audio: np.ndarray):
        """
        Transcribe audio và gửi kết quả về client

        Tham số:
            audio: Numpy array 1D float32, tần số 16kHz
        """
        await self._send_status("processing", "Đang nhận dạng...")

        try:
            # Chạy STT với các tham số chống hallucination cực mạnh
            results = await run_in_threadpool(
                self.get_engine().transcribe,
                audio,
                language=self.language,
                vad_filter=True,         
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
                temperature=0.0,         
                beam_size=5,             # beam=5: xét nhiều giả thuyết hơn → chính xác hơn
                condition_on_previous_text=False, 
                word_timestamps=True,    
                no_speech_threshold=0.3,           # Lọc silence/noise (0.3 cân bằng giữa chặt và không cắt oan)
                compression_ratio_threshold=1.8,   # Siết thêm (2.0 -> 1.8)
            )

            if results:
                # --- IN RA LOG KẾT QUẢ GỐC TỪ WHISPER (CHƯA QUA FILTER) ĐỂ USER DEBUG ---
                raw_texts = [f"[{seg.get('text', '').strip()}] (no_speech: {seg.get('no_speech_prob', 0.0):.3f}, logprob: {seg.get('avg_logprob', 0.0):.3f})" for seg in results]
                logger.info(f"🎤 [RAW WHISPER OUTPUT]: {' | '.join(raw_texts)}")
                # -------------------------------------------------------------------------

                # Blacklist các câu ảo giác kinh điển của Whisper tiếng Việt
                hallucination_blacklist = [
                    "cảm ơn các bạn", "đăng ký kênh", "ông cũng bị cáo buộc", 
                    "nguyễn thị thu", "đài truyền hình", "bản tin", "xin chào các bạn",
                    "hẹn gặp lại", "subtitles by", "tổng thống", "mỹ nhân dân",
                    "tác phẩm nghệ thuật", "đó là nguyên nhân", "tôi không ngại ngùng",
                    "tại sao lại thế", "thiết kế tư thế", "thuộc về thiên tai phong phú",
                    "những thông tin tiếp theo", "tài liệu sau", "tỏ ra bất ngờ",
                    "tên lửa vào", "bức ảnh", "thiếu nhiệt đới", "đáng tiếc",
                    # Thêm từ log thực tế
                    "chính quyền", "tàn phá", "thiêu tử", "tai nạn gian",
                    "tâm hồn của một người", "đoạn trích truyện",
                    "tác phẩm đã được triển khai", "người đàn ông không thể",
                    "sự tàn phá", "bối cảnh đó", "phong phú", "truyện ngắn",
                    "thủ tục trong các tổ chức", "được rút ra do",
                ]

                # Filter confidence + no_speech
                filtered_segments = []
                for seg in results:
                    # seg là dict với keys: text, start, end, avg_logprob, no_speech_prob, words

                    # Filter 1: Bỏ qua segment có no_speech_prob cao (> 0.4)
                    # Siết từ 0.6 → 0.4: hallucination thường có no_speech_prob 0.4-0.7
                    no_speech_prob = seg.get("no_speech_prob", 0.0)
                    if no_speech_prob > 0.4:
                        logger.info(f"Filter no_speech: prob={no_speech_prob:.3f}, text='{seg['text']}'")
                        continue

                    # Filter 2: Soft logprob filter
                    # Hard drop -0.6 gây mất tên riêng, tiếng địa phương, từ hiếm.
                    # Production fix: Chỉ drop khi avg_logprob < -1.0 (cực kỳ không tự tin)
                    # HOẶC khi logprob thấp (-0.6 đến -1.0) MÀ text bị lặp lại → mới là ảo giác.
                    avg_logprob = seg.get("avg_logprob", 0.0)
                    text_preview = seg.get("text", "").strip()
                    is_repetitive = False
                    if text_preview:
                        words_check = text_preview.split()
                        if len(words_check) >= 4:
                            unique_ratio = len(set(words_check)) / len(words_check)
                            is_repetitive = unique_ratio < 0.5

                    if avg_logprob < -0.8:
                        logger.info(f"Filter cực thấp confidence: logprob={avg_logprob:.3f}, text='{text_preview}'")
                        continue
                    elif avg_logprob < -0.6 and is_repetitive:
                        logger.info(f"Filter logprob thấp + lặp: logprob={avg_logprob:.3f}, text='{text_preview}'")
                        continue

                    # Filter 3: Bỏ qua segment rỗng
                    text = seg.get("text", "").strip()
                    if not text:
                        logger.info("Filter empty text")
                        continue

                    # Filter blacklist ảo giác
                    text_lower = text.lower()
                    is_blacklisted = any(bad_phrase in text_lower for bad_phrase in hallucination_blacklist)
                    if is_blacklisted:
                        logger.warning(f"Filter BLACKLIST ảo giác: '{text}'")
                        continue

                    # Filter: Tỷ lệ từ/giây bất thường (Whisper bịa text dài hơn audio thực tế)
                    # Tiếng Việt nói bình thường ~3-5 từ/giây, >8 từ/giây rất khả nghi
                    seg_duration = seg.get("end", 0) - seg.get("start", 0)
                    if seg_duration > 0:
                        words_count = len(text.split())
                        words_per_sec = words_count / seg_duration
                        if words_per_sec > 8.0 and words_count > 6:
                            logger.warning(f"Filter tốc độ bất thường ({words_per_sec:.1f} từ/giây, {words_count} từ trong {seg_duration:.1f}s): '{text}'")
                            continue
                        
                    # Filter 4: Phát hiện lặp từ (Hallucination loop) TRÊN TỪNG ĐOẠN
                    # Xử lý ở đây để chỉ bỏ qua đoạn ảo giác, GIỮ LẠI đoạn nói thật
                    seg_words = text.split()
                    if len(seg_words) > 5:
                        unique_words = len(set(seg_words))
                        repetition_ratio = unique_words / len(seg_words)
                        
                        # Hallucination nếu < 40% từ unique
                        if repetition_ratio < 0.4:
                            logger.warning(f"Filter Hallucination (lặp lại {repetition_ratio:.1%}): '{text}'")
                            continue
                        
                        # Kiểm tra cụm từ lặp lại liên tiếp
                        is_phrase_repeat = False
                        for i in range(len(seg_words) - 2):
                            phrase = " ".join(seg_words[i:i+3])
                            count = text.count(phrase)
                            if count > 3:
                                logger.warning(f"Filter Hallucination (cụm '{phrase}' lặp {count} lần): '{text}'")
                                is_phrase_repeat = True
                                break
                        if is_phrase_repeat:
                            continue

                    filtered_segments.append(seg)
                
                if not filtered_segments:
                    logger.info("Tất cả segments bị filter, không có kết quả")
                    await self._send_result("")
                    return
                
                full_text = " ".join(seg["text"] for seg in filtered_segments)
                
                # Filter 5: Bỏ kết quả quá ngắn (1-2 ký tự thường là hallucination)
                if len(full_text.strip()) <= 2:
                    logger.info(f"Filter text quá ngắn ({len(full_text.strip())} chars): '{full_text}'")
                    await self._send_result("")
                    return

                # Filter 6: Bỏ text chỉ chứa dấu câu/ký tự đặc biệt
                text_alpha = re.sub(r'[^\w\s]', '', full_text, flags=re.UNICODE).strip()
                if not text_alpha:
                    logger.info(f"Filter text chỉ chứa dấu câu: '{full_text}'")
                    await self._send_result("")
                    return
                if not full_text.strip():
                    logger.info("Kết quả sau khi lọc là rỗng")
                    await self._send_result("")
                    return
                
                # Log để debug
                logger.info(f"Transcription OK: '{full_text}' ({len(filtered_segments)} segments)")
                
                await self._send_result(full_text)
            else:
                await self._send_result("")

        except Exception as e:
            logger.error(f"Lỗi transcribe: {e}")
            await self._send_error(f"Lỗi nhận dạng: {str(e)}")
        finally:
            # Khôi phục trạng thái listening sau khi xử lý xong
            await self._send_status("listening", "Đang nghe...")

    async def handle(self):
        """
        Vòng lặp xử lý WebSocket chính

        Nhận message từ client, xử lý theo loại:
            - "audio": giải mã → Silero VAD confirm → transcribe → gửi kết quả
            - "config": cập nhật cấu hình (language, threshold)
            - "ping": phản hồi pong
        """
        client_id = id(self.websocket)
        logger.info(f"WebSocket [{client_id}] connected, language: {self.language}")

        # Gửi trạng thái sẵn sàng
        await self._send_status("listening", "Server sẵn sàng, bắt đầu nói...")

        try:
            async for raw_message in self.websocket.iter_text():
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning(f"WebSocket [{client_id}] nhận message không phải JSON: {raw_message[:100]}")
                    await self._send_error("Message phải là JSON hợp lệ")
                    continue

                msg_type = message.get("type", "")

                if msg_type == "audio":
                    # ===== Xử lý audio chunk =====
                    await self._handle_audio(message)

                elif msg_type == "config":
                    # ===== Cập nhật cấu hình =====
                    await self._handle_config(message)

                elif msg_type == "ping":
                    await self.websocket.send_json({"type": "pong", "timestamp": float(time.time())})

                else:
                    logger.warning(f"WebSocket [{client_id}] unknown message type: {msg_type}")

        except Exception as e:
            logger.error(f"WebSocket [{client_id}] lỗi: {e}")

        finally:
            logger.info(f"WebSocket [{client_id}] disconnected")
            await self._cleanup()

    async def _handle_audio(self, message: dict):
        """
        Xử lý audio message từ client

        Pipeline (đúng thứ tự để tránh ảo giác):
            1. Giải mã base64 → numpy array (float32 raw)
            2. Kiểm tra ngưỡng năng lượng tối thiểu (energy gate)
               → bỏ qua ngay nếu là silence tuyệt đối (dưới MIN_RAW_RMS)
            3. Silero VAD xác nhận trên audio RAW (CHƯA normalize)
               → VAD cần biên độ gốc để phân biệt speech vs noise
            4. CHỈ KHI VAD confirm: normalize + buffer + transcribe
               → Tránh noise bị boost lên mức bình thường rồi đưa vào Whisper
        """
        audio_b64 = message.get("audio")
        if not audio_b64:
            await self._send_error("Thiếu field 'audio' trong message")
            return

        # Bước 1: Giải mã audio
        try:
            audio_raw = self._decode_audio(audio_b64)
        except ValueError as e:
            await self._send_error(str(e))
            return

        duration = len(audio_raw) / SAMPLE_RATE
        rms_raw = float(np.sqrt(np.mean(audio_raw ** 2)))
        db_raw = float(20 * np.log10(rms_raw)) if rms_raw > 1e-10 else -100.0

        logger.debug(
            f"Audio raw: {duration:.3f}s, RMS={rms_raw:.6f} ({db_raw:.1f} dB)"
        )

        # Bước 2: Energy gate — Soft 2-tier thay vì hard drop.
        # Âm cuối tiếng Việt (t, c, p, dấu nhẹ) thường có RMS rất nhỏ, dễ bị cắt oan.
        #   - RMS < 0.0003        : Im lặng tuyệt đối, drop hoàn toàn.
        #   - 0.0003 <= RMS < 0.001: Vùng xám — vẫn giữ nếu VAD xác nhận có giọng.
        #   - RMS >= 0.001        : Âm thanh bình thường, cho qua luôn.
        if rms_raw < 0.0003:
            logger.debug(
                f"Energy gate HARD DROP: bỏ qua chunk (RMS={rms_raw:.6f} < 0.0003)"
            )
            # Flush buffer nếu đang tích lũy và đã im lặng đủ lâu
            if len(self.audio_buffer) > 0:
                self.audio_buffer.append(audio_raw.copy()) # Giữ lại audio padding để không cụt chữ
                self.silence_duration += duration
                # Adaptive silence timeout: buffer ngắn đợi lâu hơn
                total_dur = sum(len(c) for c in self.audio_buffer) / SAMPLE_RATE
                silence_threshold = self.silence_timeout_long if total_dur >= 3.0 else self.silence_timeout_short
                if self.silence_duration >= silence_threshold:
                    await self._flush_buffer()
            await self._send_status("silence", "Không phát hiện lời nói")
            return

        # Vùng xám: 0.0003 <= RMS < 0.001 — để VAD quyết định
        is_soft_zone = rms_raw < 0.001
        if is_soft_zone:
            logger.debug(f"Energy gate SOFT ZONE: RMS={rms_raw:.6f}, để VAD quyết định")

        # Bước 3: Silero VAD chạy trên audio RAW (biên độ gốc).
        # Điều này đảm bảo VAD thấy đúng mức năng lượng thực tế của tín hiệu.
        try:
            raw_probs = self.get_vad().get_speech_probabilities(audio_raw)
            max_prob = float(raw_probs.max()) if raw_probs.size > 0 else 0.0
            avg_prob = float(raw_probs.mean()) if raw_probs.size > 0 else 0.0
            is_speech = bool(max_prob >= self.vad_threshold)

            logger.debug(
                f"VAD (raw): max={max_prob:.4f}, avg={avg_prob:.4f}, "
                f"threshold={self.vad_threshold}, is_speech={is_speech}, "
                f"duration={duration:.3f}s, rms_raw={rms_raw:.6f}"
            )
        except Exception as e:
            logger.error(f"Lỗi Silero VAD confirm: {e}", exc_info=True)
            await self._send_status("error", f"Lỗi VAD: {e}")
            return

        if not is_speech:
            logger.debug(
                f"Silero VAD reject — bỏ qua chunk "
                f"(max_prob={max_prob:.4f} < threshold={self.vad_threshold})"
            )
            # Nếu là vùng xám RMS mà VAD cũng reject → mới thực sự là silence
            if len(self.audio_buffer) > 0:
                self.audio_buffer.append(audio_raw.copy()) # Giữ lại audio padding để không cụt chữ
                self.silence_duration += duration
                # Adaptive silence timeout: buffer ngắn đợi lâu hơn
                total_dur = sum(len(c) for c in self.audio_buffer) / SAMPLE_RATE
                silence_threshold = self.silence_timeout_long if total_dur >= 3.0 else self.silence_timeout_short
                if self.silence_duration >= silence_threshold:
                    await self._flush_buffer()
            await self._send_status("silence", "Không phát hiện lời nói")
            return

        # Nếu là vùng xám RMS nhưng VAD CONFIRM → giữ lại (âm cuối tiếng Việt)
        if is_soft_zone:
            logger.debug(f"Energy gate SOFT ZONE + VAD confirm → giữ chunk (âm cuối)")

        # Bước 4: VAD confirm — thêm audio RAW vào buffer
        # KHÔNG normalize từng chunk 30ms riêng lẻ (phá hủy dynamics tự nhiên)
        # Normalize sẽ thực hiện MỘT LẦN trên toàn bộ buffer trong _flush_buffer()
        self.audio_buffer.append(audio_raw.copy())
        self.silence_duration = 0.0
        self.is_speaking = True
        self.last_speech_time = time.time()

        total_duration = sum(len(c) for c in self.audio_buffer) / SAMPLE_RATE
        buffer_chunks = len(self.audio_buffer)

        logger.debug(
            f"Silero VAD confirm — thêm vào buffer "
            f"(max_prob={max_prob:.4f}, chunks={buffer_chunks}, total={total_duration:.2f}s)"
        )

        # Transcribe ngay khi buffer đạt 15s
        if total_duration >= 15.0:
            await self._flush_buffer(force=True)

    async def _flush_buffer(self, force: bool = False):
        """
        Transcribe và reset audio buffer

        Tham số:
            force: True → transcribe dù chưa đủ silence (dùng khi buffer đầy)
        """
        if not self.audio_buffer:
            return

        buffered_audio = np.concatenate(self.audio_buffer)
        buffered_duration = len(buffered_audio) / SAMPLE_RATE
        raw_rms = float(np.sqrt(np.mean(buffered_audio ** 2)))

        # Chỉ transcribe nếu đủ dài VÀ đủ năng lượng (kiểm tra trên audio RAW)
        if buffered_duration >= self.min_audio_length and raw_rms >= MIN_AUDIO_ENERGY:
            # Normalize toàn bộ buffer MỘT LẦN (giữ dynamics tự nhiên của speech)
            # Thay vì normalize từng chunk 30ms riêng lẻ → phá hủy tín hiệu
            if raw_rms > 1e-8:
                target_rms = 0.1
                buffered_audio = buffered_audio * (target_rms / raw_rms)
            buffer_rms = float(np.sqrt(np.mean(buffered_audio ** 2)))
            # Giảm nhiễu bằng noisereduce trước khi đưa vào Whisper
            if _noisereduce_available:
                try:
                    denoised = nr.reduce_noise(
                        y=buffered_audio,
                        sr=SAMPLE_RATE,
                        stationary=True,     # Nhiễu ổn định (quạt, ồn phòng)
                        prop_decrease=0.6,   # Giảm 60% nhiễu, giữ 40% — giữ âm thanh tự nhiên hơn tránh méo giọng
                    )
                    denoised_rms = float(np.sqrt(np.mean(denoised ** 2)))
                    logger.info(
                        f"Noisereduce: RMS trước={buffer_rms:.6f}, "
                        f"sau={denoised_rms:.6f}"
                    )
                    # Chỉ dùng audio đã denoise nếu còn đủ năng lượng
                    if denoised_rms >= MIN_AUDIO_ENERGY * 0.5:
                        buffered_audio = denoised
                    else:
                        logger.warning("Noisereduce loại bỏ quá nhiều, giữ audio gốc")
                except Exception as e:
                    logger.warning(f"Noisereduce failed: {e}, dùng audio gốc")

            logger.info(
                f"Flush buffer: {buffered_duration:.2f}s, "
                f"RMS={buffer_rms:.6f} (force={force})"
            )
            await self._transcribe_and_respond(buffered_audio)
        else:
            logger.info(
                f"Buffer không hợp lệ: duration={buffered_duration:.2f}s, "
                f"RMS={raw_rms:.6f} (min_dur={self.min_audio_length}, "
                f"min_rms={MIN_AUDIO_ENERGY}), bỏ qua"
            )

        self.audio_buffer.clear()
        self.silence_duration = 0.0
        self.is_speaking = False

    async def _handle_config(self, message: dict):
        """Xử lý cập nhật cấu hình từ client"""
        new_language = message.get("language")
        if new_language:
            self.language = new_language
            logger.info(f"WebSocket [{id(self.websocket)}] cập nhật language: {new_language}")
            await self._send_status("listening", f"Đã chuyển ngôn ngữ sang: {new_language}")

        new_threshold = message.get("vad_threshold")
        if new_threshold is not None:
            if 0.0 <= new_threshold <= 1.0:
                self.vad_threshold = new_threshold
                self.get_vad().set_threshold(new_threshold)
                logger.info(f"WebSocket [{id(self.websocket)}] cập nhật vad_threshold: {new_threshold}")
            else:
                await self._send_error(f"vad_threshold phải trong [0.0, 1.0], nhận được: {new_threshold}")

    async def _cleanup(self):
        """Dọn dẹp khi connection đóng"""
        self.audio_buffer.clear()
        self.is_speaking = False
