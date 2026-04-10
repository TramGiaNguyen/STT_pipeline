#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming_engine.py - Module streaming STT real-time từ micro

Kết hợp thu âm từ micro (sounddevice), VAD (Silero) và STT (faster-whisper)
để tạo luồng nhận dạng giọng nói real-time. Module này dùng cho CLI/test
không cần web server.

Luồng hoạt động:
    1. sounddevice liên tục thu âm từ micro theo từng chunk (~32ms)
    2. Silero VAD phát hiện đoạn có lời nói
    3. Khi phát hiện hết câu (im lặng sau lời nói) → ghép buffer → transcribe
    4. In kết quả ra console

Sử dụng:
    python -m src.streaming_engine [--language LANG] [--threshold THRESHOLD]
    python -m src.streaming_engine --help
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "sounddevice chưa được cài đặt. Chạy: pip install sounddevice>=0.4.0"
    )

from src.stt_engine import STTEngine
from src.vad_processor import SileroVAD

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Hằng số cấu hình audio
SAMPLE_RATE = 16000  # Tần số mẫu 16kHz (chuẩn cho faster-whisper và Silero VAD)
CHUNK_SIZE = 512     # 512 samples @ 16kHz = 32ms mỗi chunk
                     # Đủ nhỏ cho latency thấp, đủ lớn cho VAD ổn định
BUFFER_MAX_SIZE = 800  # Tối đa ~50s audio trong buffer (800 * 32ms = 25.6s)
                       # Ngăn buffer tăng vô hạn khi không ngừng nói
MIN_SPEECH_CHUNKS = 3  # Tối thiểu 3 chunk có lời nói (96ms) mới coi là câu hợp lệ
                       # Tránh noise ngắn bị nhận nhầm thành lời nói


class StreamingSTT:
    """
    Engine streaming STT real-time từ micro

    Lớp này phối hợp 3 thành phần: thu âm micro (sounddevice), VAD (Silero),
    và STT (faster-whisper) để tạo luồng nhận dạng giọng nói real-time.
    Thiết kế thread-safe: thu âm trên thread riêng, xử lý trên main thread.

    Thuộc tính:
        stt_engine: Instance STTEngine để nhận dạng
        vad: Instance SileroVAD để phát hiện lời nói
        audio_queue: Queue chứa audio chunk từ callback
        speech_buffer: List chứa các chunk có lời nói đang chờ
        running: Cờ kiểm soát vòng lặp chính

    Ví dụ sử dụng:
        >>> engine = StreamingSTT()
        >>> engine.start()
        # Nói vào micro, kết quả in ra console
        >>> engine.stop()
    """

    def __init__(
        self,
        language: Optional[str] = None,
        vad_threshold: float = 0.5,
        min_speech_chunks: int = MIN_SPEECH_CHUNKS,
        buffer_max_size: int = BUFFER_MAX_SIZE,
        print_prob: bool = False,
    ):
        """
        Khởi tạo StreamingSTT engine

        Tham số:
            language: Mã ngôn ngữ cho STT, ví dụ 'vi', 'en'. Mặc định None (tự nhận diện).
            vad_threshold: Ngưỡng VAD [0.0, 1.0]. Mặc định 0.5.
            min_speech_chunks: Số chunk lời nói tối thiểu để coi là câu hợp lệ.
            buffer_max_size: Số chunk tối đa trong speech_buffer (ngăn tràn bộ nhớ).
            print_prob: Có in xác suất VAD ra console không (debug).
        """
        self.language = language
        self.print_prob = print_prob
        self.min_speech_chunks = min_speech_chunks
        self.buffer_max_size = buffer_max_size

        # Queue chứa audio chunk từ callback (thread-safe)
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Buffer chứa các chunk có lời nói
        self.speech_buffer: List[np.ndarray] = []

        # Cờ kiểm soát vòng lặp
        self.running = False

        # Biến trạng thái VAD
        self.in_speech = False   # Đang trong đoạn lời nói
        self.speech_chunks_count = 0  # Đếm số chunk có lời nói liên tiếp
        self.silence_after_speech = 0  # Đếm số chunk im lặng sau đoạn lời nói

        logger.info("Khởi tạo StreamingSTT engine...")

        # Load Silero VAD
        self.vad = SileroVAD(threshold=vad_threshold)
        logger.info(f"VAD threshold: {vad_threshold}")

        # Load STT Engine
        logger.info("Đang load STT engine (có thể mất vài giây)...")
        self.stt_engine = STTEngine()
        logger.info(f"STT engine loaded, ngôn ngữ: {language or 'tự động nhận diện'}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback được gọi bởi sounddevice mỗi khi thu được chunk audio

        Callback này chạy trên thread riêng của sounddevice. Việc xử lý VAD
        và transcribe được thực hiện trên main thread để tránh race condition.

        Tham số:
            indata: numpy array shape (frames, channels), kiểu float32
            frames: Số frames trong chunk
            time_info: Thông tin thời gian từ PortAudio
            status: Trạng thái callback (âm thanh overread/overflow,...)
        """
        if status:
            logger.warning(f"sounddevice status: {status}")

        # indata shape: (frames, channels) → lấy channel đầu tiên, flatten về 1D
        audio_chunk = indata[:, 0].copy()
        self.audio_queue.put(audio_chunk)

    def _detect_speech_end(self, chunk: np.ndarray) -> bool:
        """
        Kiểm tra xem chunk hiện tại có phải là kết thúc của một câu nói không

        Logic phát hiện cuối câu:
            - Chunk trước đó là lời nói (in_speech = True)
            - Chunk hiện tại là im lặng (is_speech = False)
            - Có đủ số chunk im lặng liên tiếp (silence_after_speech >= 2)
            → Coi như kết thúc câu → trigger transcribe

        Tham số:
            chunk: Numpy array 1D float32, tần số 16kHz

        Trả về:
            True nếu phát hiện kết thúc câu, False ngược lại
        """
        is_speech = self.vad.is_speech(chunk)

        if self.print_prob:
            prob = self.vad.get_speech_probability(chunk)
            print(f"\r[VAD prob: {prob:.3f}]", end="", flush=True)

        if is_speech:
            # Phát hiện có lời nói
            if not self.in_speech:
                # Bắt đầu một đoạn lời nói mới
                logger.debug("Phát hiện bắt đầu lời nói")
            self.in_speech = True
            self.speech_chunks_count += 1
            self.silence_after_speech = 0
            return False

        else:
            # Im lặng
            if self.in_speech:
                # Đang trong đoạn lời nói, kiểm tra xem đã hết câu chưa
                self.silence_after_speech += 1
                # Đủ 2 chunk im lặng liên tiếp (~64ms) → coi là hết câu
                if self.silence_after_speech >= 2:
                    # Kiểm tra đủ chunk lời nói không (tránh noise ngắn)
                    if self.speech_chunks_count >= self.min_speech_chunks:
                        logger.debug(
                            f"Phát hiện kết thúc câu: {self.speech_chunks_count} "
                            f"chunk lời nói + {self.silence_after_speech} chunk im lặng"
                        )
                        return True
                    else:
                        logger.debug(
                            f"Bỏ qua: chỉ {self.speech_chunks_count} chunk lời nói "
                            f"(ít hơn ngưỡng {self.min_speech_chunks})"
                        )
                        self.in_speech = False
                        self.speech_chunks_count = 0
                        self.speech_buffer.clear()
            return False

    def _transcribe_buffer(self) -> Optional[str]:
        """
        Transcribe toàn bộ speech_buffer đã tích lũy

        Ghép các chunk trong buffer thành một array liên tục,
        gọi faster-whisper transcribe, in kết quả và reset buffer.

        Trả về:
            Văn bản nhận dạng được, hoặc None nếu có lỗi
        """
        if not self.speech_buffer:
            return None

        # Ghép các chunk thành audio liên tục
        audio_data = np.concatenate(self.speech_buffer)

        # Log thông tin
        duration = len(audio_data) / SAMPLE_RATE
        logger.info(f"Transcribing {duration:.2f}s audio, {len(self.speech_buffer)} chunks...")

        try:
            # Chạy STT (tắt VAD filter vì đã xử lý bằng Silero)
            results = self.stt_engine.transcribe(
                audio_data,
                language=self.language,
                vad_filter=False,  # Đã xử lý VAD bằng Silero
            )

            if results:
                full_text = " ".join(seg["text"] for seg in results)
                return full_text
            return None

        except Exception as e:
            logger.error(f"Lỗi transcribe: {e}")
            return None

    def start(self):
        """
        Bắt đầu streaming: mở micro stream và xử lý vòng lặp chính

        Phương thức này BLOCKS main thread. Xử lý:
            1. Mở InputStream với sounddevice
            2. Vòng lặp: lấy chunk từ queue → VAD check → transcribe khi hết câu
            3. Nhấn Ctrl+C hoặc Ctrl+Z để dừng
        """
        self.running = True

        # Đăng ký signal handler cho Ctrl+C
        def signal_handler(sig, frame):
            logger.info("\nNhận tín hiệu dừng, đang thoát...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"Bắt đầu thu âm micro (16kHz, chunk {CHUNK_SIZE} samples = {CHUNK_SIZE/SAMPLE_RATE*1000:.0f}ms)")
        logger.info("Nhấn Ctrl+C để dừng")
        logger.info("-" * 50)

        try:
            # Mở InputStream: liên tục thu âm từ micro
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",  # float32 [−1, 1] tương thích cả VAD và faster-whisper
                blocksize=CHUNK_SIZE,
                callback=self._audio_callback,
            )

            with stream:
                while self.running:
                    try:
                        # Lấy chunk từ queue với timeout ngắn
                        # Nếu queue rỗng sau 0.5s → vẫn tiếp tục (không block)
                        chunk = self.audio_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # Kiểm tra kết thúc câu
                    if self._detect_speech_end(chunk):
                        # Hết câu → transcribe
                        result_text = self._transcribe_buffer()
                        if result_text:
                            print(f"\n[KẾT QUẢ] {result_text}")
                        # Reset trạng thái
                        self.in_speech = False
                        self.speech_chunks_count = 0
                        self.silence_after_speech = 0

                    # Thêm chunk vào speech_buffer nếu đang trong đoạn lời nói
                    if self.in_speech:
                        self.speech_buffer.append(chunk)

                        # Ngăn buffer tăng vô hạn (khi người dùng nói quá lây)
                        if len(self.speech_buffer) > self.buffer_max_size:
                            logger.warning(
                                f"Speech buffer đạt giới hạn ({self.buffer_max_size}), "
                                "tự động transcribe..."
                            )
                            result_text = self._transcribe_buffer()
                            if result_text:
                                print(f"\n[KẾT QUẢ] {result_text}")
                            self.in_speech = False
                            self.speech_chunks_count = 0
                            self.silence_after_speech = 0

        except sd.PortAudioError as e:
            logger.error(f"Lỗi PortAudio: {e}")
            logger.error("Kiểm tra micro có hoạt động không và không bị dùng bởi app khác")
            raise
        finally:
            self.running = False
            logger.info("StreamingSTT đã dừng")


def main():
    """Entry point cho CLI streaming"""
    parser = argparse.ArgumentParser(
        description="Streaming STT real-time từ micro (không cần web server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
    python -m src.streaming_engine                        # Dùng mặc định
    python -m src.streaming_engine --language vi          # Chỉ định tiếng Việt
    python -m src.streaming_engine --vad-threshold 0.3    # VAD nhạy hơn
    python -m src.streaming_engine --print-prob           # In debug VAD probability
        """,
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Mã ngôn ngữ cho STT, ví dụ 'vi', 'en', 'zh'. Mặc định: tự nhận diện",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Ngưỡng VAD [0.0-1.0]. Mặc định: 0.5. Cao hơn → ít nhạy hơn",
    )
    parser.add_argument(
        "--min-speech-chunks",
        type=int,
        default=MIN_SPEECH_CHUNKS,
        help=f"Số chunk lời nói tối thiểu. Mặc định: {MIN_SPEECH_CHUNKS}",
    )
    parser.add_argument(
        "--print-prob",
        action="store_true",
        help="In xác suất VAD ra console (debug)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Bật log DEBUG",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate threshold
    if not 0.0 <= args.vad_threshold <= 1.0:
        parser.error(f"--vad-threshold phải trong [0.0, 1.0], nhận được: {args.vad_threshold}")

    try:
        engine = StreamingSTT(
            language=args.language,
            vad_threshold=args.vad_threshold,
            min_speech_chunks=args.min_speech_chunks,
            print_prob=args.print_prob,
        )
        engine.start()
    except KeyboardInterrupt:
        logger.info("Đã dừng (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Lỗi không xử lý được: {e}")
        raise


if __name__ == "__main__":
    main()
