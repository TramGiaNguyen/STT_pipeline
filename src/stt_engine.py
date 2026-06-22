#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stt_engine.py - Module engine STT cốt lõi

Bọc (wrap) WhisperModel của faster-whisper, cung cấp chức năng nhận dạng giọng nói
thành văn bản (Speech-to-Text). Hỗ trợ nhận dạng đa ngôn ngữ, xuất timestamp,
VAD (Voice Activity Detection - phát hiện hoạt động giọng nói).

Các chức năng chính:
    - Load mô hình cục bộ (bỏ qua HuggingFace cache)
    - Chuyển file âm thanh thành văn bản (có timestamp)
    - Hỗ trợ phát hiện hoạt động giọng nói (VAD) để lọc đoạn im lặng
    - Hỗ trợ chỉ định ngôn ngữ cố định hoặc tự động nhận diện
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List

import numpy as np

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model as download_faster_whisper_model

logger = logging.getLogger(__name__)


class STTEngine:
    """
    Lớp đóng gói engine nhận dạng giọng nói

    Lớp này bọc các chức năng cốt lõi của faster-whisper, cung cấp API đơn giản
    cho bên ngoài gọi. Hỗ trợ tăng tốc CUDA GPU (nếu môi trường hỗ trợ) và fallback CPU.

    Thuộc tính:
        model: Instance WhisperModel của faster-whisper bên dưới
        model_path: Đường dẫn cục bộ của mô hình đang được load
        device: Thiết bị suy luận hiện tại ('cuda' hoặc 'cpu')
        compute_type: Kiểu độ chính xác tính toán hiện tại

    Ví dụ sử dụng:
        >>> engine = STTEngine()                          # Dùng đường dẫn mô hình mặc định
        >>> engine = STTEngine(model_path="./my_model")   # Chỉ định đường dẫn tùy chỉnh
        >>> results = engine.transcribe("test.wav")      # Nhận dạng âm thanh
        >>> for seg in results:
        ...     print(f"[{seg['start']:.2f}s] {seg['text']}")
    """

    # Đường dẫn mặc định của mô hình tương đối với thư mục gốc dự án
    DEFAULT_MODEL_SUBDIR = Path("models") / "EraX-WoW-Turbo-V1.1-CT2"

    # Thiết bị được hỗ trợ theo thứ tự ưu tiên: GPU > CPU
    SUPPORTED_DEVICES = ["cuda", "cpu"]

    # Cấu hình VAD mặc định tối ưu cho việc tách câu tiếng Việt tự nhiên trong hội thoại nhanh/ngắn
    DEFAULT_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 15,       # Giới hạn phân đoạn tối đa 15s để tránh gom quá dài
        "min_silence_duration_ms": 600,     # Khoảng lặng tối thiểu 600ms để VAD cắt câu nhạy bén hơn
        "speech_pad_ms": 400
    }

    # Kiểu compute type khả dụng trên CUDA (sắp xếp theo tốc độ từ nhanh đến chậm)
    CUDA_COMPUTE_TYPES = ["bfloat16", "float16", "int8_float16", "float32"]
    # Kiểu compute type khả dụng trên CPU
    CPU_COMPUTE_TYPES = ["int8", "float32"]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_if_missing: bool = False,
    ):
        """
        Khởi tạo engine STT và load mô hình

        Tham số:
            model_path: Đường dẫn cục bộ của mô hình, mặc định là "models/EraX-WoW-Turbo-V1.1-CT2"
            device: Thiết bị suy luận, 'cuda' (GPU) hoặc 'cpu', mặc định tự nhận diện
            compute_type: Kiểu độ chính xác tính toán, mặc định 'bfloat16' (GPU) hoặc 'int8' (CPU)
            download_if_missing: Có tự động tải mô hình khi không tồn tại không (mặc định False)

        Ngoại lệ:
            FileNotFoundError: Đường dẫn mô hình không tồn tại và download_if_missing=False
            RuntimeError: device hoặc compute_type được chỉ định không được hỗ trợ
        """
        # Xác định đường dẫn mô hình
        if model_path is None:
            # Mặc định dùng đường dẫn tương đối với thư mục gốc dự án
            project_root = Path(__file__).resolve().parent.parent
            model_path = str(project_root / self.DEFAULT_MODEL_SUBDIR)

        self.model_path = model_path
        logger.info(f"Khởi tạo STTEngine, đường dẫn mô hình: {model_path}")

        # Kiểm tra mô hình có tồn tại không
        if not Path(model_path).exists():
            if download_if_missing:
                logger.warning(f"Thư mục mô hình không tồn tại, chuẩn bị tải tự động...")
                self._download_default_model()
            else:
                raise FileNotFoundError(
                    f"Thư mục mô hình không tồn tại: {model_path}\n"
                    f"Hãy chạy 'python scripts/download_model.py' để tải mô hình, "
                    f"hoặc đặt download_if_missing=True để tải tự động."
                )

        # Tự nhận diện thiết bị khả dụng: ưu tiên CUDA
        if device is None:
            device = self._detect_device()
        if device not in self.SUPPORTED_DEVICES:
            raise RuntimeError(f"Thiết bị không được hỗ trợ: {device}, các thiết bị: {self.SUPPORTED_DEVICES}")
        self.device = device
        logger.info(f"Sử dụng thiết bị: {self.device}")

        # Chọn compute_type mặc định dựa trên thiết bị
        if compute_type is None:
            available_types = (
                self.CUDA_COMPUTE_TYPES if self.device == "cuda"
                else self.CPU_COMPUTE_TYPES
            )
            # Chọn kiểu khả dụng đầu tiên (đã sắp xếp theo tốc độ)
            compute_type = self._select_compute_type(available_types)

        self.compute_type = compute_type
        logger.info(f"Kiểu compute: {self.compute_type}")

        # Load mô hình faster-whisper
        self.model = WhisperModel(
            model_size_or_path=self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            download_root=None,  # Không dùng HF cache, load trực tiếp từ đường dẫn cục bộ
        )
        logger.info("Load mô hình hoàn tất")

    def _detect_device(self) -> str:
        """
        Tự nhận diện thiết bị suy luận khả dụng

        Thứ tự nhận diện:
            1. Kiểm tra CUDA/GPU khả dụng (qua biến môi trường hoặc torch)
            2. Fallback sang CPU

        Trả về:
            'cuda' hoặc 'cpu'
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Phát hiện CUDA GPU khả dụng")
                return "cuda"
        except ImportError:
            pass

        # Kiểm tra biến môi trường CUDA
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            logger.info("Phát hiện biến môi trường CUDA, sẽ thử dùng GPU")
            return "cuda"

        logger.info("Không phát hiện GPU khả dụng, dùng CPU")
        return "cpu"

    def _select_compute_type(self, available_types: List[str]) -> str:
        """
        Chọn kiểu compute type tối ưu từ danh sách khả dụng

        Logic:
            - Duyệt qua danh sách ưu tiên đã định nghĩa, chọn kiểu khả dụng đầu tiên
            - GPU: bfloat16 > float16 > int8_float16 > float32
            - CPU: int8 > float32

        Tham số:
            available_types: Danh sách compute_type khả dụng

        Trả về:
            Chuỗi compute_type đã chọn
        """
        priority_list = (
            self.CUDA_COMPUTE_TYPES if self.device == "cuda"
            else self.CPU_COMPUTE_TYPES
        )

        for compute_type in priority_list:
            if compute_type in available_types:
                logger.debug(f"Chọn compute_type: {compute_type}")
                return compute_type

        # Fallback về float32
        return "float32"

    def _download_default_model(self):
        """
        Tải mô hình mặc định (được gọi khi mô hình không tồn tại và download_if_missing=True)

        Phương thức này chỉ là shortcut, logic tải thực tế ủy quyền cho scripts/download_model.py
        """
        logger.warning("Sắp gọi huggingface_hub để tải mô hình...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="erax-ai/EraX-WoW-Turbo-V1.1-CT2",
            local_dir=self.model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        word_timestamps: bool = False,
        no_speech_threshold: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
    ) -> List[dict]:
        """
        Chuyển file âm thanh hoặc numpy array thành văn bản

        Tham số:
            audio_input: Đường dẫn file âm thanh HOẶC numpy array 1D float32 [-1, 1]
                       (file: hỗ trợ mọi định dạng, pydub xử lý thành 16kHz;
                       numpy array: truyền trực tiếp cho faster-whisper)
            language: Mã ngôn ngữ nguồn, ví dụ 'vi', 'zh', 'en'.
                     Mặc định None = tự động nhận diện
            task: Loại task, 'transcribe' hoặc 'translate'
            beam_size: Độ rộng beam search (khuyến nghị 3-5)
            vad_filter: Có bật VAD filter không (khuyến nghị bật)
            vad_parameters: Tham số VAD tùy chỉnh
            temperature: Nhiệt độ sampling, 0.0 = greedy decode
            initial_prompt: Prompt ban đầu cho model
            condition_on_previous_text: Tham chiếu văn bản đoạn trước
            word_timestamps: Có xuất timestamp cấp từ không

        Trả về:
            List[dict]: Danh sách các đoạn, mỗi dict có 'text', 'start', 'end', 'words'

        Ngoại lệ:
            FileNotFoundError: File âm thanh không tồn tại (chỉ khi input là string/Path)
            RuntimeError: Lỗi trong quá trình suy luận
        """
        # Phân biệt numpy array hay file path
        is_numpy = isinstance(audio_input, np.ndarray)
        if is_numpy:
            audio_arg = audio_input
            audio_label = f"numpy_array(shape={audio_input.shape})"
        else:
            audio_arg = str(audio_input)
            if not Path(audio_arg).exists():
                raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_arg}")
            audio_label = audio_arg

        logger.info(f"Bắt đầu nhận dạng: {audio_label}, ngôn ngữ: {language or 'tự động'}")

        if vad_filter and vad_parameters is None:
            vad_parameters = self.DEFAULT_VAD_PARAMETERS

        if initial_prompt is None:
            initial_prompt = "Xin chào! Việc này là bắt buộc: viết hoa chữ cái đầu câu, thêm đủ dấu phẩy, dấu chấm. Giữ nguyên thuật ngữ tiếng Anh (RACI, KPI, Microservices, BDU Assistant, STEAM, Dashboard), không phiên âm sang tiếng Việt."

        try:
            # Gọi faster-whisper để suy luận
            # faster-whisper hỗ trợ trực tiếp numpy array nên dùng audio_arg
            segments, info = self.model.transcribe(
                audio=audio_arg,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                temperature=temperature,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                word_timestamps=word_timestamps,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
            )

            # Thu thập kết quả nhận dạng
            results = []
            for seg in segments:
                segment_dict = {
                    "text": seg.text.strip(),
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                    "avg_logprob": getattr(seg, "avg_logprob", 0.0),
                }
                # Nếu bật timestamp cấp từ, thêm thông tin từ
                if word_timestamps and hasattr(seg, "words") and seg.words:
                    segment_dict["words"] = [
                        {
                            "word": w.word.lstrip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                        }
                        for w in seg.words
                    ]
                results.append(segment_dict)

            logger.info(f"Nhận dạng hoàn tất, tổng {len(results)} đoạn")
            return results

        except Exception as e:
            logger.error(f"Nhận dạng giọng nói thất bại: {e}")
            raise RuntimeError(f"Nhận dạng giọng nói thất bại: {e}")

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        language: Optional[str] = None,
        **kwargs
    ) -> List[List[dict]]:
        """
        Nhận dạng hàng loạt nhiều file âm thanh

        Tham số:
            audio_paths: Danh sách đường dẫn file âm thanh
            language: Mã ngôn ngữ nguồn (áp dụng cho tất cả file)
            **kwargs: Các tham số khác truyền cho transcribe()

        Trả về:
            List[List[dict]]: Danh sách kết quả nhận dạng của từng file
        """
        results = []
        for i, audio_path in enumerate(audio_paths):
            logger.info(f"Xử lý hàng loạt [{i+1}/{len(audio_paths)}]: {audio_path}")
            try:
                result = self.transcribe(audio_path, language=language, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý '{audio_path}': {e}")
                results.append([])  # Một file lỗi không ảnh hưởng các file khác
        return results

    def get_full_text(self, audio_path: Union[str, Path], language: Optional[str] = None) -> str:
        """
        Lấy toàn bộ văn bản phiên âm của âm thanh (không có timestamp)

        Tham số:
            audio_path: Đường dẫn file âm thanh
            language: Mã ngôn ngữ nguồn

        Trả về:
            Chuỗi văn bản hoàn chỉnh đã nối
        """
        segments = self.transcribe(audio_path, language=language)
        return " ".join(seg["text"] for seg in segments)

    def transcribe_stream(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        beam_size: int = 8,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.35,
        compression_ratio_threshold: float = 1.8,
        vad_parameters: Optional[dict] = None,
    ):
        """
        Stream từng segment nhận dạng giọng nói ngay khi faster-whisper xử lý xong.

        Khác với transcribe() vốn phải đợi toàn bộ file, phương thức này yield
        từng dict ngay khi mỗi segment hoàn thành — phù hợp để stream SSE cho
        file dài (4 phút đến 3 tiếng) mà không bị timeout.

        Yield các dict theo thứ tự:
            {"type": "info",    "total_duration": 176.3, "language": "vi"}
            {"type": "segment", "text": "...", "start": 0.5, "end": 2.1, "progress": 12}
            (không yield "done" — caller tự xử lý khi generator kết thúc)

        Tham số:
            audio_input: Đường dẫn file âm thanh hoặc numpy array 16kHz float32
            language: Mã ngôn ngữ ('vi', 'en', ...). None = tự động phát hiện
            beam_size: Độ rộng beam search (mặc định 8)
            condition_on_previous_text: False để chống hallucination cascading
            no_speech_threshold: Ngưỡng lọc đoạn không có tiếng nói
            compression_ratio_threshold: Ngưỡng phát hiện hallucination loop
        """
        is_numpy = isinstance(audio_input, np.ndarray)
        if is_numpy:
            audio_arg = audio_input
            audio_label = f"numpy_array(shape={audio_input.shape})"
            total_duration = len(audio_input) / 16000.0
        else:
            audio_arg = str(audio_input)
            if not Path(audio_arg).exists():
                raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_arg}")
            audio_label = audio_arg
            # Ước tính duration từ file size (WAV 16kHz mono 16-bit = 32000 bytes/s)
            try:
                file_size = Path(audio_arg).stat().st_size
                total_duration = file_size / 32000.0  # ước tính sơ bộ
            except Exception:
                total_duration = 0.0

        logger.info(f"[Stream] Bắt đầu transcribe_stream: {audio_label}, ngôn ngữ: {language or 'tự động'}")

        if vad_parameters is None:
            vad_parameters = self.DEFAULT_VAD_PARAMETERS

        try:
            segments_gen, info = self.model.transcribe(
                audio=audio_arg,
                language=language,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=True,
                vad_parameters=vad_parameters,
                temperature=[0.0, 0.2, 0.4, 0.6], # Fallback tự động khi độ tin cậy thấp
                initial_prompt="Xin chào! Việc này là bắt buộc: viết hoa chữ cái đầu câu, thêm đủ dấu phẩy, dấu chấm. Giữ nguyên thuật ngữ tiếng Anh (RACI, KPI, Microservices, BDU Assistant, STEAM, Dashboard), không phiên âm sang tiếng Việt.", # Mồi chuyên ngành và ngữ pháp
                condition_on_previous_text=condition_on_previous_text,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                word_timestamps=True, # Cần cho chức năng tạo SRT
            )

            # Dùng duration từ faster-whisper info (chính xác hơn ước tính)
            real_duration = getattr(info, "duration", None) or total_duration
            detected_language = getattr(info, "language", language or "vi")

            # Event info đầu tiên để client biết tổng duration và ngôn ngữ
            yield {
                "type": "info",
                "total_duration": round(real_duration, 2),
                "language": detected_language,
            }

            segment_count = 0
            filtered_count = 0

            # Blacklist các câu ảo giác kinh điển của Whisper tiếng Việt
            _hallucination_blacklist = [
                "cảm ơn các bạn", "đăng ký kênh", "ông cũng bị cáo buộc",
                "đài truyền hình", "bản tin", "xin chào các bạn",
                "hẹn gặp lại", "subtitles by", "tổng thống",
                "tác phẩm nghệ thuật", "những thông tin tiếp theo",
                "hàng trăm tình nguyện viên", "cảnh sát đã không tìm thấy",
                "chính quyền", "tàn phá", "đoạn trích truyện",
                "homestay và thư giãn", "cơ chế văn pháp",
                "tập trận", "dấu hiệu vụ việc",
            ]

            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue

                # === BỘ LỌC HALLUCINATION (tương tự stt_ws.py) ===

                # Filter 1: no_speech_prob cao → segment là tiếng ồn, không phải lời nói
                no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
                if no_speech_prob > 0.4:
                    filtered_count += 1
                    logger.info(
                        f"[Stream] Filter no_speech: prob={no_speech_prob:.3f}, "
                        f"text='{text[:60]}'"
                    )
                    continue

                # Filter 2: avg_logprob cực thấp → Whisper rất không tự tin
                avg_logprob = getattr(seg, "avg_logprob", 0.0)
                if avg_logprob < -0.8:
                    filtered_count += 1
                    logger.info(
                        f"[Stream] Filter low confidence: logprob={avg_logprob:.3f}, "
                        f"text='{text[:60]}'"
                    )
                    continue

                # Filter 3: Blacklist ảo giác
                text_lower = text.lower()
                if any(bad in text_lower for bad in _hallucination_blacklist):
                    filtered_count += 1
                    logger.warning(f"[Stream] Filter BLACKLIST: '{text[:80]}'")
                    continue

                # Filter 4: Tốc độ từ/giây bất thường (Whisper bịa text dài hơn audio)
                seg_duration = seg.end - seg.start
                if seg_duration > 0:
                    words_count = len(text.split())
                    words_per_sec = words_count / seg_duration
                    if words_per_sec > 8.0 and words_count > 6:
                        filtered_count += 1
                        logger.warning(
                            f"[Stream] Filter speed: {words_per_sec:.1f} words/s, "
                            f"text='{text[:60]}'"
                        )
                        continue

                # Filter 5: Lặp từ (hallucination loop)
                seg_words = text.split()
                if len(seg_words) > 5:
                    unique_ratio = len(set(seg_words)) / len(seg_words)
                    if unique_ratio < 0.4:
                        filtered_count += 1
                        logger.warning(
                            f"[Stream] Filter repeat ({unique_ratio:.0%} unique): "
                            f"'{text[:60]}'"
                        )
                        continue

                    # Kiểm tra cụm từ lặp liên tiếp
                    is_phrase_repeat = False
                    for i in range(len(seg_words) - 2):
                        phrase = " ".join(seg_words[i:i+3])
                        if text.count(phrase) > 3:
                            is_phrase_repeat = True
                            logger.warning(
                                f"[Stream] Filter phrase repeat '{phrase}': "
                                f"'{text[:60]}'"
                            )
                            break
                    if is_phrase_repeat:
                        filtered_count += 1
                        continue

                # Filter 6: Text quá ngắn (1-2 ký tự) — thường là hallucination
                if len(text) <= 2:
                    filtered_count += 1
                    continue

                # === SEGMENT HỢP LỆ → yield ===

                # Lấy words để làm phụ đề chính xác (WhisperX style)
                words = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        words.append({
                            "word": w.word.lstrip(),
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "probability": round(w.probability, 2)
                        })

                # Tính progress % dựa trên timestamp kết thúc của segment
                progress = 0
                if real_duration > 0:
                    progress = min(99, int(seg.end / real_duration * 100))

                segment_count += 1
                logger.debug(f"[Stream] Segment {segment_count}: [{seg.start:.1f}s->{seg.end:.1f}s] {text[:60]}")

                yield {
                    "type": "segment",
                    "id": segment_count,
                    "text": text,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "progress": progress,
                    "words": words,
                    "no_speech_prob": round(no_speech_prob, 3),
                    "avg_logprob": round(avg_logprob, 3),
                }

            logger.info(
                f"[Stream] Hoàn tất: {segment_count} segments yield, "
                f"{filtered_count} filtered, duration={real_duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"[Stream] Lỗi transcribe_stream: {e}")
            yield {"type": "error", "message": str(e)}

    def __repr__(self) -> str:
        return (
            f"STTEngine(model_path='{self.model_path}', "
            f"device='{self.device}', compute_type='{self.compute_type}')"
        )

    def __del__(self):
        """Giải phóng tài nguyên mô hình khi hủy (Python GC sẽ tự xử lý)"""
        if hasattr(self, "model"):
            del self.model
