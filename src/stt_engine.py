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
                }
                # Nếu bật timestamp cấp từ, thêm thông tin từ
                if word_timestamps and hasattr(seg, "words") and seg.words:
                    segment_dict["words"] = [
                        {
                            "word": w.word.strip(),
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

    def __repr__(self) -> str:
        return (
            f"STTEngine(model_path='{self.model_path}', "
            f"device='{self.device}', compute_type='{self.compute_type}')"
        )

    def __del__(self):
        """Giải phóng tài nguyên mô hình khi hủy (Python GC sẽ tự xử lý)"""
        if hasattr(self, "model"):
            del self.model
