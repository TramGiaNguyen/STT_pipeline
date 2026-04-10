#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vad_processor.py - Module xử lý Voice Activity Detection (VAD)

Bọc (wrap) mô hình Silero VAD, cung cấp chức năng phát hiện hoạt động giọng nói
trong luồng âm thanh real-time. Silero VAD nhận đầu vào là numpy array 16kHz mono
và trả về xác suất có lời nói hay không.

VAD (Voice Activity Detection) được dùng trong kiến trúc hybrid:
    - WebRTC VAD (phía client/JavaScript): phát hiện nhanh, low latency
    - Silero VAD (phía server/Python): xác nhận lại, độ chính xác cao hơn

Các chức năng chính:
    - Load mô hình Silero VAD (singleton, load 1 lần khi server start)
    - Phát hiện lời nói trong audio chunk (numpy array)
    - Hỗ trợ configurable threshold để điều chỉnh độ nhạy
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Lớp bọc Silero VAD cho phát hiện hoạt động giọng nói

    Silero VAD là mô hình phát hiện lời nói của nhóm Silero, hoạt động tốt
    trong nhiều điều kiện môi trường khác nhau. Mô hình nhận âm thanh 16kHz mono
    dạng numpy float32 array và trả về xác suất có lời nói.

    Thuộc tính:
        model: Instance mô hình Silero VAD
        device: Thiết bị suy luận ('cuda' hoặc 'cpu')
        threshold: Ngưỡng xác suất để quyết định có lời nói (mặc định 0.5)

    Ví dụ sử dụng:
        >>> vad = SileroVAD()
        >>> is_speech = vad.is_speech(audio_chunk)  # audio_chunk: numpy array 16kHz
        >>> print("Có lời nói" if is_speech else "Im lặng")
    """

    # Tần số mẫu mà Silero VAD yêu cầu
    REQUIRED_SAMPLE_RATE = 16000

    def __init__(
        self,
        threshold: float = 0.5,
        device: Optional[str] = None,
        models_dir: Optional[str] = None,
    ):
        """
        Khởi tạo Silero VAD và load mô hình

        Tham số:
            threshold: Ngưỡng xác suất [0.0, 1.0]. Chunk có xác suất >= threshold
                       được coi là có lời nói. Mặc định 0.5. Giá trị cao hơn
                       → ít nhạy hơn (bỏ qua tiếng nói nhỏ), thấp hơn → nhạy hơn.
            device: Thiết bị suy luận, 'cuda' (GPU) hoặc 'cpu'. Mặc định tự nhận diện.
            models_dir: Thư mục cache mô hình. Mặc định None dùng torch cache mặc định.
        """
        self.threshold = threshold

        # Tự nhận diện thiết bị: ưu tiên CUDA
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Khởi tạo SileroVAD, thiết bị: {self.device}, threshold: {threshold}")

        # Load mô hình Silero VAD từ torch.hub
        self.model = self._load_model(models_dir)
        self.model.eval()  # Chuyển sang chế độ inference

        # Các tham số VAD nâng cao (dùng mặc định từ Silero)
        self._vad_params = {
            "threshold": threshold,
            "min_speech_duration_ms": 250,    # Clip ngắn hơn 250ms không coi là lời nói
            "min_silence_duration_ms": 200,  # Tối thiểu 200ms im lặng mới coi là hết câu
            "speech_pad_ms": 200,            # Padding thêm 200ms trước/sau đoạn speech
        }

        logger.info("Load Silero VAD hoàn tất")

    def _load_model(self, models_dir: Optional[str]) -> torch.nn.Module:
        """
        Load mô hình Silero VAD từ torch.hub

        Tham số:
            models_dir: Thư mục lưu cache mô hình

        Trả về:
            Mô hình Silero VAD đã load
        """
        try:
            # Silero cung cấp nhiều phiên bản VAD, dùng bản multiscale
            # tối ưu cho real-time streaming
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                trust_repo=True,
            )
            logger.debug("Tải Silero VAD từ torch.hub thành công")
            return model

        except Exception as e:
            logger.error(f"Không thể load Silero VAD: {e}")
            raise RuntimeError(f"Không thể load Silero VAD: {e}")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Kiểm tra chunk âm thanh có chứa lời nói hay không

        Phương thức này nhận một chunk âm thanh ngắn (thường 16-64ms),
        chuyển thành tensor và chạy qua Silero VAD. Kết quả là xác suất
        có lời nói; nếu >= threshold → True, ngược lại → False.

        Tham số:
            audio_chunk: Numpy array 1D dạng float32, giá trị trong khoảng [-1, 1]
                        hoặc [-32768, 32767] (PCM 16-bit). Tần số mẫu phải là 16kHz.
                        Độ dài chunk tối thiểu 256 samples (16ms).

        Trả về:
            True nếu chunk chứa lời nói (xác suất >= threshold), False nếu im lặng

        Lưu ý:
            - Chunk ngắn hơn 256 samples có thể cho kết quả không chính xác
            - Đảm bảo audio_chunk đã được normalize về float32 [-1, 1] trước khi gọi
        """
        # Bước 1: Đảm bảo audio_chunk là numpy array float32
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        # Bước 2: Flatten về 1D (phòng trường hợp input là 2D)
        audio_chunk = audio_chunk.flatten()

        # Bước 3: Normalize về [-1, 1] nếu là PCM 16-bit (giá trị lớn)
        # Kiểm tra nếu giá trị trung bình tuyệt đối > 1 → coi là PCM 16-bit
        if audio_chunk.size > 0:
            abs_mean = np.abs(audio_chunk).mean()
            if abs_mean > 1.0:
                audio_chunk = audio_chunk / 32768.0

        # Bước 4: Chuyển sang torch tensor, reshape thành (1, N)
        audio_tensor = torch.from_numpy(audio_chunk).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, N)

        # Bước 5: Chạy VAD inference
        with torch.no_grad():
            # Model Silero VAD yêu cầu tham số là 'sr' (không phải 'sampling_rate')
            speech_prob = self.model(audio_tensor, sr=self.REQUIRED_SAMPLE_RATE)

            # Xử lý output: có thể là scalar hoặc tensor
            if isinstance(speech_prob, torch.Tensor):
                speech_prob = speech_prob.item()

            # Quyết định có lời nói dựa trên threshold
            return speech_prob >= self.threshold

    def get_speech_probabilities(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Lấy xác suất lời nói cho từng frame (hỗ trợ chunk dài)

        Silero VAD chỉ nhận input 512 samples @ 16kHz. Phương thức này
        tự động chia chunk thành các frame 512 samples, chạy VAD cho từng
        frame, trả về mảng xác suất tương ứng.

        Tham số:
            audio_chunk: Numpy array 1D float32 [-1, 1] hoặc PCM 16-bit,
                        độ dài bất kỳ, tần số 16kHz. Độ dài phải >= 512 samples.

        Trả về:
            Numpy array 1D float32, mỗi phần tử là xác suất [0.0, 1.0]
            cho frame tương ứng. Số lượng frame = len(audio) / 512.
        """
        # Bước 1: Đảm bảo audio_chunk là numpy array float32
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        # Bước 2: Flatten về 1D
        audio_chunk = audio_chunk.flatten()

        # Bước 3: Normalize về [-1, 1] nếu là PCM 16-bit
        if audio_chunk.size > 0:
            abs_mean = np.abs(audio_chunk).mean()
            if abs_mean > 1.0:
                audio_chunk = audio_chunk / 32768.0

        # Bước 4: Chia thành các frame 512 samples
        frame_size = 512
        num_frames = len(audio_chunk) // frame_size
        
        # CRITICAL FIX: Nếu chunk < 512 samples, pad với zeros
        if num_frames == 0:
            if len(audio_chunk) > 0:
                # Pad chunk ngắn lên 512 samples
                padded = np.zeros(frame_size, dtype=np.float32)
                padded[:len(audio_chunk)] = audio_chunk
                audio_tensor = torch.from_numpy(padded).float().unsqueeze(0)  # (1, 512)
                
                with torch.no_grad():
                    prob = self.model(audio_tensor, sr=self.REQUIRED_SAMPLE_RATE)
                    if isinstance(prob, torch.Tensor):
                        prob = prob.cpu().numpy().flatten()
                    else:
                        prob = np.array([prob]).flatten()
                    return prob.astype(np.float32)
            else:
                return np.array([], dtype=np.float32)

        # Cắt bớt phần dư để chia chẵn
        audio_trimmed = audio_chunk[: num_frames * frame_size]

        # Bước 5: Reshape thành (num_frames, 512) rồi chạy batch VAD
        audio_tensor = torch.from_numpy(audio_trimmed).float().view(num_frames, frame_size)

        # Bước 6: Batch inference VAD
        with torch.no_grad():
            probs = self.model(audio_tensor, sr=self.REQUIRED_SAMPLE_RATE)

            # Output shape: (num_frames, 1) → flatten về 1D
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy().flatten()
            else:
                probs = np.array(probs).flatten()

            return probs.astype(np.float32)

    def is_speech_from_probs(self, audio_chunk: np.ndarray) -> bool:
        """
        Kiểm tra có lời nói hay không dựa trên probabilities (hỗ trợ chunk dài)

        Dùng max_prob >= threshold thay vì np.any() để tránh false positive
        khi có burst noise ngắn trong 1 frame duy nhất. Kết quả nhất quán
        với cách tính is_speech trong stt_ws.py (_handle_audio).

        Tham số:
            audio_chunk: Numpy array 1D float32, độ dài bất kỳ, 16kHz

        Trả về:
            True nếu xác suất cao nhất trong chunk >= threshold
        """
        probs = self.get_speech_probabilities(audio_chunk)
        if probs.size == 0:
            return False
        max_prob = float(probs.max())
        return max_prob >= self.threshold

    def get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Lấy xác suất có lời nói của chunk (không so sánh threshold)

        Khác với is_speech(), phương thức này trả về xác suất thô,
        hữu ích cho việc debug, logging hoặc tự điều chỉnh threshold động.

        Tham số:
            audio_chunk: Numpy array 1D float32, tần số 16kHz

        Trả về:
            Xác suất có lời nói trong khoảng [0.0, 1.0]
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)

        audio_chunk = audio_chunk.flatten()

        if audio_chunk.size > 0:
            abs_mean = np.abs(audio_chunk).mean()
            if abs_mean > 1.0:
                audio_chunk = audio_chunk / 32768.0

        audio_tensor = torch.from_numpy(audio_chunk).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            # Model Silero VAD yêu cầu tham số là 'sr' (không phải 'sampling_rate')
            prob = self.model(audio_tensor, sr=self.REQUIRED_SAMPLE_RATE)
            if isinstance(prob, torch.Tensor):
                prob = prob.item()
            return float(prob)

    def set_threshold(self, threshold: float):
        """
        Thay đổi ngưỡng phát hiện lời nói

        Tham số:
            threshold: Ngưỡng mới trong khoảng [0.0, 1.0]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold phải trong khoảng [0.0, 1.0], nhận được: {threshold}")
        self.threshold = threshold
        logger.debug(f"Cập nhật VAD threshold: {threshold}")

    def __repr__(self) -> str:
        return f"SileroVAD(device='{self.device}', threshold={self.threshold})"

    def __del__(self):
        """Giải phóng tài nguyên mô hình khi hủy"""
        if hasattr(self, "model"):
            del self.model
