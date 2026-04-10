#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_enhancer.py - Module chuẩn hóa và tăng cường âm thanh

Module này cung cấp các kỹ thuật tiền xử lý âm thanh để giảm hiện tượng
HALLUCINATION (kết quả ảo) trong STT - một vấn đề phổ biến của Whisper khi
xử lý âm thanh có nhiễu hoặc các đoạn im lặng/khoảng lặng dài.

Nguyên nhân chính của hallucination:
    1. Đoạn im lặng hoặc nhiễu nền cao → model "bịa đặt" nội dung
    2. Âm thanh có biên độ không đều → model nhầm lẫn
    3. Không lọc đoạn không phải giọng nói → phát sinh text vô nghĩa

Các kỹ thuật được áp dụng:
    1. Spectral Gating (giảm nhiễu phổ) - loại bỏ nhiễu nền
    2. Audio Normalization (chuẩn hóa) - đưa âm lượng về mức nhất quán
    3. Energy-based VAD (phát hiện giọng nói theo năng lượng) - lọc đoạn im lặng
    4. Speech Chunk Extraction - trích xuất và ghép đoạn có giọng nói
    5. Confidence-weighted filtering - lọc đoạn có độ tự tin thấp

Sử dụng trong pipeline:
    audio_enhancer.py (tiền xử lý) -> audio_utils.py (chuyển đổi 16k) -> stt_engine.py (nhận dạng)
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)


class AudioEnhancer:
    """
    Lớp chuẩn hóa và tăng cường âm thanh

    Lớp này kết hợp nhiều kỹ thuật xử lý âm thanh để cải thiện chất lượng
    đầu vào cho STT, giảm hallucination đáng kể.

    Các bước xử lý mặc định:
        1. Tính toán phổ nhiễu từ các đoạn im lặng (noise profile)
        2. Áp dụng spectral gating để giảm nhiễu
        3. Normalize biên độ về mức chuẩn
        4. Lọc đoạn năng lượng thấp (im lặng/khoảng lặng)
        5. Trích xuất các đoạn có giọng nói

    Thuộc tính:
        sample_rate: Tần số mẫu của âm thanh (mặc định 16000Hz)
        noise_threshold: Ngưỡng năng lượng để phân biệt tiếng nói/im lặng
        min_speech_duration: Thời lượng tối thiểu của đoạn có giọng nói (giây)

    Ví dụ sử dụng:
        >>> enhancer = AudioEnhancer(sample_rate=16000)
        >>> enhanced = enhancer.enhance(audio_data)  # numpy array 1D
        >>> # Hoặc xử lý từng bước:
        >>> enhancer.estimate_noise(audio_data)
        >>> denoised = enhancer.spectral_gate(audio_data)
        >>> normalized = enhancer.normalize(denoised)
        >>> speech_chunks = enhancer.extract_speech_chunks(normalized)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_threshold: float = 0.01,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 0.2,
        normalize_target_db: float = -20.0,
        apply_denoising: bool = True,
        apply_normalization: bool = True,
        apply_vad_filtering: bool = True,
    ):
        """
        Khởi tạo AudioEnhancer với các tham số điều chỉnh

        Tham số:
            sample_rate: Tần số mẫu của âm thanh đầu vào (Hz)
            noise_threshold: Ngưỡng năng lượng RMS để phát hiện im lặng.
                            Giá trị cao hơn → ít nhạy hơn (bỏ qua tiếng nhỏ).
                            Mặc định 0.01 phù hợp cho âm thanh đã normalize.
            min_speech_duration: Thời lượng tối thiểu (giây) để coi là đoạn có giọng nói.
                               Đoạn ngắn hơn sẽ bị loại bỏ. Mặc định 0.3s.
            min_silence_duration: Thời lượng tối thiểu (giây) để coi là hết một câu.
                                 Dùng để ghép các đoạn speech gần nhau. Mặc định 0.2s.
            normalize_target_db: Mức âm lượng mục tiêu khi normalize (dB).
                                -20dB là mức phát biểu thoải mái. Thấp hơn → to hơn.
            apply_denoising: Có áp dụng spectral gating giảm nhiễu không
            apply_normalization: Có normalize biên độ không
            apply_vad_filtering: Có lọc đoạn im lặng bằng năng lượng không
        """
        self.sample_rate = sample_rate
        self.noise_threshold = noise_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.normalize_target_db = normalize_target_db
        self.apply_denoising = apply_denoising
        self.apply_normalization = apply_normalization
        self.apply_vad_filtering = apply_vad_filtering

        # Noise profile sẽ được ước tính từ dữ liệu thực tế
        self._noise_profile: Optional[np.ndarray] = None

        # Tham số FFT cho spectral gating
        self._n_fft = 512
        self._hop_length = 256
        self._n_mels = 64

        logger.debug(
            f"Khởi tạo AudioEnhancer: sr={sample_rate}, noise_th={noise_threshold}, "
            f"min_speech={min_speech_duration}s, target_db={normalize_target_db}dB"
        )

    # ========== PHƯƠNG THỨC CHÍNH ==========

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Pipeline chuẩn hóa hoàn chỉnh: giảm nhiễu -> normalize -> VAD filter

        Phương thức này thực hiện tất cả các bước xử lý theo thứ tự tối ưu
        để tối đa hóa chất lượng đầu vào cho STT.

        Tham số:
            audio: Numpy array 1D float32 [-1, 1] hoặc PCM 16-bit, tần số 16kHz

        Trả về:
            Numpy array 1D float32 đã được xử lý hoàn chỉnh
        """
        # Bước 1: Đảm bảo định dạng chuẩn
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            logger.warning("Âm thanh rỗng, trả về mảng rỗng")
            return audio

        # Bước 2: Tự động ước tính noise profile từ các đoạn im lặng
        if self.apply_denoising and self._noise_profile is None:
            self.estimate_noise(audio)

        # Bước 3: Giảm nhiễu phổ (spectral gating)
        if self.apply_denoising and self._noise_profile is not None:
            audio = self._spectral_subtraction(audio)

        # Bước 4: Normalize biên độ
        if self.apply_normalization:
            audio = self._rms_normalize(audio)

        # Bước 5: Trích xuất đoạn có giọng nói (loại bỏ im lặng)
        if self.apply_vad_filtering:
            audio = self._filter_silence(audio)

        # Bước 6: Đảm bảo âm thanh vẫn hợp lệ sau xử lý
        if len(audio) == 0:
            logger.warning("Sau xử lý VAD, âm thanh bị loại bỏ hoàn toàn, giữ nguyên đầu vào")
            audio = self._prepare_audio(audio)

        return audio

    # ========== KỸ THUẬT 1: ƯỚC TÍNH NHIỄU ==========

    def estimate_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Ước tính noise profile từ các đoạn im lặng trong âm thanh

        Phương thức này tự động phát hiện các đoạn im lặng trong âm thanh,
        tính toán phổ nhiễu trung bình từ đó. Noise profile dùng cho
        spectral gating để loại bỏ nhiễu nền một cách thích ứng.

        Thuật toán:
            1. Tính RMS theo từng window
            2. Xác định ngưỡng dựa trên percentile thấp (5-25%)
            3. Chọn các window có RMS < ngưỡng làm đoạn im lặng
            4. Tính trung bình phổ của các đoạn này

        Tham số:
            audio: Numpy array 1D float32, tần số 16kHz

        Trả về:
            Noise profile dưới dạng phổ tần số (magnitude spectrogram)
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            logger.warning("Âm thanh rỗng, không thể ước tính nhiễu")
            return np.zeros(self._n_fft // 2 + 1, dtype=np.float32)

        # Tính STFT
        stft = self._compute_stft(audio)
        magnitude = np.abs(stft)

        # Tính năng lượng trung bình theo thời gian
        time_energy = np.mean(magnitude, axis=0)

        # Xác định ngưỡng nhiễu dựa trên percentile thấp
        # Các đoạn có năng lượng thấp nhất được coi là im lặng
        noise_percentile = np.percentile(time_energy, 15)
        threshold = noise_percentile * 1.5

        # Tìm các frame có năng lượng thấp (im lặng)
        noise_frames = magnitude[:, time_energy < threshold]

        if noise_frames.shape[1] < 10:
            # Không đủ frame im lặng → dùng toàn bộ âm thanh ước tính
            logger.debug("Không đủ frame im lặng, dùng toàn bộ âm thanh làm noise profile")
            self._noise_profile = np.mean(magnitude, axis=1).astype(np.float32)
        else:
            # Tính noise profile trung bình
            self._noise_profile = np.mean(noise_frames, axis=1).astype(np.float32)

        logger.debug(
            f"Ước tính noise profile từ {noise_frames.shape[1]} frame, "
            f"ngưỡng={threshold:.4f}"
        )
        return self._noise_profile

    def set_noise_profile(self, noise_profile: np.ndarray):
        """
        Đặt noise profile thủ công (cho phép dùng profile đã tính sẵn)

        Tham số:
            noise_profile: Numpy array 1D float32, độ dài = n_fft // 2 + 1
        """
        self._noise_profile = np.array(noise_profile, dtype=np.float32)
        logger.debug(f"Đặt noise profile thủ công, độ dài={len(self._noise_profile)}")

    # ========== KỸ THUẬT 2: SPECTRAL GATING (GIẢM NHIỄU) ==========

    def spectral_gate(self, audio: np.ndarray, noise_reduce_factor: float = 1.5) -> np.ndarray:
        """
        Áp dụng spectral gating để giảm nhiễu nền

        Spectral gating là kỹ thuật trừ phổ đơn giản nhưng hiệu quả:
            - Tính STFT của tín hiệu
            - So sánh từng bin tần số với noise profile
            - Giảm các bin có năng lượng không đáng kể so với nhiễu

        Tham số:
            audio: Numpy array 1D float32 [-1, 1], tần số 16kHz
            noise_reduce_factor: Hệ số giảm nhiễu. Lớn hơn → giảm nhiễu nhiều hơn
                                 nhưng có thể làm méo tiếng nói. Khuyến nghị 1.0-2.0

        Trả về:
            Âm thanh đã giảm nhiễu, cùng độ dài với đầu vào
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            return audio

        if self._noise_profile is None:
            logger.warning("Chưa có noise profile, tự động ước tính từ âm thanh")
            self.estimate_noise(audio)

        return self._spectral_subtraction(audio, noise_reduce_factor)

    def _spectral_subtraction(
        self, audio: np.ndarray, reduce_factor: float = 1.5
    ) -> np.ndarray:
        """
        Triển khai thuật toán spectral subtraction

        Công thức:
            |X_filtered(f)| = max(|X(f)| - reduce_factor * |N(f)|, floor)
            X_filtered = |X_filtered(f)| * exp(j * angle(X(f)))

        Tham số:
            audio: Âm thanh đầu vào
            reduce_factor: Hệ số giảm (multiplier cho noise floor)

        Trả về:
            Âm thanh đã trừ nhiễu
        """
        stft = self._compute_stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Tính noise floor: noise_profile * reduce_factor
        noise_floor = self._noise_profile * reduce_factor

        # Trừ nhiễu: chỉ giữ lại phần vượt trên noise floor
        # Sử dụng flooring để tránh âm thanh bị giảm quá mức
        magnitude_subtracted = np.maximum(magnitude - noise_floor[:, np.newaxis], 0.1 * magnitude)

        # Tái tạo tín hiệu từ phổ đã xử lý
        stft_filtered = magnitude_subtracted * np.exp(1j * phase)
        audio_filtered = self._inverse_stft(stft_filtered)

        # Trim về độ dài ban đầu (inverse STFT có thể có padding)
        if len(audio_filtered) > len(audio):
            audio_filtered = audio_filtered[: len(audio)]

        return audio_filtered.astype(np.float32)

    # ========== KỸ THUẬT 3: NORMALIZE BIÊN ĐỘ ==========

    def normalize(self, audio: np.ndarray, target_db: Optional[float] = None) -> np.ndarray:
        """
        Chuẩn hóa âm lượng về mức mục tiêu (dB)

        Phương thức này điều chỉnh biên độ âm thanh sao cho RMS đạt mức
        tương ứng với target_db, đảm bảo âm lượng nhất quán giữa các file.

        Công thức:
            target_rms = 10^(target_db / 20)
            scale = target_rms / current_rms
            normalized = audio * scale

        Tham số:
            audio: Numpy array 1D float32, tần số 16kHz
            target_db: Mức mục tiêu (dB). Mặc định dùng self.normalize_target_db

        Trả về:
            Âm thanh đã normalize về [-1, 1] và RMS tương ứng target_db
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            return audio

        if target_db is None:
            target_db = self.normalize_target_db

        # Tính RMS hiện tại
        current_rms = np.sqrt(np.mean(audio ** 2))

        # Tránh chia cho 0
        if current_rms < 1e-8:
            logger.debug("Âm thanh gần như im lặng, bỏ qua normalize")
            return audio

        # Tính target RMS từ dB
        target_rms = 10 ** (target_db / 20)

        # Tính hệ số scale
        scale = target_rms / current_rms

        # Áp dụng scale và clip về [-1, 1]
        normalized = audio * scale
        normalized = np.clip(normalized, -1.0, 1.0)

        logger.debug(
            f"Normalize: {target_db}dB, RMS trước={current_rms:.4f}, "
            f"sau={np.sqrt(np.mean(normalized**2)):.4f}"
        )
        return normalized

    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Alias cho normalize(), RMS-based normalization
        """
        return self.normalize(audio)

    def peak_normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Chuẩn hóa theo đỉnh (peak normalization) thay vì RMS

        Khác với RMS normalize: peak normalize điều chỉnh sao cho giá trị
        đỉnh (max absolute) đạt mức tương ứng. Phù hợp khi cần giữ nguyên
        dynamic range nhưng đảm bảo không bị clipping.

        Tham số:
            audio: Numpy array 1D float32
            target_db: Mức đỉnh mục tiêu (dBFS). -3dB là an toàn, tránh clipping

        Trả về:
            Âm thanh đã peak-normalize
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            return audio

        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio

        target_peak = 10 ** (target_db / 20)
        scale = target_peak / peak
        normalized = audio * scale

        return np.clip(normalized, -1.0, 1.0)

    # ========== KỸ THUẬT 4: VAD DỰA TRÊN NĂNG LƯỢNG ==========

    def extract_speech_chunks(
        self,
        audio: np.ndarray,
        min_duration: Optional[float] = None,
        merge_gap: Optional[float] = None,
    ) -> List[dict]:
        """
        Trích xuất các đoạn có giọng nói từ âm thanh

        Phương thức này chia âm thanh thành các đoạn, đánh dấu đoạn nào
        có giọng nói (speech) và đoạn nào là im lặng (silence). Trả về
        danh sách dict chứa thông tin timestamp của từng đoạn.

        Thuật toán:
            1. Tính RMS theo từng window 30ms
            2. So sánh với noise_threshold để phân loại speech/silence
            3. Gộp các đoạn speech gần nhau (khoảng cách < merge_gap)
            4. Loại bỏ đoạn speech quá ngắn

        Tham số:
            audio: Numpy array 1D float32, tần số 16kHz
            min_duration: Thời gian tối thiểu để coi là speech (giây)
            merge_gap: Khoảng cách tối đa để gộp 2 đoạn speech gần nhau (giây)

        Trả về:
            List[dict], mỗi dict có keys: 'text' (placeholder), 'start', 'end', 'is_speech'
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            return []

        if min_duration is None:
            min_duration = self.min_speech_duration
        if merge_gap is None:
            merge_gap = self.min_silence_duration

        # Cửa sổ 30ms để tính năng lượng
        window_size = int(0.03 * self.sample_rate)  # 30ms
        hop_size = window_size // 2  # 50% overlap

        # Tính năng lượng RMS theo từng window
        num_windows = (len(audio) - window_size) // hop_size + 1
        if num_windows <= 0:
            return []

        rms_values = []
        for i in range(num_windows):
            start = i * hop_size
            end = start + window_size
            chunk = audio[start:end]
            rms = np.sqrt(np.mean(chunk ** 2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)

        # Phân loại speech/silence dựa trên threshold
        is_speech = rms_values > self.noise_threshold

        # Chuyển index window -> timestamp
        def idx_to_time(idx: int) -> float:
            return (idx * hop_size) / self.sample_rate

        # Tìm các đoạn speech liên tục
        chunks: List[dict] = []
        in_speech = False
        chunk_start = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                # Bắt đầu đoạn speech
                in_speech = True
                chunk_start = i
            elif not speech and in_speech:
                # Kết thúc đoạn speech
                in_speech = False
                chunks.append({
                    "start": idx_to_time(chunk_start),
                    "end": idx_to_time(i),
                    "is_speech": True,
                })

        # Xử lý đoạn speech cuối cùng nếu kéo dài đến cuối
        if in_speech:
            chunks.append({
                "start": idx_to_time(chunk_start),
                "end": idx_to_time(len(is_speech)),
                "is_speech": True,
            })

        # Gộp các đoạn speech gần nhau
        if len(chunks) > 1:
            merged: List[dict] = [chunks[0]]
            for chunk in chunks[1:]:
                if chunk["start"] - merged[-1]["end"] <= merge_gap:
                    # Gộp với đoạn trước
                    merged[-1]["end"] = chunk["end"]
                else:
                    merged.append(chunk)
            chunks = merged

        # Lọc đoạn quá ngắn
        chunks = [c for c in chunks if c["end"] - c["start"] >= min_duration]

        logger.debug(f"Trích xuất {len(chunks)} đoạn speech, tổng thời lượng={sum(c['end']-c['start'] for c in chunks):.2f}s")
        return chunks

    def _filter_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Lọc bỏ các đoạn im lặng, chỉ giữ lại đoạn có giọng nói

        Tham số:
            audio: Numpy array 1D float32

        Trả về:
            Âm thanh chỉ chứa các đoạn có giọng nói (đã nối)
        """
        chunks = self.extract_speech_chunks(audio)

        if not chunks:
            logger.warning("Không tìm thấy đoạn speech nào, giữ nguyên âm thanh gốc")
            return audio

        # Ghép các đoạn speech lại
        speech_segments = []
        for chunk in chunks:
            start_sample = int(chunk["start"] * self.sample_rate)
            end_sample = int(chunk["end"] * self.sample_rate)
            # Thêm buffer 0.05s (50ms) ở đầu để tránh cắt đầu câu
            start_sample = max(0, start_sample - int(0.05 * self.sample_rate))
            # Thêm buffer ở cuối
            end_sample = min(len(audio), end_sample + int(0.05 * self.sample_rate))
            speech_segments.append(audio[start_sample:end_sample])

        if not speech_segments:
            return audio

        # Nối các đoạn speech với khoảng lặng ngắn 0.1s ở giữa
        result = []
        for i, seg in enumerate(speech_segments):
            if i > 0:
                # Thêm 0.1s im lặng giữa các đoạn
                silence = np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)
                result.append(silence)
            result.append(seg)

        return np.concatenate(result) if result else audio

    # ========== KỸ THUẬT 5: TRÍCH XUẤT ĐOẠN CÓ CHẤT LƯỢNG TỐT ==========

    def get_speech_quality_score(self, audio: np.ndarray) -> float:
        """
        Tính điểm chất lượng giọng nói (0.0 - 1.0)

        Điểm chất lượng được tính dựa trên:
            1. Tỷ lệ speech/silence (cao hơn → tốt hơn)
            2. SNR ước tính (signal-to-noise ratio)
            3. Độ ổn định của biên độ

        Tham số:
            audio: Numpy array 1D float32

        Trả về:
            Điểm chất lượng trong khoảng [0.0, 1.0]
        """
        audio = self._prepare_audio(audio)

        if len(audio) == 0:
            return 0.0

        # Tính RMS overall
        overall_rms = np.sqrt(np.mean(audio ** 2))

        # Tính RMS trung bình của các window
        window_size = int(0.03 * self.sample_rate)
        hop_size = window_size // 2
        num_windows = (len(audio) - window_size) // hop_size

        if num_windows <= 0:
            return 0.0

        rms_per_window = []
        for i in range(num_windows):
            start = i * hop_size
            chunk = audio[start:start + window_size]
            rms_per_window.append(np.sqrt(np.mean(chunk ** 2)))

        rms_per_window = np.array(rms_per_window)

        # Thành phần 1: Tỷ lệ speech (window có RMS > threshold)
        speech_ratio = np.mean(rms_per_window > self.noise_threshold)

        # Thành phần 2: SNR ước tính
        # SNR = mean(speech_rms) / mean(noise_rms)
        speech_rms = rms_per_window[rms_per_window > self.noise_threshold]
        noise_rms = rms_per_window[rms_per_window <= self.noise_threshold]

        if len(speech_rms) > 0 and len(noise_rms) > 0:
            snr = np.mean(speech_rms) / (np.mean(noise_rms) + 1e-8)
            snr_score = min(snr / 10.0, 1.0)  # Normalize: SNR=10 → score=1.0
        else:
            snr_score = 0.5

        # Thành phần 3: Độ ổn định (coefficient of variation)
        if overall_rms > 1e-8:
            cv = np.std(rms_per_window) / (np.mean(rms_per_window) + 1e-8)
            stability_score = max(0, 1.0 - cv)
        else:
            stability_score = 0.0

        # Tổng hợp điểm (trọng số)
        quality_score = (
            speech_ratio * 0.3 +
            snr_score * 0.4 +
            stability_score * 0.3
        )

        logger.debug(
            f"Quality score: {quality_score:.3f} "
            f"(speech_ratio={speech_ratio:.2f}, snr_score={snr_score:.2f}, "
            f"stability={stability_score:.2f})"
        )
        return float(quality_score)

    # ========== CÁC HÀM TIỆN ÍCH ==========

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Chuẩn bị âm thanh: flatten, type conversion, normalize về [-1, 1]
        """
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        audio = audio.flatten().astype(np.float32)

        # Normalize PCM 16-bit về [-1, 1]
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        return audio

    def _compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """
        Tính Short-Time Fourier Transform (STFT)

        Tham số:
            audio: Numpy array 1D float32

        Trả về:
            STFT matrix, shape = (n_fft//2+1, num_frames)
        """
        n = len(audio)
        num_frames = 1 + (n - self._n_fft) // self._hop_length

        # Window function (Hann window)
        window = np.hanning(self._n_fft).astype(np.float32)

        stft = np.zeros((self._n_fft // 2 + 1, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * self._hop_length
            end = start + self._n_fft
            if end > n:
                break
            frame = audio[start:end] * window
            stft[:, i] = np.fft.rfft(frame)

        return stft

    def _inverse_stft(self, stft: np.ndarray) -> np.ndarray:
        """
        Tính Inverse STFT để tái tạo tín hiệu âm thanh

        Tham số:
            stft: STFT matrix

        Trả về:
            Audio signal 1D
        """
        n_frames = stft.shape[1]
        n_fft = (stft.shape[0] - 1) * 2

        window = np.hanning(n_fft).astype(np.float32)
        signal_length = n_fft + (n_frames - 1) * self._hop_length
        signal = np.zeros(signal_length, dtype=np.float32)
        window_sum = np.zeros(signal_length, dtype=np.float32)

        for i in range(n_frames):
            start = i * self._hop_length
            frame = np.fft.irfft(stft[:, i], n=n_fft)
            # Áp dụng window và cộng dồn (overlap-add)
            signal[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2

        # Normalize bằng window sum (tránh divide by zero)
        window_sum[window_sum == 0] = 1.0
        signal = signal / window_sum

        return signal

    def reset(self):
        """
        Reset noise profile, chuẩn bị cho âm thanh mới

        Gọi phương thức này trước khi xử lý một âm thanh mới nếu muốn
        noise profile được ước tính lại từ đầu.
        """
        self._noise_profile = None
        logger.debug("Đã reset noise profile")

    def __repr__(self) -> str:
        return (
            f"AudioEnhancer(sr={self.sample_rate}, noise_th={self.noise_threshold}, "
            f"min_speech={self.min_speech_duration}s, "
            f"normalize_db={self.normalize_target_db}dB)"
        )


# ========== HÀM TIỆN ÍCH CẤP MODULE ==========

def enhance_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    target_db: float = -20.0,
    apply_denoising: bool = True,
) -> str:
    """
    Hàm tiện ích: chuẩn hóa một file âm thanh và lưu ra file mới

    Tham số:
        input_path: Đường dẫn file âm thanh đầu vào
        output_path: Đường dẫn file đầu ra. Mặc định: thêm "_enhanced" vào tên
        sample_rate: Tần số mẫu
        target_db: Mức normalize (dB)
        apply_denoising: Có giảm nhiễu không

    Trả về:
        Đường dẫn file đã xử lý
    """
    from pydub import AudioSegment

    # Load âm thanh
    audio = AudioSegment.from_file(input_path)

    # Chuyển về 16kHz mono nếu cần
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    if audio.channels > 1:
        audio = audio.split_to_mono()[0]

    # Chuyển sang numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.sample_width == 2:
        samples = samples / 32768.0

    # Xử lý với AudioEnhancer
    enhancer = AudioEnhancer(
        sample_rate=sample_rate,
        normalize_target_db=target_db,
        apply_denoising=apply_denoising,
    )
    enhanced = enhancer.enhance(samples)

    # Chuyển lại sang AudioSegment
    enhanced_samples = (enhanced * 32767).astype(np.int16)
    enhanced_audio = AudioSegment(
        data=enhanced_samples.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )

    # Xác định đường dẫn output
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_enhanced{p.suffix}")

    # Xuất file
    enhanced_audio.export(output_path, format="wav")
    logger.info(f"Đã lưu âm thanh chuẩn hóa: {output_path}")

    return output_path
