#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_utils.py - Module tiền xử lý âm thanh

Cung cấp các chức năng tiền xử lý âm thanh như chuyển đổi định dạng, điều chỉnh
tần số mẫu, xử lý kênh âm thanh, chuẩn hóa và giảm nhiễu, đảm bảo file âm thanh
phù hợp với yêu cầu đầu vào của mô hình faster-whisper.

Định dạng đầu vào được faster-whisper khuyến nghị:
    - Tần số mẫu: 16kHz
    - Kênh: Đơn âm (Mono)
    - Mã hóa: PCM 16-bit (định dạng wav)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from src.audio_enhancer import AudioEnhancer, enhance_audio_file

# Cấu hình logger cho module
logger = logging.getLogger(__name__)


def convert_to_16k_mono(
    audio_path: str,
    output_path: Optional[str] = None,
    bit_depth: int = 16
) -> str:
    """
    Chuyển đổi file âm thanh bất kỳ sang định dạng 16kHz đơn âm PCM

    Các bước chuyển đổi:
        1. Tự động nhận diện định dạng đầu vào và giải mã
        2. Trích xuất kênh trái (nếu là âm thanh nổi)
        3. Resample về 16000Hz
        4. Chuyển đổi sang mã hóa 16-bit PCM
        5. Xuất ra định dạng WAV

    Tham số:
        audio_path: Đường dẫn file âm thanh đầu vào, hỗ trợ mp3/wav/ogg/flac/m4a
        output_path: Đường dẫn file đầu ra (nếu không có extension sẽ tự thêm .wav)
                    Mặc định là None, sẽ tạo file temp_16k.wav trong cùng thư mục
        bit_depth: Độ sâu bit đầu ra, mặc định 16-bit (giá trị khuyến nghị của faster-whisper)

    Trả về:
        Chuỗi đường dẫn tuyệt đối của file âm thanh đã chuyển đổi

    Ngoại lệ:
        FileNotFoundError: File đầu vào không tồn tại
        CouldntDecodeError: pydub không thể giải mã định dạng âm thanh này
        ValueError: Tham số tần số mẫu hoặc độ sâu bit không hợp lệ
    """
    # Kiểm tra tham số: đảm bảo file đầu vào tồn tại
    input_path = Path(audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_path}")

    # Xác định đường dẫn đầu ra: mặc định tạo file có timestamp trong cùng thư mục
    if output_path is None:
        output_path = str(input_path.parent / "temp_16k.wav")
    else:
        output_path = str(output_path)
        # Nếu đường dẫn được chỉ định nhưng không có extension, tự động thêm .wav
        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Bắt đầu chuyển đổi âm thanh: {audio_path} -> {output_path}")

    try:
        # Bước 1: Load âm thanh (pydub sẽ tự chọn decoder dựa trên extension)
        audio = AudioSegment.from_file(str(input_path))
        original_sr = audio.frame_rate
        original_channels = audio.channels
        original_duration = len(audio) / 1000.0  # pydub trả về mili-giây

        logger.debug(f"Thông tin âm thanh gốc - Tần số mẫu: {original_sr}Hz, "
                     f"Số kênh: {original_channels}, "
                     f"Thời lượng: {original_duration:.2f}s")

        # Bước 2: Chuyển đổi sang đơn âm
        if audio.channels > 1:
            # split_to_mono() trả về list [kênh trái, kênh phải], lấy kênh đầu tiên
            audio = audio.split_to_mono()[0]
            logger.debug("Đã chuyển sang đơn âm")
        elif audio.channels == 1:
            logger.debug("Đầu vào đã là đơn âm, bỏ qua bước chuyển kênh")
        else:
            raise ValueError(f"Số kênh không được hỗ trợ: {audio.channels}")

        # Bước 3: Resample về 16kHz
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
            logger.debug(f"Đã resample về 16000Hz (tần số gốc: {original_sr}Hz)")
        else:
            logger.debug("Đầu vào đã là 16kHz, bỏ qua resample")

        # Bước 4: Cài đặt tham số mã hóa PCM 16-bit
        # sample_width=2 tức 16-bit (2 bytes = 16 bits)
        audio = audio.set_sample_width(bit_depth // 8)

        # Bước 5: Xuất ra định dạng WAV
        audio.export(output_path, format="wav", codec="pcm_s16le")
        logger.info(f"Chuyển đổi âm thanh hoàn tất, file đầu ra: {output_path}, "
                    f"thời lượng: {len(AudioSegment.from_file(output_path)) / 1000:.2f}s")

        return output_path

    except CouldntDecodeError as e:
        logger.error(f"Không thể giải mã file âm thanh '{audio_path}', hãy kiểm tra định dạng: {e}")
        raise CouldntDecodeError(f"Không thể giải mã file âm thanh '{audio_path}': {e}")
    except Exception as e:
        logger.error(f"Lỗi không xác định trong quá trình chuyển đổi: {e}")
        raise


def convert_to_wav_mono(
    audio_path: str,
    output_path: Optional[str] = None,
    bit_depth: int = 16
) -> str:
    """
    Chuyển đổi file âm thanh bất kỳ sang định dạng WAV đơn âm PCM và giữ nguyên tần số mẫu gốc.
    Được sử dụng cho Pyannote Diarization để tránh lỗi resample của pydub làm méo tiếng.

    Tham số:
        audio_path: Đường dẫn file âm thanh đầu vào, hỗ trợ mp3/wav/ogg/flac/m4a
        output_path: Đường dẫn file đầu ra
        bit_depth: Độ sâu bit đầu ra, mặc định 16-bit

    Trả về:
        Chuỗi đường dẫn tuyệt đối của file âm thanh đã chuyển đổi
    """
    input_path = Path(audio_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_path}")

    if output_path is None:
        output_path = str(input_path.parent / "temp_wav_mono.wav")
    else:
        output_path = str(output_path)
        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Bắt đầu convert_to_wav_mono: {audio_path} -> {output_path}")

    try:
        audio = AudioSegment.from_file(str(input_path))
        original_sr = audio.frame_rate
        original_channels = audio.channels

        # Chuyển sang đơn âm nếu cần
        if audio.channels > 1:
            audio = audio.split_to_mono()[0]
            logger.debug("Đã chuyển sang đơn âm")
        elif audio.channels == 1:
            logger.debug("Đầu vào đã là đơn âm, bỏ qua bước chuyển kênh")
        else:
            raise ValueError(f"Số kênh không được hỗ trợ: {audio.channels}")

        # Giữ nguyên sample_rate gốc, chỉ thiết lập độ sâu bit
        audio = audio.set_sample_width(bit_depth // 8)

        # Xuất ra định dạng WAV
        audio.export(output_path, format="wav", codec="pcm_s16le")
        logger.info(f"Convert hoàn tất: {output_path}, SR: {original_sr}Hz")

        return output_path

    except Exception as e:
        logger.error(f"Lỗi trong convert_to_wav_mono: {e}")
        raise


def get_audio_info(audio_path: str) -> dict:
    """
    Lấy thông tin cơ bản của file âm thanh (tần số mẫu, số kênh, thời lượng,...)

    Tham số:
        audio_path: Đường dẫn file âm thanh

    Trả về:
        Dictionary chứa các key sau:
            - sample_rate: Tần số mẫu (Hz)
            - channels: Số kênh (1=đơn âm, 2=âm thanh nổi)
            - duration: Thời lượng (giây)
            - bit_depth: Độ sâu bit (bits)
            - format: Định dạng âm thanh (mp3/wav...)
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_path}")

    audio = AudioSegment.from_file(str(path))
    return {
        "sample_rate": audio.frame_rate,
        "channels": audio.channels,
        "duration": len(audio) / 1000.0,
        "bit_depth": audio.sample_width * 8,
        "format": path.suffix.lstrip(".").lower()
    }


def is_compatible_with_whisper(audio_path: str) -> bool:
    """
    Kiểm tra file âm thanh có phù hợp với định dạng đầu vào khuyến nghị của faster-whisper

    Điều kiện kiểm tra:
        - Tần số mẫu = 16000Hz
        - Số kênh = 1 (đơn âm)
        - Định dạng là WAV

    Tham số:
        audio_path: Đường dẫn file âm thanh

    Trả về:
        True nếu phù hợp với định dạng khuyến nghị, False nếu không
    """
    try:
        info = get_audio_info(audio_path)
        is_compatible = (
            info["sample_rate"] == 16000
            and info["channels"] == 1
            and info["format"] == "wav"
        )
        if not is_compatible:
            logger.debug(f"Âm thanh '{audio_path}' không hoàn toàn phù hợp: {info}")
        return is_compatible
    except Exception:
        return False


# ========== CÁC HÀM CHUẨN HÓA VÀ TĂNG CƯỜNG ÂM THANH ==========
# Nhóm hàm này giải quyết vấn đề HALLUCINATION (kết quả ảo) trong STT
# Tham khảo: audio_enhancer.py để biết chi tiết thuật toán

def enhance_audio(
    audio_data: Union[np.ndarray, str],
    noise_threshold: float = 0.01,
    normalize_db: float = -20.0,
    apply_denoising: bool = True,
    apply_normalization: bool = True,
    apply_vad_filtering: bool = True,
    min_speech_duration: float = 0.3,
) -> np.ndarray:
    """
    Chuẩn hóa và tăng cường âm thanh để giảm hallucination trong STT

    Đây là pipeline hoàn chỉnh giải quyết vấn đề kết quả ảo:
        1. Tự động ước tính noise profile từ các đoạn im lặng
        2. Áp dụng spectral gating để giảm nhiễu nền
        3. Normalize biên độ về mức nhất quán (mặc định -20dB)
        4. Lọc bỏ các đoạn im lặng/khoảng lặng dài

    Khi nào nên dùng:
        - Âm thanh có nhiều khoảng lặng, im lặng
        - Âm thanh bị nhiễu nền (quạt máy, ồn môi trường)
        - Âm thanh có âm lượng không đều (lúc to, lúc nhỏ)
        - STT cho kết quả ảo (hallucination) nhiều

    Tham số:
        audio_data: Numpy array 1D float32 [-1, 1] HOẶC đường dẫn file âm thanh
        noise_threshold: Ngưỡng năng lượng phát hiện im lặng. Mặc định 0.01.
                         Tăng lên nếu muốn bỏ qua tiếng nhỏ, giảm xuống để nhạy hơn.
        normalize_db: Mức âm lượng mục tiêu khi normalize (dB). Mặc định -20dB.
                     Giá trị âm hơn → to hơn. Khoảng -30dB đến -15dB.
        apply_denoising: Có giảm nhiễu nền bằng spectral gating không. Mặc định True.
        apply_normalization: Có chuẩn hóa âm lượng không. Mặc định True.
        apply_vad_filtering: Có lọc đoạn im lặng không. Mặc định True.
        min_speech_duration: Thời lượng tối thiểu để coi là đoạn speech (giây).
                            Mặc định 0.3s. Đoạn ngắn hơn sẽ bị loại.

    Trả về:
        Numpy array 1D float32 đã được chuẩn hóa hoàn chỉnh

    Ví dụ:
        >>> # Xử lý numpy array trực tiếp
        >>> audio = np.random.randn(16000).astype(np.float32) * 0.1
        >>> enhanced = enhance_audio(audio, normalize_db=-20.0)
        >>>
        >>> # Xử lý file âm thanh
        >>> enhanced = enhance_audio("test.mp3", apply_denoising=True)
    """
    # Xác định input là file hay numpy array
    if isinstance(audio_data, (str, Path)):
        # Load file và chuyển về numpy array
        audio_path = str(audio_data)
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"File âm thanh không tồn tại: {audio_path}")

        # Load bằng pydub rồi chuyển về numpy
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(16000)
        if audio_segment.channels > 1:
            audio_segment = audio_segment.split_to_mono()[0]

        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        if audio_segment.sample_width == 2:
            samples = samples / 32768.0
        audio = samples
        logger.debug(f"Load âm thanh từ file: {len(audio)} samples, {len(audio)/16000:.2f}s")
    else:
        audio = audio_data

    # Chuẩn bị AudioEnhancer
    enhancer = AudioEnhancer(
        sample_rate=16000,
        noise_threshold=noise_threshold,
        normalize_target_db=normalize_db,
        min_speech_duration=min_speech_duration,
        apply_denoising=apply_denoising,
        apply_normalization=apply_normalization,
        apply_vad_filtering=apply_vad_filtering,
    )

    # Thực hiện enhancement pipeline
    enhanced = enhancer.enhance(audio)

    # Log thông tin chất lượng
    quality = enhancer.get_speech_quality_score(enhanced)
    logger.info(
        f"Audio enhancement hoàn tất: {len(enhanced)/16000:.2f}s, "
        f"quality_score={quality:.3f}"
    )

    return enhanced


def enhance_audio_file(
    input_path: str,
    output_path: Optional[str] = None,
    noise_threshold: float = 0.01,
    normalize_db: float = -20.0,
    apply_denoising: bool = True,
    apply_normalization: bool = True,
    apply_vad_filtering: bool = True,
) -> str:
    """
    Chuẩn hóa file âm thanh và lưu ra file mới (phiên bản mở rộng)

    Hàm này là wrapper mở rộng cho hàm cùng tên trong audio_enhancer,
    hỗ trợ thêm các tham số VAD filtering.

    Các bước xử lý:
        1. Load file âm thanh (pydub)
        2. Resample về 16kHz, chuyển mono
        3. Ước tính noise profile từ các đoạn im lặng
        4. Spectral gating giảm nhiễu
        5. Normalize biên độ về target_db
        6. Lọc đoạn im lặng bằng VAD năng lượng
        7. Xuất file WAV đã xử lý

    Tham số:
        input_path: Đường dẫn file âm thanh đầu vào
        output_path: Đường dẫn file đầu ra. Mặc định: thêm "_enhanced" vào tên gốc
        noise_threshold: Ngưỡng năng lượng VAD. Mặc định 0.01.
        normalize_db: Mức normalize mục tiêu (dB). Mặc định -20dB.
        apply_denoising: Có giảm nhiễu không. Mặc định True.
        apply_normalization: Có normalize không. Mặc định True.
        apply_vad_filtering: Có lọc im lặng không. Mặc định True.

    Trả về:
        Đường dẫn file đã xử lý

    Ví dụ:
        >>> # Xử lý đơn giản
        >>> out = enhance_audio_file("test.mp3")
        >>>
        >>> # Xử lý với tham số tùy chỉnh
        >>> out = enhance_audio_file(
        ...     input_path="test.mp3",
        ...     output_path="test_clean.wav",
        ...     normalize_db=-18.0,
        ...     apply_denoising=True,
        ...     apply_vad_filtering=True,
        ... )
    """
    # Xác định output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_enhanced.wav")
    else:
        output_path = str(output_path)
        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"

    # Đảm bảo thư mục đầu ra tồn tại
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load và xử lý
    audio = enhance_audio(
        input_path,
        noise_threshold=noise_threshold,
        normalize_db=normalize_db,
        apply_denoising=apply_denoising,
        apply_normalization=apply_normalization,
        apply_vad_filtering=apply_vad_filtering,
    )

    # Chuyển numpy array về AudioSegment và xuất
    enhanced_samples = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    enhanced_audio = AudioSegment(
        data=enhanced_samples.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1,
    )
    enhanced_audio.export(output_path, format="wav", codec="pcm_s16le")

    logger.info(f"Đã lưu âm thanh chuẩn hóa: {output_path} ({len(audio)/16000:.2f}s)")
    return output_path


def get_audio_quality_score(audio_path: str) -> float:
    """
    Kiểm tra nhanh điểm chất lượng âm thanh

    Tham số:
        audio_path: Đường dẫn file âm thanh

    Trả về:
        Điểm chất lượng [0.0 - 1.0]:
        - >= 0.7: Chất lượng tốt
        - 0.4 - 0.7: Chất lượng trung bình, nên enhance
        - < 0.4: Chất lượng kém, cần enhance trước STT
    """
    audio = enhance_audio(audio_path, apply_denoising=False,
                         apply_normalization=False, apply_vad_filtering=False)
    enhancer = AudioEnhancer(sample_rate=16000)
    enhancer.estimate_noise(audio)
    return enhancer.get_speech_quality_score(audio)
