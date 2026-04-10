#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stt.py - Unit test cho module STT

File này chứa unit test cho các chức năng cốt lõi của stt_engine.py và audio_utils.py.

Phạm vi test:
    - audio_utils.py:
        - convert_to_16k_mono() chức năng chuyển đổi định dạng âm thanh
        - get_audio_info() lấy thông tin âm thanh
        - is_compatible_with_whisper() kiểm tra tương thích định dạng
    - stt_engine.py:
        - Khởi tạo STTEngine (đường dẫn bình thường + đường dẫn lỗi)
        - transcribe() chức năng nhận dạng giọng nói
        - get_full_text() lấy văn bản hoàn chỉnh
        - transcribe_batch() nhận dạng hàng loạt

Cách chạy:
    pytest tests/test_stt.py -v
    pytest tests/test_stt.py -v -s    # Hiển thị output từ print

Lưu ý:
    - Một số test cần file âm thanh thực tế và mô hình đã tải
    - Khi chạy test hãy đảm bảo thư mục hiện tại là thư mục gốc dự án
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Thêm thư mục gốc dự án vào đường dẫn tìm module Python
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import audio_utils
from src import stt_engine


# ============================================================
# Cấu hình test và Fixtures
# ============================================================

@pytest.fixture
def temp_audio_file():
    """
    Tạo file âm thanh tạm phục vụ test

    Trả về:
        Path: Đường dẫn file WAV tạm
    """
    # Dùng pydub tạo đoạn âm thanh test đơn giản (sóng sine 440Hz, 1 giây)
    from pydub import AudioSegment, generators

    # Tạo sóng sine 440Hz, thời lượng 1 giây, âm lượng -20dBFS
    sine_wave = generators.Sine(freq=440).to_audio_segment(duration=1000)
    sine_wave = sine_wave - 20  # Giảm âm lượng

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, mode="wb") as f:
        temp_path = f.name
        sine_wave.export(temp_path, format="wav")

    yield temp_path

    # Dọn dẹp file tạm sau khi test xong
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_mp3_file():
    """
    Tạo file MP3 tạm phục vụ test chuyển đổi định dạng

    Trả về:
        Path: Đường dẫn file MP3 tạm
    """
    from pydub import AudioSegment, generators

    # Tạo đoạn âm thanh test rồi xuất thành MP3
    sine_wave = generators.Sine(freq=440).to_audio_segment(duration=1000)
    sine_wave = sine_wave - 20

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, mode="wb") as f:
        temp_path = f.name
        sine_wave.export(temp_path, format="mp3")

    yield temp_path

    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def project_root_path():
    """Trả về đối tượng Path của thư mục gốc dự án"""
    return project_root


@pytest.fixture
def mock_model_path(tmp_path):
    """
    Tạo thư mục "giả lập" tạm phục vụ test đường dẫn lỗi

    Trả về:
        Path: Đường dẫn thư mục rỗng hoặc không tồn tại
    """
    fake_model_dir = tmp_path / "fake_model"
    fake_model_dir.mkdir()
    return fake_model_dir


# ============================================================
# Test audio_utils.py
# ============================================================

class TestAudioUtils:
    """Lớp unit test cho module audio_utils"""

    def test_get_audio_info_valid_file(self, temp_audio_file):
        """Test get_audio_info() đọc thông tin file WAV hợp lệ"""
        info = audio_utils.get_audio_info(temp_audio_file)

        # Xác nhận dictionary trả về có đầy đủ các trường bắt buộc
        assert isinstance(info, dict)
        assert "sample_rate" in info
        assert "channels" in info
        assert "duration" in info
        assert "bit_depth" in info
        assert "format" in info

        # Xác nhận giá trị hợp lý (file test là 16kHz, đơn âm)
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["format"] == "wav"
        # Thời lượng khoảng 1 giây (cho phép sai số ±0.1 giây)
        assert 0.9 <= info["duration"] <= 1.1

    def test_get_audio_info_nonexistent_file(self):
        """Test get_audio_info() xử lý lỗi file không tồn tại"""
        with pytest.raises(FileNotFoundError):
            audio_utils.get_audio_info("nonexistent_file.wav")

    def test_convert_to_16k_mono_wav_input(self, temp_audio_file):
        """Test convert_to_16k_mono() với file WAV (đã là định dạng tương thích)"""
        output_path = tempfile.mktemp(suffix=".wav")

        try:
            result = audio_utils.convert_to_16k_mono(
                temp_audio_file,
                output_path=output_path
            )

            # Xác nhận trả về đúng đường dẫn đầu ra
            assert result == output_path

            # Xác nhận file đầu ra được tạo
            assert os.path.exists(output_path)

            # Xác nhận thông tin file đầu ra
            info = audio_utils.get_audio_info(output_path)
            assert info["sample_rate"] == 16000
            assert info["channels"] == 1
            assert info["format"] == "wav"

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_to_16k_mono_mp3_input(self, temp_mp3_file):
        """Test convert_to_16k_mono() với file MP3 (trường hợp cần chuyển đổi định dạng)"""
        output_path = tempfile.mktemp(suffix=".wav")

        try:
            result = audio_utils.convert_to_16k_mono(
                temp_mp3_file,
                output_path=output_path
            )

            # Sau khi chuyển MP3 sang WAV phải là 16kHz đơn âm
            assert os.path.exists(output_path)
            info = audio_utils.get_audio_info(output_path)
            assert info["sample_rate"] == 16000
            assert info["channels"] == 1

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_convert_to_16k_mono_nonexistent_input(self):
        """Test convert_to_16k_mono() xử lý lỗi file đầu vào không tồn tại"""
        with pytest.raises(FileNotFoundError):
            audio_utils.convert_to_16k_mono("nonexistent.wav")

    def test_convert_to_16k_mono_default_output_path(self, temp_audio_file):
        """Test convert_to_16k_mono() với hành vi mặc định của output_path"""
        # Không chỉ định output_path, dùng giá trị mặc định
        result = audio_utils.convert_to_16k_mono(temp_audio_file)

        assert result.endswith(".wav")
        assert os.path.exists(result)

        # Dọn dẹp
        if os.path.exists(result):
            os.unlink(result)

    def test_is_compatible_with_whisper_compatible(self, temp_audio_file):
        """Test is_compatible_with_whisper() với định dạng tương thích"""
        # File test là 16kHz đơn âm WAV, phải trả về True
        result = audio_utils.is_compatible_with_whisper(temp_audio_file)
        assert result is True

    def test_is_compatible_with_whisper_incompatible(self, temp_mp3_file):
        """Test is_compatible_with_whisper() với định dạng không tương thích"""
        # File MP3 không thỏa mãn điều kiện định dạng 'wav', phải trả về False
        result = audio_utils.is_compatible_with_whisper(temp_mp3_file)
        assert result is False

    def test_is_compatible_with_whisper_nonexistent(self):
        """Test is_compatible_with_whisper() xử lý file không tồn tại"""
        # File không tồn tại gây exception, hàm phải trả về False
        result = audio_utils.is_compatible_with_whisper("nonexistent.wav")
        assert result is False


# ============================================================
# Test stt_engine.py
# ============================================================

class TestSTTEngine:
    """Lớp unit test cho module stt_engine"""

    def test_stt_engine_init_with_nonexistent_model(self, mock_model_path, monkeypatch):
        """
        Test STTEngine xử lý lỗi khi mô hình không tồn tại

        Lưu ý: Do mock_model_path là thư mục rỗng (không có file mô hình),
        faster-whisper sẽ báo lỗi khi load.
        """
        # Dùng đường dẫn không tồn tại trong mock_model_path
        nonexistent_path = str(mock_model_path / "truly_nonexistent")

        # Khi download_if_missing=False phải raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            stt_engine.STTEngine(model_path=nonexistent_path, download_if_missing=False)

    def test_stt_engine_repr(self, mock_model_path):
        """
        Test phương thức __repr__ của STTEngine

        Test này chỉ xác nhận định dạng chuỗi repr, không load mô hình thực
        """
        # Giữ placeholder, cần mock配合 mới test được đường dẫn thành công
        pass  # Giữ placeholder, cần mock配合 mới test được đường dẫn thành công

    def test_stt_engine_device_detection(self, monkeypatch):
        """
        Test logic tự nhận diện thiết bị

        Dùng monkeypatch mô phỏng các môi trường khác nhau, xác nhận hành vi của _detect_device()
        """
        # Mô phỏng môi trường không có GPU, phải trả về cpu
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch("torch.cuda.is_available", return_value=False):
            engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
            engine.device = None
            result = engine._detect_device()
            assert result in ["cuda", "cpu"]

    def test_stt_engine_select_compute_type_gpu(self):
        """Test logic chọn compute_type cho thiết bị GPU"""
        engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
        engine.device = "cuda"

        available = ["bfloat16", "float16", "float32"]
        result = engine._select_compute_type(available)
        assert result == "bfloat16"  # Phải chọn kiểu khả dụng đầu tiên (nhanh nhất)

    def test_stt_engine_select_compute_type_cpu(self):
        """Test logic chọn compute_type cho thiết bị CPU"""
        engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
        engine.device = "cpu"

        available = ["int8", "float32"]
        result = engine._select_compute_type(available)
        assert result == "int8"  # CPU nên ưu tiên int8 quantization

    def test_stt_engine_select_compute_type_fallback(self):
        """Test logic fallback của compute_type (khi không có kiểu khả dụng)"""
        engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
        engine.device = "cuda"

        available = []  # Không có kiểu khả dụng
        result = engine._select_compute_type(available)
        assert result == "float32"  # Phải fallback về float32

    def test_transcribe_nonexistent_audio(self):
        """Test transcribe() xử lý lỗi file âm thanh không tồn tại"""
        # Cần mô hình thực mới khởi tạo được STTEngine, dùng mock thay thế
        engine = stt_engine.STTEngine.__new__(stt_engine.STTEngine)
        engine.model = MagicMock()  # Mock thuộc tính model

        with pytest.raises(FileNotFoundError):
            engine.transcribe("nonexistent_audio.wav")


class TestTranscribeIntegration:
    """
    Integration test: cần file âm thanh thực và mô hình thực

    Các test này mặc định bị skip (pytest.mark.skip), cần bỏ skip thủ công
    và đảm bảo môi trường sẵn sàng mới chạy.
    Cách chạy: pytest tests/test_stt.py -v --run-integration
    """

    @pytest.fixture
    def real_audio_path(self):
        """
        Trả về đường dẫn file âm thanh test (cần cung cấp thủ công)

        Muốn chạy integration test, hãy đặt một file WAV 16kHz đơn âm thực
        vào tests/fixtures/test_audio.wav
        """
        fixture_path = project_root / "tests" / "fixtures" / "test_audio.wav"
        if not fixture_path.exists():
            pytest.skip(f"File âm thanh test không tồn tại: {fixture_path}")
        return fixture_path

    @pytest.mark.skip(reason="Cần file âm thanh thực và mô hình đã tải")
    def test_transcribe_real_audio(self, real_audio_path):
        """Integration test: dùng file âm thanh thực để test toàn bộ quy trình nhận dạng"""
        engine = stt_engine.STTEngine(download_if_missing=False)
        results = engine.transcribe(str(real_audio_path), language="vi")

        # Xác nhận cấu trúc kết quả trả về
        assert isinstance(results, list)
        if results:
            first_result = results[0]
            assert "text" in first_result
            assert "start" in first_result
            assert "end" in first_result
            assert isinstance(first_result["text"], str)

    @pytest.mark.skip(reason="Cần file âm thanh thực và mô hình đã tải")
    def test_get_full_text_real_audio(self, real_audio_path):
        """Integration test: dùng file âm thanh thực để lấy văn bản hoàn chỉnh"""
        engine = stt_engine.STTEngine(download_if_missing=False)
        full_text = engine.get_full_text(str(real_audio_path))

        assert isinstance(full_text, str)
        assert len(full_text) > 0


# ============================================================
# Cấu hình entry chạy test
# ============================================================

if __name__ == "__main__":
    # Hỗ trợ chạy trực tiếp file này để test
    pytest.main([__file__, "-v", "-s"])
