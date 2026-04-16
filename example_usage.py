#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_usage.py - Script ví dụ sử dụng module STT

Script này minh họa cách sử dụng cơ bản của STT_module, bao gồm:
    1. Tải mô hình (lần đầu sử dụng)
    2. Tiền xử lý âm thanh (chuyển về 16kHz đơn âm)
    3. Nhận dạng giọng nói và xuất timestamp
    4. Xử lý hàng loạt nhiều file âm thanh

Chuẩn bị trước khi chạy:
    1. Cài đặt dependencies: pip install -r requirements.txt
    2. Tải mô hình: python scripts/download_model.py
    3. Chuẩn bị file âm thanh cần nhận dạng (hỗ trợ mp3/wav/ogg...)

Cách sử dụng:
    # Nhận dạng một file
    python example_usage.py --input test.wav

    # Chỉ định ngôn ngữ
    python example_usage.py --input test.mp3 --language vi

    # Xử lý hàng loạt
    python example_usage.py --batch file1.wav file2.mp3 file3.ogg
"""

import argparse
import io
import logging
import sys
from pathlib import Path

# 修复 Windows 控制台 UTF-8 输出编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Thêm thư mục gốc dự án vào đường dẫn tìm module Python
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.audio_utils import (
    convert_to_16k_mono,
    is_compatible_with_whisper,
    enhance_audio,
    enhance_audio_file,
    get_audio_quality_score,
)
from src.stt_engine import STTEngine

# Cấu hình log: mặc định hiển thị cấp INFO, xuất ra console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def transcribe_single_file(
    audio_path: str,
    language: str = "vi",
    output_format: str = "timestamped",
    show_word_timestamps: bool = False,
    enhance: bool = True,
    normalize_db: float = -20.0,
    apply_denoising: bool = True,
    apply_vad_filtering: bool = True,
    save_enhanced: bool = False,
    initial_prompt: str = None,
    beam_size: int = 5,
    model_path: str = None,
    condition_on_previous_text: bool = True,
):
    """
    Nhận dạng một file âm thanh và xuất kết quả

    Tham số:
        audio_path: Đường dẫn file âm thanh
        language: Mã ngôn ngữ ('vi'=Tiếng Việt, 'zh'=Tiếng Trung, 'en'=Tiếng Anh)
        output_format: Định dạng xuất ('timestamped'=có timestamp, 'text'=chỉ văn bản)
        show_word_timestamps: Có hiển thị timestamp cấp từ không
        enhance: Có áp dụng chuẩn hóa âm thanh (giảm hallucination) không. Mặc định True.
        normalize_db: Mức âm lượng chuẩn hóa (dB). Mặc định -20dB.
        apply_denoising: Có giảm nhiễu nền không. Mặc định True.
        apply_vad_filtering: Có lọc đoạn im lặng không. Mặc định True.
        save_enhanced: Có lưu file đã chuẩn hóa ra đĩa không. Mặc định False.
    """
    logger.info(f"Đang xử lý file âm thanh: {audio_path}")

    # Kiểm tra file âm thanh có tồn tại không
    if not Path(audio_path).exists():
        logger.error(f"File âm thanh không tồn tại: {audio_path}")
        return

    # Tiền xử lý âm thanh: nếu không phải 16kHz đơn âm WAV, chuyển đổi trước
    if not is_compatible_with_whisper(audio_path):
        logger.info("Định dạng âm thanh không phù hợp với faster-whisper, bắt đầu chuyển đổi...")
        converted_path = convert_to_16k_mono(audio_path)
        logger.info(f"Âm thanh đã chuyển đổi: {converted_path}")
        working_path = converted_path
    else:
        working_path = audio_path
        logger.info("Kiểm tra định dạng âm thanh đạt, dùng trực tiếp file gốc")

    # ========== BƯỚC CHUẨN HÓA ÂM THANH (GIẢM HALLUCINATION) ==========
    # CẢNH BÁO: Chỉ bật enhance khi thật sự cần (âm thanh có nhiễu nặng).
    # Với âm thanh sạch, bật enhance có thể LÀM GIẢM chất lượng STT do:
    #   1. Spectral gating làm méo tiếng nói
    #   2. Energy VAD (noise_threshold=0.01) có thể CẮT SAI các đoạn speech có âm lượng thấp
    #   3. Double VAD (enhance VAD + faster-whisper Silero VAD) gây mất nội dung
    # Mặc định: TẮT enhance, dùng logic đơn giản như code gốc của người dùng
    if enhance:
        logger.info(f"Bắt đầu chuẩn hóa âm thanh: normalize_db={normalize_db}dB, "
                    f"denoise={apply_denoising}, vad={apply_vad_filtering}")

        try:
            # Chuẩn hóa âm thanh bằng AudioEnhancer
            enhanced_audio = enhance_audio(
                working_path,
                noise_threshold=0.01,
                normalize_db=normalize_db,
                apply_denoising=apply_denoising,
                apply_normalization=True,
                apply_vad_filtering=apply_vad_filtering,
            )

            # Nếu yêu cầu lưu file đã chuẩn hóa
            if save_enhanced:
                enhanced_path = enhance_audio_file(
                    working_path,
                    normalize_db=normalize_db,
                    apply_denoising=apply_denoising,
                    apply_vad_filtering=apply_vad_filtering,
                )
                logger.info(f"Đã lưu file chuẩn hóa: {enhanced_path}")

            # Dùng audio đã chuẩn hóa cho STT (faster-whisper nhận trực tiếp numpy array)
            stt_input = enhanced_audio
            logger.info(f"Chuẩn hóa hoàn tất, thời lượng: {len(enhanced_audio)/16000:.2f}s")
        except Exception as e:
            logger.warning(f"Chuẩn hóa thất bại ({e}), dùng file gốc cho STT")
            stt_input = working_path
    else:
        # ✨ Logic đơn giản như code gốc: chỉ chuyển 16k, không enhance, không energy VAD
        # Silero VAD trong faster-whisper sẽ tự xử lý VAD
        stt_input = working_path
        logger.info("Bỏ qua chuẩn hóa, dùng audio gốc (logic giống code gốc)")

    # Khởi tạo engine STT (load mô hình)
    # Lưu ý: Lần đầu chạy sẽ tốn thời gian (load mô hình), các lần sau sẽ dùng lại
    engine = STTEngine(model_path=model_path) if model_path else STTEngine()

    # Thực hiện nhận dạng giọng nói
    logger.info(f"Bắt đầu nhận dạng, ngôn ngữ: {language or 'tự động'}")
    if initial_prompt:
        logger.info(f"Sử dụng initial prompt: {initial_prompt[:100]}...")
    
    results = engine.transcribe(
        stt_input,
        language=language if language != "auto" else None,
        word_timestamps=show_word_timestamps,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
    )

    # Xuất kết quả
    print("\n" + "=" * 70)
    print(f"Kết quả nhận dạng - {Path(audio_path).name}")
    print("=" * 70)

    if output_format == "text":
        # Chỉ xuất văn bản thuần
        full_text = " ".join(seg["text"] for seg in results)
        print(full_text)
    else:
        # Xuất từng đoạn kèm timestamp
        for i, seg in enumerate(results, 1):
            print(f"[{i:03d}] [{seg['start']:>6.2f}s -> {seg['end']:>6.2f}s] {seg['text']}")
            # Nếu bật timestamp cấp từ, hiển thị thụt vào
            if show_word_timestamps and "words" in seg:
                for word in seg["words"]:
                    print(f"       |  [{word['start']:.2f}s -> {word['end']:.2f}s] {word['word']}")

    print("=" * 70)
    print(f"Tổng cộng nhận dạng được {len(results)} đoạn\n")


def transcribe_batch_files(audio_paths: list, language: str = "vi"):
    """
    Nhận dạng hàng loạt nhiều file âm thanh

    Tham số:
        audio_paths: Danh sách đường dẫn file âm thanh
        language: Mã ngôn ngữ
    """
    logger.info(f"Bắt đầu xử lý hàng loạt, tổng {len(audio_paths)} file")

    # Khởi tạo engine một lần cho tất cả file (tránh load mô hình lặp lại)
    engine = STTEngine()

    for i, audio_path in enumerate(audio_paths, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(audio_paths)}] Xử lý: {Path(audio_path).name}")
        print("=" * 70)

        try:
            transcribe_single_file(audio_path, language=language)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file '{audio_path}': {e}")
            print(f"[Lỗi] {e}")


def main():
    """Hàm chính: parse tham số dòng lệnh và thực thi tương ứng"""
    parser = argparse.ArgumentParser(
        description="Script ví dụ STT Module - Công cụ demo nhận dạng giọng nói",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
    Nhận dạng một file:
        python example_usage.py --input test.wav

    Chỉ định tiếng Việt:
        python example_usage.py --input test.wav --language vi

    Xử lý hàng loạt:
        python example_usage.py --batch file1.wav file2.mp3

    Xuất chỉ văn bản (không timestamp):
        python example_usage.py --input test.wav --format text

    Hiển thị timestamp cấp từ:
        python example_usage.py --input test.wav --word-timestamps

Mã ngôn ngữ được hỗ trợ:
    vi  - Tiếng Việt
    zh  - Tiếng Trung
    en  - Tiếng Anh
    ja  - Tiếng Nhật
    ko  - Tiếng Hàn
    auto - Tự động nhận diện ngôn ngữ
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Đường dẫn file âm thanh đầu vào (một file)"
    )
    parser.add_argument(
        "--batch", "-b",
        nargs="+",
        help="Danh sách file âm thanh để xử lý hàng loạt"
    )
    parser.add_argument(
        "--language", "-l",
        default="vi",
        help="Mã ngôn ngữ (vi/zh/en/ja/ko/auto), mặc định: vi"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["timestamped", "text"],
        default="timestamped",
        help="Định dạng xuất: timestamped(có timestamp) hoặc text(chỉ văn bản), mặc định: timestamped"
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Hiển thị timestamp cấp từ (sẽ tăng thời gian xử lý)"
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Initial prompt để hướng dẫn mô hình về ngữ cảnh "
             "(ví dụ: 'Nội dung về quyên góp, từ thiện, hoạt động cộng đồng'). "
             "Giúp giảm lỗi nhận dạng sai từ."
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Độ rộng beam search (3-10). Càng lớn càng chính xác nhưng chậm hơn. Mặc định: 5"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Đường dẫn mô hình (mặc định: models/EraX-WoW-Turbo-V1.1-CT2). "
             "Thử: models/PhoWhisper-large-ct2 cho tiếng Việt tốt hơn"
    )
    parser.add_argument(
        "--no-condition-prev-text",
        action="store_true",
        help="Tắt condition_on_previous_text. Ngăn lỗi hallucination nhân lên "
             "giữa các segment. Khuyến nghị dùng khi output bị lặp lại hoặc toàn ký tự lạ."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị thông tin debug chi tiết"
    )
    # ========== NHÓM THAM SỐ CHUẨN HÓA ÂM THANH ==========
    # Nhóm tham số này kiểm soát việc chuẩn hóa âm thanh để giảm HALLUCINATION
    # Tham khảo: src/audio_enhancer.py để biết chi tiết từng kỹ thuật
    enhancement_group = parser.add_argument_group(
        title="Tùy chọn chuẩn hóa âm thanh (giảm kết quả ảo)",
        description="Các tham số này giúp giảm hiện tượng HALLUCINATION - "
                    "kết quả ảo/hallucination trong STT"
    )
    enhancement_group.add_argument(
        "--no-enhance",
        action="store_true",
        help="Tắt chuẩn hóa âm thanh (mặc định bật). Dùng khi đã xử lý riêng."
    )
    enhancement_group.add_argument(
        "--normalize-db",
        type=float,
        default=-20.0,
        help="Mức âm lượng chuẩn hóa (dB), mặc định: -20. "
             "Giá trị âm hơn (-25, -30) → to hơn, dương hơn (-15) → nhỏ hơn."
    )
    enhancement_group.add_argument(
        "--no-denoise",
        action="store_true",
        help="Tắt giảm nhiễu nền bằng spectral gating. "
             "Dùng khi âm thanh đã sạch."
    )
    enhancement_group.add_argument(
        "--no-vad",
        action="store_true",
        help="Tắt lọc đoạn im lặng. Mặc định bật VAD filter."
    )
    enhancement_group.add_argument(
        "--save-enhanced",
        action="store_true",
        help="Lưu file âm thanh đã chuẩn hóa ra đĩa (thêm _enhanced vào tên)"
    )
    enhancement_group.add_argument(
        "--quality-check",
        action="store_true",
        help="Chỉ kiểm tra điểm chất lượng âm thanh, không chạy STT. "
             "Điểm >= 0.7: tốt, 0.4-0.7: trung bình, < 0.4: kém"
    )
    parser.add_argument(
        "--help-enhance",
        action="store_true",
        help="Hiển thị hướng dẫn chi tiết về tùy chọn chuẩn hóa âm thanh"
    )

    args = parser.parse_args()

    # Hiển thị hướng dẫn enhancement nếu được yêu cầu
    if args.help_enhance:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║        HƯỚNG DẪN TÙY CHỌN CHUẨN HÓA ÂM THANH (GIẢM HALLUCINATION)   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  HALLUCINATION là hiện tượng STT tạo ra văn bản ảo, không có      ║
║  trong âm thanh thực tế. Nguyên nhân chính:                       ║
║    - Đoạn im lặng dài → model "bịa đặt" nội dung                  ║
║    - Nhiễu nền cao → model nhầm lẫn                               ║
║    - Âm lượng không đều → model đọc sai                            ║
║                                                                      ║
║  CÁC KỸ THUẬT ĐƯỢC ÁP DỤNG:                                       ║
║    1. Spectral Gating: Giảm nhiễu nền bằng phổ tần số             ║
║    2. Audio Normalization: Đưa âm lượng về mức nhất quán (-20dB)  ║
║    3. Energy VAD: Lọc bỏ các đoạn im lặng/khoảng lặng dài          ║
║                                                                      ║
║  VÍ DỤ SỬ DỤNG:                                                    ║
║    # Chuẩn hóa đầy đủ (mặc định) - khuyến nghị dùng              ║
║    python example_usage.py -i test.mp3                               ║
║                                                                      ║
║    # Tắt giảm nhiễu, chỉ normalize                                  ║
║    python example_usage.py -i test.mp3 --no-denoise                 ║
║                                                                      ║
║    # Tắt VAD filter, giữ nguyên các đoạn im lặng                  ║
║    python example_usage.py -i test.mp3 --no-vad                     ║
║                                                                      ║
║    # Tăng âm lượng normalize lên (-18dB to hơn)                    ║
║    python example_usage.py -i test.mp3 --normalize-db -18.0          ║
║                                                                      ║
║    # Lưu file đã chuẩn hóa để debug                                ║
║    python example_usage.py -i test.mp3 --save-enhanced               ║
║                                                                      ║
║    # Kiểm tra chất lượng âm thanh trước khi STT                    ║
║    python example_usage.py -i test.mp3 --quality-check               ║
║                                                                      ║
║  KHI NÀO NÊN TẮT CÁC TÙY CHỌN:                                    ║
║    --no-enhance:     Đã tự xử lý âm thanh trước                     ║
║    --no-denoise:     Âm thanh đã sạch, không có nhiễu nền          ║
║    --no-vad:         Cần giữ nguyên các khoảng lặng trong text       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        return
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Xác định chế độ chạy: một file / hàng loạt / không tham số (hiển thị help)
    if args.batch:
        # Chế độ hàng loạt
        transcribe_batch_files(args.batch, language=args.language)
    elif args.input:
        # Chế độ một file
        # Kiểm tra chế độ quality check
        if args.quality_check:
            print(f"\n{'='*70}")
            print(f"Kiểm tra chất lượng âm thanh: {args.input}")
            print("=" * 70)
            score = get_audio_quality_score(args.input)
            if score >= 0.7:
                quality_label = "TỐT"
            elif score >= 0.4:
                quality_label = "TRUNG BÌNH (nên enhance)"
            else:
                quality_label = "KÉM (cần enhance trước STT)"
            print(f"Điểm chất lượng: {score:.3f} / 1.0")
            print(f"Đánh giá: {quality_label}")
            print("=" * 70)
            return

        # Chế độ STT thông thường với tùy chọn enhancement
        transcribe_single_file(
            args.input,
            language=args.language,
            output_format=args.format,
            show_word_timestamps=args.word_timestamps,
            enhance=not args.no_enhance,
            normalize_db=args.normalize_db,
            apply_denoising=not args.no_denoise,
            apply_vad_filtering=not args.no_vad,
            save_enhanced=args.save_enhanced,
            initial_prompt=args.prompt,
            beam_size=args.beam_size,
            model_path=args.model,
            condition_on_previous_text=not args.no_condition_prev_text,
        )
    else:
        # Không có tham số thì hiển thị help
        parser.print_help()


if __name__ == "__main__":
    main()
