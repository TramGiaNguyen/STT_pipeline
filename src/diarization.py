import logging
import os
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Hàm worker – chạy trong Process con riêng biệt                    #
# ------------------------------------------------------------------ #

def _run_pyannote_process(audio_path: str, token: str) -> List[Dict]:
    """
    Chạy pyannote Speaker Diarization trong một Process riêng để tránh
    xung đột DLL CuDNN với Faster-Whisper đang chạy ở process chính.

    Trả về: list của dict {"start": float, "end": float, "speaker": str}
    """
    # Import bên trong process con để tránh xung đột bộ nhớ dùng chung
    import sys
    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    # Force UTF-8 stdout to prevent Windows console UnicodeEncodeError
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # --- 1. Nạp âm thanh thủ công vào RAM ---
    waveform, sample_rate = torchaudio.load(audio_path)

    # Chuyển sang mono nếu cần
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize volume: đưa biên độ về ~0.95 peak để Pyannote hoạt động ổn định hơn
    # với audio điện thoại thường có volume không đều giữa 2 đầu dây
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

    # --- 2. Nạp model Diarization ---
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    if pipeline is None:
        raise RuntimeError(
            "Không thể tải model pyannote/speaker-diarization-3.1. "
            "Hãy kiểm tra: (1) HF_TOKEN có đúng không, "
            "(2) Bạn đã Accept điều khoản tại https://hf.co/pyannote/speaker-diarization-3.1 chưa."
        )

    # --- 3. Chuyển sang GPU nếu có ---
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # --- 4. Chạy Diarization ---
    # Với audio điện thoại 2 người: cố định min=max=2 để Pyannote không "sáng tạo"
    # ra speaker thứ 3, thứ 4 không có thật — vốn là nguyên nhân chính gây gộp nhầm
    diarization = pipeline(audio_in_memory, min_speakers=2, max_speakers=2)

    # --- 5. Chuyển đổi kết quả sang danh sách dict ---
    timeline: List[Dict] = []
    speaker_map: Dict[str, str] = {}
    speaker_counter = 1

    # Tính tổng thời lượng mỗi speaker để debug
    spk_durations = {}

    # Pyannote 4.x trả về biến obj DiarizeOutput thay vì trực tiếp đối tượng Annotation
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Người nói {speaker_counter}"
            speaker_counter += 1

        dur = round(turn.end - turn.start, 2)
        mapped_spk = speaker_map[speaker]
        spk_durations[mapped_spk] = spk_durations.get(mapped_spk, 0) + dur

        timeline.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": mapped_spk
        })

    print(f"[Diarization Process] Found {len(speaker_map)} speakers:")
    for spk, dur in spk_durations.items():
        print(f"  - {spk}: total {dur:.1f} sec")

    # --- Speaker Consolidation (Hợp nhất Diễn giả Phụ) ---
    # Bước này chỉ chạy khi Pyannote vẫn tạo ra > 2 speaker dù đã set max_speakers=2
    # (có thể xảy ra với một số phiên bản pyannote)
    if len(speaker_map) > 2:
        print("[Diarization Process] Merging minor speakers...")
        sorted_speakers = sorted(spk_durations.items(), key=lambda x: x[1], reverse=True)

        main_speakers = set()
        main_speakers.add(sorted_speakers[0][0])
        main_speakers.add(sorted_speakers[1][0])

        consolidated_timeline = []
        for turn in timeline:
            spk_name = turn["speaker"]
            if spk_name in main_speakers:
                consolidated_timeline.append(dict(turn))
            else:
                # Tìm speaker chính có TỔNG OVERLAP lớn nhất với segment này
                # (chính xác hơn so với tìm khoảng cách gần nhất)
                overlap_scores: Dict[str, float] = {s: 0.0 for s in main_speakers}
                for other_turn in timeline:
                    if other_turn["speaker"] in main_speakers:
                        o_start = max(turn["start"], other_turn["start"])
                        o_end = min(turn["end"], other_turn["end"])
                        overlap = max(0.0, o_end - o_start)
                        if overlap > 0:
                            overlap_scores[other_turn["speaker"]] = (
                                overlap_scores.get(other_turn["speaker"], 0.0) + overlap
                            )

                # Nếu có overlap thực sự → chọn speaker có overlap lớn nhất
                # Nếu không có overlap → chọn speaker có segment gần nhất về thời gian
                if max(overlap_scores.values()) > 0:
                    best_speaker = max(overlap_scores, key=overlap_scores.get)
                else:
                    midpoint = (turn["start"] + turn["end"]) / 2.0
                    best_speaker = min(
                        main_speakers,
                        key=lambda s: min(
                            abs(midpoint - t["start"])
                            for t in timeline if t["speaker"] == s
                        )
                    )

                new_turn = dict(turn)
                new_turn["speaker"] = best_speaker
                consolidated_timeline.append(new_turn)
    else:
        consolidated_timeline = [dict(t) for t in timeline]

    # --- Merge các segment liền nhau của cùng 1 speaker ---
    # Điều kiện: cùng speaker + khoảng lặng < 0.4s + tổng độ dài sau merge < 8s
    # Ngưỡng 0.4s: đủ để nối từ bị cắt giữa chừng nhưng không nuốt lượt thoại người khác
    # Giới hạn 8s: tránh tạo block quá dài làm subtitle_formatter gán nhầm speaker
    MAX_MERGE_GAP = 0.4
    MAX_MERGED_DURATION = 8.0

    merged_timeline = []
    for turn in consolidated_timeline:
        if not merged_timeline:
            merged_timeline.append(dict(turn))
        else:
            last_turn = merged_timeline[-1]
            gap = turn["start"] - last_turn["end"]
            merged_duration = turn["end"] - last_turn["start"]
            if (turn["speaker"] == last_turn["speaker"]
                    and gap < MAX_MERGE_GAP
                    and merged_duration < MAX_MERGED_DURATION):
                last_turn["end"] = max(last_turn["end"], turn["end"])
            else:
                merged_timeline.append(dict(turn))

    timeline = merged_timeline

    # In kết quả cuối
    final_durations: Dict[str, float] = {}
    for turn in timeline:
        dur = turn["end"] - turn["start"]
        final_durations[turn["speaker"]] = final_durations.get(turn["speaker"], 0.0) + dur
    print(f"[Diarization Process] Final result ({len(final_durations)} speakers, {len(timeline)} segments):")
    for spk_name, dur in sorted(final_durations.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {spk_name}: total {dur:.1f} sec")

    return timeline


# ------------------------------------------------------------------ #
#  DiarizationProcessor – Singleton wrapper                           #
# ------------------------------------------------------------------ #

class DiarizationProcessor:
    """
    Wrapper cho Speaker Diarization bằng pyannote.audio.
    Luôn chạy qua một Process con riêng để tránh xung đột DLL CuDNN.
    """

    _instance: Optional["DiarizationProcessor"] = None

    def __init__(self) -> None:
        # Token được đọc tại thời điểm khởi tạo instance (không phải lúc load module)
        # Ưu tiên biến môi trường HF_TOKEN trước, sau đó mới dùng giá trị mặc định
        self._token: str = os.environ.get(
            "HF_TOKEN",
            "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",
        )

    @classmethod
    def get_instance(cls) -> "DiarizationProcessor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_audio(self, audio_path: str, timeout_sec: int = 600) -> List[Dict]:
        """
        Phân tách người nói qua Process con độc lập.

        Args:
            audio_path:   Đường dẫn file WAV đã convert sang 16 kHz mono.
            timeout_sec:  Thời gian chờ tối đa (giây). Mặc định 600s (10 phút).

        Returns:
            List[Dict] mỗi phần tử có dạng {"start", "end", "speaker"}.
        """
        logger.info(f"Bắt đầu phân tách người nói: {audio_path}")

        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _run_pyannote_process, audio_path, self._token
                )
                try:
                    timeline = future.result(timeout=timeout_sec)
                except TimeoutError:
                    raise RuntimeError(
                        f"Diarization vượt quá thời gian cho phép ({timeout_sec}s). "
                        "Thử với file ngắn hơn."
                    )

            logger.info(f"Phân tách người nói hoàn tất: {len(timeline)} đoạn")
            return timeline

        except RuntimeError:
            raise  # re-raise lỗi rõ ràng đã tạo bên trên
        except Exception as e:
            logger.error(f"Lỗi Diarization: {e}", exc_info=True)
            raise RuntimeError(f"Lỗi phân tách giọng nói: {e}") from e