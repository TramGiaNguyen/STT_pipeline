import logging
import multiprocessing
import os
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, TimeoutError

logger = logging.getLogger(__name__)


def _merge_into_main(timeline: List[Dict], main_speakers: set) -> List[Dict]:
    """
    Gán lại mọi segment KHÔNG thuộc main_speakers về một trong các speaker chính:
    chọn speaker có TỔNG OVERLAP thời gian lớn nhất với segment đó
    (fallback: speaker có segment gần nhất theo thời gian nếu không overlap).

    Dùng chung cho cả 2 chế độ:
      - Cuộc gọi (2 người): main_speakers = 2 người nói nhiều nhất.
      - Họp (nhiều người):  main_speakers = tất cả người nói thật (đã bỏ cụm rác).
    """
    if not main_speakers:
        return [dict(t) for t in timeline]

    result: List[Dict] = []
    for turn in timeline:
        if turn["speaker"] in main_speakers:
            result.append(dict(turn))
            continue

        overlap_scores: Dict[str, float] = {s: 0.0 for s in main_speakers}
        for other in timeline:
            if other["speaker"] in main_speakers:
                o_start = max(turn["start"], other["start"])
                o_end = min(turn["end"], other["end"])
                overlap = max(0.0, o_end - o_start)
                if overlap > 0:
                    overlap_scores[other["speaker"]] += overlap

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
        result.append(new_turn)
    return result


# ------------------------------------------------------------------ #
#  Hàm worker – chạy trong Process con riêng biệt                    #
# ------------------------------------------------------------------ #

def _run_pyannote_process(audio_path: str, token: str, mode: str = "phone") -> List[Dict]:
    """
    Chạy pyannote Speaker Diarization trong một Process riêng để tránh
    xung đột DLL CuDNN với Faster-Whisper đang chạy ở process chính.

    Args:
        mode: "phone"   → cuộc gọi 8kHz 2 người (ép về đúng 2 người nói chính).
              "meeting" → họp nhiều người (KHÔNG giới hạn số người, giữ tất cả
                          người nói thật, chỉ bỏ các cụm rác cực nhỏ).

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
    if mode == "meeting":
        # Chế độ HỌP nhiều người: KHÔNG giới hạn số người nói — để pyannote tự xác định
        # số cụm tự nhiên. Phù hợp khi không biết trước có bao nhiêu người trong cuộc họp.
        diarization = pipeline(audio_in_memory)
    else:
        # Chế độ CUỘC GỌI (audio điện thoại 8kHz 2 người):
        # Ép cứng min=max=2 NGHE thì hợp lý nhưng thực tế lại PHÁ clustering: trên giọng
        # băng hẹp 8kHz, embedding 2 người khá giống nhau, ép đúng 2 cụm khiến thuật toán
        # gom ~90% giọng vào MỘT người (đo thực tế trên 4.wav: 90/10) → gộp giọng "dài thòng lòng".
        # Để max_speakers nới ra (=4) thì pyannote tự tách 2 cụm CÂN BẰNG đúng (đo được 50/48)
        # cộng vài cụm rác nhỏ; bước "Speaker Consolidation" bên dưới sẽ gộp các cụm phụ
        # vào đúng 2 người chính → kết quả 2 người cân bằng, ít gộp nhầm hơn hẳn.
        diarization = pipeline(audio_in_memory, min_speakers=2, max_speakers=4)

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
    if mode == "meeting":
        # Chế độ HỌP: GIỮ TẤT CẢ người nói thật. Chỉ gộp các cụm "rác" cực nhỏ
        # (artefact do nhiễu/overlap, tổng thời lượng < 2s) vào người nói chính gần nhất.
        # KHÔNG ép về 2 người như chế độ cuộc gọi → tránh nuốt mất người nói ít lời.
        JUNK_MAX_DURATION = 2.0
        keep = {s for s, d in spk_durations.items() if d >= JUNK_MAX_DURATION}
        if not keep:  # an toàn: luôn giữ ít nhất người nói nhiều nhất
            keep = {max(spk_durations, key=spk_durations.get)}
        n_junk = len(spk_durations) - len(keep)
        if n_junk > 0:
            print(f"[Diarization Process] Meeting mode: giữ {len(keep)} người nói, "
                  f"gộp {n_junk} cụm rác (<{JUNK_MAX_DURATION}s)")
        consolidated_timeline = _merge_into_main(timeline, keep)
    else:
        # Chế độ CUỘC GỌI: ép về đúng 2 người nói chính (như cũ).
        # Chỉ chạy khi pyannote vẫn tạo ra > 2 cụm.
        if len(speaker_map) > 2:
            print("[Diarization Process] Phone mode: gộp cụm phụ vào 2 người nói chính...")
            sorted_speakers = sorted(spk_durations.items(), key=lambda x: x[1], reverse=True)
            keep = {sorted_speakers[0][0], sorted_speakers[1][0]}
            consolidated_timeline = _merge_into_main(timeline, keep)
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

    # --- Đánh số lại nhãn cho liền mạch "Người nói 1..N" theo thứ tự xuất hiện ---
    # Sau khi gộp cụm phụ/rác, các nhãn còn lại có thể bị nhảy số (vd 1, 2, 4).
    # Đổi tên lại theo thứ tự ai nói trước để hiển thị gọn gàng cho cả họp nhiều người.
    relabel: Dict[str, str] = {}
    next_id = 1
    for turn in timeline:
        if turn["speaker"] not in relabel:
            relabel[turn["speaker"]] = f"Người nói {next_id}"
            next_id += 1
    for turn in timeline:
        turn["speaker"] = relabel[turn["speaker"]]

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

    def process_audio(self, audio_path: str, timeout_sec: int = 600,
                      mode: str = "phone") -> List[Dict]:
        """
        Phân tách người nói qua Process con độc lập.

        Args:
            audio_path:   Đường dẫn file WAV đã convert sang 16 kHz mono.
            timeout_sec:  Thời gian chờ tối đa (giây). Mặc định 600s (10 phút).
            mode:         "phone" (cuộc gọi 2 người) hoặc "meeting" (họp nhiều người).

        Returns:
            List[Dict] mỗi phần tử có dạng {"start", "end", "speaker"}.
        """
        mode = "meeting" if mode == "meeting" else "phone"
        logger.info(f"Bắt đầu phân tách người nói: {audio_path} (mode={mode})")

        try:
            # QUAN TRỌNG: ép start method = "spawn" cho process con.
            # Trên Linux mặc định là "fork" → process con kế thừa CUDA đã được init ở process
            # cha (STT/VAD đã nạp lên GPU) → lỗi "Cannot re-initialize CUDA in forked subprocess".
            # "spawn" tạo process Python mới hoàn toàn, tự init CUDA sạch — giống hệt hành vi mặc
            # định trên Windows (nơi code này vốn chạy tốt). Không ảnh hưởng khi chạy CPU.
            mp_ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
                future = executor.submit(
                    _run_pyannote_process, audio_path, self._token, mode
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