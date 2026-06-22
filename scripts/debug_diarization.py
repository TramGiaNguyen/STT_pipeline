"""
Script chẩn đoán Diarization — chạy Pyannote trong Process con (giống production)
để phân tích timeline người nói.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from concurrent.futures import ProcessPoolExecutor
import json

AUDIO_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "temp_16k.wav"
))
TOKEN = os.environ.get("HF_TOKEN", "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")


def _worker(audio_path, token):
    """Chạy trong process con để tránh WinError 127."""
    import torch
    import torchaudio
    from pyannote.audio import Pipeline

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration = waveform.shape[1] / sample_rate
    info = f"Audio: shape={list(waveform.shape)}, sr={sample_rate}, duration={duration:.1f}s, CUDA={torch.cuda.is_available()}"

    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    results = {}

    # ===== TEST 1: min_speakers=2 (cài đặt hiện tại) =====
    diar1 = pipeline(audio_in_memory, min_speakers=2, max_speakers=5)
    results["test1_min2_max5"] = _extract(diar1)

    # ===== TEST 2: num_speakers=2 (ÉP ĐÚNG 2) =====
    diar2 = pipeline(audio_in_memory, num_speakers=2)
    results["test2_force2"] = _extract(diar2)

    return {"info": info, "results": results}


def _extract(diarization):
    """Trích xuất timeline từ Pyannote result."""
    turns = []
    speaker_map = {}
    counter = 1
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker_{counter}"
            counter += 1
        turns.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "dur": round(turn.end - turn.start, 2),
            "spk": speaker_map[speaker],
        })
    return turns


def format_ts(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def analyze(turns, tag):
    speakers = set(t["spk"] for t in turns)
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(f"Số người nói: {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"Tổng đoạn: {len(turns)}")

    for spk in sorted(speakers):
        spk_turns = [t for t in turns if t["spk"] == spk]
        total = sum(t["dur"] for t in spk_turns)
        avg = total / len(spk_turns) if spk_turns else 0
        print(f"  {spk}: {len(spk_turns)} đoạn, tổng={total:.1f}s, avg={avg:.1f}s")

    # 30 đoạn đầu
    print(f"\n--- 30 đoạn đầu ---")
    for t in turns[:30]:
        print(f"  [{format_ts(t['start'])} -> {format_ts(t['end'])}] {t['dur']:5.2f}s  {t['spk']}")

    if len(turns) > 40:
        print(f"  ... ({len(turns)-40} đoạn ẩn) ...")
        print(f"\n--- 10 đoạn cuối ---")
        for t in turns[-10:]:
            print(f"  [{format_ts(t['start'])} -> {format_ts(t['end'])}] {t['dur']:5.2f}s  {t['spk']}")

    # Đếm speaker switches
    switches = sum(1 for i in range(1, len(turns)) if turns[i]["spk"] != turns[i-1]["spk"])
    print(f"\nTổng chuyển người nói: {switches}")


if __name__ == "__main__":
    print(f"=== Chẩn đoán Diarization (ProcessPoolExecutor) ===")
    print(f"File: {AUDIO_FILE}")
    print(f"Đang chạy... (có thể mất 2-5 phút)\n")

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker, AUDIO_FILE, TOKEN)
        data = future.result(timeout=600)

    print(data["info"])

    for tag, turns in data["results"].items():
        analyze(turns, tag)
