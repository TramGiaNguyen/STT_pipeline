import re
import datetime
from typing import List, Dict

def format_time_srt(seconds: float) -> str:
    """Chuyển đổi giây sang định dạng SRT: HH:MM:SS,mmm"""
    delta = datetime.timedelta(seconds=seconds)
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    secs = int(delta.total_seconds() % 60)
    millis = int(round((delta.total_seconds() - int(delta.total_seconds())) * 1000))
    # Ràng buộc lỗi làm tròn 1000ms
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def process_text_glossary(text: str) -> str:
    """Lọc hậu kỳ: thay đổi thuật ngữ bị phiên âm sai bằng từ chuẩn"""
    # === Glossary chuyên ngành (thuật ngữ tiếng Anh) ===
    tech_replacements = [
        (r'(?i)\b(bi đi u|pi đi u)\s*(a sít từn|assistant)?\b', 'BDU Assistant'),
        (r'(?i)\b(ai đi ti ai)\b', 'IDTI'),
        (r'(?i)\b(cây|kê|ki)\s*pi\s*ai\b', 'KPI'),
        (r'(?i)\bđát\s*(bót|bo)\b', 'Dashboard'),
        (r'(?i)\bmai cờ rồ sơ\s*(vít|vích|vices)\b', 'Microservices'),
        (r'(?i)\b(ri pê|chi pi ây)\b', 'GPA'),
        (r'(?i)\b(rây|rê)\s*(si|xi)\s*(ai|i)?\b', 'RACI'),
        (r'(?i)\bsờ tim(\s*ây\s*ai)?\b', 'STEAM AI'),
    ]
    
    # === Glossary tiếng Việt (từ hay bị nghe sai trong cuộc gọi điện thoại) ===
    vn_replacements = [
        # Học thuật
        (r'(?i)\btiếng chỉ\b', 'tín chỉ'),
        (r'(?i)\btiếng chữ\b', 'tín chỉ'),
        (r'(?i)\btính chỉ\b', 'tín chỉ'),
        (r'(?i)\btính chữ\b', 'tín chỉ'),
        (r'(?i)\bchín chỉ\b', 'tín chỉ'),
        (r'(?i)\btrứng chịu\b', 'tín chỉ'),
        (r'(?i)\bthính chỉ\b', 'tín chỉ'),
        (r'(?i)\bthính vị\b', 'tín chỉ'),
        # Zalo / liên lạc
        (r'(?i)\bnha lô\b', 'Zalo'),
        (r'(?i)\byao lôi\b', 'Zalo'),
        (r'(?i)\byao lô\b', 'Zalo'),
        (r'(?i)\bgia lô\b', 'Zalo'),
        (r'(?i)\bnhắn lô\b', 'Zalo'),
        (r'(?i)\bcác bạn zalo\b', 'kết bạn Zalo'),
        (r'(?i)\bcác bạn nghe lôi\b', 'kết bạn Zalo'),
        (r'(?i)\bkết bạn gia lô\b', 'kết bạn Zalo'),
        # Email / mạng xã hội
        (r'(?i)\bmeo\b', 'mail'),
        (r'(?i)\bphây\b', 'Facebook'),
        (r'(?i)\btrang phây\b', 'trang Facebook'),
        (r'(?i)\btrang face\b', 'trang Facebook'),
        # Tên riêng thường bị sai
        (r'(?i)\bviệt tinh\b', 'Viettin'),
        (r'(?i)\bviệc tinh\b', 'Viettin'),
        # Từ bị nghe sai phổ biến
        (r'(?i)\bsút sai\b', 'xuất file'),
        (r'(?i)\blịch hội\b', 'mật khẩu'),
        (r'(?i)\bnhẫn huyết\b', 'nhắn miết'),
        (r'(?i)\btrang goét\b', 'trang web'),
        (r'(?i)\bon lai\b', 'online'),
        (r'(?i)\bcân bằng hai\b', 'văn bằng hai'),
        (r'(?i)\bhoa băng\b', 'Habana'),
        (r'(?i)\bbốc\s+phót\b', 'bóc phốt'),
        (r'(?i)\bbốc\s+phốt\b', 'bóc phốt'),
        (r'(?i)\brất môn\b', 'rớt môn'),
        (r'(?i)\bhổ\s+chợ\b', 'hỗ trợ'),
        (r'(?i)\bhỗ\s+chợ\b', 'hỗ trợ'),
        (r'(?i)\bnạo\s+môn\b', 'nợ môn'),
        (r'(?i)\bnạo\s+quá\b', 'nợ quá'),
    ]
    
    for pattern, replacement in tech_replacements + vn_replacements:
        text = re.sub(pattern, replacement, text)
    
    return text

def get_speaker_for_word(word_start: float, word_end: float, diarization: List[Dict]) -> str:
    """Tìm ID diễn giả tương ứng với một từ dựa trên overlap thời gian"""
    if not diarization:
        return "Unknown"
        
    candidates = []
    for seg in diarization:
        o_start = max(word_start, seg["start"])
        o_end = min(word_end, seg["end"])
        overlap = max(0.0, o_end - o_start)
        if overlap > 0:
            seg_duration = seg["end"] - seg["start"]
            candidates.append({
                "speaker": seg["speaker"],
                "overlap": overlap,
                "seg_duration": seg_duration
            })
            
    if candidates:
        # Ưu tiên overlap lớn nhất, nếu bằng nhau thì ưu tiên segment NGẮN NHẤT 
        # (để bắt được các câu ngắt lời/chuyển thoại thay vì bị đè bởi người nói dài)
        candidates.sort(key=lambda x: (-x["overlap"], x["seg_duration"]))
        best_speaker = candidates[0]["speaker"]
    else:
        # Nếu không có overlap, tìm segment gần nhất
        closest_seg = min(diarization, key=lambda s: min(abs(word_start - s["end"]), abs(word_end - s["start"])))
        best_speaker = closest_seg["speaker"]
        
    return best_speaker

def generate_srt(segments: List[Dict], diarization: List[Dict] = None, max_words=12, max_duration=5.0) -> str:
    """
    Sinh file SRT hoàn chỉnh từ kết quả STT, hỗ trợ chia khối (segmentation)
    và gán nhãn người nói (diarization).
    
    - max_words: Tối đa số từ trong 1 block SRT (để hiển thị đẹp trên màn hình 1-2 dòng)
    - max_duration: Tối đa số giây 1 block SRT
    """
    # 1. Thu thập tất cả các 'words' 
    all_words = []
    for seg in segments:
        if "words" in seg:
            all_words.extend(seg["words"])
            
    if not all_words:
        # Fallback nếu Whisper không cấu hình cấp từ (word-level)
        return _generate_srt_fallback(segments, diarization)

    # Gán nhãn người nói cho từng chữ
    if diarization:
        for w in all_words:
            w["speaker"] = get_speaker_for_word(w["start"], w["end"], diarization)
        
        # === SMOOTHING: Loại bỏ backchannel cắt ngang ===
        # CHỈ hấp thụ nếu: đúng 1 từ, thời lượng < 0.8s,
        # VÀ cả trước+sau đều là cùng 1 speaker với run >= 3 từ
        speakers = [w.get("speaker", "Không rõ") for w in all_words]
        smoothed = list(speakers)
        
        i = 0
        while i < len(smoothed):
            if i > 0 and smoothed[i] != smoothed[i - 1]:
                run_start = i
                run_speaker = smoothed[i]
                j = i
                while j < len(smoothed) and smoothed[j] == run_speaker:
                    j += 1
                run_len = j - run_start
                
                if run_len == 1 and j < len(smoothed) and smoothed[j] == smoothed[i - 1]:
                    word_duration = all_words[i]["end"] - all_words[i]["start"]
                    if word_duration < 0.8:
                        prev_run_len = 0
                        k = i - 1
                        while k >= 0 and smoothed[k] == smoothed[i - 1]:
                            prev_run_len += 1
                            k -= 1
                        if prev_run_len >= 3:
                            smoothed[i] = smoothed[i - 1]
                            i = j
                            continue
            i += 1
        
        for idx, w in enumerate(all_words):
            w["speaker"] = smoothed[idx]

    # 2. Xây dựng các khối phụ đề (Subtitle Blocks)
    blocks = []
    current_block_words = []
    current_speaker = None

    def add_block(speaker_to_save: str):
        """Đóng block hiện tại, lưu speaker được truyền vào thay vì đọc biến ngoài."""
        if not current_block_words:
            return
        start_time = current_block_words[0]["start"]
        end_time = current_block_words[-1]["end"]
        raw_text = " ".join(w["word"] for w in current_block_words)

        # Sửa hậu kỳ
        clean_text = process_text_glossary(raw_text)
        # Hoa chữ đầu dòng nếu prompt chưa làm
        if clean_text and clean_text[0].islower():
            clean_text = clean_text[0].upper() + clean_text[1:]

        blocks.append({
            "speaker": speaker_to_save,
            "start": start_time,
            "end": end_time,
            "text": clean_text
        })
        current_block_words.clear()

    for w in all_words:
        word_speaker = w.get("speaker", "") if diarization else ""

        # Quyết định cắt khối (Cut-off conditions)
        needs_cut = False

        if current_block_words:
            # Cắt nếu đổi người nói — ưu tiên cao nhất
            if current_speaker != word_speaker:
                needs_cut = True
            # Cắt nếu đủ chữ
            elif len(current_block_words) >= max_words:
                needs_cut = True
            # Cắt nếu đủ giây
            elif (w["end"] - current_block_words[0]["start"]) > max_duration:
                needs_cut = True
            # Cắt nếu có khoảng lặng giữa các chữ > 1.5s
            elif (w["start"] - current_block_words[-1]["end"]) > 1.5:
                needs_cut = True
            # Cắt nếu từ TRƯỚC kết thúc bằng dấu câu (đặt cuối để không override đổi người nói)
            elif (current_block_words[-1]["word"].endswith('.')
                  or current_block_words[-1]["word"].endswith('?')
                  or current_block_words[-1]["word"].endswith('!')):
                needs_cut = True

        if needs_cut:
            add_block(current_speaker)   # lưu đúng speaker của block đang đóng
            current_speaker = word_speaker  # cập nhật speaker cho block mới

        if not current_block_words:
            current_speaker = word_speaker

        current_block_words.append(w)
        
    # Thêm block cuối
    if current_block_words:
        add_block(current_speaker)
        
    # 3. Kết xuất ra SRT Format
    out_lines = []
    for i, b in enumerate(blocks, 1):
        out_lines.append(str(i))
        start_srt = format_time_srt(b['start'])
        end_srt = format_time_srt(b['end'])
        out_lines.append(f"{start_srt} --> {end_srt}")
        if b['speaker']:
            out_lines.append(f"[{b['speaker']}]: {b['text']}")
        else:
            out_lines.append(b['text'])
        out_lines.append("")
        
    return "\n".join(out_lines)

def _generate_srt_fallback(segments: List[Dict], diarization: List[Dict]) -> str:
    """Dùng khi không có word timestamps"""
    out_lines = []
    for i, seg in enumerate(segments, 1):
        out_lines.append(str(i))
        start_srt = format_time_srt(seg['start'])
        end_srt = format_time_srt(seg['end'])
        out_lines.append(f"{start_srt} --> {end_srt}")
        spk = get_speaker_for_word(seg['start'], seg['end'], diarization) if diarization else ""
        text = process_text_glossary(seg['text'])
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        if spk:
            out_lines.append(f"[{spk}]: {text}")
        else:
            out_lines.append(text)
        out_lines.append("")
        
    return "\n".join(out_lines)