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
    # Mặc dù prompt đã ép model, dùng thêm hậu kỳ bằng regex luôn chắc cú.
    replacements = [
        (r'(?i)\b(bi đi u|pi đi u)\s*(a sít từn|assistant)?\b', 'BDU Assistant'),
        (r'(?i)\b(ai đi ti ai)\b', 'IDTI'),
        (r'(?i)\b(cây|kê|ki)\s*pi\s*ai\b', 'KPI'),
        (r'(?i)\bđát\s*(bót|bo)\b', 'Dashboard'),
        (r'(?i)\bmai cờ rồ sơ\s*(vít|vích|vices)\b', 'Microservices'),
        (r'(?i)\b(ri pê|chi pi ây)\b', 'GPA'),
        (r'(?i)\b(rây|rê)\s*(si|xi)\s*(ai|i)?\b', 'RACI'),
        (r'(?i)\bsờ tim(\s*ây\s*ai)?\b', 'STEAM AI'),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    
    # Loại bỏ một số từ thừa vô nghĩa (tùy chọn)
    # text = re.sub(r'(?i)\b(ờm|ờ|ừm)\b', '', text)
    # text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_speaker_for_word(word_start: float, word_end: float, diarization: List[Dict]) -> str:
    """Tìm ID diễn giả tương ứng với một từ dựa trên overlap thời gian"""
    if not diarization:
        return "Unknown"
        
    best_speaker = "Unknown"
    max_overlap = 0.0
    
    # Tinh chỉnh: chọn khoảng loa đè thời gian lớn nhất với từ
    for seg in diarization:
        # Tính overlap
        o_start = max(word_start, seg["start"])
        o_end = min(word_end, seg["end"])
        overlap = max(0.0, o_end - o_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]
            
    # Nếu overlap 0, tìm cái có khoảng cách nhỏ nhất
    if max_overlap == 0:
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