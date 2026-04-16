import logging
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

def _run_pyannote_process(audio_path: str, token: str) -> List[Dict]:
    """Hàm chạy độc lập trong Process riêng để tránh xung đột DLL CuDNN với Faster-Whisper"""
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    
    # Nạp thủ công âm thanh vào RAM để lách lỗi thiếu bộ giải mã (bug của Pyannote 4.x)
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Đảm bảo là mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
    
    # Init pipeline inside child process
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token
    )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        
    diarization = pipeline(audio_in_memory)
    
    timeline = []
    speaker_map = {}
    speaker_counter = 1

    # Pyannote 4.x trả về biến obj DiarizeOutput thay vì trực tiếp đối tượng Annotation
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Người nói {speaker_counter}"
            speaker_counter += 1
            
        timeline.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker_map[speaker]
        })
    return timeline

class DiarizationProcessor:
    """
    Class hỗ trợ phân tách người nói (Speaker Diarization) sử dụng pyannote.audio.
    Hỗ trợ process isolation để tránh lỗi WinError 127 CuDNN.
    """
    _instance = None
    _token = "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"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Khởi tạo Child Process để chạy Pyannote. Môi trường nhớ tách biệt.
        """
        logger.info(f"Đang phân tách người nói qua Process độc lập: {audio_path}")
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_pyannote_process, audio_path, self._token)
                timeline = future.result()
            
            logger.info("Phân tách người nói hoàn tất.")
            return timeline
        except Exception as e:
            logger.error(f"Lỗi Process Diarization: {e}")
            raise RuntimeError(f"Lỗi phân tách giọng nói: {e}")
