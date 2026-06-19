"""
voice_input.py — Giai đoạn 3: Voice input với Whisper
Speech-to-text cho phép user nói thay vì gõ.

Hỗ trợ 2 backend:
1. faster-whisper (khuyến nghị, nhanh hơn, ít RAM hơn)
2. openai-whisper (chính thức, dễ cài)

Cài:
  pip install faster-whisper
  # hoặc: pip install openai-whisper
"""

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict

# Lazy load model
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()
_WHISPER_BACKEND = None
_WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")  # tiny/base/small/medium
_WHISPER_LANG = os.getenv("WHISPER_LANG", "vi")  # mặc định tiếng Việt


def _load_whisper():
    """Lazy load Whisper model (chỉ load lần đầu)."""
    global _WHISPER_MODEL, _WHISPER_BACKEND

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL, _WHISPER_BACKEND

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL, _WHISPER_BACKEND

        # Thử faster-whisper trước (nhanh hơn)
        try:
            from faster_whisper import WhisperModel
            print(f"  ⏳ Loading faster-whisper model: {_WHISPER_MODEL_NAME}...")
            _WHISPER_MODEL = WhisperModel(_WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
            _WHISPER_BACKEND = "faster-whisper"
            print(f"  ✅ faster-whisper ready")
            return _WHISPER_MODEL, _WHISPER_BACKEND
        except ImportError:
            pass

        # Fallback: openai-whisper
        try:
            import whisper
            print(f"  ⏳ Loading openai-whisper model: {_WHISPER_MODEL_NAME}...")
            _WHISPER_MODEL = whisper.load_model(_WHISPER_MODEL_NAME)
            _WHISPER_BACKEND = "openai-whisper"
            print(f"  ✅ openai-whisper ready")
            return _WHISPER_MODEL, _WHISPER_BACKEND
        except ImportError:
            pass

        print("  ⚠️ No Whisper backend installed. Run:")
        print("     pip install faster-whisper")
        print("     # hoặc: pip install openai-whisper")
        return None, None


def is_available() -> bool:
    """Check voice input có khả dụng không."""
    model, _ = _load_whisper()
    return model is not None


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Dict:
    """
    Transcribe audio file → text.

    Args:
        audio_path: đường dẫn file audio (wav, mp3, m4a, webm, ogg...)
        language: mã ngôn ngữ ISO (vi, en, ...). None = auto-detect
        initial_prompt: gợi ý ngữ cảnh để tăng độ chính xác
                        (VD: "Thư viện QNU, sách, tài liệu học thuật")

    Returns:
        dict với: text, language, language_probability, duration, segments
    """
    model, backend = _load_whisper()
    if not model:
        return {
            "text": "",
            "error": "Whisper chưa được cài đặt. Chạy: pip install faster-whisper",
        }

    if not Path(audio_path).exists():
        return {"text": "", "error": f"File không tồn tại: {audio_path}"}

    lang = language or _WHISPER_LANG
    prompt = initial_prompt or "Thư viện QNU, sách giáo trình, tài liệu học thuật, nghiên cứu khoa học."

    try:
        if backend == "faster-whisper":
            segments, info = model.transcribe(
                audio_path,
                language=lang,
                initial_prompt=prompt,
                beam_size=5,
                vad_filter=True,  # lọc im lặng
            )
            segments_list = list(segments)
            text = " ".join([s.text.strip() for s in segments_list]).strip()
            return {
                "text": text,
                "language": info.language,
                "language_probability": float(info.language_probability),
                "duration": float(info.duration),
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in segments_list
                ],
                "backend": "faster-whisper",
            }
        else:  # openai-whisper
            result = model.transcribe(
                audio_path,
                language=lang,
                initial_prompt=prompt,
                fp16=False,
            )
            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", lang),
                "language_probability": 1.0,
                "duration": 0.0,
                "segments": result.get("segments", []),
                "backend": "openai-whisper",
            }
    except Exception as e:
        return {"text": "", "error": str(e)}


def save_uploaded_audio(audio_bytes: bytes, suffix: str = ".webm") -> str:
    """
    Lưu audio bytes từ frontend vào file tạm.
    Trả về đường dẫn file.
    """
    tmp_dir = Path(tempfile.gettempdir()) / "qnu_voice"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"voice_input_{os.getpid()}_{suffix.lstrip('.')}"
    tmp_path = tmp_path.with_suffix(suffix)
    tmp_path.write_bytes(audio_bytes)
    return str(tmp_path)


def cleanup_audio(path: str) -> None:
    """Xoá file audio tạm sau khi xử lý."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
