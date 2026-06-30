"""
lib/config.py — Environment variables, API keys, feature flags.
Zero business logic, zero FastAPI imports.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
OPENROUTER_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "meta-llama/llama-3.3-70b-instruct:free,"
        "qwen/qwen-2.5-72b-instruct:free,"
        "mistralai/mistral-small-3.2-24b-instruct:free,"
        "google/gemini-2.0-flash-exp:free,"
        "openai/gpt-oss-20b:free"
    ).split(",")
    if m.strip()
]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "QNU Library Assistant")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123").strip()

LLM_CIRCUIT_THRESHOLD = int(os.getenv("LLM_CIRCUIT_THRESHOLD", "5"))
LLM_CIRCUIT_COOLDOWN = int(os.getenv("LLM_CIRCUIT_COOLDOWN", "60"))

MAX_HISTORY = 3

VERCEL_READ_ONLY_MESSAGE = (
    "Bản demo trên Vercel chạy ở chế độ read-only: không thể upload hoặc xóa PDF "
    "vì Vercel Functions chỉ có filesystem tạm thời. Hãy deploy lên server riêng "
    "hoặc dùng storage ngoài nếu cần chức năng này."
)


def is_vercel_runtime() -> bool:
    return os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))
