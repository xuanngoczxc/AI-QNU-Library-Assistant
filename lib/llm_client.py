"""
lib/llm_client.py — OpenRouter LLM client with retry and circuit breaker.
"""

import time
import requests
from typing import Optional

from fastapi import HTTPException

from .config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_BASE_URL, OPENROUTER_APP_TITLE, OPENROUTER_HTTP_REFERER,
)
from .state import llm_stats, llm_circuit
from .logging_setup import logger


def record_llm_failure(error_msg: str) -> None:
    """Track LLM failure, open circuit breaker if threshold reached."""
    llm_stats["failures"] += 1
    llm_stats["last_error"] = error_msg
    llm_circuit["failures"] += 1
    if (
        llm_circuit["state"] in ("closed", "half_open")
        and llm_circuit["failures"] >= llm_circuit["failure_threshold"]
    ):
        llm_circuit["state"] = "open"
        llm_circuit["opened_at"] = time.time()
        logger.error(
            "LLM circuit → OPEN (failures=%d >= threshold=%d, cooldown=%ss)",
            llm_circuit["failures"],
            llm_circuit["failure_threshold"],
            llm_circuit["cooldown_seconds"],
        )


def check_backend() -> None:
    """Validate OpenRouter API key at startup."""
    if not OPENROUTER_API_KEY:
        from .config import is_vercel_runtime
        if is_vercel_runtime():
            logger.warning("OPENROUTER_API_KEY not set at build time — will check at runtime")
            return
        raise RuntimeError("OPENROUTER_API_KEY is required")
    logger.info("OpenRouter ready | Model: %s", OPENROUTER_MODEL)


def call_llm(messages: list, temperature: float = 0.3) -> str:
    """
    Call OpenRouter with retry, fallback models, and circuit breaker.
    Returns the response text content.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")

    cb = llm_circuit
    if cb["state"] == "open":
        if time.time() - cb["opened_at"] < cb["cooldown_seconds"]:
            logger.warning(
                "LLM circuit OPEN, refusing call (cooldown %ss left)",
                int(cb["cooldown_seconds"] - (time.time() - cb["opened_at"])),
            )
            raise HTTPException(503, "LLM service tạm thời không khả dụng, vui lòng thử lại sau ít phút.")
        cb["state"] = "half_open"
        logger.info("LLM circuit → half_open (retry attempt)")

    llm_stats["calls"] += 1
    request_start = time.time()
    last_error: Optional[str] = None

    model_chain = [OPENROUTER_MODEL] + list(OPENROUTER_FALLBACK_MODELS)
    max_retries = 3

    for model_idx, model_name in enumerate(model_chain):
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                        "X-Title": OPENROUTER_APP_TITLE,
                    },
                    json={"model": model_name, "messages": messages, "temperature": temperature},
                    timeout=300,
                )

                if response.status_code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    llm_stats["retries"] += 1
                    logger.warning("Rate limited on %s (429), retrying in %ss (attempt %d/%d)", model_name, wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue

                if response.status_code in (400, 404):
                    last_error = f"Model '{model_name}' invalid [{response.status_code}]: {response.text[:200]}"
                    logger.warning("LLM %s → fallback to next model: %s", model_name, last_error)
                    break

                if response.status_code != 200:
                    last_error = f"OpenRouter error [{response.status_code}]: {response.text[:200]}"
                    record_llm_failure(last_error)
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        llm_stats["retries"] += 1
                        logger.warning("LLM error, retrying in %ss (attempt %d/%d): %s", wait, attempt + 1, max_retries, last_error)
                        time.sleep(wait)
                        continue
                    break

                payload = response.json()
                content = payload["choices"][0]["message"].get("content", "")
                elapsed = time.time() - request_start
                tag = "primary" if model_idx == 0 else f"fallback#{model_idx}"
                logger.info("LLM call OK in %.2fs (%s, model=%s, len=%d)", elapsed, tag, model_name, len(content))
                if cb["state"] in ("open", "half_open"):
                    logger.info("LLM circuit → closed (recovered)")
                cb["state"] = "closed"
                cb["failures"] = 0
                return content

            except HTTPException:
                raise
            except Exception as e:
                last_error = str(e)[:200]
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    llm_stats["retries"] += 1
                    logger.warning("LLM error on %s, retrying in %ss (attempt %d/%d): %s", model_name, wait, attempt + 1, max_retries, last_error)
                    time.sleep(wait)
                    continue
                break

    record_llm_failure(last_error or "All models failed")
    logger.error("LLM call FAILED after %d models in %.2fs: %s", len(model_chain), time.time() - request_start, last_error)
    raise HTTPException(503, f"LLM API error: {last_error}")
