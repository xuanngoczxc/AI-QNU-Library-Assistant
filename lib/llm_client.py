"""
lib/llm_client.py — OpenRouter LLM client with async support, true SSE streaming, retry, and circuit breaker.
"""

import time
import json
import asyncio
import httpx
from typing import Optional, AsyncGenerator

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


def _check_circuit_breaker() -> None:
    """Check if the circuit breaker is open, raise 503 if cooling down."""
    cb = llm_circuit
    if cb["state"] == "open":
        elapsed = time.time() - cb["opened_at"]
        if elapsed < cb["cooldown_seconds"]:
            cooldown_left = int(cb["cooldown_seconds"] - elapsed)
            logger.warning(
                "LLM circuit OPEN, refusing call (cooldown %ss left)",
                cooldown_left,
            )
            raise HTTPException(503, "LLM service tạm thời không khả dụng, vui lòng thử lại sau ít phút.")
        cb["state"] = "half_open"
        logger.info("LLM circuit → half_open (retry attempt)")


def _on_llm_success(model_name: str, elapsed: float, tag: str, length: int) -> None:
    """Reset circuit breaker and log successful call."""
    logger.info("LLM call OK in %.2fs (%s, model=%s, len=%d)", elapsed, tag, model_name, length)
    if llm_circuit["state"] in ("open", "half_open"):
        logger.info("LLM circuit → closed (recovered)")
    llm_circuit["state"] = "closed"
    llm_circuit["failures"] = 0


def check_backend() -> None:
    """Validate OpenRouter API key at startup."""
    if not OPENROUTER_API_KEY:
        from .config import is_vercel_runtime
        if is_vercel_runtime():
            logger.warning("OPENROUTER_API_KEY not set at build time — will check at runtime")
            return
        raise RuntimeError("OPENROUTER_API_KEY is required")
    logger.info("OpenRouter ready | Model: %s", OPENROUTER_MODEL)


def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }


async def call_llm_async(messages: list, temperature: float = 0.3) -> str:
    """
    Non-blocking async call to OpenRouter with retry, fallback models, and circuit breaker.
    Uses httpx.AsyncClient so FastAPI's event loop is never blocked.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")

    _check_circuit_breaker()
    llm_stats["calls"] += 1
    request_start = time.time()
    last_error: Optional[str] = None

    model_chain = [OPENROUTER_MODEL] + list(OPENROUTER_FALLBACK_MODELS)
    max_retries = 3

    timeout_config = httpx.Timeout(120.0, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        for model_idx, model_name in enumerate(model_chain):
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers=_get_headers(),
                        json={"model": model_name, "messages": messages, "temperature": temperature},
                    )

                    if response.status_code == 429 and attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        llm_stats["retries"] += 1
                        logger.warning("Rate limited on %s (429), retrying in %ss (attempt %d/%d)", model_name, wait, attempt + 1, max_retries)
                        await asyncio.sleep(wait)
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
                            await asyncio.sleep(wait)
                            continue
                        break

                    try:
                        payload = response.json()
                    except Exception:
                        raw_clean = response.text.split("data: [DONE]")[0].strip()
                        payload = json.loads(raw_clean)
                    content = payload["choices"][0]["message"].get("content", "") or ""
                    elapsed = time.time() - request_start
                    tag = "primary" if model_idx == 0 else f"fallback#{model_idx}"
                    _on_llm_success(model_name, elapsed, tag, len(content))
                    return content

                except HTTPException:
                    raise
                except Exception as e:
                    last_error = str(e)[:200]
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        llm_stats["retries"] += 1
                        logger.warning("LLM error on %s, retrying in %ss (attempt %d/%d): %s", model_name, wait, attempt + 1, max_retries, last_error)
                        await asyncio.sleep(wait)
                        continue
                    break

    record_llm_failure(last_error or "All models failed")
    logger.error("LLM call FAILED after %d models in %.2fs: %s", len(model_chain), time.time() - request_start, last_error)
    raise HTTPException(503, f"LLM API error: {last_error}")


async def stream_llm_async(messages: list, temperature: float = 0.3) -> AsyncGenerator[str, None]:
    """
    True SSE streaming generator from OpenRouter.
    Yields text delta chunks in real-time as the LLM produces them.
    Falls back to next model if initial stream connection fails.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")

    _check_circuit_breaker()
    llm_stats["calls"] += 1
    request_start = time.time()
    last_error: Optional[str] = None

    model_chain = [OPENROUTER_MODEL] + list(OPENROUTER_FALLBACK_MODELS)
    timeout_config = httpx.Timeout(120.0, connect=15.0)

    for model_idx, model_name in enumerate(model_chain):
        try:
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                async with client.stream(
                    "POST",
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=_get_headers(),
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        last_error = f"Model {model_name} HTTP {response.status_code}: {err_body.decode(errors='replace')[:200]}"
                        logger.warning("Stream failed on %s: %s -> trying fallback", model_name, last_error)
                        record_llm_failure(last_error)
                        continue

                    # Stream succeeded, yield chunks
                    streamed_total = 0
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str or data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    text_chunk = delta.get("content", "")
                                    if text_chunk:
                                        streamed_total += len(text_chunk)
                                        yield text_chunk
                            except Exception:
                                continue

                    elapsed = time.time() - request_start
                    tag = "primary" if model_idx == 0 else f"fallback#{model_idx}"
                    _on_llm_success(model_name, elapsed, tag, streamed_total)
                    return

        except Exception as e:
            last_error = f"Stream exception on {model_name}: {str(e)[:200]}"
            logger.warning("%s -> trying fallback", last_error)
            record_llm_failure(last_error)
            continue

    record_llm_failure(last_error or "All stream models failed")
    logger.error("LLM stream FAILED after %d models in %.2fs: %s", len(model_chain), time.time() - request_start, last_error)
    raise HTTPException(503, f"LLM Stream API error: {last_error}")


def call_llm(messages: list, temperature: float = 0.3) -> str:
    """
    Synchronous fallback wrapper for call_llm (keeps backwards compatibility for sync code).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If already in an event loop, run in a separate thread to prevent event loop lock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, call_llm_async(messages, temperature)).result()
    else:
        return asyncio.run(call_llm_async(messages, temperature))
