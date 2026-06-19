#!/usr/bin/env python
"""
Run smoke-test queries against /chat endpoint and verify expectations.

Usage:
    python eval/run_eval.py                       # uses defaults (localhost:8000)
    python eval/run_eval.py --base http://host:p  # custom base
    python eval/run_eval.py --only content_001    # run single case
    python eval/run_eval.py --verbose             # print full response

Verifies:
    1. HTTP 200 from /chat
    2. Response contains expected substrings
    3. LLM call counter (via /stats) increments when expect_llm_call=true
    4. Source count meets min_sources threshold
    5. No unexpected substring in answer
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

# Force UTF-8 on stdout/stderr so the ✓/✗ icons don't crash under Windows
# cp1252 console encoding (Tee-Object, piped output, etc.).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")


def post_json(url: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    """POST JSON with UTF-8 body (avoid charmap issues on Windows)."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 10) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cases(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def run_case(case: Dict[str, Any], base: str, baseline_llm_calls: int, verbose: bool) -> tuple:
    """Run one test case. Returns (status: pass/fail/skip, message)."""
    cid = case["id"]
    req = case["request"]
    expect = case.get("expect", {})

    url = f"{base.rstrip('/')}/chat"
    t0 = time.time()
    try:
        resp = post_json(url, req, timeout=180)
    except urllib.error.HTTPError as e:
        return "fail", f"{cid}: HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return "fail", f"{cid}: connection error: {e.reason}"
    except Exception as e:
        return "fail", f"{cid}: {type(e).__name__}: {e}"
    elapsed = time.time() - t0

    answer = resp.get("answer", "")
    sources = resp.get("sources", [])
    summary = resp.get("summary", "")
    failures = []

    # 1. min_sources
    min_src = expect.get("min_sources")
    if min_src is not None and len(sources) < min_src:
        failures.append(f"sources {len(sources)} < min_sources {min_src}")

    # 2. expect_min_sources (alias for "may have zero")
    min0 = expect.get("expect_min_sources")
    if min0 is not None and len(sources) < min0:
        failures.append(f"sources {len(sources)} < expect_min_sources {min0}")

    # 3. expect_answer_contains
    for needle in expect.get("expect_answer_contains", []) or []:
        if needle not in answer:
            failures.append(f"answer missing substring: {needle!r}")

    # 4. expect_answer_not_contains
    for needle in expect.get("expect_answer_not_contains", []) or []:
        if needle in answer:
            failures.append(f"answer contains forbidden substring: {needle!r}")

    # 5. expect_summary_min_len
    min_summary = expect.get("expect_summary_min_len")
    if min_summary is not None and len(summary) < min_summary:
        failures.append(f"summary len {len(summary)} < min {min_summary}")

    # 6. LLM call counter check
    if expect.get("expect_llm_call"):
        try:
            stats = get_json(f"{base.rstrip('/')}/stats")
            current = stats.get("llm", {}).get("calls", 0)
            if current <= baseline_llm_calls:
                failures.append(
                    f"LLM not called: stats.calls={current}, baseline={baseline_llm_calls}"
                )
        except Exception as e:
            failures.append(f"could not fetch /stats: {e}")

    status = "pass" if not failures else "fail"
    msg_parts = [f"{cid}: {status}", f"elapsed={elapsed:.2f}s", f"answer_len={len(answer)}", f"sources={len(sources)}"]
    if failures:
        msg_parts.append("FAILURES: " + "; ".join(failures))
    if verbose:
        msg_parts.append(f"\n  answer={answer[:200]!r}{'...' if len(answer) > 200 else ''}")
        msg_parts.append(f"\n  sources[0]={sources[0] if sources else None}")
    return status, "\n  ".join(msg_parts)


def main():
    p = argparse.ArgumentParser(description="QNU Library Assistant eval runner")
    p.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    p.add_argument("--queries", default="eval/sample_queries.json", help="path to sample_queries.json")
    p.add_argument("--only", default=None, help="run only case with this id")
    p.add_argument("--verbose", action="store_true", help="print full responses")
    args = p.parse_args()

    base = args.base
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"ERROR: queries file not found: {queries_path}", file=sys.stderr)
        sys.exit(2)

    cases = load_cases(queries_path)
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"ERROR: no case with id {args.only!r}", file=sys.stderr)
            sys.exit(2)

    # Baseline LLM counter
    try:
        baseline = get_json(f"{base.rstrip('/')}/stats").get("llm", {}).get("calls", 0)
    except Exception as e:
        print(f"WARNING: could not read baseline /stats: {e}", file=sys.stderr)
        baseline = 0

    print(f"=== Eval runner ===")
    print(f"Base URL:   {base}")
    print(f"Cases:      {len(cases)}")
    print(f"Baseline llm.calls: {baseline}")
    print()

    pass_count = fail_count = skip_count = 0
    for case in cases:
        status, msg = run_case(case, base, baseline, args.verbose)
        icon = {"pass": "✓", "fail": "✗", "skip": "○"}.get(status, "?")
        print(f"[{icon}] {msg}")
        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            skip_count += 1

    print()
    print(f"=== Summary: {pass_count} passed, {fail_count} failed, {skip_count} skipped ===")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
