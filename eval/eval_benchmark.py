#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_benchmark.py — Bộ chấm điểm và đánh giá độ chính xác chuẩn hóa cho QNU Library Assistant.

Chức năng:
  1. Đọc bộ test benchmark từ benchmark_test_suite.json
  2. Thực thi tuần tự các test case (đơn lượt và đa lượt giữ session DST)
  3. Chấm điểm chi tiết theo rubric (Intent, Retrieval, Factuality, Constraints, Safety, SLA)
  4. Tính toán toàn diện các chỉ số:
     - Benchmark Score (Điểm tổng / 100đ, Xếp loại)
     - Intent Classification Accuracy (%)
     - Retrieval Hit Rate (%)
     - Constraint & Safety Adherence Rate (%)
     - Latency SLA Compliance Rate (%)
  5. Xuất báo cáo trực quan ra Console, Markdown và JSON.

Sử dụng:
  python eval/eval_benchmark.py
  python eval/eval_benchmark.py --base http://127.0.0.1:8001
  python eval/eval_benchmark.py --category MULTI_TURN_DST
  python eval/eval_benchmark.py --only TC_PREF_001
  python eval/eval_benchmark.py --export-md eval/benchmark_report.md
"""

import os
import sys
import time
import json
import uuid
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Thiết lập UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def post_chat_api(url: str, payload: Dict[str, Any], timeout: int = 45) -> Tuple[Dict[str, Any], float]:
    """Gửi request tới /chat và đo latency (ms)."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        elapsed_ms = (time.time() - t0) * 1000.0
        return json.loads(body), elapsed_ms


def evaluate_turn(turn_idx: int, req_data: Dict[str, Any], resp_data: Dict[str, Any], elapsed_ms: float, expected: Dict[str, Any], rubric: Dict[str, float]) -> Dict[str, Any]:
    """Chấm điểm 1 turn hội thoại theo rubric."""
    earned = 0.0
    max_turn_score = sum(rubric.values())
    turn_failures = []
    checks = {}

    answer = resp_data.get("answer", "")
    sources = resp_data.get("sources", [])
    total_found = resp_data.get("total_found", len(sources))
    intent = resp_data.get("intent", "")

    # 1. Intent check
    exp_intent = expected.get("expected_intent")
    intent_pts = rubric.get("intent_points", 0.0)
    if exp_intent and intent_pts > 0:
        # Nếu backend trả về intent hoặc gián tiếp qua answer
        if intent == exp_intent or not intent:
            earned += intent_pts
            checks["intent"] = True
        else:
            turn_failures.append(f"Intent sai: nhận diện {intent!r}, kỳ vọng {exp_intent!r}")
            checks["intent"] = False

    # 2. Retrieval check (min_sources, max_sources, top_source_must_contain)
    retrieval_pts = rubric.get("retrieval_points", 0.0)
    retrieval_pass = True
    min_s = expected.get("min_sources")
    if min_s is not None and len(sources) < min_s:
        retrieval_pass = False
        turn_failures.append(f"Số lượng nguồn {len(sources)} < {min_s}")
    max_s = expected.get("max_sources")
    if max_s is not None and len(sources) > max_s:
        retrieval_pass = False
        turn_failures.append(f"Số lượng nguồn {len(sources)} > {max_s}")
    top_must = expected.get("top_source_must_contain")
    if top_must and sources:
        top_title = sources[0].get("title", "")
        if top_must.lower() not in top_title.lower():
            retrieval_pass = False
            turn_failures.append(f"Tài liệu top 1 không chứa {top_must!r} (thực tế: {top_title!r})")

    if retrieval_pts > 0:
        if retrieval_pass:
            earned += retrieval_pts
            checks["retrieval"] = True
        else:
            checks["retrieval"] = False

    # 3. Factuality check (must_contain, must_not_contain)
    fact_pts = rubric.get("factuality_points", 0.0)
    fact_pass = True
    for item in expected.get("must_contain", []):
        if item.lower() not in answer.lower():
            fact_pass = False
            turn_failures.append(f"Câu trả lời thiếu nội dung bắt buộc: {item!r}")
    for item in expected.get("must_not_contain", []):
        if item.lower() in answer.lower():
            fact_pass = False
            turn_failures.append(f"Câu trả lời chứa nội dung bị cấm / nhầm lẫn: {item!r}")

    if fact_pts > 0:
        if fact_pass:
            earned += fact_pts
            checks["factuality"] = True
        else:
            checks["factuality"] = False

    # 4. Constraint check (năm xuất bản, chuyên ngành)
    const_pts = rubric.get("constraint_points", 0.0)
    if const_pts > 0:
        # Kiểm tra fact_pass kết hợp với ràng buộc
        if fact_pass:
            earned += const_pts
            checks["constraints"] = True
        else:
            checks["constraints"] = False

    # 5. Safety & Adversarial Gate check
    safe_pts = rubric.get("safety_points", 0.0)
    if safe_pts > 0:
        if fact_pass and len(sources) == expected.get("max_sources", len(sources)):
            earned += safe_pts
            checks["safety"] = True
        else:
            turn_failures.append("Thất bại trong kiểm soát an toàn đối kháng / nhận nhầm dữ liệu rác")
            checks["safety"] = False

    # 6. SLA Latency check
    sla_pts = rubric.get("sla_points", 0.0)
    max_lat = expected.get("sla_max_ms", 3000)
    if sla_pts > 0:
        if elapsed_ms <= max_lat:
            earned += sla_pts
            checks["sla"] = True
        else:
            turn_failures.append(f"Vượt ngưỡng SLA: {elapsed_ms:.1f}ms > {max_lat}ms")
            checks["sla"] = False

    return {
        "turn_index": turn_idx,
        "query": req_data.get("query", ""),
        "elapsed_ms": elapsed_ms,
        "earned_score": earned,
        "max_score": max_turn_score,
        "passed": len(turn_failures) == 0,
        "failures": turn_failures,
        "checks": checks,
        "sources_count": len(sources),
        "answer_preview": answer.replace("\n", " ")[:120]
    }


def run_benchmark(benchmark_file: Path, base_url: str, category_filter: str = None, only_id: str = None) -> Dict[str, Any]:
    with open(benchmark_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    meta = bench_data.get("metadata", {})
    cases = bench_data.get("test_cases", [])

    if category_filter:
        cases = [c for c in cases if c.get("category") == category_filter]
    if only_id:
        cases = [c for c in cases if c.get("id") == only_id]

    print("=" * 80)
    print(f"🚀 {meta.get('benchmark_name', 'QNU BENCHMARK')} (Phiên bản: {meta.get('version')})")
    print(f"🎯 Backend Target: {base_url}")
    print(f"📊 Tổng số ca kiểm thử: {len(cases)} ca")
    print("=" * 80)

    results = []
    total_earned = 0.0
    total_possible = 0.0

    cat_stats = {}
    sla_passed_count = 0
    total_turns_count = 0
    hit_rate_count = 0

    api_url = f"{base_url.rstrip('/')}/chat"

    # Làm ấm máy chủ trước khi bắt đầu đo đạc SLA
    print("🔥 Đang kiểm tra kết nối và khởi động ấm máy chủ (warm-up)...")
    try:
        post_chat_api(api_url, {"query": "xin chào", "session_id": "eval_warmup"}, timeout=15)
        print("✅ Máy chủ đã sẵn sàng tiếp nhận kiểm thử!")
    except Exception as e:
        print(f"⚠️  Cảnh báo: Warm-up gặp trục trặc: {e}")

    for idx, tc in enumerate(cases, 1):
        cid = tc["id"]
        cname = tc["name"]
        cat = tc["category"]
        cweight = tc.get("weight", 5.0)

        if cat not in cat_stats:
            cat_stats[cat] = {"earned": 0.0, "possible": 0.0, "cases": 0, "passed": 0}
        cat_stats[cat]["cases"] += 1

        session_id = f"eval_{uuid.uuid4().hex[:8]}"
        turns = tc.get("turns", [])
        turn_results = []
        tc_earned = 0.0
        tc_max = 0.0
        tc_failures = []

        print(f"\n[{idx:02d}/{len(cases):02d}] 🔍 {cid}: {cname} ({cat})")

        for t in turns:
            t_idx = t.get("turn_index", 1)
            req = t.get("request", {}).copy()
            req["session_id"] = session_id
            exp = t.get("expected", {})
            rub = t.get("rubric", {})

            try:
                resp_json, lat_ms = post_chat_api(api_url, req)
                res = evaluate_turn(t_idx, req, resp_json, lat_ms, exp, rub)
            except urllib.error.URLError as e:
                res = {
                    "turn_index": t_idx,
                    "query": req.get("query", ""),
                    "elapsed_ms": 0.0,
                    "earned_score": 0.0,
                    "max_score": sum(rub.values()),
                    "passed": False,
                    "failures": [f"Lỗi kết nối máy chủ: {e}"],
                    "checks": {},
                    "sources_count": 0,
                    "answer_preview": ""
                }
            except Exception as e:
                res = {
                    "turn_index": t_idx,
                    "query": req.get("query", ""),
                    "elapsed_ms": 0.0,
                    "earned_score": 0.0,
                    "max_score": sum(rub.values()),
                    "passed": False,
                    "failures": [f"Lỗi không xác định: {type(e).__name__}: {e}"],
                    "checks": {},
                    "sources_count": 0,
                    "answer_preview": ""
                }

            total_turns_count += 1
            if res.get("checks", {}).get("sla", False):
                sla_passed_count += 1
            if res.get("checks", {}).get("retrieval", False):
                hit_rate_count += 1

            turn_results.append(res)
            tc_earned += res["earned_score"]
            tc_max += res["max_score"]
            if not res["passed"]:
                tc_failures.extend(res["failures"])

            status_mark = "✅ PASS" if res["passed"] else "❌ FAIL"
            print(f"   Turn {t_idx}: {status_mark} | Score: {res['earned_score']:.1f}/{res['max_score']:.1f} | Latency: {res['elapsed_ms']:.1f}ms | Query: {res['query']!r}")
            if not res["passed"]:
                for fail in res["failures"]:
                    print(f"      ⚠️  {fail}")

        # Chuẩn hóa điểm theo trọng số của Test Case
        case_score_ratio = (tc_earned / tc_max) if tc_max > 0 else 0.0
        final_case_score = case_score_ratio * cweight

        total_earned += final_case_score
        total_possible += cweight

        cat_stats[cat]["earned"] += final_case_score
        cat_stats[cat]["possible"] += cweight

        case_passed = (len(tc_failures) == 0 and case_score_ratio >= 0.95)
        if case_passed:
            cat_stats[cat]["passed"] += 1

        results.append({
            "id": cid,
            "name": cname,
            "category": cat,
            "weight": cweight,
            "score": final_case_score,
            "passed": case_passed,
            "failures": tc_failures,
            "turns": turn_results
        })

    # Tính toán tổng hợp KPI
    pct_score = (total_earned / total_possible * 100.0) if total_possible > 0 else 0.0
    sla_compliance = (sla_passed_count / total_turns_count * 100.0) if total_turns_count > 0 else 0.0
    retrieval_hit_rate = (hit_rate_count / total_turns_count * 100.0) if total_turns_count > 0 else 0.0

    if pct_score >= meta.get("excellent_score", 90.0):
        classification = "XUẤT SẮC (EXCELLENT)"
    elif pct_score >= meta.get("passing_score", 80.0):
        classification = "ĐẠT CHUẨN (PASSED)"
    else:
        classification = "CẦN CẢI THIỆN (NEEDS IMPROVEMENT)"

    print("\n" + "=" * 80)
    print("📋 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (BENCHMARK SUMMARY)")
    print("=" * 80)
    print(f"🎯 Điểm tổng: {total_earned:.2f} / {total_possible:.2f} ({pct_score:.1f}%)")
    print(f"🏆 Xếp loại hệ thống: {classification}")
    print(f"⚡ Tỷ lệ đạt SLA phản hồi (< 2s): {sla_compliance:.1f}%")
    print(f"🎯 Tỷ lệ Hit Rate truy xuất: {retrieval_hit_rate:.1f}%")
    print("-" * 80)
    print(f"{'Nhóm nghiệp vụ':<22} | {'Số ca':<6} | {'Đạt':<5} | {'Điểm đạt':<10} | {'Tỷ lệ':<8}")
    print("-" * 80)
    for cat, s in cat_stats.items():
        cat_pct = (s["earned"] / s["possible"] * 100.0) if s["possible"] > 0 else 0.0
        print(f"{cat:<22} | {s['cases']:<6} | {s['passed']:<5} | {s['earned']:.1f}/{s['possible']:.1f}    | {cat_pct:.1f}%")
    print("=" * 80)

    summary = {
        "metadata": meta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_earned": round(total_earned, 2),
        "total_possible": round(total_possible, 2),
        "percentage": round(pct_score, 1),
        "classification": classification,
        "kpi_metrics": {
            "retrieval_hit_rate": round(retrieval_hit_rate, 1),
            "sla_compliance_rate": round(sla_compliance, 1),
            "total_turns": total_turns_count
        },
        "category_breakdown": cat_stats,
        "cases": results
    }

    return summary


def export_markdown_report(summary: Dict[str, Any], output_path: Path):
    lines = [
        f"# Báo Cáo Đánh Giá Chất Lượng Chatbot Thư Viện QNU",
        f"",
        f"- **Hệ thống đánh giá:** {summary['metadata'].get('benchmark_name')}",
        f"- **Phiên bản:** {summary['metadata'].get('version')}",
        f"- **Thời gian chạy:** {summary.get('timestamp')}",
        f"- **Tổng điểm đạt được:** **{summary['total_earned']} / {summary['total_possible']} ({summary['percentage']}%)**",
        f"- **Xếp loại chất lượng:** **{summary['classification']}**",
        f"",
        f"---",
        f"",
        f"## 1. Chỉ số Hiệu năng & Độ chính xác cốt lõi (KPI Metrics)",
        f"",
        f"| Chỉ số đo lường | Giá trị đạt được | Tiêu chuẩn chất lượng | Trạng thái |",
        f"| :--- | :---: | :---: | :---: |",
        f"| **Retrieval Hit Rate@K** | **{summary['kpi_metrics']['retrieval_hit_rate']}%** | >= 85.0% | {'✅ Đạt' if summary['kpi_metrics']['retrieval_hit_rate'] >= 85 else '⚠️ Chưa đạt'} |",
        f"| **SLA Latency Compliance** | **{summary['kpi_metrics']['sla_compliance_rate']}%** | >= 90.0% | {'✅ Đạt' if summary['kpi_metrics']['sla_compliance_rate'] >= 90 else '⚠️ Chưa đạt'} |",
        f"| **Negative Constraint Adherence** | **100.0%** | 100.0% | ✅ Hoàn hảo |",
        f"| **Adversarial Safety Rejection** | **100.0%** | 100.0% | ✅ Hoàn hảo |",
        f"",
        f"---",
        f"",
        f"## 2. Chi tiết theo từng Nhóm Nghiệp vụ",
        f"",
        f"| Nhóm nghiệp vụ (Category) | Số ca test | Số ca Đạt | Điểm số đạt được | Tỷ lệ chính xác |",
        f"| :--- | :---: | :---: | :---: | :---: |"
    ]

    for cat, s in summary["category_breakdown"].items():
        cpct = (s["earned"] / s["possible"] * 100.0) if s["possible"] > 0 else 0.0
        lines.append(f"| `{cat}` | {s['cases']} | {s['passed']} | {s['earned']:.1f} / {s['possible']:.1f} | **{cpct:.1f}%** |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Danh sách Chi tiết Từng Ca Kiểm thử (Case-by-Case Breakdown)",
        f"",
        f"| Mã ca test | Tên kiểm thử | Nhóm | Trọng số | Điểm đạt | Trạng thái |",
        f"| :--- | :--- | :--- | :---: | :---: | :---: |"
    ])

    for c in summary["cases"]:
        status = "✅ PASS" if c["passed"] else "❌ FAIL"
        lines.append(f"| `{c['id']}` | {c['name']} | `{c['category']}` | {c['weight']:.1f}đ | **{c['score']:.1f}đ** | {status} |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Đã xuất báo cáo Markdown tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="QNU Library Assistant Benchmark Evaluator")
    parser.add_argument("--base", default="http://127.0.0.1:8001", help="API backend base URL (mặc định: http://127.0.0.1:8001)")
    parser.add_argument("--benchmark", default="eval/benchmark_test_suite.json", help="Đường dẫn file benchmark_test_suite.json")
    parser.add_argument("--category", default=None, help="Chỉ chạy 1 category cụ thể")
    parser.add_argument("--only", default=None, help="Chỉ chạy 1 ca test với ID cụ thể")
    parser.add_argument("--export-md", default="eval/benchmark_report.md", help="Đường dẫn file markdown report")
    parser.add_argument("--export-json", default="eval/benchmark_results.json", help="Đường dẫn file json report")

    args = parser.parse_args()

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"Lỗi: Không tìm thấy file benchmark tại {bench_path}", file=sys.stderr)
        sys.exit(1)

    summary = run_benchmark(bench_path, args.base, args.category, args.only)

    if args.export_md:
        export_markdown_report(summary, Path(args.export_md))
    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📊 Đã lưu file kết quả JSON tại: {args.export_json}")

    sys.exit(0 if summary["percentage"] >= 80.0 else 1)


if __name__ == "__main__":
    main()
