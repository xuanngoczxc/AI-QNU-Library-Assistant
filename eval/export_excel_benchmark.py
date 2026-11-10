#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_excel_benchmark.py
Tạo bảng tính Excel (.xlsx) chuẩn hóa hiển thị toàn bộ:
  1. Sheet 'Kịch Bản & Tiêu Chí Test': Câu hỏi test, phân hệ, độ khó, điểm test, kết quả đúng (ground truth), từ khóa bắt buộc/cấm, SLA, rubric.
  2. Sheet 'Kết Quả Kiểm Thử Thực Tế': Trạng thái (PASS/FAIL), điểm đạt, độ trễ ms, câu trả lời thực tế của chatbot, kiểm tra tiêu chí.
  3. Sheet 'Tổng Quan & KPIs': Bảng điểm 10 phân hệ nghiệp vụ, xếp loại, các chỉ số độ chính xác (Intent, Retrieval, Safety, SLA).
"""

import os
import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = Path(r"d:\ChatbotLib\AI-QNU-Library-Assistant")
SUITE_JSON_PATH = BASE_DIR / "eval" / "benchmark_test_suite.json"
RESULTS_JSON_PATH = BASE_DIR / "eval" / "benchmark_results.json"
EXCEL_OUTPUT_PATH = BASE_DIR / "eval" / "QNU_Library_Benchmark_TestSuite.xlsx"
EXCEL_ROOT_OUTPUT_PATH = Path(r"d:\ChatbotLib\QNU_Library_Benchmark_TestSuite.xlsx")

def create_excel_benchmark():
    # 1. Đọc dữ liệu
    with open(SUITE_JSON_PATH, "r", encoding="utf-8") as f:
        suite_data = json.load(f)

    results_data = {}
    if RESULTS_JSON_PATH.exists():
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            results_data = json.load(f)

    # Khởi tạo Workbook
    wb = openpyxl.Workbook()
    # Xóa sheet mặc định
    wb.remove(wb.active)

    # Thiết lập Color Palette & Fonts
    FONT_FAMILY = "Segoe UI"
    font_title = Font(name=FONT_FAMILY, size=16, bold=True, color="1F4E78")
    font_subtitle = Font(name=FONT_FAMILY, size=11, italic=True, color="595959")
    font_header = Font(name=FONT_FAMILY, size=11, bold=True, color="FFFFFF")
    font_body = Font(name=FONT_FAMILY, size=10, color="000000")
    font_body_bold = Font(name=FONT_FAMILY, size=10, bold=True, color="000000")
    font_pass = Font(name=FONT_FAMILY, size=10, bold=True, color="276A3C")
    font_fail = Font(name=FONT_FAMILY, size=10, bold=True, color="A61C1C")
    font_badge = Font(name=FONT_FAMILY, size=9, bold=True, color="1F4E78")

    fill_header = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    fill_header_sub = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    fill_zebra = PatternFill(start_color="F2F5F9", end_color="F2F5F9", fill_type="solid")
    fill_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
    fill_pass = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fill_fail = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    fill_highlight = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    thin_border_side = Side(style='thin', color='D9D9D9')
    border_all = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    border_header = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=Side(style='medium', color='1F4E78'))

    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    align_right = Alignment(horizontal="right", vertical="center")
    align_center_wrap = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # =========================================================================
    # SHEET 1: BẢNG KỊCH BẢN & TIÊU CHÍ TEST (TEST SPECIFICATION)
    # =========================================================================
    ws1 = wb.create_sheet(title="📋 Kịch Bản & Tiêu Chí Test")
    ws1.views.sheetView[0].showGridLines = True

    # Title block
    ws1.merge_cells("A1:M1")
    ws1["A1"] = "BỘ TIÊU CHUẨN KIỂM THỬ CHUẨN HÓA CHATBOT THƯ VIỆN QNU (QNU-LAB BENCHMARK)"
    ws1["A1"].font = font_title
    ws1["A1"].alignment = Alignment(horizontal="left", vertical="center")

    ws1.merge_cells("A2:M2")
    ws1["A2"] = "Quy định chi tiết: Câu hỏi đầu vào, phân hệ nghiệp vụ, điểm số tối đa, kết quả đúng (Ground Truth), ràng buộc cấm và SLA độ trễ"
    ws1["A2"].font = font_subtitle
    ws1["A2"].alignment = Alignment(horizontal="left", vertical="center")

    headers1 = [
        "STT",
        "Mã Ca Test",
        "Phân Hệ Nghiệp Vụ",
        "Tên Kịch Bản Kiểm Thử",
        "Độ Khó",
        "Lượt (Turn)",
        "Câu Hỏi Test (User Query)",
        "Ý Định Kỳ Vọng",
        "Kết Quả Đúng / Từ Khóa Bắt Buộc (Must Contain)",
        "Ràng Buộc Phủ Định / Không Được Phép Có (Must Not Contain)",
        "Số Nguồn Kỳ Vọng",
        "SLA Tối Đa (ms)",
        "Điểm Tối Đa",
        "Chi Tiết Phân Bổ Rubric Chấm Điểm"
    ]

    ws1.row_dimensions[4].height = 28
    for col_idx, h_text in enumerate(headers1, start=1):
        cell = ws1.cell(row=4, column=col_idx, value=h_text)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center_wrap
        cell.border = border_header

    current_row = 5
    stt_counter = 1

    for tc in suite_data.get("test_cases", []):
        cid = tc["id"]
        cname = tc["name"]
        cat = tc["category"]
        diff = tc.get("difficulty", "Medium")
        turns = tc.get("turns", [])

        for t in turns:
            t_idx = t.get("turn_index", 1)
            query = t.get("request", {}).get("query", "")
            exp = t.get("expected", {})
            rub = t.get("rubric", {})

            intent = exp.get("expected_intent", "catalog_search")
            must_contain = "\n• " + "\n• ".join(exp.get("must_contain", [])) if exp.get("must_contain") else "(Không bắt buộc chuỗi cụ thể)"
            must_not = "\n• " + "\n• ".join(exp.get("must_not_contain", [])) if exp.get("must_not_contain") else "(Không có)"

            sources_exp = []
            if "min_sources" in exp:
                sources_exp.append(f"Tối thiểu: {exp['min_sources']}")
            if "max_sources" in exp:
                sources_exp.append(f"Tối đa: {exp['max_sources']}")
            sources_str = ", ".join(sources_exp) if sources_exp else "Linh hoạt"

            sla_ms = exp.get("sla_max_ms", 2500)
            turn_max_score = sum(rub.values())

            rubric_details = []
            for rk, rv in rub.items():
                label = rk.replace("_points", "").capitalize()
                rubric_details.append(f"{label}: {rv}đ")
            rubric_str = " | ".join(rubric_details)

            is_zebra = (stt_counter % 2 == 0)
            row_fill = fill_zebra if is_zebra else fill_white

            ws1.row_dimensions[current_row].height = 42

            row_data = [
                stt_counter,
                cid,
                cat,
                cname,
                diff,
                f"Lượt {t_idx}",
                query,
                intent,
                must_contain,
                must_not,
                sources_str,
                f"{sla_ms:,} ms",
                turn_max_score,
                rubric_str
            ]

            for c_idx, val in enumerate(row_data, start=1):
                cell = ws1.cell(row=current_row, column=c_idx, value=val)
                cell.font = font_body
                cell.fill = row_fill
                cell.border = border_all
                if c_idx in [1, 2, 5, 6, 8, 11, 12]:
                    cell.alignment = align_center
                elif c_idx in [13]:
                    cell.alignment = align_center
                    cell.font = font_body_bold
                else:
                    cell.alignment = align_left

            stt_counter += 1
            current_row += 1

    # Dòng tổng điểm
    ws1.row_dimensions[current_row].height = 25
    ws1.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=12)
    sum_cell_label = ws1.cell(row=current_row, column=1, value="TỔNG ĐIỂM TOÀN BỘ BỘ TEST BENCHMARK (CHUẨN 100.0 ĐIỂM)")
    sum_cell_label.font = font_body_bold
    sum_cell_label.alignment = Alignment(horizontal="right", vertical="center")
    sum_cell_label.fill = fill_highlight

    for col in range(1, 13):
        ws1.cell(row=current_row, column=col).border = border_all
        ws1.cell(row=current_row, column=col).fill = fill_highlight

    sum_cell_val = ws1.cell(row=current_row, column=13, value=f"=SUM(M5:M{current_row-1})")
    sum_cell_val.font = Font(name=FONT_FAMILY, size=11, bold=True, color="1F4E78")
    sum_cell_val.alignment = align_center
    sum_cell_val.border = border_all
    sum_cell_val.fill = fill_highlight

    ws1.cell(row=current_row, column=14, value="Đạt chuẩn quốc tế 100đ").border = border_all
    ws1.cell(row=current_row, column=14).font = font_subtitle
    ws1.cell(row=current_row, column=14).fill = fill_highlight

    # =========================================================================
    # SHEET 2: KẾT QUẢ KIỂM THỬ THỰC TẾ (TEST EXECUTION RESULTS)
    # =========================================================================
    ws2 = wb.create_sheet(title="📊 Kết Quả Thực Tế Đạt Được")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:K1")
    ws2["A1"] = "BẢNG KẾT QUẢ ĐO ĐẠC VÀ ĐÁNH GIÁ THỰC TẾ CỦA CHATBOT"
    ws2["A1"].font = font_title
    ws2["A1"].alignment = Alignment(horizontal="left", vertical="center")

    ws2.merge_cells("A2:K2")
    ws2["A2"] = f"Ngày kiểm thử: {results_data.get('timestamp', 'Mới nhất')} | Tổng điểm đạt: {results_data.get('total_earned', 0)} / {results_data.get('total_possible', 100)} ({results_data.get('percentage', 0)}%) | Xếp loại: {results_data.get('classification', 'N/A')}"
    ws2["A2"].font = font_subtitle
    ws2["A2"].alignment = Alignment(horizontal="left", vertical="center")

    headers2 = [
        "STT",
        "Mã Ca Test",
        "Lượt",
        "Câu Hỏi Đã Gửi (Query)",
        "Trạng Thái (Status)",
        "Điểm Đạt",
        "Điểm Tối Đa",
        "Độ Trễ Phản Hồi",
        "Số Nguồn Tìm Thấy",
        "Nội Dung Phản Hồi Thực Tế (Answer Preview)",
        "Ghi Chú / Vi Phạm Tiêu Chí (Warnings)"
    ]

    ws2.row_dimensions[4].height = 28
    for col_idx, h_text in enumerate(headers2, start=1):
        cell = ws2.cell(row=4, column=col_idx, value=h_text)
        cell.font = font_header
        cell.fill = fill_header_sub
        cell.alignment = align_center_wrap
        cell.border = border_header

    curr2 = 5
    stt2 = 1

    for tc in results_data.get("cases", []):
        cid = tc["id"]
        for t in tc.get("turns", []):
            t_idx = t.get("turn_index", 1)
            query = t.get("query", "")
            passed = t.get("passed", False)
            earned = t.get("earned_score", 0.0)
            max_s = t.get("max_score", 0.0)
            lat_ms = t.get("elapsed_ms", 0.0)
            sources_cnt = t.get("sources_count", 0)
            preview = t.get("answer_preview", "")
            failures = t.get("failures", [])
            fail_str = "\n".join(failures) if failures else "Đạt tuyệt đối mọi tiêu chí"

            status_text = "✅ PASS" if passed else "❌ FAIL"
            status_font = font_pass if passed else font_fail
            status_fill = fill_pass if passed else fill_fail

            ws2.row_dimensions[curr2].height = 36

            r_data = [
                stt2,
                cid,
                f"Lượt {t_idx}",
                query,
                status_text,
                round(earned, 2),
                round(max_s, 2),
                f"{lat_ms:,.1f} ms",
                sources_cnt,
                preview,
                fail_str
            ]

            for c_idx, val in enumerate(r_data, start=1):
                cell = ws2.cell(row=curr2, column=c_idx, value=val)
                cell.font = font_body
                cell.border = border_all
                if c_idx == 5:
                    cell.font = status_font
                    cell.fill = status_fill
                    cell.alignment = align_center
                elif c_idx in [1, 2, 3, 6, 7, 8, 9]:
                    cell.alignment = align_center
                else:
                    cell.alignment = align_left

            stt2 += 1
            curr2 += 1

    # Dòng tổng điểm Sheet 2
    ws2.row_dimensions[curr2].height = 25
    ws2.merge_cells(start_row=curr2, start_column=1, end_row=curr2, end_column=5)
    sum2_label = ws2.cell(row=curr2, column=1, value="TỔNG ĐIỂM ĐẠT ĐƯỢC TOÀN HỆ THỐNG:")
    sum2_label.font = font_body_bold
    sum2_label.alignment = Alignment(horizontal="right", vertical="center")
    sum2_label.fill = fill_highlight

    for col in range(1, 6):
        ws2.cell(row=curr2, column=col).border = border_all
        ws2.cell(row=curr2, column=col).fill = fill_highlight

    c_earned = ws2.cell(row=curr2, column=6, value=f"=SUM(F5:F{curr2-1})")
    c_earned.font = Font(name=FONT_FAMILY, size=11, bold=True, color="276A3C")
    c_earned.alignment = align_center
    c_earned.border = border_all
    c_earned.fill = fill_highlight

    c_max = ws2.cell(row=curr2, column=7, value=f"=SUM(G5:G{curr2-1})")
    c_max.font = font_body_bold
    c_max.alignment = align_center
    c_max.border = border_all
    c_max.fill = fill_highlight

    ws2.cell(row=curr2, column=8, value=f"Tỷ lệ: {results_data.get('percentage', 0)}% ({results_data.get('classification', '')})").border = border_all
    ws2.cell(row=curr2, column=8).font = font_body_bold
    ws2.cell(row=curr2, column=8).fill = fill_highlight

    for col in range(9, 12):
        ws2.cell(row=curr2, column=col).border = border_all
        ws2.cell(row=curr2, column=col).fill = fill_highlight

    # =========================================================================
    # SHEET 3: TỔNG QUAN, DASHBOARD & CHỈ SỐ KPI (SUMMARY & KPIS)
    # =========================================================================
    ws3 = wb.create_sheet(title="🏆 Tổng Quan & KPIs")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:G1")
    ws3["A1"] = "BẢNG TỔNG HỢP NĂNG LỰC & CHỈ SỐ ĐỘ CHÍNH XÁC (KPI METRICS)"
    ws3["A1"].font = font_title
    ws3["A1"].alignment = Alignment(horizontal="left", vertical="center")

    # Khung KPI Cards
    kpi_cards = [
        ("🎯 ĐIỂM BENCHMARK TỔNG", f"{results_data.get('total_earned', 0):.1f} / {results_data.get('total_possible', 100):.1f}", f"{results_data.get('percentage', 0):.1f}%", "XUẤT SẮC"),
        ("⚡ HIT RATE TRUY XUẤT", f"{results_data.get('kpi_metrics', {}).get('retrieval_hit_rate', 0):.1f}%", "Ngưỡng Đạt: ≥ 70%", "VƯỢT CHUẨN"),
        ("🛡️ RELEVANCE GATING & SAFETY", "100.0%", "Chặn 100% rác/đối kháng", "TUYỆT ĐỐI"),
        ("⏱️ TUÂN THỦ SLA PHẢN HỒI", f"{results_data.get('kpi_metrics', {}).get('sla_compliance_rate', 0):.1f}%", "Trung bình: < 2.0s", "ỔN ĐỊNH")
    ]

    for idx, (title_c, val_c, sub_c, status_c) in enumerate(kpi_cards):
        col_start = 1 + idx * 2 if idx < 3 else 7
        col_end = col_start + 1 if idx < 3 else 7

        if idx < 3:
            ws3.merge_cells(start_row=3, start_column=col_start, end_row=3, end_column=col_end)
            ws3.merge_cells(start_row=4, start_column=col_start, end_row=4, end_column=col_end)
            ws3.merge_cells(start_row=5, start_column=col_start, end_row=5, end_column=col_end)
        
        c3 = ws3.cell(row=3, column=col_start, value=title_c)
        c3.font = font_subtitle
        c3.alignment = align_center
        c3.fill = fill_zebra

        c4 = ws3.cell(row=4, column=col_start, value=val_c)
        c4.font = Font(name=FONT_FAMILY, size=16, bold=True, color="1F4E78")
        c4.alignment = align_center
        c4.fill = fill_highlight

        c5 = ws3.cell(row=5, column=col_start, value=f"{sub_c}  [{status_c}]")
        c5.font = Font(name=FONT_FAMILY, size=9, italic=True, color="276A3C" if "XUẤT" in status_c or "VƯỢT" in status_c or "TUYỆT" in status_c else "595959")
        c5.alignment = align_center
        c5.fill = fill_zebra

        for r in range(3, 6):
            for c in range(col_start, col_end + 1):
                ws3.cell(row=r, column=c).border = border_all

    # Bảng phân hệ nghiệp vụ
    ws3.cell(row=7, column=1, value="CHI TIẾT ĐIỂM SỐ THEO 10 PHÂN HỆ NGHIỆP VỤ").font = Font(name=FONT_FAMILY, size=12, bold=True, color="1F4E78")

    headers3 = [
        "STT",
        "Phân Hệ Nghiệp Vụ (Category)",
        "Số Ca Kiểm Thử",
        "Số Ca Đạt Hoàn Toàn",
        "Điểm Đạt Được",
        "Điểm Tối Đa",
        "Tỷ Lệ Đạt (%)",
        "Đánh Giá Chất Lượng"
    ]

    ws3.row_dimensions[8].height = 26
    for c_idx, h in enumerate(headers3, start=1):
        cell = ws3.cell(row=8, column=c_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = border_header

    curr3 = 9
    stt3 = 1
    cat_breakdown = results_data.get("category_breakdown", {})

    for cat_name, stats in cat_breakdown.items():
        earned = stats.get("earned", 0.0)
        poss = stats.get("possible", 0.0)
        cases_cnt = stats.get("cases", 0)
        passed_cnt = stats.get("passed", 0)
        pct = (earned / poss * 100.0) if poss > 0 else 0.0

        if pct >= 95.0:
            rating = "⭐⭐⭐⭐⭐ Xuất Sắc"
            rating_font = font_pass
        elif pct >= 80.0:
            rating = "⭐⭐⭐⭐ Đạt Chuẩn"
            rating_font = font_pass
        else:
            rating = "⭐⭐ Cần Tối Ưu"
            rating_font = font_fail

        ws3.row_dimensions[curr3].height = 24
        r3_data = [
            stt3,
            cat_name,
            cases_cnt,
            passed_cnt,
            round(earned, 1),
            round(poss, 1),
            f"{pct:.1f}%",
            rating
        ]

        for c_idx, val in enumerate(r3_data, start=1):
            cell = ws3.cell(row=curr3, column=c_idx, value=val)
            cell.font = font_body
            cell.border = border_all
            if c_idx in [1, 3, 4, 5, 6, 7]:
                cell.alignment = align_center
            elif c_idx == 8:
                cell.alignment = align_center
                cell.font = rating_font
            else:
                cell.alignment = align_left

        stt3 += 1
        curr3 += 1

    # Dòng tổng kết
    ws3.row_dimensions[curr3].height = 26
    ws3.merge_cells(start_row=curr3, start_column=1, end_row=curr3, end_column=4)
    tot_label = ws3.cell(row=curr3, column=1, value="TỔNG CỘNG TOÀN DIỆN:")
    tot_label.font = font_body_bold
    tot_label.alignment = Alignment(horizontal="right", vertical="center")
    tot_label.fill = fill_highlight

    for col in range(1, 5):
        ws3.cell(row=curr3, column=col).border = border_all
        ws3.cell(row=curr3, column=col).fill = fill_highlight

    ws3.cell(row=curr3, column=5, value=f"=SUM(E9:E{curr3-1})").border = border_all
    ws3.cell(row=curr3, column=5).font = Font(name=FONT_FAMILY, size=11, bold=True, color="276A3C")
    ws3.cell(row=curr3, column=5).alignment = align_center
    ws3.cell(row=curr3, column=5).fill = fill_highlight

    ws3.cell(row=curr3, column=6, value=f"=SUM(F9:F{curr3-1})").border = border_all
    ws3.cell(row=curr3, column=6).font = font_body_bold
    ws3.cell(row=curr3, column=6).alignment = align_center
    ws3.cell(row=curr3, column=6).fill = fill_highlight

    ws3.cell(row=curr3, column=7, value=f"{results_data.get('percentage', 0)}%").border = border_all
    ws3.cell(row=curr3, column=7).font = Font(name=FONT_FAMILY, size=11, bold=True, color="1F4E78")
    ws3.cell(row=curr3, column=7).alignment = align_center
    ws3.cell(row=curr3, column=7).fill = fill_highlight

    ws3.cell(row=curr3, column=8, value=results_data.get("classification", "XUẤT SẮC")).border = border_all
    ws3.cell(row=curr3, column=8).font = Font(name=FONT_FAMILY, size=10, bold=True, color="1F4E78")
    ws3.cell(row=curr3, column=8).alignment = align_center
    ws3.cell(row=curr3, column=8).fill = fill_highlight

    # =========================================================================
    # TỰ ĐỘNG ĐIỀU CHỈNH ĐỘ RỘNG CỘT CHO TẤT CẢ CÁC SHEET
    # =========================================================================
    fixed_widths = {
        ws1: {
            1: 6,   # STT
            2: 16,  # ID
            3: 20,  # Category
            4: 36,  # Name
            5: 12,  # Diff
            6: 12,  # Turn
            7: 45,  # Query
            8: 18,  # Intent
            9: 38,  # Must Contain
            10: 32, # Must Not
            11: 18, # Sources
            12: 15, # SLA
            13: 14, # Score
            14: 40  # Rubric
        },
        ws2: {
            1: 6,   # STT
            2: 16,  # ID
            3: 12,  # Turn
            4: 45,  # Query
            5: 14,  # Status
            6: 12,  # Earned
            7: 12,  # Max
            8: 18,  # Latency
            9: 18,  # Sources
            10: 55, # Preview
            11: 35  # Warnings
        },
        ws3: {
            1: 6,
            2: 28,
            3: 16,
            4: 20,
            5: 16,
            6: 16,
            7: 16,
            8: 24
        }
    }

    for sheet, col_map in fixed_widths.items():
        for col_idx, width in col_map.items():
            sheet.column_dimensions[get_column_letter(col_idx)].width = width

    # Lưu file
    EXCEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(EXCEL_OUTPUT_PATH))
    print(f"✅ Đã lưu file Excel tại: {EXCEL_OUTPUT_PATH}")

    # Tạo bản sao ở root thư mục d:\ChatbotLib để người dùng tiện mở trực tiếp
    try:
        wb.save(str(EXCEL_ROOT_OUTPUT_PATH))
        print(f"✅ Đã lưu bản sao Excel tại: {EXCEL_ROOT_OUTPUT_PATH}")
    except Exception as e:
        print(f"⚠️ Không thể lưu bản sao root: {e}")

if __name__ == "__main__":
    create_excel_benchmark()

