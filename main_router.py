#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from grid_table_extractor import (
    extract_tables_from_page,
    load_pages,
    table_result_to_entry,
)
from ocr_table_extractor import process_page3_style_image

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route PDF pages to the right table extractor.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF (or image) to process")
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Directory to write extracted outputs (default: ./out)",
    )
    return parser.parse_args(argv)


def classify_page(page_img_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, gray.shape[1] // 40), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, gray.shape[0] // 40))
    )

    horizontal = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, vertical_kernel)

    line_pixels = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    page_pixels = gray.shape[0] * gray.shape[1]
    line_ratio = line_pixels / max(page_pixels, 1)

    if line_ratio > 0.03:
        return "generic_grid"
    return "ocr_recursive"

JUNK_CELL_RE = re.compile(r"^[\s\|\_\-\~\.\,\:;]+$")

def normalize_cell(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\|\s*", " | ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "" or JUNK_CELL_RE.fullmatch(text):
        return pd.NA

    return text

def row_signal_score(row: pd.Series) -> float:
    vals = [str(x).strip() for x in row.tolist() if pd.notna(x) and str(x).strip()]
    if not vals:
        return -1.0

    alpha_like = sum(bool(re.search(r"[A-Za-z]", v)) for v in vals)
    numeric_like = sum(bool(re.fullmatch(r"[\d\.\,\-\<\>\=\(\)%/]+", v)) for v in vals)
    unique_ratio = len(set(vals)) / max(len(vals), 1)
    filled_ratio = len(vals) / max(len(row), 1)

    return (1.6 * alpha_like) - (0.5 * numeric_like) + (0.8 * unique_ratio) + (0.6 * filled_ratio)

def choose_header_row(df: pd.DataFrame, search_rows: int = 4) -> int | None:
    if df.empty:
        return None

    max_rows = min(search_rows, len(df))
    best_idx = None
    best_score = float("-inf")

    for i in range(max_rows):
        score = row_signal_score(df.iloc[i])
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx

def maybe_combine_header_rows(df: pd.DataFrame, header_idx: int) -> list[str]:
    header_row = df.iloc[header_idx].copy()

    second_header = None
    if header_idx + 1 < len(df):
        next_row = df.iloc[header_idx + 1]
        next_vals = [str(x).strip() for x in next_row.tolist() if pd.notna(x) and str(x).strip()]
        if next_vals:
            alpha_like = sum(bool(re.search(r"[A-Za-z]", v)) for v in next_vals)
            if alpha_like >= max(1, len(next_vals) // 2):
                second_header = next_row

    headers: list[str] = []
    for col_idx in range(df.shape[1]):
        top = "" if pd.isna(header_row.iloc[col_idx]) else str(header_row.iloc[col_idx]).strip()
        bottom = ""
        if second_header is not None and col_idx < len(second_header):
            bottom = "" if pd.isna(second_header.iloc[col_idx]) else str(second_header.iloc[col_idx]).strip()

        if top and bottom and bottom.lower() != top.lower():
            merged = f"{top} {bottom}"
        else:
            merged = top or bottom or f"column_{col_idx + 1}"

        merged = re.sub(r"\s+", " ", merged).strip()
        headers.append(merged if merged else f"column_{col_idx + 1}")

    seen: dict[str, int] = {}
    deduped: list[str] = []
    for h in headers:
        if h not in seen:
            seen[h] = 1
            deduped.append(h)
        else:
            seen[h] += 1
            deduped.append(f"{h}_{seen[h]}")
    return deduped

def clean_dataframe_general(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df = df.apply(lambda col: col.map(normalize_cell))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if df.empty:
        return pd.DataFrame()

    if df.shape[1] >= 4:
        min_non_null = max(1, int(0.15 * len(df)))
        keep_cols = [c for c in df.columns if df[c].notna().sum() >= min_non_null]
        if keep_cols:
            df = df[keep_cols]

    if df.shape[1] >= 3:
        min_filled = max(1, int(0.2 * df.shape[1]))
        df = df[df.notna().sum(axis=1) >= min_filled]

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)

    header_idx = choose_header_row(df)
    if header_idx is not None:
        headers = maybe_combine_header_rows(df, header_idx)

        rows_to_skip = 1
        if header_idx + 1 < len(df):
            next_row = df.iloc[header_idx + 1]
            next_vals = [str(x).strip() for x in next_row.tolist() if pd.notna(x) and str(x).strip()]
            if next_vals:
                alpha_like = sum(bool(re.search(r"[A-Za-z]", v)) for v in next_vals)
                if alpha_like >= max(1, len(next_vals) // 2):
                    rows_to_skip = 2

        data_start = header_idx + rows_to_skip
        df = df.iloc[data_start:].reset_index(drop=True)
        df.columns = headers
    else:
        df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]

    if not df.empty:
        header_lower = [str(c).strip().lower() for c in df.columns]
        keep_mask = []
        for _, row in df.iterrows():
            row_vals = ["" if pd.isna(v) else str(v).strip().lower() for v in row.tolist()]
            same_count = sum(1 for a, b in zip(row_vals, header_lower) if a and b and a == b)
            keep_mask.append(same_count < max(1, len(header_lower) // 2))
        df = df.loc[keep_mask].reset_index(drop=True)

    if not df.empty and df.shape[1] >= 3:
        sparse_rows = df.notna().sum(axis=1) <= max(1, df.shape[1] // 3)
        if sparse_rows.any():
            df.loc[sparse_rows] = df.loc[sparse_rows].ffill(axis=1)

    df = df.apply(lambda col: col.map(normalize_cell))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)

    return df

def first_existing_key(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    if not text:
        return default

    if text.isdigit():
        return int(text)

    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))

    return default

def normalize_table_entry(entry: dict[str, Any], fallback_index: int = 0) -> dict[str, Any]:
    page_number_raw = first_existing_key(entry, ["page_number", "page", "page_num"])
    table_index_raw = first_existing_key(entry, ["table_index", "table_id", "index"])
    extractor = first_existing_key(entry, ["extractor", "method", "source"]) or "unknown"

    csv_path = first_existing_key(
        entry,
        ["csv_path", "table_csv", "path", "output_csv", "csv_file"],
    )

    crop_path = first_existing_key(
        entry,
        ["crop_path", "image_path", "table_image", "crop_file"],
    )

    return {
        "page_number": safe_int(page_number_raw, default=0),
        "table_index": safe_int(table_index_raw, default=fallback_index),
        "table_id": str(table_index_raw) if table_index_raw is not None else f"table_{fallback_index}",
        "extractor": str(extractor),
        "csv_path": str(csv_path) if csv_path else None,
        "crop_path": str(crop_path) if crop_path else None,
        "raw_entry": entry,
    }

def load_table_dataframe(entry: dict[str, Any]) -> pd.DataFrame:
    csv_path = entry.get("csv_path")
    if csv_path:
        csv_path_obj = Path(csv_path)
        if csv_path_obj.exists():
            try:
                return pd.read_csv(csv_path_obj, header=None, dtype=str, keep_default_na=False)
            except Exception:
                pass

    raw_entry = entry.get("raw_entry", {})
    for key in ("data", "rows", "cells", "table"):
        value = raw_entry.get(key)
        if isinstance(value, list) and value:
            try:
                return pd.DataFrame(value)
            except Exception:
                pass

    return pd.DataFrame()

def dataframe_to_llm_markdown(df: pd.DataFrame, title: str) -> str:
    parts: list[str] = [f"## {title}", ""]

    if df.empty:
        parts.append("No structured table content found.")
        parts.append("")
        return "\n".join(parts)

    columns = [str(c).strip() for c in df.columns.tolist()]

    parts.append("### Columns")
    parts.append("")
    parts.append(", ".join(columns))
    parts.append("")

    parts.append("### Rows")
    parts.append("")
    row_count = 0

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        pairs = []
        for col, val in zip(columns, row.tolist()):
            if pd.isna(val):
                continue
            text = str(val).strip()
            if not text:
                continue
            pairs.append(f"{col} = {text}")

        if pairs:
            parts.append(f"- Row {row_idx}: " + "; ".join(pairs))
            row_count += 1

    if row_count == 0:
        parts.append("- No non-empty rows found.")
    parts.append("")

    if df.shape[1] == 2:
        parts.append("### Key-Value Pairs")
        parts.append("")
        c1, c2 = columns
        kv_count = 0
        for _, row in df.iterrows():
            k = row.iloc[0]
            v = row.iloc[1]
            if pd.notna(k) and pd.notna(v):
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs:
                    parts.append(f"- {ks}: {vs}")
                    kv_count += 1
        if kv_count == 0:
            parts.append("- No key-value pairs found.")
        parts.append("")

    return "\n".join(parts)

def write_combined_llm_markdown(entries: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Extracted Tables",
        "",
        "This file is structured for question answering using row-wise table records.",
        "",
    ]

    for idx, entry in enumerate(entries, start=1):
        df_raw = load_table_dataframe(entry)
        df_clean = clean_dataframe_general(df_raw)

        page_number = entry.get("page_number", "?")
        table_index = entry.get("table_index", idx - 1)
        extractor = entry.get("extractor", "unknown")

        title = f"Table {idx} (Page {page_number}, Table {table_index}, Extractor: {extractor})"
        lines.append(dataframe_to_llm_markdown(df_clean, title))

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_combined_summary(entries: list[dict[str, Any]], out_path: Path) -> None:
    summary = {
        "total_tables": len(entries),
        "tables": [],
    }

    for idx, entry in enumerate(entries, start=1):
        df_raw = load_table_dataframe(entry)
        df_clean = clean_dataframe_general(df_raw)

        summary["tables"].append(
            {
                "global_table_index": idx,
                "page_number": entry.get("page_number"),
                "table_index": entry.get("table_index"),
                "table_id": entry.get("table_id"),
                "extractor": entry.get("extractor"),
                "csv_path": entry.get("csv_path"),
                "crop_path": entry.get("crop_path"),
                "raw_shape": list(df_raw.shape) if not df_raw.empty else [0, 0],
                "clean_shape": list(df_clean.shape) if not df_clean.empty else [0, 0],
                "status": "ok" if not df_clean.empty else "empty_after_cleaning",
            }
        )

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def cleanup_nested_outputs(root_out_dir: Path) -> None:
    for path in root_out_dir.rglob("llm_tables.md"):
        if path.resolve() != (root_out_dir / "llm_tables.md").resolve():
            try:
                path.unlink()
                print(f"Removed nested markdown: {path}")
            except Exception as e:
                print(f"Warning: could not remove nested markdown {path}: {e}")

    for path in root_out_dir.rglob("extraction_summary.json"):
        if path.resolve() != (root_out_dir / "extraction_summary.json").resolve():
            try:
                path.unlink()
                print(f"Removed nested summary: {path}")
            except Exception as e:
                print(f"Warning: could not remove nested summary {path}: {e}")

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pdf_path = args.pdf_path
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_table_entries: list[dict[str, Any]] = []
    pages = load_pages(pdf_path, dpi=250)

    for page_number, page_img in pages:
        page_type = classify_page(page_img)
        print(f"Page {page_number}: {page_type}")

        if page_type == "generic_grid":
            page_results = extract_tables_from_page(
                page_img=page_img,
                page_number=page_number,
                tables_dir=tables_dir,
                lang="eng",
            )
            normalized = [
                normalize_table_entry(table_result_to_entry(t), fallback_index=i)
                for i, t in enumerate(page_results)
            ]
            all_table_entries.extend(normalized)
        else:
            page_out_dir = out_dir / f"page_{page_number}_ocr"
            ocr_result = process_page3_style_image(
                page_img_bgr=page_img,
                page_number=page_number,
                output_dir=page_out_dir,
                lang="eng",
                debug=False,
            )
            raw_tables = ocr_result.get("tables", [])
            normalized = [
                normalize_table_entry(t, fallback_index=i)
                for i, t in enumerate(raw_tables)
            ]
            all_table_entries.extend(normalized)

    write_combined_llm_markdown(all_table_entries, out_dir / "llm_tables.md")
    write_combined_summary(all_table_entries, out_dir / "extraction_summary.json")
    cleanup_nested_outputs(out_dir)

    print("Done.")
    print(f"Output root: {out_dir}")
    print(f"Combined markdown: {out_dir / 'llm_tables.md'}")
    print(f"Combined summary: {out_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    main()