#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract


@dataclass
class Cell:
    page_number: int
    table_id: str
    row_index: int
    col_index: int
    bbox: Tuple[int, int, int, int]
    text: str
    extractor: str
    confidence: Optional[float] = None


@dataclass
class TableResult:
    page_number: int
    table_id: str
    bbox: Tuple[int, int, int, int]
    extractor: str
    n_rows: int
    n_cols: int
    csv_path: str
    cells: List[Cell]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def natural_text_cleanup(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\x0c", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pages(input_path: Path, dpi: int = 250) -> List[Tuple[int, np.ndarray]]:
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(str(input_path))
        pages: List[Tuple[int, np.ndarray]] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            pages.append((i + 1, bgr))
        return pages

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not read input: {input_path}")
    return [(1, img)]


def to_gray_and_binary(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        12,
    )
    return gray, bw_inv


def extract_line_masks(bw_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bw_inv.shape[:2]
    h_kernel_len = max(20, w // 40)
    v_kernel_len = max(20, h // 40)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    horizontal = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(bw_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
    return horizontal, vertical


def union_table_mask(horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    mask = cv2.add(horizontal, vertical)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    return mask


def merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.2,
    proximity: int = 10,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    changed = True
    merged = boxes[:]
    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(merged)

        for i, a in enumerate(merged):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                near = not (
                    cur[2] + proximity < b[0]
                    or b[2] + proximity < cur[0]
                    or cur[3] + proximity < b[1]
                    or b[3] + proximity < cur[1]
                )
                if iou(cur, b) >= iou_threshold or near:
                    cur = (
                        min(cur[0], b[0]),
                        min(cur[1], b[1]),
                        max(cur[2], b[2]),
                        max(cur[3], b[3]),
                    )
                    used[j] = True
                    changed = True
            used[i] = True
            new_boxes.append(cur)
        merged = new_boxes

    return merged


def find_table_regions(image_bgr: np.ndarray, line_union_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(line_union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    page_area = h * w

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < page_area * 0.01:
            continue
        if ww < w * 0.12 or hh < h * 0.05:
            continue
        boxes.append((x, y, x + ww, y + hh))

    boxes = merge_overlapping_boxes(boxes, iou_threshold=0.15, proximity=20)
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def cluster_positions(vals: Sequence[int], tolerance: int = 8) -> List[int]:
    if not vals:
        return []
    vals = sorted(int(v) for v in vals)
    groups: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - groups[-1][-1]) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(round(sum(g) / len(g))) for g in groups]


def get_lines_from_mask(mask: np.ndarray, axis: str, min_coverage: float = 0.2) -> List[int]:
    h, w = mask.shape[:2]
    if axis == "horizontal":
        proj = np.sum(mask > 0, axis=1)
        threshold = max(10, int(w * min_coverage))
        coords = np.where(proj >= threshold)[0].tolist()
        tol = max(5, h // 300)
    else:
        proj = np.sum(mask > 0, axis=0)
        threshold = max(10, int(h * min_coverage))
        coords = np.where(proj >= threshold)[0].tolist()
        tol = max(5, w // 300)
    return cluster_positions(coords, tolerance=tol)


def crop_with_pad(img: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 + pad)
    y1 = max(0, y1 + pad)
    x2 = min(w, x2 - pad)
    y2 = min(h, y2 - pad)
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


def ocr_cell(cell_img: np.ndarray, lang: str = "eng") -> Tuple[str, Optional[float]]:
    if cell_img.size == 0:
        return "", None

    gray = cell_img if len(cell_img.shape) == 2 else cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if min(h, w) < 40:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    proc = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    data = pytesseract.image_to_data(proc, lang=lang, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)

    words: List[str] = []
    confs: List[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        txt = natural_text_cleanup(text)
        if not txt:
            continue
        try:
            c = float(conf)
        except Exception:
            c = -1
        words.append(txt)
        if c >= 0:
            confs.append(c)

    text = natural_text_cleanup(" ".join(words))
    confidence = round(float(sum(confs) / len(confs)), 2) if confs else None
    return text, confidence


def save_grid_csv(grid: List[List[str]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(grid)


def make_table_id(page_number: int, table_index_zero_based: int) -> str:
    return f"page_{page_number}_table_{table_index_zero_based}"


def extract_ruled_table(
    page_img: np.ndarray,
    page_number: int,
    table_bbox: Tuple[int, int, int, int],
    table_index_zero_based: int,
    tables_dir: Path,
    lang: str = "eng",
) -> Optional[TableResult]:
    x1, y1, x2, y2 = table_bbox
    table_img = page_img[y1:y2, x1:x2]
    if table_img.size == 0:
        return None

    _, bw_inv = to_gray_and_binary(table_img)
    horizontal, vertical = extract_line_masks(bw_inv)

    ys = get_lines_from_mask(horizontal, "horizontal", min_coverage=0.25)
    xs = get_lines_from_mask(vertical, "vertical", min_coverage=0.25)

    h, w = table_img.shape[:2]
    if not ys or ys[0] > 10:
        ys = [0] + ys
    if not xs or xs[0] > 10:
        xs = [0] + xs
    if ys[-1] < h - 10:
        ys = ys + [h - 1]
    if xs[-1] < w - 10:
        xs = xs + [w - 1]

    ys = cluster_positions(ys, tolerance=max(5, h // 300))
    xs = cluster_positions(xs, tolerance=max(5, w // 300))

    if len(ys) < 2 or len(xs) < 2:
        return None

    n_rows = len(ys) - 1
    n_cols = len(xs) - 1
    if n_rows < 2 or n_cols < 2 or n_rows > 300 or n_cols > 100:
        return None

    table_id = make_table_id(page_number, table_index_zero_based)
    csv_path = tables_dir / f"{table_id}.csv"

    cells: List[Cell] = []
    grid: List[List[str]] = []

    for r in range(n_rows):
        row_vals: List[str] = []
        for c in range(n_cols):
            cx1, cy1, cx2, cy2 = xs[c], ys[r], xs[c + 1], ys[r + 1]
            crop = crop_with_pad(table_img, (cx1, cy1, cx2, cy2), pad=2)
            text, conf = ocr_cell(crop, lang=lang)
            row_vals.append(text)
            cells.append(
                Cell(
                    page_number=page_number,
                    table_id=table_id,
                    row_index=r,
                    col_index=c,
                    bbox=(x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2),
                    text=text,
                    extractor="ruled_grid",
                    confidence=conf,
                )
            )
        grid.append(row_vals)

    save_grid_csv(grid, csv_path)
    return TableResult(
        page_number=page_number,
        table_id=table_id,
        bbox=table_bbox,
        extractor="ruled_grid",
        n_rows=n_rows,
        n_cols=n_cols,
        csv_path=str(csv_path),
        cells=cells,
    )


def extract_weak_table_via_ocr_boxes(
    page_img: np.ndarray,
    page_number: int,
    table_bbox: Tuple[int, int, int, int],
    table_index_zero_based: int,
    tables_dir: Path,
    lang: str = "eng",
) -> Optional[TableResult]:
    x1, y1, x2, y2 = table_bbox
    roi = page_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, lang=lang, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data["text"])):
        txt = natural_text_cleanup(data["text"][i])
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1
        if conf < 0:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        words.append(
            {
                "text": txt,
                "conf": conf,
                "x1": left,
                "y1": top,
                "x2": left + width,
                "y2": top + height,
                "xc": left + width / 2.0,
                "yc": top + height / 2.0,
            }
        )

    if len(words) < 4:
        return None

    median_h = np.median([max(1, w["y2"] - w["y1"]) for w in words])
    row_tol = max(8, int(median_h * 0.8))
    row_bands = cluster_positions([int(w["yc"]) for w in words], tolerance=row_tol)
    rows: List[List[Dict[str, Any]]] = [[] for _ in row_bands]
    for w in words:
        idx = min(range(len(row_bands)), key=lambda i: abs(row_bands[i] - w["yc"]))
        rows[idx].append(w)
    rows = [sorted(r, key=lambda z: z["x1"]) for r in rows if r]
    if len(rows) < 2:
        return None

    x_positions = []
    widths = []
    for r in rows:
        for w in r:
            x_positions.extend([int(w["x1"]), int(w["xc"])])
            widths.append(max(1, int(w["x2"] - w["x1"])))
    col_tol = max(12, int(np.median(widths) * 0.9)) if widths else 12
    col_bands = cluster_positions(x_positions, tolerance=col_tol)
    if len(col_bands) < 2:
        return None

    table_id = make_table_id(page_number, table_index_zero_based)
    csv_path = tables_dir / f"{table_id}.csv"
    cells: List[Cell] = []
    grid: List[List[str]] = []

    for r_idx, row_words in enumerate(rows):
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for w in row_words:
            c_idx = min(range(len(col_bands)), key=lambda i: abs(col_bands[i] - w["xc"]))
            buckets.setdefault(c_idx, []).append(w)

        row_vals: List[str] = []
        for c_idx in range(len(col_bands)):
            ws = sorted(buckets.get(c_idx, []), key=lambda z: z["x1"])
            text = natural_text_cleanup(" ".join(w["text"] for w in ws))
            row_vals.append(text)

            if ws:
                bx1 = min(w["x1"] for w in ws)
                by1 = min(w["y1"] for w in ws)
                bx2 = max(w["x2"] for w in ws)
                by2 = max(w["y2"] for w in ws)
                confs = [w["conf"] for w in ws if w["conf"] >= 0]
                conf = round(float(sum(confs) / len(confs)), 2) if confs else None
            else:
                bx1 = by1 = bx2 = by2 = 0
                conf = None

            cells.append(
                Cell(
                    page_number=page_number,
                    table_id=table_id,
                    row_index=r_idx,
                    col_index=c_idx,
                    bbox=(x1 + bx1, y1 + by1, x1 + bx2, y1 + by2),
                    text=text,
                    extractor="ocr_box_fallback",
                    confidence=conf,
                )
            )
        grid.append(row_vals)

    if grid:
        df = pd.DataFrame(grid)
        keep_cols = []
        for c in range(df.shape[1]):
            non_empty = sum(bool(natural_text_cleanup(str(v))) and str(v) != "nan" for v in df.iloc[:, c].tolist())
            if non_empty >= max(1, math.ceil(df.shape[0] * 0.15)):
                keep_cols.append(c)
        if keep_cols:
            df = df.iloc[:, keep_cols]
            old_to_new = {old: new for new, old in enumerate(keep_cols)}
            for cell in cells:
                if cell.col_index in old_to_new:
                    cell.col_index = old_to_new[cell.col_index]
                else:
                    cell.col_index = -1
            cells = [c for c in cells if c.col_index >= 0]
            grid = df.fillna("").astype(str).values.tolist()

    if not grid or len(grid) < 2 or len(grid[0]) < 2:
        return None

    save_grid_csv(grid, csv_path)
    return TableResult(
        page_number=page_number,
        table_id=table_id,
        bbox=table_bbox,
        extractor="ocr_box_fallback",
        n_rows=len(grid),
        n_cols=len(grid[0]),
        csv_path=str(csv_path),
        cells=cells,
    )


def compute_line_density(horizontal: np.ndarray, vertical: np.ndarray) -> float:
    return float(np.count_nonzero(horizontal) + np.count_nonzero(vertical)) / max(1, horizontal.size)


def detect_candidate_regions(page_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    _, bw_inv = to_gray_and_binary(page_img)
    horizontal, vertical = extract_line_masks(bw_inv)
    table_mask = union_table_mask(horizontal, vertical)
    boxes = find_table_regions(page_img, table_mask)
    if boxes:
        return boxes
    h, w = page_img.shape[:2]
    return [(0, 0, w, h)]


def extract_tables_from_page(page_img: np.ndarray, page_number: int, tables_dir: Path, lang: str = "eng") -> List[TableResult]:
    ensure_dir(tables_dir)
    results: List[TableResult] = []
    candidate_boxes = detect_candidate_regions(page_img)

    for idx, bbox in enumerate(candidate_boxes):
        x1, y1, x2, y2 = bbox
        roi = page_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        _, bw_inv = to_gray_and_binary(roi)
        hmask, vmask = extract_line_masks(bw_inv)
        density = compute_line_density(hmask, vmask)
        ordered_extractors = ["ruled", "fallback"] if density >= 0.01 else ["fallback", "ruled"]

        table_result: Optional[TableResult] = None
        for name in ordered_extractors:
            if name == "ruled":
                table_result = extract_ruled_table(page_img, page_number, bbox, idx, tables_dir, lang=lang)
            else:
                table_result = extract_weak_table_via_ocr_boxes(page_img, page_number, bbox, idx, tables_dir, lang=lang)
            if table_result is not None:
                break

        if table_result is not None:
            results.append(table_result)

    return results


# ---------- Unified LLM markdown helpers ----------

def md_escape(val: Any) -> str:
    if val is None:
        return ""
    return str(val).replace("\n", " ").replace("|", "\\|").strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.fillna("").astype(str)
    df = df.apply(lambda col: col.map(lambda x: natural_text_cleanup(x) if isinstance(x, str) else x))
    df = df.loc[~(df == "").all(axis=1)]
    df = df.loc[:, ~(df == "").all(axis=0)]
    if df.empty:
        return df
    df.columns = [natural_text_cleanup(str(c)) if natural_text_cleanup(str(c)) else f"column_{i+1}" for i, c in enumerate(df.columns)]
    return df


def infer_headers_from_dataframe(df: pd.DataFrame) -> Tuple[List[str], List[List[str]]]:
    if df.empty:
        return [], []

    grid = df.fillna("").astype(str).values.tolist()
    first_row = [natural_text_cleanup(x) for x in grid[0]]
    non_empty_first = sum(bool(x) for x in first_row)
    unique_first = len(set(x for x in first_row if x))

    looks_like_header = (
        len(grid) >= 2
        and non_empty_first >= max(2, len(first_row) // 2)
        and unique_first >= max(2, len(first_row) // 2)
    )

    if looks_like_header:
        headers = [h if h else f"col_{i}" for i, h in enumerate(first_row)]
        body_rows = grid[1:]
    else:
        headers = [natural_text_cleanup(str(c)) if natural_text_cleanup(str(c)) else f"column_{i+1}" for i, c in enumerate(df.columns)]
        body_rows = grid

    return headers, body_rows


def table_result_to_entry(table: TableResult) -> Dict[str, Any]:
    return {
        "page_number": table.page_number,
        "table_id": table.table_id,
        "bbox": list(table.bbox),
        "extractor": table.extractor,
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "csv_path": table.csv_path,
        "status": "success",
    }


def write_combined_llm_markdown(table_entries: List[Dict[str, Any]], out_path: Path) -> None:
    sections: List[str] = [
        "# Extracted Tables for LLM QA",
        "",
        "All extractors write into this single consolidated file.",
        "",
    ]

    for entry in sorted(table_entries, key=lambda x: (x.get("page_number", 0), x.get("table_id", ""))):
        table_id = entry.get("table_id", "unknown_table")
        page_number = entry.get("page_number", "")
        extractor = entry.get("extractor", "unknown")
        status = entry.get("status", "unknown")
        notes = entry.get("notes")
        csv_path = entry.get("csv_path")

        sections.append(f"## {table_id}")
        sections.append("")
        sections.append(f"- Page: {page_number}")
        sections.append(f"- Extractor: {extractor}")
        sections.append(f"- Status: {status}")
        if notes:
            sections.append(f"- Notes: {notes}")
        sections.append("")

        if status != "success" or not csv_path or not Path(csv_path).exists():
            sections.append("### Clean Table")
            sections.append("")
            sections.append("No structured table could be extracted.")
            sections.append("")
            sections.append("---")
            sections.append("")
            continue

        try:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
        except Exception:
            sections.append("### Clean Table")
            sections.append("")
            sections.append("Table CSV exists but could not be parsed.")
            sections.append("")
            sections.append("---")
            sections.append("")
            continue

        df = clean_dataframe(df)
        if df.empty:
            sections.append("### Clean Table")
            sections.append("")
            sections.append("Extractor produced an empty table.")
            sections.append("")
            sections.append("---")
            sections.append("")
            continue

        headers, body_rows = infer_headers_from_dataframe(df)

        sections.append("### Clean Table")
        sections.append("")
        sections.append("| " + " | ".join(md_escape(h) for h in headers) + " |")
        sections.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in body_rows:
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            padded = padded[:len(headers)]
            sections.append("| " + " | ".join(md_escape(x) for x in padded) + " |")
        sections.append("")

        if len(df) <= 25:
            sections.append("### Row-wise Records")
            sections.append("")
            data_rows = body_rows
            for idx, row in enumerate(data_rows, start=1):
                parts = []
                for h, v in zip(headers, row):
                    v = natural_text_cleanup(str(v))
                    if v:
                        parts.append(f"{h} = {v}")
                sections.append(f"- Row {idx}: " + ("; ".join(parts) if parts else "[empty row]"))
            sections.append("")

        sections.append("---")
        sections.append("")

    out_path.write_text("\n".join(sections), encoding="utf-8")


def write_combined_summary(table_entries: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps({"tables": table_entries}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic PDF/image table extractor.")
    parser.add_argument("input_path", help="Input PDF or image path")
    parser.add_argument("--output_dir", default="out", help="Output directory, default: out")
    parser.add_argument("--lang", default="eng", help="Tesseract language code")
    parser.add_argument("--dpi", type=int, default=250, help="PDF render DPI, default: 250")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    tables_dir = output_dir / "tables"
    ensure_dir(tables_dir)

    pages = load_pages(input_path, dpi=args.dpi)
    all_tables: List[TableResult] = []

    for page_number, page_img in pages:
        page_tables = extract_tables_from_page(page_img, page_number, tables_dir, lang=args.lang)
        all_tables.extend(page_tables)

    table_entries = [table_result_to_entry(t) for t in all_tables]
    write_combined_llm_markdown(table_entries, output_dir / "llm_tables.md")
    write_combined_summary(table_entries, output_dir / "extraction_summary.json")

    print(f"Rendered and processed {len(pages)} page(s).")
    print(f"Extracted {len(all_tables)} table(s).")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
