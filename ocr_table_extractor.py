#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import pytesseract
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
from PIL import Image

from grid_table_extractor import extract_tables_from_page

MIN_WIDTH = 40
MIN_HEIGHT = 40
MIN_AREA_RATIO = 0.01


ocr_engine = TesseractOCR(n_threads=1, lang="eng")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_cell(x: Any) -> str:
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.apply(lambda col: col.map(clean_cell))
    df = df.replace("", pd.NA)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.fillna("")
    df.columns = [clean_cell(c) if str(c).strip() else f"column_{i+1}" for i, c in enumerate(df.columns)]
    return df


def make_unique_columns(columns: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


def crop_from_bbox(img: Image.Image, bbox: Any) -> Image.Image:
    return img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))


def clamp_bbox_to_image(bbox_xyxy: tuple[int, int, int, int], img: Image.Image) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), img.width))
    y1 = max(0, min(int(y1), img.height))
    x2 = max(0, min(int(x2), img.width))
    y2 = max(0, min(int(y2), img.height))
    if x2 <= x1:
        x2 = min(img.width, x1 + 1)
    if y2 <= y1:
        y2 = min(img.height, y1 + 1)
    return x1, y1, x2, y2


def extract_tables_from_pil(
    pil_img: Image.Image,
    ocr: Optional[TesseractOCR] = None,
    implicit_rows: bool = False,
    implicit_columns: bool = False,
    borderless_tables: bool = False,
    min_confidence: int = 50,
):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_img_path = tmp_file.name
        pil_img.save(temp_img_path)

    try:
        doc = Img2TableImage(src=temp_img_path)
        tables = doc.extract_tables(
            ocr=ocr,
            implicit_rows=implicit_rows,
            implicit_columns=implicit_columns,
            borderless_tables=borderless_tables,
            min_confidence=min_confidence,
        )
    finally:
        os.remove(temp_img_path)

    return tables


def is_meaningful_child(parent_img: Image.Image, child_bbox: Any) -> bool:
    w = child_bbox.x2 - child_bbox.x1
    h = child_bbox.y2 - child_bbox.y1
    child_area = max(0, w) * max(0, h)
    parent_area = parent_img.width * parent_img.height

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False
    if parent_area == 0:
        return False
    if child_area / parent_area < MIN_AREA_RATIO:
        return False
    return True


def autorotate_and_deskew(pil_img: Image.Image, max_deskew: float = 20, debug: bool = False) -> Image.Image:
    img = pil_img.convert("RGB")

    coarse_angle = 0
    try:
        osd = pytesseract.image_to_osd(img)
        match = re.search(r"Rotate: (\d+)", osd)
        if match:
            coarse_angle = int(match.group(1)) % 360
            if coarse_angle in (90, 180, 270):
                img = img.rotate(-coarse_angle, expand=True, fillcolor="white")
    except Exception:
        coarse_angle = 0

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    coords = cv2.findNonZero(bw)

    fine_angle = 0.0
    if coords is not None:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        fine_angle = -angle

    if abs(fine_angle) > max_deskew:
        fine_angle = 0.0

    if fine_angle != 0.0:
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), fine_angle, 1.0)
        rotated = cv2.warpAffine(
            cv_img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        img = Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    if debug and (coarse_angle or fine_angle):
        print(f"  Auto-rotation applied: coarse={coarse_angle}°, fine={fine_angle:.2f}°")

    return img


def preprocess_full_page(pil_img: Image.Image, debug: bool = False) -> Image.Image:
    return autorotate_and_deskew(pil_img, debug=debug)


def ocr_table_crop_to_dataframe(
    pil_img: Image.Image,
    lang: str = "eng",
    min_confidence: int = 25,
    implicit_rows: bool = True,
    implicit_columns: bool = True,
    borderless_tables: bool = False,
    debug: bool = False,
) -> Optional[pd.DataFrame]:
    local_ocr_engine = TesseractOCR(n_threads=1, lang=lang)
    pil_img = autorotate_and_deskew(pil_img, debug=debug)

    tables = extract_tables_from_pil(
        pil_img,
        ocr=local_ocr_engine,
        implicit_rows=implicit_rows,
        implicit_columns=implicit_columns,
        borderless_tables=borderless_tables,
        min_confidence=min_confidence,
    )

    if not tables:
        return None

    df = tables[0].df.copy()
    df = normalize_dataframe(df)
    if df.empty:
        return None

    df.columns = make_unique_columns([str(c) for c in df.columns])
    return df


def dataframe_to_llm_markdown(table_id: int, df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append(f"## Table {table_id}")
    lines.append("")
    lines.append("### Original Table")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")

    lines.append("### Columns")
    lines.append("")
    for idx, col in enumerate(df.columns, start=1):
        lines.append(f"- {idx}. {col}")
    lines.append("")

    lines.append("### Row-wise Records")
    lines.append("")
    for row_idx, row in df.iterrows():
        non_empty_parts = []
        for col in df.columns:
            val = clean_cell(row[col])
            if val:
                non_empty_parts.append(f"{col} = {val}")
        if non_empty_parts:
            lines.append(f"- Row {row_idx + 1}: " + "; ".join(non_empty_parts))
        else:
            lines.append(f"- Row {row_idx + 1}: [empty row]")
    lines.append("")

    lines.append("### Column-wise Values")
    lines.append("")
    for col in df.columns:
        vals = [clean_cell(v) for v in df[col].tolist()]
        vals = [v for v in vals if v]
        if vals:
            lines.append(f"- {col}: " + " | ".join(vals))
        else:
            lines.append(f"- {col}: [no values]")
    lines.append("")

    if len(df.columns) == 2:
        lines.append("### Key-Value Interpretation")
        lines.append("")
        for _, row in df.iterrows():
            k = clean_cell(row.iloc[0])
            v = clean_cell(row.iloc[1])
            if k or v:
                lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines)

def process_page3_style_image(
    page_img_bgr: np.ndarray,
    page_number: int,
    output_dir: str | Path,
    lang: str = "eng",
    debug: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    crops_dir = output_dir / "crops"
    tmp_v2_tables_dir = output_dir / "tmp_v2_tables"

    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    ensure_dir(crops_dir)
    ensure_dir(tmp_v2_tables_dir)

    original_img = Image.fromarray(cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2RGB))
    original_img = preprocess_full_page(original_img, debug=debug)

    top_level_tables = extract_tables_from_pil(
        original_img,
        ocr=None,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=50,
    )

    if debug:
        print(f"Top-level tables found: {len(top_level_tables)}")

    # Keep all detected top-level crops
    all_final_tables: List[Image.Image] = []
    for table in top_level_tables:
        table_img = crop_from_bbox(original_img, table.bbox)
        all_final_tables.append(table_img)

    # Fallback: if nothing detected, process full page
    if not all_final_tables:
        all_final_tables = [original_img]

    successful_tables = 0
    global_idx = 1
    summary_tables: List[Dict[str, Any]] = []

    for tbl_img in all_final_tables:
        tbl_bgr = cv2.cvtColor(np.array(tbl_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        sub_results = extract_tables_from_page(
            page_img=tbl_bgr,
            page_number=page_number,
            tables_dir=tmp_v2_tables_dir,
            lang=lang,
        )

        if not sub_results:
            df = ocr_table_crop_to_dataframe(
                tbl_img,
                lang=lang,
                min_confidence=25,
                implicit_rows=True,
                implicit_columns=True,
                borderless_tables=False,
                debug=debug,
            )

            img_path = crops_dir / f"table_{global_idx:02d}.png"
            tbl_img.save(img_path)

            csv_path = tables_dir / f"page_{page_number}_table_{global_idx - 1}.csv"

            if df is None:
                csv_path.write_text("", encoding="utf-8")
                summary_tables.append(
                    {
                        "page_number": page_number,
                        "table_id": f"page_{page_number}_table_{global_idx - 1}",
                        "table_index": global_idx - 1,
                        "extractor": "page3_ocr_fallback",
                        "csv_path": str(csv_path),
                        "crop_path": str(img_path),
                        "status": "failed",
                    }
                )
                global_idx += 1
                continue

            df.to_csv(csv_path, index=False)
            successful_tables += 1
            summary_tables.append(
                {
                    "page_number": page_number,
                    "table_id": f"page_{page_number}_table_{global_idx - 1}",
                    "table_index": global_idx - 1,
                    "extractor": "page3_ocr_fallback",
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                    "csv_path": str(csv_path),
                    "crop_path": str(img_path),
                    "status": "success",
                }
            )
            global_idx += 1
            continue

        for sub in sub_results:
            src_csv = Path(sub.csv_path)
            dest_csv = tables_dir / f"page_{page_number}_table_{global_idx - 1}.csv"

            if src_csv.exists():
                shutil.copyfile(src_csv, dest_csv)
            else:
                dest_csv.write_text("", encoding="utf-8")

            img_path = crops_dir / f"table_{global_idx:02d}.png"
            if hasattr(sub, "bbox") and sub.bbox is not None and len(sub.bbox) == 4:
                sx1, sy1, sx2, sy2 = clamp_bbox_to_image(tuple(sub.bbox), tbl_img)
                sub_crop = tbl_img.crop((sx1, sy1, sx2, sy2))
                sub_crop.save(img_path)
            else:
                tbl_img.save(img_path)

            try:
                df = pd.read_csv(dest_csv)
                df = normalize_dataframe(df)
                if not df.empty:
                    df.columns = make_unique_columns([str(c) for c in df.columns])
            except Exception:
                df = None

            extractor_name = getattr(sub, "extractor", "grid_subtable")

            if df is not None and not df.empty:
                successful_tables += 1
                summary_tables.append(
                    {
                        "page_number": page_number,
                        "table_id": f"page_{page_number}_table_{global_idx - 1}",
                        "table_index": global_idx - 1,
                        "extractor": extractor_name,
                        "n_rows": int(df.shape[0]),
                        "n_cols": int(df.shape[1]),
                        "csv_path": str(dest_csv),
                        "crop_path": str(img_path),
                        "status": "success",
                    }
                )
            else:
                summary_tables.append(
                    {
                        "page_number": page_number,
                        "table_id": f"page_{page_number}_table_{global_idx - 1}",
                        "table_index": global_idx - 1,
                        "extractor": extractor_name,
                        "csv_path": str(dest_csv),
                        "crop_path": str(img_path),
                        "status": "empty",
                    }
                )

            global_idx += 1

    shutil.rmtree(tmp_v2_tables_dir, ignore_errors=True)

    return {
        "page_number": page_number,
        "successful_tables": successful_tables,
        "total_tables": global_idx - 1,
        "tables_dir": str(tables_dir),
        "crops_dir": str(crops_dir),
        "tables": summary_tables,
    }