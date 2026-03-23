# ocr-table-pipeline

A PDF table extraction pipeline that uses OpenCV-based page classification to route each page to either a grid-line detector or a recursive OCR extractor (Tesseract + img2table), outputting structured CSVs with an LLM-formatted markdown summary.

Built for environmental groundwater analysis reports, but works on any multi-page PDF with tabular data.

---

## How it works

```
PDF
 └── rasterized page-by-page (PyMuPDF)
      └── classify_page()
           ├── line density > threshold → grid_table_extractor.py
           │    detects row/col separators from line projections,
           │    forms a grid, runs cell-level Tesseract OCR
           │
           └── line density ≤ threshold → ocr_table_extractor.py
                detects table regions with img2table,
                retries grid extractor per region,
                falls back to spatial word clustering (Tesseract bbox)

Output: out/tables/*.csv  |  out/llm_tables.md  |  out/extraction_summary.json
```

The classifier works by binarizing each page with Otsu thresholding, then applying directional morphological filters to isolate horizontal and vertical lines. The ratio of line pixels to total pixels acts as a proxy for layout structure — no ML model needed.

---

## Setup

Prerequisites: Python 3.8+, and Tesseract installed on your system.

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

Then install Python dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
python3 main_router.py Sample_Ground_Water_Analysis.pdf
```

Output goes to `./out/` by default. Use `-o` to change:

```bash
python3 main_router.py your_file.pdf -o results/
```

---

## Output

| File | Description |
|------|-------------|
| `out/tables/page_N_table_M.csv` | Extracted table as CSV |
| `out/llm_tables.md` | All tables formatted for LLM input |
| `out/extraction_summary.json` | Metadata: page count, tables found, extractor used per page |

---

## Project structure

```
├── main_router.py          # Page classifier + orchestrator
├── grid_table_extractor.py # CV-based grid detection + cell OCR
├── ocr_table_extractor.py  # img2table detection + spatial reconstruction fallback
├── requirements.txt
└── out/                    # Generated outputs (gitignored)
```

---

## Dependencies

- `opencv-python` — image processing, morphological ops
- `pytesseract` — OCR (requires Tesseract system install)
- `img2table` — table bounding box detection
- `pymupdf` — PDF to image rasterization
- `pandas`, `numpy`, `pillow`

---

## Sample results

Tested on a 3-page groundwater analytical results report (RACER format):

- Page 1 & 2: grid extractor — extracted 2 wide multi-column tables cleanly
- Page 3: OCR fallback — detected **12 tables** from a dense site map layout with borderless grids

---

## Author

Smruthi Bangalore Thandava Murthy
