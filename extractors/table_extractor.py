"""
extractors/table_extractor.py
------------------------------
Deterministic 3-stage table extraction cascade.

Stage 1: pdfplumber  (fast, no external deps)
Stage 2: camelot     (better for bordered/lattice tables)
Stage 3: Vision LLM  (Ollama llama3.2-vision — guaranteed fallback)

Each stage produces a list[dict] (rows as dicts with column headers as keys).
A stage is accepted if the fill rate (non-null cells / total cells) >= TABLE_MIN_FILL.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pdfplumber
import camelot
import pandas as pd
from loguru import logger

from config.settings import TABLE_MIN_FILL, PAGE_DPI
from utils.ollama_client import vision_extract_table_json


# ─── Acceptance check ─────────────────────────────────────────────────────────

def _fill_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    total = sum(len(r) for r in rows)
    filled = sum(1 for r in rows for v in r.values() if v is not None and str(v).strip() != "")
    return filled / total if total > 0 else 0.0


def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate and nan column headers.
    - nan headers (from merged cells) → col_1, col_2, etc.
    - Duplicate headers → Value, Value_2, Value_3 etc.
    Common in engineering tables with merged or repeated header cells.
    """
    import math
    seen: dict[str, int] = {}
    new_cols = []
    col_counter = 0
    for col in df.columns:
        col_counter += 1
        # Handle nan/None/float nan from merged cells
        if col is None or (isinstance(col, float) and math.isnan(col)):
            col_str = f"col_{col_counter}"
        else:
            col_str = str(col).strip()
            # pdfplumber sometimes returns "nan" as string
            if col_str.lower() in ("nan", "none", ""):
                col_str = f"col_{col_counter}"

        if col_str in seen:
            seen[col_str] += 1
            new_cols.append(f"{col_str}_{seen[col_str]}")
        else:
            seen[col_str] = 1
            new_cols.append(col_str)
    df.columns = new_cols
    return df


def _df_to_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to list-of-dicts, using first row as header if needed."""
    if df is None or df.empty:
        return []
    # If headers are 0,1,2,... (unnamed), promote first row to header
    if all(isinstance(c, int) for c in df.columns):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    # Deduplicate column names before conversion (avoids pandas UserWarning + data loss)
    df = _dedup_columns(df)
    # Replace empty strings with None
    df = df.replace(r"^\s*$", None, regex=True)
    return df.to_dict(orient="records")


# ─── Stage 1: pdfplumber ──────────────────────────────────────────────────────

def _try_pdfplumber(pdf_path: Path, page_num: int) -> list[dict]:
    """page_num is 0-indexed."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return []
            page = pdf.pages[page_num]

            # Try explicit table settings first (lattice-like)
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
            })
            if not tables:
                # Fallback to text-based detection
                tables = page.extract_tables()

            all_rows = []
            for tbl in tables:
                if not tbl:
                    continue
                df = pd.DataFrame(tbl[1:], columns=tbl[0]) if tbl else pd.DataFrame()
                rows = _df_to_rows(df)
                all_rows.extend(rows)
            return all_rows
    except Exception as e:
        logger.debug(f"pdfplumber table extraction failed p{page_num}: {e}")
        return []


# ─── Stage 2: camelot ────────────────────────────────────────────────────────

def _try_camelot(pdf_path: Path, page_num: int) -> list[dict]:
    """page_num is 0-indexed; camelot uses 1-indexed pages."""
    try:
        page_1indexed = page_num + 1
        # Try lattice first (bordered tables)
        tables = camelot.read_pdf(
            str(pdf_path), pages=str(page_1indexed),
            flavor="lattice", suppress_stdout=True
        )
        if tables.n == 0:
            # Try stream (borderless / text-aligned)
            tables = camelot.read_pdf(
                str(pdf_path), pages=str(page_1indexed),
                flavor="stream", suppress_stdout=True,
                edge_tol=50, row_tol=10
            )

        all_rows = []
        for tbl in tables:
            rows = _df_to_rows(tbl.df)
            all_rows.extend(rows)
        return all_rows
    except Exception as e:
        logger.debug(f"camelot table extraction failed p{page_num}: {e}")
        return []


# ─── Stage 3: Vision LLM fallback ────────────────────────────────────────────

def _rasterise_page(pdf_path: Path, page_num: int, dpi: int = PAGE_DPI) -> Optional[Path]:
    """
    Rasterise a single PDF page using pdftoppm (from poppler-utils).
    Returns path to the generated JPEG, or None on failure.
    page_num is 0-indexed.
    """
    page_1indexed = page_num + 1
    tmp_dir = Path(tempfile.mkdtemp())
    prefix = str(tmp_dir / "page")

    cmd = [
        "pdftoppm",
        "-jpeg",
        "-r", str(dpi),
        "-f", str(page_1indexed),
        "-l", str(page_1indexed),
        str(pdf_path),
        prefix,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning(f"pdftoppm failed: {result.stderr.decode()[:200]}")
        return None

    # pdftoppm zero-pads: page-01.jpg, page-001.jpg depending on total pages
    candidates = list(tmp_dir.glob("page-*.jpg"))
    if not candidates:
        candidates = list(tmp_dir.glob("page-*.jpeg"))
    return candidates[0] if candidates else None


def _try_vision_llm(pdf_path: Path, page_num: int, context: str = "") -> list[dict]:
    """Rasterise page and send to vision model. Final guaranteed fallback."""
    logger.info(f"  → Vision LLM fallback for table on page {page_num + 1}")
    img_path = _rasterise_page(pdf_path, page_num)
    if img_path is None:
        logger.error(f"Could not rasterise page {page_num + 1}")
        return []

    rows = vision_extract_table_json(img_path, context=context)

    # Cleanup temp file
    try:
        img_path.unlink()
        img_path.parent.rmdir()
    except Exception:
        pass

    return rows or []


# ─── Public API: extract_tables_from_page ────────────────────────────────────

def extract_tables_from_page(
    pdf_path: Path,
    page_num: int,
    context: str = "",
    min_fill: float = TABLE_MIN_FILL,
) -> list[dict]:
    """
    Extract all tables from a single PDF page using the 3-stage cascade.

    Args:
        pdf_path:  Path to the PDF file.
        page_num:  0-indexed page number.
        context:   Short description of what the page is about (helps vision LLM).
        min_fill:  Minimum fill rate to accept a stage's output (0–1).

    Returns:
        List of row dicts. Empty list if no tables found.
    """
    log_prefix = f"Page {page_num + 1}:"

    # Stage 1 — pdfplumber
    logger.debug(f"{log_prefix} Trying pdfplumber...")
    rows = _try_pdfplumber(pdf_path, page_num)
    fill = _fill_rate(rows)
    if rows and fill >= min_fill:
        logger.debug(f"{log_prefix} pdfplumber accepted (fill={fill:.0%}, rows={len(rows)})")
        return rows
    logger.debug(f"{log_prefix} pdfplumber fill={fill:.0%} < threshold, trying camelot...")

    # Stage 2 — camelot
    rows = _try_camelot(pdf_path, page_num)
    fill = _fill_rate(rows)
    if rows and fill >= min_fill:
        logger.debug(f"{log_prefix} camelot accepted (fill={fill:.0%}, rows={len(rows)})")
        return rows
    logger.debug(f"{log_prefix} camelot fill={fill:.0%} < threshold, falling back to vision LLM...")

    # Stage 3 — Vision LLM (guaranteed fallback)
    rows = _try_vision_llm(pdf_path, page_num, context=context)
    fill = _fill_rate(rows)
    logger.info(f"{log_prefix} vision LLM result: {len(rows)} rows, fill={fill:.0%}")
    return rows


def extract_all_tables(
    pdf_path: Path,
    page_nums: list[int],
    context_map: dict[int, str] = None,
) -> dict[int, list[dict]]:
    """
    Extract tables from multiple pages. Returns {page_num: [rows]}.
    context_map: optional {page_num: context_string} for vision fallback.
    """
    results = {}
    for pn in page_nums:
        ctx = (context_map or {}).get(pn, "")
        results[pn] = extract_tables_from_page(pdf_path, pn, context=ctx)
    return results