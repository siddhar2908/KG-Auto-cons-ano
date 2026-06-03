"""
extractors/page_classifier.py
------------------------------
Rule-based page classifier — no LLM, runs in <1ms per page.
Categorizes each page to route it to the right extractor.

Categories:
    SKIP   — TOC, blank, header-only, purely administrative
    TEXT   — text-rich engineering content → fact extractor
    TABLE  — table-dominant, minimal prose → table extractor only
    MIXED  — both text and tables → both extractors
    IMAGE  — low text, embedded images → vision model

Used by run_extraction.py to avoid wasting LLM calls on
pages that have no extractable engineering content.
"""

import re
from dataclasses import dataclass
from enum import Enum

# ─── Page category ────────────────────────────────────────────────────────────

class PageCategory(str, Enum):
    SKIP  = "SKIP"
    TEXT  = "TEXT"
    TABLE = "TABLE"
    MIXED = "MIXED"
    IMAGE = "IMAGE"


@dataclass
class PageClassification:
    category:     PageCategory
    reason:       str
    text_density: float   # chars per line
    table_score:  float   # 0-1, likelihood of table content
    image_score:  float   # 0-1, likelihood of image-dominant page
    line_count:   int
    char_count:   int


# ─── Pattern library ──────────────────────────────────────────────────────────

# TOC patterns — lines that are purely "Section Title .... 42"
_TOC_LINE = re.compile(
    r"^.{3,80}[.\s]{3,}\s*\d{1,4}\s*$",
    re.MULTILINE
)

# Blank / header-only indicators
_HEADER_ONLY_PATTERNS = [
    re.compile(r"^\s*chapter\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*section\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*annexure\s*[-\d]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*appendix\s+[a-z\d]", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*list\s+of\s+(figures|tables|abbreviations|contents)", re.IGNORECASE),
    re.compile(r"^\s*table\s+of\s+contents\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*index\s*$", re.IGNORECASE | re.MULTILINE),
]

# Administrative / non-engineering content patterns
_ADMIN_PATTERNS = [
    re.compile(r"declaration|certificate of|this is to certify", re.IGNORECASE),
    re.compile(r"acknowledgement|acknowledgment|foreword|preface", re.IGNORECASE),
    re.compile(r"^\s*signature\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"name\s*:\s*_+|designation\s*:\s*_+|date\s*:\s*_+", re.IGNORECASE),
]

# Table indicator patterns
_TABLE_PATTERNS = [
    re.compile(r"\|"),                          # pipe separators
    re.compile(r"\t.+\t"),                      # tab-separated columns
    re.compile(r"sr\.?\s*no\.?", re.IGNORECASE),  # serial number column
    re.compile(r"s\.?\s*no\.?", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*[\.\)]\s+\w", re.MULTILINE),  # numbered list rows
    re.compile(r"(qty|quantity|unit|rate|amount|cost|total)\s*[\|:]", re.IGNORECASE),
    re.compile(r"sl\.?\s*no", re.IGNORECASE),
    re.compile(r"description\s+of\s+work", re.IGNORECASE),
]

# Engineering content indicators — these pages are worth extracting
_ENGINEERING_PATTERNS = [
    re.compile(r"\d+\.?\d*\s*(mm|cm|m|km|kn|kpa|mpa|mw|kva|rpm|kmph|km/h)", re.IGNORECASE),
    re.compile(r"(design|proposed|existing|required|minimum|maximum)\s+\w+\s+\w*\s*=?\s*\d", re.IGNORECASE),
    re.compile(r"(grade|class|type|category)\s+[a-z0-9]+", re.IGNORECASE),
    re.compile(r"(IRC|IS|RDSO|BIS|DMRC|MMRDA)\s*[:\-]?\s*\d+", re.IGNORECASE),
    re.compile(r"(bearing capacity|shear strength|compressive strength|tensile)", re.IGNORECASE),
    re.compile(r"(viaduct|tunnel|station|depot|alignment|corridor|platform)", re.IGNORECASE),
]

# Figure / image page indicators
_IMAGE_PAGE_PATTERNS = [
    re.compile(r"^\s*fig(ure)?\s*[\d\.\-]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(drawing|plan|section|elevation|layout)\s+no", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*plate\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"not\s+to\s+scale|n\.t\.s\.", re.IGNORECASE),
    re.compile(r"legend\s*:", re.IGNORECASE),
]


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _toc_ratio(text: str) -> float:
    """Fraction of lines that look like TOC entries."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    toc_lines = sum(1 for l in lines if _TOC_LINE.match(l.strip()))
    return toc_lines / len(lines)


def _table_score(text: str, has_grid_lines: bool, rect_count: int) -> float:
    """
    Score 0-1 for how table-dominant the page is.
    Indian DPR PDFs predominantly use rect-based tables (not line-based),
    so rect_count is the primary structural signal.
    """
    score = 0.0
    # Structural signals — rects are the primary indicator in Indian DPRs
    # Most DPR tables use filled/bordered rectangles as cells, not line objects
    if rect_count > 50:
        score += 0.5   # very likely a table-heavy page
    elif rect_count > 20:
        score += 0.35  # probably has a table
    elif rect_count > 6:
        score += 0.2   # may have a small table
    # Line-based tables (less common in Indian DPRs)
    if has_grid_lines:
        score += 0.3
    # Text pattern signals
    pattern_hits = sum(1 for p in _TABLE_PATTERNS if p.search(text))
    score += min(0.3, pattern_hits * 0.08)
    return min(1.0, score)


def _image_score(text: str, image_count: int, char_count: int) -> float:
    """Score 0-1 for how image-dominant the page is."""
    score = 0.0
    if image_count > 0:
        score += min(0.5, image_count * 0.2)
    if char_count < 100 and image_count > 0:
        score += 0.4
    pattern_hits = sum(1 for p in _IMAGE_PAGE_PATTERNS if p.search(text))
    score += min(0.2, pattern_hits * 0.1)
    return min(1.0, score)


def _engineering_density(text: str) -> float:
    """Score 0-1 for engineering content richness."""
    hits = sum(1 for p in _ENGINEERING_PATTERNS if p.search(text))
    return min(1.0, hits / len(_ENGINEERING_PATTERNS))


# ─── Main classifier ──────────────────────────────────────────────────────────

def classify_page(
    text: str,
    page_num: int,
    image_count: int = 0,
    has_grid_lines: bool = False,
    rect_count: int = 0,
) -> PageClassification:
    """
    Classify a single page into a category.

    Args:
        text:           Extracted text content of the page.
        page_num:       0-indexed page number (for context).
        image_count:    Number of embedded images on the page.
        has_grid_lines: True if pdfplumber detected line-based grid.
        rect_count:     Number of rectangles (cell borders) detected.

    Returns:
        PageClassification with category and diagnostic scores.
    """
    char_count = len(text.strip())
    lines      = [l for l in text.split("\n") if l.strip()]
    line_count = len(lines)
    text_density = char_count / max(line_count, 1)

    # ── SKIP: blank page
    if char_count < 30:
        return PageClassification(
            category=PageCategory.SKIP, reason="blank or near-blank page",
            text_density=0, table_score=0, image_score=0,
            line_count=line_count, char_count=char_count
        )

    # ── SKIP: TOC page (>50% of lines are TOC entries)
    if _toc_ratio(text) > 0.5:
        return PageClassification(
            category=PageCategory.SKIP, reason="table of contents page",
            text_density=text_density, table_score=0, image_score=0,
            line_count=line_count, char_count=char_count
        )

    # ── SKIP: header-only page
    for pat in _HEADER_ONLY_PATTERNS:
        if pat.search(text) and char_count < 200:
            return PageClassification(
                category=PageCategory.SKIP,
                reason=f"header-only page ({pat.pattern[:30]})",
                text_density=text_density, table_score=0, image_score=0,
                line_count=line_count, char_count=char_count
            )

    # ── SKIP: purely administrative content
    admin_hits = sum(1 for p in _ADMIN_PATTERNS if p.search(text))
    if admin_hits >= 2 and char_count < 500:
        return PageClassification(
            category=PageCategory.SKIP, reason="administrative page (declaration/signature)",
            text_density=text_density, table_score=0, image_score=0,
            line_count=line_count, char_count=char_count
        )

    # ── Compute scores
    t_score = _table_score(text, has_grid_lines, rect_count)
    i_score = _image_score(text, image_count, char_count)
    e_density = _engineering_density(text)

    # ── IMAGE: low text, image signals
    if i_score > 0.6 and char_count < 150:
        return PageClassification(
            category=PageCategory.IMAGE,
            reason=f"image-dominant page (i_score={i_score:.2f}, chars={char_count})",
            text_density=text_density, table_score=t_score, image_score=i_score,
            line_count=line_count, char_count=char_count
        )

    # ── TABLE: high table score, low engineering text
    if t_score > 0.6 and e_density < 0.2:
        return PageClassification(
            category=PageCategory.TABLE,
            reason=f"table-dominant page (t_score={t_score:.2f})",
            text_density=text_density, table_score=t_score, image_score=i_score,
            line_count=line_count, char_count=char_count
        )

    # ── MIXED: both table and text signals
    if t_score > 0.3 and (e_density > 0.2 or char_count > 300):
        return PageClassification(
            category=PageCategory.MIXED,
            reason=f"mixed text+table (t_score={t_score:.2f}, e_density={e_density:.2f})",
            text_density=text_density, table_score=t_score, image_score=i_score,
            line_count=line_count, char_count=char_count
        )

    # ── TEXT: default for pages with sufficient content
    if char_count >= 30:
        return PageClassification(
            category=PageCategory.TEXT,
            reason=f"text page (chars={char_count}, e_density={e_density:.2f})",
            text_density=text_density, table_score=t_score, image_score=i_score,
            line_count=line_count, char_count=char_count
        )

    # ── Fallback SKIP
    return PageClassification(
        category=PageCategory.SKIP, reason="insufficient content",
        text_density=text_density, table_score=t_score, image_score=i_score,
        line_count=line_count, char_count=char_count
    )


def classify_pages(pages: list) -> dict[int, PageClassification]:
    """
    Classify a list of PageContent objects.
    Returns {page_num: PageClassification}.
    PageContent objects may have rect_count and has_grid_lines set by the loader.
    """
    results = {}
    for page in pages:
        # Use rect_count and has_grid_lines if available on the PageContent object
        rect_count    = getattr(page, 'rect_count',    0)
        has_grid_lines = getattr(page, 'has_grid_lines', False)
        clf = classify_page(
            text=page.text,
            page_num=page.page_num,
            image_count=page.image_count,
            has_grid_lines=has_grid_lines,
            rect_count=rect_count,
        )
        results[page.page_num] = clf
    return results


def summarize_classifications(classifications: dict) -> dict:
    """Print-friendly summary of page classification results."""
    from collections import Counter
    counts = Counter(clf.category.value for clf in classifications.values())
    return dict(counts)