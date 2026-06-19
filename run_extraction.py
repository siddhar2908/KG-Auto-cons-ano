#!/usr/bin/env python3
"""
run_extraction.py
-----------------
STEP 1A: Raw extraction — zero arguments needed for normal usage.

Default behaviour (just run: python run_extraction.py):
  - Scans data/dpr/       for DPR PDFs     → extracts facts + tables
  - Scans data/rulebooks/ for rulebook PDFs → extracts rules
  - Uses DPR_START_PAGE, DPR_MAX_PAGES from config/settings.py
  - Parallel workers from EXTRACTION_WORKERS in settings.py

Override via flags when needed:
  --dpr path/to/specific.pdf        process one specific DPR
  --rulebook path/to/specific.pdf   process one specific rulebook
  --max-pages 30                    override for quick testing
  --start-page 1                    override start page
  --workers 6                       override worker count
  --sector "Bridges"                skip auto-classification

No DB writes — outputs raw JSON to data/processed/<doc_id>/
Next step: python run_push.py
"""

import sys
import argparse
import uuid
import json
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TimeElapsedColumn
)

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    SECTORS, DPR_INPUT_DIR, RULEBOOKS_INPUT_DIR, PROCESSED_DIR,
    DPR_START_PAGE, DPR_MAX_PAGES, RULEBOOK_START_PAGE, RULEBOOK_MAX_PAGES,
    EXTRACTION_WORKERS,
)
from utils.ollama_client import classify_sector, vision_describe_image
from extractors.document_loader import load_document
from extractors.table_extractor import extract_tables_from_page
from extractors.fact_extractor import extract_facts_from_page
from extractors.rule_extractor import extract_rules_from_text, detect_standard_name
from extractors.page_classifier import classify_pages, PageCategory, summarize_classifications

console = Console()
_lock = threading.Lock()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DPR Validation — Step 1A: Raw Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default (no flags):
  Scans data/dpr/ and data/rulebooks/ automatically.
  Page range and workers come from config/settings.py.

Override examples:
  python run_extraction.py --dpr data/dpr/specific.pdf
  python run_extraction.py --max-pages 30              # quick test
  python run_extraction.py --start-page 1              # from page 1
  python run_extraction.py --workers 6                 # more parallelism
        """
    )
    p.add_argument("--dpr",        type=Path, help="Process one specific DPR file")
    p.add_argument("--rulebook",   type=Path, help="Process one specific rulebook file")
    p.add_argument("--standard",   type=str,  help="Standard name for --rulebook (auto-detected if not given)")
    p.add_argument("--sector",     type=str,  help="Force sector (skips auto-classification)")
    p.add_argument("--doc-id",     type=str,  help="Custom document ID")
    p.add_argument("--start-page", type=int,  default=None, help=f"Override start page (default: {DPR_START_PAGE})")
    p.add_argument("--max-pages",  type=int,  default=None, help=f"Override max pages (default: {DPR_MAX_PAGES or 'all'})")
    p.add_argument("--workers",    type=int,  default=None, help=f"Override workers (default: {EXTRACTION_WORKERS})")
    p.add_argument("--skip-tables", action="store_true")
    p.add_argument("--skip-facts",  action="store_true")
    return p.parse_args()


# ─── Page range ───────────────────────────────────────────────────────────────

def resolve_page_range(total: int, start: int, max_p: int) -> list[int]:
    start_0 = max(0, start - 1)
    pages   = list(range(start_0, total))
    return pages[:max_p] if max_p else pages


# ─── Save ─────────────────────────────────────────────────────────────────────

def get_doc_dir(doc_id: str) -> Path:
    d = PROCESSED_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Per-page worker ──────────────────────────────────────────────────────────

def _should_use_fast_model(text: str) -> bool:
    """
    Route simple pages to llama3.1:8b instead of qwen2.5:14b.
    Simple = short text with few numeric values.
    2x faster, sufficient quality for low-density pages.
    """
    import re as _re
    if not text:
        return True
    char_count = len(text.strip())
    if char_count > 600:
        return False  # rich content → heavy model
    numeric_hits = len(_re.findall(r'\d+\.?\d*\s*(?:m|km|mm|%|kmph|Rs|crore|lakh|ha|cum)', text, _re.I))
    return numeric_hits < 3  # few engineering values → fast model ok


def _has_extractable_content(text: str) -> bool:
    """
    Quick pre-filter: does this page have enough signal for the LLM?
    Saves LLM calls on pages that will return [] anyway.
    """
    import re as _re
    if not text or len(text.strip()) < 80:
        return False
    # Must have at least one numeric value
    has_numbers = bool(_re.search(r'\d+\.?\d*', text))
    if not has_numbers:
        return False
    # Must not be pure header/title page
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return False
    return True


def _process_page(page, doc_id, sector, dpr_path, skip_facts, skip_tables, clf):
    cat    = clf.category
    facts  = []
    tables = []
    try:
        if not skip_facts and cat in (PageCategory.TEXT, PageCategory.MIXED):
            if page.text and _has_extractable_content(page.text):
                facts = extract_facts_from_page(
                    text=page.text, doc_id=doc_id, sector=sector,
                    page_num=page.page_num, write_to_db=False,
                )
                for f in facts:
                    f["doc_id"]      = doc_id
                    f["source_page"] = page.page_num + 1
                    f["sector"]      = sector        # ← critical: same bug as rules had

        if not skip_tables and cat in (PageCategory.TABLE, PageCategory.MIXED):
            tables = extract_tables_from_page(
                pdf_path=dpr_path,
                page_num=page.page_num,
                context=f"{sector} DPR page {page.page_num + 1}",
            )

        # IMAGE pages: vision model (llama3.2-vision) skipped if unsupported
        # mllama architecture requires Ollama >= 0.3.0 with vision support
        # These pages (maps, diagrams) have no extractable engineering text anyway
        if not skip_facts and cat == PageCategory.IMAGE and page.image_count > 0:
            pass  # skip gracefully — image pages contain maps/diagrams not engineering facts

    except Exception as e:
        logger.warning(f"Page {page.page_num + 1}: {e}")

    return {"page_num": page.page_num, "category": cat.value,
            "facts": facts, "table_rows": tables}


# ─── DPR extraction ───────────────────────────────────────────────────────────

def extract_dpr(
    dpr_path: Path,
    sector: str = None,
    doc_id: str = None,
    start_page: int = None,
    max_pages: int = None,
    workers: int = None,
    skip_facts: bool = False,
    skip_tables: bool = False,
) -> dict:

    doc_id    = doc_id or str(uuid.uuid4())[:8]
    doc_dir   = get_doc_dir(doc_id)
    start     = start_page if start_page is not None else DPR_START_PAGE
    max_p     = max_pages  if max_pages  is not None else DPR_MAX_PAGES
    n_workers = workers    if workers    is not None else EXTRACTION_WORKERS

    console.rule(f"[bold]DPR: {dpr_path.name}[/bold]")
    console.print(f"   Start page: {start}  |  Max pages: {max_p or 'all'}  |  Workers: {n_workers}")

    doc = load_document(dpr_path, doc_id)

    # Sector classification
    if sector:
        sector_conf = 1.0
        console.print(f"   Sector (manual): [bold green]{sector}[/bold green]")
    else:
        console.print("🔍 Classifying sector...")
        sample = " ".join(p.text for p in doc.pages[:5])[:4000]
        sector, sector_conf = classify_sector(sample, SECTORS)
        console.print(f"   Sector: [bold green]{sector}[/bold green] ({sector_conf:.0%})")

    pages_idx    = resolve_page_range(doc.total_pages, start, max_p)
    pages_subset = [doc.pages[i] for i in pages_idx if i < len(doc.pages)]
    console.print(f"   Pages: {pages_idx[0]+1}–{pages_idx[-1]+1} ({len(pages_subset)} of {doc.total_pages})")

    # Page classification
    classifications = classify_pages(pages_subset)
    summary         = summarize_classifications(classifications)
    skip_count      = summary.get("SKIP", 0)
    pages_active    = [p for p in pages_subset
                       if classifications[p.page_num].category != PageCategory.SKIP]

    console.print(f"   Classification: {summary} → [green]{len(pages_active)} active[/green], [dim]{skip_count} skipped[/dim]")

    # Save metadata
    save_json({
        "doc_id": doc_id, "filename": dpr_path.name,
        "doc_type": doc.doc_type, "total_pages": doc.total_pages,
        "processed_pages": [p+1 for p in pages_idx],
        "page_range": f"{pages_idx[0]+1}–{pages_idx[-1]+1}",
        "sector": sector, "sector_confidence": sector_conf,
        "is_scanned": doc.is_scanned,
        "page_classification_summary": summary,
        "extracted_at": datetime.now().isoformat(),
        "doc_kind": "dpr", "workers": n_workers,
    }, doc_dir / "metadata.json")

    # Parallel extraction
    all_facts  = []
    all_tables = {}

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("Extracting...", total=len(pages_active))

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_process_page, page, doc_id, sector, dpr_path,
                          skip_facts, skip_tables, classifications[page.page_num]): page.page_num
                for page in pages_active
            }
            for future in as_completed(futures):
                r = future.result()
                with _lock:
                    prog.advance(task)
                    prog.update(task, description=f"Page {r['page_num']+1} ({r['category']})")
                all_facts.extend(r["facts"])
                if r["table_rows"]:
                    all_tables[str(r["page_num"] + 1)] = r["table_rows"]

    save_json(all_facts,  doc_dir / "facts_raw.json")
    save_json(all_tables, doc_dir / "tables_raw.json")

    total_rows = sum(len(v) for v in all_tables.values())
    console.print(
        f"\n✅ [bold]{dpr_path.name}[/bold] done:\n"
        f"   Facts: [green]{len(all_facts)}[/green]  "
        f"Table rows: [green]{total_rows}[/green]  "
        f"Skipped pages: [dim]{skip_count}[/dim]\n"
        f"   Saved → [cyan]{doc_dir}[/cyan]"
    )
    return {"doc_id": doc_id, "sector": sector,
            "total_facts": len(all_facts), "table_pages": len(all_tables)}


# ─── Rulebook extraction ──────────────────────────────────────────────────────

def extract_rulebook(
    rb_path: Path,
    standard_name: str = None,
    sector: str = None,
    doc_id: str = None,
    start_page: int = None,
    max_pages: int = None,
    workers: int = None,
) -> dict:

    doc_id    = (doc_id or str(uuid.uuid4())[:8]) + "_rules"
    doc_dir   = get_doc_dir(doc_id)
    start     = start_page if start_page is not None else RULEBOOK_START_PAGE
    max_p     = max_pages  if max_pages  is not None else RULEBOOK_MAX_PAGES
    n_workers = workers    if workers    is not None else EXTRACTION_WORKERS

    console.rule(f"[bold]Rulebook: {rb_path.name}[/bold]")

    doc = load_document(rb_path, doc_id)
    standard_name = standard_name or detect_standard_name(doc.raw_text[:3000])
    console.print(f"   Standard: [bold cyan]{standard_name}[/bold cyan]")

    if not sector:
        sector, _ = classify_sector(doc.raw_text[:3000], SECTORS)
    console.print(f"   Sector: [bold green]{sector}[/bold green]")

    pages_idx    = resolve_page_range(doc.total_pages, start, max_p)
    pages_subset = [doc.pages[i] for i in pages_idx if i < len(doc.pages)]
    classifications = classify_pages(pages_subset)
    pages_active    = [p for p in pages_subset
                       if classifications[p.page_num].category != PageCategory.SKIP]
    console.print(f"   Pages: {len(pages_active)} active of {len(pages_subset)}")

    all_rules = []
    rule_lock = threading.Lock()

    def _extract_page_rules(page):
        if not page.text or len(page.text.strip()) < 50:
            return []
        rules = extract_rules_from_text(page.text, standard_name, sector, write_to_db=False)
        for r in rules:
            r["doc_id"]      = doc_id
            r["source_page"] = page.page_num + 1
            r["sector"]      = sector          # ← critical: sector must be on every rule
        return rules

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("Extracting rules...", total=len(pages_active))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_extract_page_rules, p): p.page_num for p in pages_active}
            for future in as_completed(futures):
                rules = future.result()
                with rule_lock:
                    all_rules.extend(rules)
                    prog.advance(task)

    save_json({"doc_id": doc_id, "filename": rb_path.name,
               "standard_name": standard_name, "sector": sector,
               "total_pages": doc.total_pages,
               "extracted_at": datetime.now().isoformat(),
               "doc_kind": "rulebook"}, doc_dir / "metadata.json")
    save_json(all_rules, doc_dir / "rules_raw.json")

    console.print(
        f"\n✅ [bold]{rb_path.name}[/bold] done:\n"
        f"   Rules: [green]{len(all_rules)}[/green]  "
        f"Saved → [cyan]{doc_dir}[/cyan]"
    )
    return {"doc_id": doc_id, "standard_name": standard_name,
            "sector": sector, "total_rules": len(all_rules)}


# ─── Folder scanner ───────────────────────────────────────────────────────────

def scan_and_extract_all(args) -> dict:
    """
    Scan data/dpr/ and data/rulebooks/ and process everything found.
    This is the default zero-argument mode.
    """
    results = {"dpr": None, "rulebooks": []}

    # ── Rulebooks first (so rules exist in Neo4j before DPR validation)
    # Scan for all supported document types
    rb_files = sorted([
        p for ext in ("*.pdf", "*.docx", "*.doc", "*.txt")
        for p in RULEBOOKS_INPUT_DIR.glob(ext)
    ])
    if rb_files:
        console.print(f"\n📚 Found {len(rb_files)} rulebook(s) in [cyan]{RULEBOOKS_INPUT_DIR}[/cyan]")
        for rb_path in rb_files:
            r = extract_rulebook(
                rb_path=rb_path,
                start_page=args.start_page,
                max_pages=args.max_pages,
                workers=args.workers,
            )
            results["rulebooks"].append(r)
    else:
        console.print(f"[dim]No rulebooks found in {RULEBOOKS_INPUT_DIR} — skipping[/dim]")

    # ── DPR files
    dpr_files = sorted([
        p for ext in ("*.pdf", "*.docx", "*.doc", "*.txt")
        for p in DPR_INPUT_DIR.glob(ext)
    ])
    if not dpr_files:
        console.print(f"[red]No DPR PDFs found in {DPR_INPUT_DIR}[/red]")
        console.print(f"[yellow]Drop your DPR PDF into: {DPR_INPUT_DIR}[/yellow]")
        sys.exit(1)

    console.print(f"\n📄 Found {len(dpr_files)} DPR(s) in [cyan]{DPR_INPUT_DIR}[/cyan]")

    # Process first DPR found (most common case: one DPR at a time)
    # For multiple DPRs, each gets its own doc_id and pipeline run
    for dpr_path in dpr_files:
        r = extract_dpr(
            dpr_path=dpr_path,
            sector=args.sector,
            start_page=args.start_page,
            max_pages=args.max_pages,
            workers=args.workers,
            skip_facts=args.skip_facts,
            skip_tables=args.skip_tables,
        )
        results["dpr"] = r
        # Only process first DPR in auto mode — subsequent DPRs need separate runs
        if len(dpr_files) > 1:
            console.print(
                f"[yellow]Note: {len(dpr_files)-1} more DPR(s) found. "
                f"Process them with --dpr flag separately.[/yellow]"
            )
        break

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    results = {}

    # ── Explicit file mode (--dpr or --rulebook flags given)
    if args.dpr or args.rulebook:
        if args.rulebook:
            if not args.rulebook.exists():
                console.print(f"[red]Not found: {args.rulebook}[/red]")
                sys.exit(1)
            results["rulebook"] = extract_rulebook(
                rb_path=args.rulebook,
                standard_name=args.standard,
                sector=args.sector,
                doc_id=args.doc_id,
                start_page=args.start_page,
                max_pages=args.max_pages,
                workers=args.workers,
            )
        if args.dpr:
            if not args.dpr.exists():
                console.print(f"[red]Not found: {args.dpr}[/red]")
                sys.exit(1)
            results["dpr"] = extract_dpr(
                dpr_path=args.dpr,
                sector=args.sector,
                doc_id=args.doc_id,
                start_page=args.start_page,
                max_pages=args.max_pages,
                workers=args.workers,
                skip_facts=args.skip_facts,
                skip_tables=args.skip_tables,
            )

    # ── Auto folder scan mode (no flags given)
    else:
        results = scan_and_extract_all(args)

    # Save state
    state_file = Path("output/.extraction_state.json")
    state_file.parent.mkdir(exist_ok=True)
    state_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    dpr_id = results.get("dpr", {}).get("doc_id", "none") if results.get("dpr") else "none"
    console.print(Panel(
        json.dumps(results, indent=2),
        title="[bold green]Extraction Complete[/bold green]",
        border_style="green",
    ))
    console.print(f"\nNext step: [bold]python run_push.py[/bold]")


if __name__ == "__main__":
    main()