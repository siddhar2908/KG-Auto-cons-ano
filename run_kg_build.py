#!/usr/bin/env python3
"""
run_kg_build.py
---------------
STEP 1C: Build Knowledge Graph from extracted text.
Runs AFTER run_push.py, BEFORE run_engines.py.

What this does:
  1. Reloads page text from the original PDF (already parsed — no re-OCR)
  2. Runs triple extraction per page in parallel (entity-relation + event-entity)
  3. Runs concept induction (schema generation) per page
  4. Writes Entity nodes + TRIPLE edges + OntologyClass nodes to Neo4j
  5. Builds FAISS indexes (node + edge) using mxbai-embed-large

Usage:
    python run_kg_build.py --from-state
    python run_kg_build.py --doc-id <id>
    python run_kg_build.py --doc-id <id> --workers 4 --skip-faiss
    python run_kg_build.py --doc-id <id> --skip-concepts   (faster, no schema induction)
"""

import sys
import argparse
import json
import threading
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import PROCESSED_DIR, NodeLabel
from utils.neo4j_client import init_schema, run_read
from extractors.document_loader import load_document
from extractors.kg_builder import build_kg_from_page, get_kg_stats
from extractors.kg_embeddings import build_kg_index

console = Console()
_progress_lock = threading.Lock()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPR Validation — Step 1C: KG Build (triples + concepts + FAISS)",
    )
    parser.add_argument("--doc-id",       type=str,  help="Document ID")
    parser.add_argument("--from-state",   action="store_true")
    parser.add_argument("--workers",      type=int, default=4,
                        help="Parallel workers for triple extraction (default 4)")
    parser.add_argument("--skip-concepts", action="store_true",
                        help="Skip concept induction (faster, no schema layer)")
    parser.add_argument("--skip-faiss",   action="store_true",
                        help="Skip FAISS index build")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Rebuild even if triples already exist")
    return parser.parse_args()


# ─── Load metadata ────────────────────────────────────────────────────────────

def load_metadata(doc_id: str) -> dict:
    meta_path = PROCESSED_DIR / doc_id / "metadata.json"
    if not meta_path.exists():
        console.print(f"[red]No metadata for {doc_id}. Run run_extraction.py first.[/red]")
        sys.exit(1)
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ─── Per-page KG build worker ─────────────────────────────────────────────────

def _kg_build_page(page, doc_id: str, sector: str, induce_schema: bool) -> dict:
    """Worker: build KG for one page. Returns stats dict."""
    try:
        result = build_kg_from_page(
            text=page.text,
            doc_id=doc_id,
            sector=sector,
            page_num=page.page_num,
            write_to_db=True,
            induce_schema=induce_schema,
        )
        return {
            "page_num":  page.page_num,
            "triples":   len(result.triples),
            "entities":  len(result.all_entities),
            "concepts":  len(result.concept_map),
            "error":     None,
        }
    except Exception as e:
        logger.warning(f"KG build failed on page {page.page_num + 1}: {e}")
        return {
            "page_num": page.page_num,
            "triples":  0,
            "entities": 0,
            "concepts": 0,
            "error":    str(e),
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    doc_id = args.doc_id

    # Auto-load from state if no doc_id given
    state_file = Path("output/.extraction_state.json")
    if doc_id is None and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        doc_id = state.get("dpr", {}).get("doc_id")
        if doc_id:
            console.print(f"📂 Auto-loaded: doc_id=[cyan]{doc_id}[/cyan]")

    if not doc_id:
        console.print("[red]No document found. Run run_extraction.py and run_push.py first or pass --doc-id explicitly.[/red]")
        sys.exit(1)

    console.print("🔌 Connecting to Neo4j...")
    init_schema()

    # Load metadata
    metadata = load_metadata(doc_id)
    sector   = metadata.get("sector", "")
    filename = metadata.get("filename", "")

    console.print(
        f"\n📋 Document: [cyan]{filename}[/cyan] | "
        f"Sector: [green]{sector}[/green]"
    )

    # Check if already built
    if not args.force_rebuild:
        stats = get_kg_stats(doc_id)
        if stats["triples"] > 0:
            console.print(
                f"[yellow]KG already exists: "
                f"{stats['triples']} triples, {stats['entities']} entities, "
                f"{stats['concepts']} concepts.[/yellow]\n"
                f"Use --force-rebuild to rebuild."
            )
            if not args.skip_faiss:
                console.print("\n🔢 Building FAISS indexes...")
                idx_stats = build_kg_index(doc_id, force_rebuild=args.force_rebuild)
                console.print(f"   FAISS: {idx_stats['nodes']} nodes, {idx_stats['edges']} edges")
            sys.exit(0)

    # Reload document to get page text
    # (already parsed in run_extraction — no re-OCR, just re-reading parsed text)
    source_path = metadata.get("filename", "")
    # Find the actual file
    possible_paths = [
        Path("data/uploads") / source_path,
        Path(source_path),
    ]
    doc_path = next((p for p in possible_paths if p.exists()), None)

    if doc_path is None:
        console.print(f"[red]Cannot find source document: {source_path}[/red]")
        console.print("[yellow]Tip: Make sure the original PDF is still in data/uploads/[/yellow]")
        sys.exit(1)

    console.print(f"📄 Loading document text (no re-OCR)...")
    doc = load_document(doc_path, doc_id)

    # Only process pages that were originally extracted
    processed_pages_1indexed = set(metadata.get("processed_pages", []))
    pages_to_build = [
        p for p in doc.pages
        if (p.page_num + 1) in processed_pages_1indexed
        and p.text
        and len(p.text.strip()) > 50
    ]

    console.print(
        f"   Pages to build KG for: [bold]{len(pages_to_build)}[/bold] "
        f"(from {len(processed_pages_1indexed)} extracted pages)"
    )
    console.print(f"   Workers: [cyan]{args.workers}[/cyan]")
    console.print(f"   Concept induction: {'[dim]disabled[/dim]' if args.skip_concepts else '[green]enabled[/green]'}")

    # ── Parallel KG build
    total_triples  = 0
    total_entities = 0
    total_concepts = 0
    errors         = 0

    console.rule("[bold]Triple Extraction + Concept Induction[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building KG...", total=len(pages_to_build))

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _kg_build_page,
                    page, doc_id, sector,
                    not args.skip_concepts,
                ): page.page_num
                for page in pages_to_build
            }

            for future in as_completed(futures):
                result = future.result()
                with _progress_lock:
                    progress.advance(task)
                    progress.update(
                        task,
                        description=f"Page {result['page_num']+1} "
                                    f"({result['triples']} triples)"
                    )
                total_triples  += result["triples"]
                total_entities += result["entities"]
                total_concepts += result["concepts"]
                if result["error"]:
                    errors += 1

    # Final Neo4j stats
    stats = get_kg_stats(doc_id)
    console.print(
        f"\n✅ KG build complete:\n"
        f"   Triples extracted:  [green]{total_triples}[/green]\n"
        f"   Neo4j entities:     [green]{stats['entities']}[/green]\n"
        f"   Neo4j triples:      [green]{stats['triples']}[/green]\n"
        f"   Concept classes:    [green]{stats['concepts']}[/green]\n"
        f"   Page errors:        [{'red' if errors else 'dim'}]{errors}[/{'red' if errors else 'dim'}]"
    )

    # ── Build FAISS indexes
    if not args.skip_faiss:
        console.rule("[bold]FAISS Index Build[/bold]")
        console.print(f"   Embedding with mxbai-embed-large...")
        idx_stats = build_kg_index(doc_id, force_rebuild=True)
        console.print(
            f"   Node index: [green]{idx_stats['nodes']}[/green] vectors\n"
            f"   Edge index: [green]{idx_stats['edges']}[/green] vectors\n"
            f"   Saved to:   [cyan]{PROCESSED_DIR / doc_id / 'faiss'}[/cyan]"
        )
    else:
        console.print("[dim]FAISS index: skipped (use without --skip-faiss to enable)[/dim]")

    # Update state file
    state_file = Path("output/.extraction_state.json")
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {}
    state["kg_build"] = {
        "doc_id":   doc_id,
        "triples":  stats["triples"],
        "entities": stats["entities"],
        "concepts": stats["concepts"],
        "faiss_built": not args.skip_faiss,
    }
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    console.print(Panel(
        f"Entities:  {stats['entities']}\n"
        f"Triples:   {stats['triples']}\n"
        f"Concepts:  {stats['concepts']}\n"
        f"FAISS:     {'built' if not args.skip_faiss else 'skipped'}",
        title="[bold green]KG Build Complete[/bold green]",
        border_style="green",
    ))
    console.print(f"\nNext step: [bold]python run_engines.py --doc-id {doc_id}[/bold]")


if __name__ == "__main__":
    main()