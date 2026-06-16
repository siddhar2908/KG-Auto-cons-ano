"""
config/settings.py
------------------
Central configuration for the DPR Validation System.
All settings here — no hardcoding elsewhere in the codebase.

TO CONFIGURE FOR YOUR RUN:
  1. Set NEO4J_PASSWORD to your Neo4j password
  2. Place DPR PDFs in:       data/dpr/
  3. Place rulebooks in:      data/rulebooks/
  4. Set DPR_MAX_PAGES = 50 for testing, None for full production run
  5. Run: python run_extraction.py  (no flags needed)
"""

from pathlib import Path
from typing import Optional

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR    = BASE_DIR / "output"

# ─── Input folders ────────────────────────────────────────────────────────────
# Drop your files here — run_extraction.py scans these automatically

DPR_INPUT_DIR       = DATA_DIR / "dpr"        # ← put DPR PDFs here
RULEBOOKS_INPUT_DIR = DATA_DIR / "rulebooks"  # ← put standards/rulebooks here

# Create all directories on import
for _d in [DPR_INPUT_DIR, RULEBOOKS_INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Extraction page defaults ─────────────────────────────────────────────────
# These apply when running python run_extraction.py with no flags.
# Override per-run with --start-page / --max-pages flags if needed.

DPR_START_PAGE  = 8   # Skip cover page + TOC (usually pages 1-20)
                        # Set to 1 to process from the very first page

DPR_MAX_PAGES   = None  # None = process entire document from DPR_START_PAGE
                        # Set to 50 for a quick test run
                        # Set to None for full production run

RULEBOOK_START_PAGE = 1    # Rulebooks don't have TOC pages to skip
RULEBOOK_MAX_PAGES  = None # None = process entire rulebook

EXTRACTION_WORKERS  = 6    # Parallel workers for page extraction
                           # Increase to 6-8 on powerful machines
                           # Decrease to 2 if Ollama becomes unstable

# ─── Neo4j ────────────────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "dontgetmebanned2002"   # ← CHANGE THIS to your Neo4j password
NEO4J_DATABASE = "neo4j"      # Community Edition: always "neo4j"

# ─── Ollama ───────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL     = "http://localhost:11434"
OLLAMA_TEXT_MODEL   = "qwen2.5:14b"          # heavy model: fact + triple extraction
OLLAMA_VISION_MODEL = "llama3.2-vision:11b"  # vision: tables + scanned pages
OLLAMA_EMBED_MODEL  = "mxbai-embed-large"    # embeddings for FAISS index

OLLAMA_TIMEOUT     = 120   # seconds per request
OLLAMA_MAX_RETRIES = 3

# ─── Extraction settings ──────────────────────────────────────────────────────

CHUNK_SIZE     = 1200   # tokens per chunk sent to LLM for fact extraction
CHUNK_OVERLAP  = 150
PAGE_DPI       = 200    # DPI for rasterising pages for vision fallback
TABLE_MIN_FILL = 0.75   # pdfplumber/camelot table acceptance threshold (0-1)

# ─── 8 RITES Sectors ─────────────────────────────────────────────────────────

SECTORS = [
    "Rail Infrastructure",
    "Bridges",
    "Tunnels",
    "Metro",
    "Mobility / Logistics",
    "Highways",
    "Ports",
    "Airports",
]

SECTOR_KEYS = {
    "Rail Infrastructure":  "rail",
    "Bridges":              "bridges",
    "Tunnels":              "tunnels",
    "Metro":                "metro",
    "Mobility / Logistics": "mobility",
    "Highways":             "highways",
    "Ports":                "ports",
    "Airports":             "airports",
}

SECTOR_STANDARDS = {
    "rail":      ["RDSO", "IRS", "Indian Railways"],
    "bridges":   ["IRC:5", "IRC:6", "IRC:78", "IS:456", "IS:800", "RDSO bridges"],
    "tunnels":   ["IRC:SP:91", "ITA guidelines", "NATM"],
    "metro":     ["DMRC standards", "RDSO metro", "IS:14665"],
    "mobility":  ["MoRTH guidelines", "IRC standards", "logistics norms"],
    "highways":  ["IRC:37", "IRC:58", "IRC:86", "MoRTH", "IS:1343"],
    "ports":     ["IS:4651", "PIANC", "MoPSW guidelines"],
    "airports":  ["ICAO Annex 14", "AAI standards", "IS:875"],
}

# ─── Sector hierarchy ─────────────────────────────────────────────────────────
# Defines which sectors inherit rules from parent sectors.
# A Metro DPR fact is validated against Metro rules AND Rail Infrastructure rules.
# This is the single source of truth — no hardcoding in validation logic.
#
# Key relationships:
#   Metro      ← Rail Infrastructure (signalling, track, rolling stock standards)
#   Bridges    ← Highways (IRC loading standards apply to both)
#   Tunnels    ← Rail Infrastructure (for rail tunnels)
#   Airports   ← (standalone — ICAO/AAI only)
#   Ports      ← (standalone — IS:4651/PIANC only)

SECTOR_PARENTS = {
    "Metro":               ["Rail Infrastructure"],
    "Bridges":             ["Highways"],
    "Tunnels":             ["Rail Infrastructure"],
    "Mobility / Logistics": ["Highways", "Rail Infrastructure"],
    # These sectors inherit from themselves only
    "Rail Infrastructure": [],
    "Highways":            [],
    "Ports":               [],
    "Airports":            [],
}

def get_applicable_sectors(sector: str) -> list[str]:
    """
    Return all sectors whose rules apply to a given DPR sector.
    Includes the sector itself plus all parent sectors.
    Example: get_applicable_sectors("Metro") → ["Metro", "Rail Infrastructure"]
    """
    parents = SECTOR_PARENTS.get(sector, [])
    return [sector] + parents

# ─── Consistency + anomaly thresholds ────────────────────────────────────────

ANOMALY_SIGMA_THRESHOLD  = 3.0   # z-score flag threshold
ORDER_OF_MAGNITUDE_RATIO = 10.0  # OOM error multiplier
NUMERIC_DUPLICATE_RATIO  = 0.85  # duplicate value ratio flag

# ─── Validation ───────────────────────────────────────────────────────────────

MIN_FACT_CONFIDENCE = 0.45   # facts below this dropped before validation

# ─── Severity levels ──────────────────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    INFO     = "INFO"

# ─── Neo4j node labels ────────────────────────────────────────────────────────

class NodeLabel:
    DOCUMENT = "Document"
    FACT     = "Fact"
    RULE     = "Rule"
    ENTITY   = "Entity"
    ONTOLOGY = "OntologyClass"
    SECTOR   = "Sector"
    VIOLATION= "Violation"

# ─── Neo4j relationship types ─────────────────────────────────────────────────

class RelType:
    HAS_FACT      = "HAS_FACT"
    HAS_RULE      = "HAS_RULE"
    HAS_ENTITY    = "HAS_ENTITY"   # Document → Entity (KG builder)
    TRIPLE        = "TRIPLE"       # Entity → Entity (KG triples)
    BELONGS_TO    = "BELONGS_TO"
    DEPENDS_ON    = "DEPENDS_ON"
    COMPLIES_WITH = "COMPLIES_WITH"
    VIOLATES      = "VIOLATES"
    CONTRADICTS   = "CONTRADICTS"
    DERIVED_FROM  = "DERIVED_FROM"
    PART_OF       = "PART_OF"
    INSTANCE_OF   = "INSTANCE_OF"