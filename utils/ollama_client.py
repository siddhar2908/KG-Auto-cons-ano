"""
utils/ollama_client.py
----------------------
Thin wrapper around the Ollama Python client.
Handles retries, timeouts, JSON parsing, and vision calls.
"""

import json
import re
import base64
from pathlib import Path
from typing import Any

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from config.settings import (
    OLLAMA_TEXT_MODEL, OLLAMA_VISION_MODEL,
    OLLAMA_TIMEOUT, OLLAMA_MAX_RETRIES
)

# ─── Model router ─────────────────────────────────────────────────────────────
# Routes tasks to appropriate model size.
# Heavy structured extraction → 14b (quality critical)
# Simple classification/short prompts → 8b (fast, sufficient)

_FAST_MODEL  = "llama3.1:8b"    # sector classification, concept induction
_HEAVY_MODEL = OLLAMA_TEXT_MODEL # qwen2.5:14b — fact/triple extraction

class TaskType:
    EXTRACTION   = "extraction"    # fact/triple extraction → heavy model
    CONCEPT      = "concept"       # concept induction → fast model
    CLASSIFY     = "classify"      # sector classification → fast model
    CONSISTENCY  = "consistency"   # consistency/anomaly LLM checks → heavy model
    VALIDATION   = "validation"    # validation reasoning → heavy model

def get_model_for_task(task: str) -> str:
    """Route a task to the appropriate Ollama model."""
    fast_tasks = {TaskType.CONCEPT, TaskType.CLASSIFY}
    return _FAST_MODEL if task in fast_tasks else _HEAVY_MODEL


# ─── Text generation ──────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate(prompt: str, system: str = "", model: str = None, temperature: float = 0.1) -> str:
    """Generate a text response from Ollama. Low temperature for structured extraction."""
    m = model or OLLAMA_TEXT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=m,
        messages=messages,
        options={"temperature": temperature, "num_ctx": 8192},
    )
    return response["message"]["content"].strip()


def generate_json(prompt: str, system: str = "", model: str = None) -> dict | list | None:
    """
    Generate and parse JSON from Ollama.
    Strips markdown fences, handles partial JSON gracefully.
    """
    json_system = (system + "\n\n" if system else "") + (
        "You MUST respond with valid JSON only. "
        "No markdown, no explanation, no preamble. "
        "Start your response directly with { or [."
    )
    raw = generate(prompt, system=json_system, model=model, temperature=0.05)

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract first JSON object/array from the string
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        raw = match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}. Raw snippet: {raw[:200]}")
        return None


# ─── Vision calls (for table / image extraction) ─────────────────────────────

@retry(
    stop=stop_after_attempt(OLLAMA_MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def vision_extract(image_path: Path, prompt: str, model: str = None) -> str:
    """Send an image + prompt to the vision model. Returns raw text."""
    m = model or OLLAMA_VISION_MODEL

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = ollama.chat(
        model=m,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        options={"temperature": 0.05, "num_ctx": 4096},
    )
    return response["message"]["content"].strip()


def vision_extract_table_json(image_path: Path, context: str = "") -> list[dict] | None:
    """
    Extract a table from an image and return it as a list of row dicts.
    Used as the final fallback when pdfplumber and camelot both fail.
    """
    prompt = (
        f"{'Context: ' + context + chr(10) if context else ''}"
        "This image contains an engineering table from a Detailed Project Report (DPR). "
        "Extract ALL rows and columns from this table. "
        "Return ONLY a JSON array where each element is an object representing one row. "
        "Use the column headers as keys. Preserve numeric values exactly as shown. "
        "If a cell is empty, use null. "
        "Do not add any explanation. Start directly with [."
    )
    raw = vision_extract(image_path, prompt)

    # Strip fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("Vision table extraction: JSON parse failed")
            return None
    return None


def vision_describe_image(image_path: Path, sector: str = "") -> str:
    """Get a textual description of an engineering diagram/image for fact extraction."""
    prompt = (
        f"This is an engineering diagram from a {'[' + sector + '] ' if sector else ''}DPR. "
        "Describe all engineering information, measurements, labels, and annotations visible. "
        "Be specific about numbers, units, materials, and structural elements."
    )
    return vision_extract(image_path, prompt)


# ─── Sector classification ────────────────────────────────────────────────────

def classify_sector(text_sample: str, sectors: list[str]) -> tuple[str, float]:
    """
    Zero-shot sector classification. Returns (sector_name, confidence 0-1).
    Uses fast model (llama3.1:8b) — classification is a simple task.
    text_sample: first ~3000 chars of the document.
    """
    sectors_formatted = "\n".join(f"- {s}" for s in sectors)
    prompt = (
        f"You are classifying an infrastructure project document.\n\n"
        f"Available sectors:\n{sectors_formatted}\n\n"
        f"Document excerpt:\n\"\"\"\n{text_sample[:3000]}\n\"\"\"\n\n"
        "Which sector does this document belong to? "
        "Respond with a JSON object: "
        '{"sector": "<exact sector name from the list>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}'
    )
    result = generate_json(prompt, model=get_model_for_task(TaskType.CLASSIFY))
    if result and "sector" in result:
        return result["sector"], float(result.get("confidence", 0.5))
    # Fallback: return first sector with low confidence
    logger.warning("Sector classification failed, defaulting to first sector")
    return sectors[0], 0.1