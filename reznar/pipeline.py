"""Extraction pipeline for Reznar's Arcane Oddities.

Pipeline:

PDF -> rendered page PNGs -> OCR provenance text -> Gemini VLM source extraction
-> Pydantic validation -> Gemini LLM semantic enrichment -> Postgres.

The VLM is the structured source extractor. OCR is intentionally auxiliary: it
is saved for audit/debugging and can be passed to Gemini as noisy context, but
the prompt tells Gemini to use the page image as the source of truth and to
return concise, non-verbatim mechanics summaries instead of full prose.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from psycopg.types.json import Jsonb
from pydantic import ValidationError

import db
from reznar.ontology import CatalogPageExtraction, ItemMechanics, MagicItem, SourceMagicItem


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ENRICH_ITEM_ATTEMPTS = 3


SOURCE_EXTRACTION_SHAPE = {
    "page_number": 1,
    "items": [
        {
            "name": "Item Name",
            "printed_type_line": "Wondrous item, rare (requires attunement)",
            "item_kind": "wondrous_item | weapon | armor | ring | potion | scroll | other",
            "subtype": "shield | plate | dagger | any axe | null",
            "rarity": "common | uncommon | rare | very_rare | legendary | artifact | varies | unknown",
            "requires_attunement": True,
            "attunement_requirement": "elf | bard | null",
            "is_cursed": False,
            "description": "Concise non-verbatim mechanics summary for this item.",
            "source_pages": [1],
            "continues_from_previous_page": False,
            "continues_on_next_page": False,
            "confidence": 0.95,
            "warnings": [],
        }
    ],
    "page_warnings": [],
}


MECHANICS_SHAPE = {
    "equipment_slots": ["head", "neck", "ring", "weapon"],
    "bonuses": [{"target": "AC", "value": "+2", "context": "while wearing"}],
    "damage_types": ["necrotic", "radiant"],
    "defenses": [{"kind": "resistance", "target": "necrotic damage", "context": "while wearing"}],
    "spells_granted": ["disguise self"],
    "conditions_inflicted": ["frightened"],
    "target_creatures": ["fiends", "undead"],
    "environment_tags": ["wooded environment"],
    "usage_limits": [{"kind": "charges", "amount": "3", "reset": "midnight", "context": None}],
    "action_economy": ["action", "bonus action", "reaction"],
    "notes": [],
}


@dataclass(frozen=True)
class Paths:
    root: Path
    pdf: Path
    work_dir: Path
    page_dir: Path
    ocr_dir: Path
    vlm_dir: Path
    source_items: Path
    enriched_items: Path


def build_paths(root: Path, pdf: Path, work_dir: Path) -> Paths:
    return Paths(
        root=root,
        pdf=pdf,
        work_dir=work_dir,
        page_dir=work_dir / "page_images",
        ocr_dir=work_dir / "ocr",
        vlm_dir=work_dir / "vlm_pages",
        source_items=work_dir / "items_source.json",
        enriched_items=work_dir / "items_enriched.json",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"{name!r} was not found on PATH. Install it before running this stage."
        )
    return path


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def pdf_page_count(paths: Paths) -> int:
    require_tool("pdfinfo")
    result = run(["pdfinfo", str(paths.pdf)], cwd=paths.root)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"could not determine page count for {paths.pdf}")
    return int(match.group(1))


def page_number_from_path(path: Path) -> int:
    match = re.search(r"page-(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"could not infer page number from {path}")
    return int(match.group(1))


def normalized_name(name: str) -> str:
    text = name.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def existing_page_images(page_dir: Path) -> list[Path]:
    return sorted(page_dir.glob("page-*.png"), key=page_number_from_path)


def selected_existing_page_images(
    page_dir: Path,
    pages: list[int] | None = None,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Path]:
    images = existing_page_images(page_dir)
    if pages:
        allowed = set(pages)
        images = [path for path in images if page_number_from_path(path) in allowed]
    if first_page is not None:
        images = [path for path in images if page_number_from_path(path) >= first_page]
    if last_page is not None:
        images = [path for path in images if page_number_from_path(path) <= last_page]
    return images


def render_pages(
    paths: Paths,
    dpi: int,
    force: bool,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Path]:
    require_tool("pdftoppm")
    ensure_dir(paths.page_dir)
    total_pages = pdf_page_count(paths)
    start_page = first_page or 1
    end_page = last_page or total_pages
    if start_page < 1 or end_page < start_page:
        raise ValueError(f"invalid page range: {start_page}-{end_page}")
    if end_page > total_pages:
        raise ValueError(f"page range ends at {end_page}, but PDF has {total_pages} pages")

    existing = selected_existing_page_images(paths.page_dir, first_page=first_page, last_page=last_page)
    expected_pages = set(range(start_page, end_page + 1))
    existing_pages = {page_number_from_path(path) for path in existing}
    if existing and not force:
        if expected_pages.issubset(existing_pages):
            print(f"render: using {len(existing)} existing page images in {paths.page_dir}")
            return existing
        missing = sorted(expected_pages - existing_pages)
        print(f"render: {len(missing)} page images missing; rendering requested range")

    if force and existing:
        for png in existing:
            png.unlink()

    tmp_dir = paths.page_dir / "_render_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    prefix = tmp_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if first_page is not None:
        cmd.extend(["-f", str(first_page)])
    if last_page is not None:
        cmd.extend(["-l", str(last_page)])
    cmd.extend([str(paths.pdf), str(prefix)])
    print("render:", " ".join(cmd))
    run(cmd, cwd=paths.root)

    rendered = sorted(tmp_dir.glob("page-*.png"))
    stable_paths: list[Path] = []
    for index, path in enumerate(rendered, start=start_page):
        stable = paths.page_dir / f"page-{index:03d}.png"
        if path != stable:
            if stable.exists():
                stable.unlink()
            path.rename(stable)
        stable_paths.append(stable)
    shutil.rmtree(tmp_dir)

    print(f"render: wrote {len(stable_paths)} page images")
    return stable_paths


def ocr_pages(
    paths: Paths,
    psm: int,
    force: bool,
    pages: list[int] | None = None,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Path]:
    require_tool("tesseract")
    ensure_dir(paths.ocr_dir)
    outputs: list[Path] = []
    for image_path in selected_existing_page_images(
        paths.page_dir, pages=pages, first_page=first_page, last_page=last_page
    ):
        page = page_number_from_path(image_path)
        out_path = paths.ocr_dir / f"page-{page:03d}.txt"
        outputs.append(out_path)
        if out_path.exists() and not force:
            continue
        cmd = ["tesseract", str(image_path), "stdout", "--psm", str(psm)]
        print(f"ocr: page {page}")
        result = run(cmd, cwd=paths.root)
        out_path.write_text(result.stdout, encoding="utf-8")
    print(f"ocr: ready for {len(outputs)} pages")
    return outputs


class GeminiClient:
    def __init__(self, api_key: str, model: str, api_base: str = DEFAULT_API_BASE):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")

    def generate_json(
        self,
        prompt: str,
        image_path: Path | None = None,
        max_attempts: int = 5,
    ) -> tuple[Any, dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image_path is not None:
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                }
            )

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        }
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"Gemini API error {exc.code}: {detail}")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                    raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Gemini API network error: {exc}")
                if attempt == max_attempts:
                    raise last_error from exc
            except (TimeoutError, socket.timeout) as exc:
                last_error = RuntimeError(f"Gemini API timeout: {exc}")
                if attempt == max_attempts:
                    raise last_error from exc
            time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Gemini API failed: {last_error}")

        text = extract_gemini_text(raw)
        parsed = parse_jsonish(text)
        return parsed, raw


def extract_gemini_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            if "text" in part:
                texts.append(str(part["text"]))
    if texts:
        return "\n".join(texts)
    if "text" in raw:
        return str(raw["text"])
    raise ValueError(f"Gemini response did not contain text: {raw}")


def parse_jsonish(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON from Gemini response: {text[:500]}")


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "Set GEMINI_API_KEY before running extract/enrich/all.\n"
            "Example: export GEMINI_API_KEY='...'"
        )
    return key


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def is_successful_page_record(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("status") == "ok" and payload.get("validated"))


def extraction_error_sidecar_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.name}.error")


def write_extraction_record(
    out_path: Path,
    record: dict[str, Any],
    previous_record: dict[str, Any] | None,
) -> None:
    if record.get("status") == "ok":
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        sidecar = extraction_error_sidecar_path(out_path)
        if sidecar.exists():
            sidecar.unlink()
        return

    if is_successful_page_record(previous_record):
        sidecar = extraction_error_sidecar_path(out_path)
        sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        page = record.get("page_number", "?")
        print(
            f"extract: page {page} failed; kept previous successful artifact "
            f"and wrote {sidecar.name}"
        )
        return

    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def select_pages(
    paths: Paths,
    pages: list[int] | None,
    limit: int | None,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Path]:
    images = selected_existing_page_images(
        paths.page_dir, pages=pages, first_page=first_page, last_page=last_page
    )
    if limit is not None:
        images = images[:limit]
    return images


def load_previous_page_context(paths: Paths, page_number: int) -> str:
    previous = paths.vlm_dir / f"page-{page_number - 1:03d}.json"
    if page_number <= 1 or not previous.exists():
        return "No previous page context available."
    try:
        payload = json.loads(previous.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            return "Previous page extraction exists but was not successful."
        extraction = CatalogPageExtraction.model_validate(payload["validated"])
    except Exception as exc:  # noqa: BLE001
        return f"Previous page extraction could not be read: {exc!r}"

    if not extraction.items:
        return "Previous page had no extracted items."

    lines = []
    for item in extraction.items:
        lines.append(
            "- "
            f"name={item.name!r}; "
            f"type_line={item.printed_type_line!r}; "
            f"item_kind={item.item_kind.value!r}; "
            f"rarity={item.rarity.value!r}; "
            f"requires_attunement={item.requires_attunement!r}; "
            f"attunement_requirement={item.attunement_requirement!r}; "
            f"continues_on_next_page={item.continues_on_next_page!r}"
        )
    return "\n".join(lines)


def build_source_prompt(
    page_number: int,
    ocr_text: str,
    previous_context: str,
    recitation_retry: bool = False,
) -> str:
    shape = json.dumps(SOURCE_EXTRACTION_SHAPE, indent=2)
    retry_guard = ""
    if recitation_retry:
        retry_guard = """
This is a retry after the model refused a too-verbatim response. Return only
structured catalog facts. Make each description a short list of non-verbatim
facts under 35 words. Do not quote any complete sentence from the page.
"""
    return f"""You are extracting a fantasy magic item catalog from a scanned PDF page.

Use the PAGE IMAGE as the source of truth. The OCR text below is noisy helper
context and may contain incorrect words, broken lines, watermarks, or image
artifacts. Ignore watermark/order text.

Do not transcribe full item prose. The pipeline stores complete page OCR text
separately for auditability, so your job is to turn the page image into
structured catalog facts. For each item's description field, write a concise
non-verbatim mechanics summary of the visible effects: numeric bonuses, damage
types and amounts, spells, charges, daily limits, action economy, attunement or
cursed constraints, target creature types, and whether the entry continues.
Preserve exact item names, printed type lines, spell names, numbers, dice, and
game terms, but do not copy long sentences from the page. Keep each description
under 90 words.
{retry_guard}

Previous page context:
{previous_context}

Extract every distinct magic item entry that starts on page {page_number}. A new
item must have its own heading plus an italic type/rarity line such as
"Wondrous item, rare" or "Armor (shield), uncommon".

If the page only continues prose for an item that started on an earlier page,
use the previous page context to return the continued mechanics as the same item:
copy the previous item's name, type line, item_kind, rarity, attunement fields,
set continues_from_previous_page to true, and summarize only the visible
continuation mechanics from this page. Do not invent a new product name from
story text. Do not create placeholder items named "continued", "the drum",
"war drum", or a character name when there is no fresh type/rarity line. The
collector will stitch continuation summaries into the earlier item.

Do not invent missing text. Set source_pages to exactly [{page_number}]. Do not
use printed footer numbers inside the catalog art.

Normalize item_kind and rarity to the allowed values. Preserve the original
printed type line exactly as well as you can read it. Return JSON only in this
shape:

{shape}

OCR helper text for page {page_number}:
\"\"\"
{ocr_text}
\"\"\"
"""


def build_repair_prompt(raw_json_text: str, error: str, schema_name: str) -> str:
    return f"""Repair the JSON so it validates as {schema_name}. Return JSON only.

Validation error:
{error}

Broken JSON:
{raw_json_text}
"""


def is_recitation_stop(exc: Exception) -> bool:
    return "RECITATION" in repr(exc)


def extract_pages(
    paths: Paths,
    client: GeminiClient,
    pages: list[int] | None,
    limit: int | None,
    first_page: int | None,
    last_page: int | None,
    force: bool,
    sleep_seconds: float,
    use_ocr_context: bool,
) -> list[Path]:
    ensure_dir(paths.vlm_dir)
    outputs: list[Path] = []
    for image_path in select_pages(paths, pages, limit, first_page=first_page, last_page=last_page):
        page = page_number_from_path(image_path)
        out_path = paths.vlm_dir / f"page-{page:03d}.json"
        outputs.append(out_path)
        previous_record = read_json_file(out_path)
        if out_path.exists() and not force:
            print(f"extract: page {page} exists, skipping")
            continue

        ocr_path = paths.ocr_dir / f"page-{page:03d}.txt"
        ocr_text = (
            ocr_path.read_text(encoding="utf-8")
            if use_ocr_context and ocr_path.exists()
            else ""
        )
        previous_context = load_previous_page_context(paths, page)
        prompt = build_source_prompt(page, ocr_text, previous_context)
        print(f"extract: page {page}")
        record: dict[str, Any] = {
            "page_number": page,
            "image_path": str(image_path),
            "ocr_path": str(ocr_path),
            "ocr_context_used": bool(ocr_text),
            "status": "pending",
        }
        try:
            try:
                parsed, raw_response = client.generate_json(prompt, image_path=image_path)
            except ValueError as exc:
                if not is_recitation_stop(exc):
                    raise
                record["recitation_retry_error"] = repr(exc)
                parsed, raw_response = client.generate_json(
                    build_source_prompt(
                        page,
                        ocr_text="",
                        previous_context=previous_context,
                        recitation_retry=True,
                    ),
                    image_path=image_path,
                )
                record["recitation_retry_used"] = True
            try:
                extraction = CatalogPageExtraction.model_validate(parsed)
            except ValidationError as exc:
                repaired, repair_raw = client.generate_json(
                    build_repair_prompt(json.dumps(parsed, indent=2), str(exc), "CatalogPageExtraction")
                )
                extraction = CatalogPageExtraction.model_validate(repaired)
                record["repair_raw_response"] = repair_raw
                record["repaired_json"] = repaired

            for item in extraction.items:
                item.source_pages = [page]
            record.update(
                {
                    "status": "ok",
                    "raw_response": raw_response,
                    "validated": extraction.model_dump(mode="json"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "error", "error": repr(exc)})
        write_extraction_record(out_path, record, previous_record)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return outputs


def merge_text(existing: str, incoming: str) -> str:
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming in existing:
        return existing
    return f"{existing}\n\n{incoming}"


def clean_ocr_continuation_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "Michael Burnam-Fink" in stripped:
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        if re.fullmatch(r"[^A-Za-z0-9]+", stripped):
            continue
        cleaned_lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def is_continuation_only(item: SourceMagicItem) -> bool:
    """Return true when a VLM item is page-continuation text, not a product.

    In this catalog, real product entries have a printed type/rarity line. The
    common failure mode is a continuation page being emitted as "The Drum" or
    "Pouch (Continued)" with no type line. Those should be stitched into the
    previous real product.
    """

    name = normalized_name(item.name)
    type_line = item.printed_type_line.strip()
    if not type_line:
        return True
    if "continued" in name:
        return True
    if item.item_kind.value == "other" and item.rarity.value == "unknown":
        return True
    return False


def merge_item_fields(prior: SourceMagicItem, item: SourceMagicItem) -> None:
    prior.description = merge_text(prior.description, item.description)
    prior.source_pages = sorted(set(prior.source_pages + item.source_pages))
    prior.warnings = sorted(set(prior.warnings + item.warnings))
    prior.requires_attunement = prior.requires_attunement or item.requires_attunement
    prior.is_cursed = prior.is_cursed or item.is_cursed
    prior.continues_from_previous_page = (
        prior.continues_from_previous_page or item.continues_from_previous_page
    )
    prior.continues_on_next_page = prior.continues_on_next_page or item.continues_on_next_page
    if not prior.printed_type_line and item.printed_type_line:
        prior.printed_type_line = item.printed_type_line
    if prior.confidence and item.confidence:
        prior.confidence = min(prior.confidence, item.confidence)
    else:
        prior.confidence = prior.confidence or item.confidence


def merge_continuation_into(
    target: SourceMagicItem,
    continuation_name: str,
    continuation_text: str,
    source_pages: list[int],
    warnings: list[str] | None = None,
    confidence: float = 0.0,
) -> None:
    target.description = merge_text(target.description, continuation_text)
    target.source_pages = sorted(set(target.source_pages + source_pages))
    target.continues_on_next_page = False
    target.warnings = sorted(
        set(
            target.warnings
            + (warnings or [])
            + [
                "Merged continuation text from "
                f"{continuation_name!r} on page(s) {source_pages}."
            ]
        )
    )
    if target.confidence and confidence:
        target.confidence = min(target.confidence, confidence)


def collect_source_items(paths: Paths) -> list[SourceMagicItem]:
    grouped: dict[str, SourceMagicItem] = {}
    last_real_key: str | None = None
    page_errors: list[dict[str, Any]] = []
    for page_file in sorted(paths.vlm_dir.glob("page-*.json")):
        page = int(page_file.stem.split("-")[-1])
        payload = json.loads(page_file.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            page_errors.append(payload)
            continue
        extraction = CatalogPageExtraction.model_validate(payload["validated"])
        if not extraction.items and last_real_key and last_real_key in grouped:
            target = grouped[last_real_key]
            if target.continues_on_next_page:
                ocr_path = paths.ocr_dir / f"page-{page:03d}.txt"
                ocr_text = (
                    clean_ocr_continuation_text(ocr_path.read_text(encoding="utf-8"))
                    if ocr_path.exists()
                    else ""
                )
                if ocr_text:
                    merge_continuation_into(
                        target,
                        continuation_name=f"OCR-only continuation page {page}",
                        continuation_text=ocr_text,
                        source_pages=[page],
                        warnings=extraction.page_warnings
                        + ["No VLM item was returned; merged cleaned OCR as continuation text."],
                    )
            continue
        for item in extraction.items:
            item.source_pages = [page]
            key = normalized_name(item.name)
            if not key:
                continue

            if is_continuation_only(item):
                if last_real_key and last_real_key in grouped:
                    target = grouped[last_real_key]
                    merge_continuation_into(
                        target,
                        continuation_name=item.name,
                        continuation_text=item.description,
                        source_pages=item.source_pages,
                        warnings=item.warnings,
                        confidence=item.confidence,
                    )
                    target.continues_on_next_page = item.continues_on_next_page
                    continue
                item.warnings = sorted(
                    set(item.warnings + ["Continuation-like extraction had no prior item to merge into."])
                )

            if key not in grouped:
                grouped[key] = item
                last_real_key = key
                continue
            prior = grouped[key]
            merge_item_fields(prior, item)
            last_real_key = key

    items = sorted(grouped.values(), key=lambda item: (item.source_pages or [9999], item.name))
    artifact = {
        "items": [item.model_dump(mode="json") for item in items],
        "page_errors": page_errors,
        "item_count": len(items),
    }
    paths.source_items.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"collect: wrote {len(items)} source items to {paths.source_items}")
    if page_errors:
        print(f"collect: {len(page_errors)} page extraction errors recorded")
    return items


def build_mechanics_prompt(item: SourceMagicItem) -> str:
    shape = json.dumps(MECHANICS_SHAPE, indent=2)
    source = json.dumps(item.model_dump(mode="json"), indent=2)
    return f"""You are adding semantic ontology tags to one fantasy magic item.

Do not rewrite the source item. Infer only compact query tags from its name,
printed type line, and description. If a tag is not supported by the text,
leave that list empty. Return JSON only in this shape:

{shape}

Source item:
{source}
"""


def load_source_items(paths: Paths) -> list[SourceMagicItem]:
    payload = json.loads(paths.source_items.read_text(encoding="utf-8"))
    return [SourceMagicItem.model_validate(item) for item in payload["items"]]


def source_page_errors(paths: Paths) -> list[dict[str, Any]]:
    if not paths.source_items.exists():
        return []
    payload = json.loads(paths.source_items.read_text(encoding="utf-8"))
    return payload.get("page_errors") or []


def ensure_no_source_page_errors(paths: Paths) -> None:
    errors = source_page_errors(paths)
    if not errors:
        return
    first = errors[0]
    first_page = first.get("page_number", "unknown")
    first_error = str(first.get("error", "unknown error"))[:400]
    raise SystemExit(
        "Source extraction has "
        f"{len(errors)} page error(s); fix extraction before enrich/load.\n"
        f"First failed page: {first_page}\n"
        f"First error: {first_error}"
    )


def enrich_items(
    paths: Paths,
    client: GeminiClient,
    limit: int | None,
    force: bool,
    sleep_seconds: float,
) -> list[MagicItem]:
    ensure_no_source_page_errors(paths)
    if paths.enriched_items.exists() and not force:
        payload = json.loads(paths.enriched_items.read_text(encoding="utf-8"))
        print(f"enrich: using existing {paths.enriched_items}")
        return [MagicItem.model_validate(item) for item in payload["items"]]

    source_items = load_source_items(paths)
    if limit is not None:
        source_items = source_items[:limit]

    enriched: list[MagicItem] = []
    errors: list[dict[str, Any]] = []
    for index, source in enumerate(source_items, start=1):
        print(f"enrich: {index}/{len(source_items)} {source.name}")
        last_error: Exception | None = None
        for attempt in range(1, ENRICH_ITEM_ATTEMPTS + 1):
            try:
                parsed, raw_response = client.generate_json(build_mechanics_prompt(source))
                try:
                    mechanics = ItemMechanics.model_validate(parsed)
                except ValidationError as exc:
                    repaired, repair_raw = client.generate_json(
                        build_repair_prompt(
                            json.dumps(parsed, indent=2), str(exc), "ItemMechanics"
                        )
                    )
                    mechanics = ItemMechanics.model_validate(repaired)
                    raw_response = {"original": raw_response, "repair": repair_raw}
                item_data = source.model_dump(mode="json")
                item_data["id"] = str(uuid5(NAMESPACE_URL, f"reznar:{normalized_name(source.name)}"))
                item_data["mechanics"] = mechanics.model_dump(mode="json")
                item = MagicItem.model_validate(item_data)
                enriched.append(item)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == ENRICH_ITEM_ATTEMPTS:
                    break
                print(
                    "enrich: retrying "
                    f"{source.name} after error on attempt {attempt}: {exc!r}"
                )
                time.sleep(max(sleep_seconds, 5.0))

        if last_error is not None:
            errors.append(
                {
                    "id": str(uuid4()),
                    "item_name": source.name,
                    "error_type": "enrichment_error",
                    "message": repr(last_error),
                    "raw_payload": source.model_dump(mode="json"),
                }
            )
            item_data = source.model_dump(mode="json")
            item_data["id"] = str(uuid5(NAMESPACE_URL, f"reznar:{normalized_name(source.name)}"))
            item_data["mechanics"] = ItemMechanics().model_dump(mode="json")
            enriched.append(MagicItem.model_validate(item_data))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    artifact = {
        "items": [item.model_dump(mode="json") for item in enriched],
        "errors": errors,
        "item_count": len(enriched),
    }
    paths.enriched_items.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"enrich: wrote {len(enriched)} enriched items to {paths.enriched_items}")
    if errors:
        print(f"enrich: {len(errors)} enrichment errors recorded")
    return enriched


def load_enriched_items(paths: Paths) -> tuple[list[MagicItem], list[dict[str, Any]]]:
    payload = json.loads(paths.enriched_items.read_text(encoding="utf-8"))
    return [MagicItem.model_validate(item) for item in payload["items"]], payload.get("errors", [])


def load_page_records(paths: Paths) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page_file in sorted(paths.vlm_dir.glob("page-*.json")):
        page = int(page_file.stem.split("-")[-1])
        ocr_path = paths.ocr_dir / f"page-{page:03d}.txt"
        image_path = paths.page_dir / f"page-{page:03d}.png"
        payload = json.loads(page_file.read_text(encoding="utf-8"))
        warnings: list[str] = []
        raw_vlm_json: dict[str, Any] = payload
        if payload.get("status") == "ok":
            validated = payload.get("validated") or {}
            warnings.extend(validated.get("page_warnings") or [])
        elif payload.get("error"):
            warnings.append(payload["error"])
        records.append(
            {
                "page_number": page,
                "image_path": str(image_path),
                "ocr_text": ocr_path.read_text(encoding="utf-8") if ocr_path.exists() else "",
                "raw_vlm_json": raw_vlm_json,
                "status": payload.get("status", "unknown"),
                "warnings": warnings,
            }
        )
    return records


def load_postgres(paths: Paths) -> None:
    ensure_no_source_page_errors(paths)
    items, enrichment_errors = load_enriched_items(paths)
    page_records = load_page_records(paths)
    extraction_errors: list[dict[str, Any]] = []
    for record in page_records:
        if record["status"] != "ok":
            extraction_errors.append(
                {
                    "id": str(uuid4()),
                    "page_number": record["page_number"],
                    "item_name": None,
                    "error_type": "page_extraction_error",
                    "message": "; ".join(record["warnings"]) or "page extraction failed",
                    "raw_payload": record["raw_vlm_json"],
                }
            )
    extraction_errors.extend(enrichment_errors)

    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists extraction_page (
                page_number integer primary key,
                image_path text not null,
                ocr_text text not null,
                raw_vlm_json jsonb not null,
                status text not null,
                warnings jsonb not null
            )
            """
        )
        cur.execute(
            """
            create table if not exists magic_item (
                id uuid primary key,
                name text not null,
                item_kind text not null,
                rarity text not null,
                requires_attunement boolean not null,
                source_pages integer[] not null,
                data jsonb not null
            )
            """
        )
        cur.execute(
            """
            create table if not exists extraction_error (
                id uuid primary key,
                page_number integer,
                item_name text,
                error_type text not null,
                message text not null,
                raw_payload jsonb not null
            )
            """
        )
        cur.execute("create index if not exists magic_item_kind_idx on magic_item (item_kind)")
        cur.execute("create index if not exists magic_item_rarity_idx on magic_item (rarity)")
        cur.execute("create index if not exists magic_item_data_gin on magic_item using gin (data)")

        cur.execute("delete from extraction_error")
        cur.execute("delete from magic_item")
        cur.execute("delete from extraction_page")

        for record in page_records:
            cur.execute(
                """
                insert into extraction_page
                    (page_number, image_path, ocr_text, raw_vlm_json, status, warnings)
                values (%s, %s, %s, %s, %s, %s)
                on conflict (page_number) do update set
                    image_path = excluded.image_path,
                    ocr_text = excluded.ocr_text,
                    raw_vlm_json = excluded.raw_vlm_json,
                    status = excluded.status,
                    warnings = excluded.warnings
                """,
                (
                    record["page_number"],
                    record["image_path"],
                    record["ocr_text"],
                    Jsonb(record["raw_vlm_json"]),
                    record["status"],
                    Jsonb(record["warnings"]),
                ),
            )

        for item in items:
            data = item.model_dump(mode="json")
            cur.execute(
                """
                insert into magic_item
                    (id, name, item_kind, rarity, requires_attunement, source_pages, data)
                values (%s, %s, %s, %s, %s, %s, %s)
                on conflict (id) do update set
                    name = excluded.name,
                    item_kind = excluded.item_kind,
                    rarity = excluded.rarity,
                    requires_attunement = excluded.requires_attunement,
                    source_pages = excluded.source_pages,
                    data = excluded.data
                """,
                (
                    item.id,
                    item.name,
                    item.item_kind.value,
                    item.rarity.value,
                    item.requires_attunement,
                    item.source_pages,
                    Jsonb(data),
                ),
            )

        for error in extraction_errors:
            cur.execute(
                """
                insert into extraction_error
                    (id, page_number, item_name, error_type, message, raw_payload)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (
                    error["id"],
                    error.get("page_number"),
                    error.get("item_name"),
                    error["error_type"],
                    error["message"],
                    Jsonb(error.get("raw_payload", {})),
                ),
            )
        conn.commit()
    print(
        f"load: inserted {len(page_records)} page records, "
        f"{len(items)} magic items, {len(extraction_errors)} errors"
    )


def print_summary(paths: Paths) -> None:
    if not paths.enriched_items.exists() and not paths.source_items.exists():
        print("summary: no item artifact found yet")
        return
    path = paths.enriched_items if paths.enriched_items.exists() else paths.source_items
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    counts: dict[str, defaultdict[str, int]] = {
        "item_kind": defaultdict(int),
        "rarity": defaultdict(int),
    }
    for item in items:
        counts["item_kind"][item.get("item_kind", "unknown")] += 1
        counts["rarity"][item.get("rarity", "unknown")] += 1
    print(f"summary: {len(items)} items from {path}")
    for key, values in counts.items():
        print(key + ":")
        for value, count in sorted(values.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {value}: {count}")


def validate_pipeline_outputs(paths: Paths) -> None:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"validate: ok - {message}")
        else:
            print(f"validate: FAIL - {message}")
            failures.append(message)

    check(paths.source_items.exists(), f"source artifact exists: {paths.source_items}")
    check(paths.enriched_items.exists(), f"enriched artifact exists: {paths.enriched_items}")
    if failures:
        raise SystemExit(1)

    source_payload = json.loads(paths.source_items.read_text(encoding="utf-8"))
    enriched_payload = json.loads(paths.enriched_items.read_text(encoding="utf-8"))
    source_items = [
        SourceMagicItem.model_validate(item) for item in source_payload.get("items", [])
    ]
    enriched_items = [
        MagicItem.model_validate(item) for item in enriched_payload.get("items", [])
    ]
    page_records = load_page_records(paths)

    check(len(source_items) == len(enriched_items), "source and enriched item counts match")
    check(not source_payload.get("page_errors"), "no page extraction errors in source artifact")
    check(not enriched_payload.get("errors"), "no enrichment errors in enriched artifact")

    expected_pages = pdf_page_count(paths) if paths.pdf.exists() else len(page_records)
    ok_pages = [record for record in page_records if record["status"] == "ok"]
    check(len(page_records) == expected_pages, f"{expected_pages} page records present")
    check(len(ok_pages) == len(page_records), "all page extraction records have status ok")

    page_numbers = {record["page_number"] for record in page_records}
    bad_page_refs = [
        f"{item.name}: {page}"
        for item in enriched_items
        for page in item.source_pages
        if page not in page_numbers
    ]
    check(not bad_page_refs, "all item source_pages refer to loaded page records")

    name_counts: defaultdict[str, int] = defaultdict(int)
    for item in enriched_items:
        name_counts[normalized_name(item.name)] += 1
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    check(not duplicate_names, "no duplicate normalized item names")

    missing_required = [
        item.name
        for item in enriched_items
        if not item.name or not item.description or not item.source_pages
    ]
    check(not missing_required, "all enriched items have name, description, and source_pages")

    empty_ocr_pages = [
        record["page_number"] for record in page_records if not record["ocr_text"].strip()
    ]
    if empty_ocr_pages:
        print(f"validate: warn - empty OCR text on pages {empty_ocr_pages}")
    else:
        print("validate: ok - OCR provenance text exists for every page")

    with db.connect() as conn, conn.cursor() as cur:
        for table in ["extraction_page", "magic_item", "extraction_error"]:
            cur.execute("select to_regclass(%s)", (f"public.{table}",))
            check(cur.fetchone()[0] is not None, f"Postgres table exists: {table}")

        if failures:
            raise SystemExit(1)

        cur.execute("select count(*) from extraction_page")
        db_page_count = cur.fetchone()[0]
        cur.execute("select count(*) from magic_item")
        db_item_count = cur.fetchone()[0]
        cur.execute("select count(*) from extraction_error")
        db_error_count = cur.fetchone()[0]

        check(db_page_count == len(page_records), "Postgres page count matches artifacts")
        check(db_item_count == len(enriched_items), "Postgres magic_item count matches artifacts")
        check(db_error_count == 0, "Postgres extraction_error table is empty")

        cur.execute(
            """
            select count(*)
            from magic_item
            where name = ''
               or coalesce(array_length(source_pages, 1), 0) = 0
               or data is null
            """
        )
        check(cur.fetchone()[0] == 0, "Postgres magic_item rows have required fields")

        print("validate: Postgres item_kind counts")
        cur.execute(
            """
            select item_kind, count(*)
            from magic_item
            group by item_kind
            order by count(*) desc, item_kind
            """
        )
        for item_kind, count in cur.fetchall():
            print(f"  {item_kind}: {count}")

        print("validate: Postgres rarity counts")
        cur.execute(
            """
            select rarity, count(*)
            from magic_item
            group by rarity
            order by count(*) desc, rarity
            """
        )
        for rarity, count in cur.fetchall():
            print(f"  {rarity}: {count}")

    if failures:
        print(f"validate: failed with {len(failures)} issue(s)")
        raise SystemExit(1)
    print(
        "validate: passed "
        f"({len(page_records)} pages, {len(enriched_items)} items, 0 DB errors)"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "render",
            "ocr",
            "extract",
            "collect",
            "enrich",
            "load",
            "all",
            "summary",
            "validate",
        ],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--pdf", type=Path, default=Path("data/items_combined.pdf"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--psm", type=int, default=11)
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("GEMINI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--page", type=int, action="append", help="Page number to process.")
    parser.add_argument("--first-page", type=int, help="First PDF page to render/OCR.")
    parser.add_argument("--last-page", type=int, help="Last PDF page to render/OCR.")
    parser.add_argument("--limit", type=int, help="Limit the number of pages/items processed.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing artifacts for the stage.")
    parser.add_argument(
        "--no-ocr-context",
        action="store_true",
        help="Do not include OCR text in Gemini extraction prompts; the PNG remains the source.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    root = Path.cwd()
    paths = build_paths(root=root, pdf=args.pdf, work_dir=args.work_dir)
    ensure_dir(paths.work_dir)

    if args.stage in {"render", "all"}:
        render_pages(
            paths,
            dpi=args.dpi,
            force=args.force,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    if args.stage in {"ocr", "all"}:
        if not existing_page_images(paths.page_dir):
            render_pages(
                paths,
                dpi=args.dpi,
                force=False,
                first_page=args.first_page,
                last_page=args.last_page,
            )
        ocr_pages(
            paths,
            psm=args.psm,
            force=args.force,
            pages=args.page,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    if args.stage in {"extract", "all"}:
        if not existing_page_images(paths.page_dir):
            render_pages(
                paths,
                dpi=args.dpi,
                force=False,
                first_page=args.first_page,
                last_page=args.last_page,
            )
        if not list(paths.ocr_dir.glob("page-*.txt")):
            ocr_pages(
                paths,
                psm=args.psm,
                force=False,
                pages=args.page,
                first_page=args.first_page,
                last_page=args.last_page,
            )
        client = GeminiClient(api_key=get_api_key(), model=args.model, api_base=args.api_base)
        extract_pages(
            paths,
            client=client,
            pages=args.page,
            limit=args.limit,
            first_page=args.first_page,
            last_page=args.last_page,
            force=args.force,
            sleep_seconds=args.sleep,
            use_ocr_context=not args.no_ocr_context,
        )
    if args.stage in {"collect", "all"}:
        collect_source_items(paths)
    if args.stage in {"enrich", "all"}:
        if not paths.source_items.exists():
            collect_source_items(paths)
        ensure_no_source_page_errors(paths)
        client = GeminiClient(api_key=get_api_key(), model=args.model, api_base=args.api_base)
        enrich_items(paths, client=client, limit=args.limit, force=args.force, sleep_seconds=args.sleep)
    if args.stage in {"load", "all"}:
        if not paths.enriched_items.exists():
            raise SystemExit("Run the enrich stage before load.")
        load_postgres(paths)
    if args.stage in {"summary", "all"}:
        print_summary(paths)
    if args.stage == "validate":
        validate_pipeline_outputs(paths)


if __name__ == "__main__":
    main()
