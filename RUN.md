# Part 1 Run Instructions

This pipeline populates the local Postgres database from `data/items_combined.pdf`.
The PDF is image-based, so direct text extraction is not reliable. The pipeline
renders each PDF page to PNG, runs OCR to preserve full page text as provenance,
then uses Gemini vision as the primary structured source extractor.

## 1. Prerequisites

Install Python dependencies:

```bash
uv sync --python 3.12
```

If `uv` picks Python 3.14 on macOS, point it at Homebrew Python 3.12 instead:

```bash
uv sync --python /opt/homebrew/opt/python@3.12/bin/python3.12
```

Install command-line tools:

```bash
brew install poppler tesseract
```

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Optional model override:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

## 2. Verify Postgres

```bash
uv run python verify.py
```

Expected:

```text
postgres is ready
```

## 3. Recommended Extraction Workflow

For a clean exam-style run from the PDF into a fresh artifact directory:

```bash
uv run python -m reznar.pipeline all --pdf data/items_combined.pdf --work-dir data/extracted_fresh --force --no-ocr-context --model gemini-2.5-flash --workers 16 --sleep 0
```

This single command renders all 39 PDF pages, runs OCR for page-level
provenance, extracts structured item records with Gemini vision, validates and
collects source items, enriches semantic mechanics tags, loads Postgres, writes
CSV/SQL exports, and prints a summary.

`--workers 16` parallelizes the expensive stages while keeping `--dpi 200`.
Rendering runs pages concurrently and caps local render workers to the machine's
CPU count. Page extraction uses a parallel first pass, then reruns
continuation/error-sensitive pages sequentially with previous-page context
before collection. Enrichment is also parallel because each item can be tagged
independently. If the Gemini API returns temporary 429/503 rate-limit errors,
rerun the failed stage with `--workers 8` or `--workers 4`.

Validate the generated artifacts and loaded Postgres tables:

```bash
uv run python -m reznar.pipeline validate --work-dir data/extracted_fresh
```

Expected successful validation:

```text
validate: passed (39 pages, 80 items, 0 DB errors)
```

### Optional development workflow

Run local, non-API stages first:

```bash
uv run python -m reznar.pipeline render --work-dir data/extracted_fresh
uv run python -m reznar.pipeline ocr --work-dir data/extracted_fresh
```

For a quick smoke test, render/OCR only the first two pages:

```bash
uv run python -m reznar.pipeline render --work-dir data/extracted_fresh --first-page 1 --last-page 2 --force
uv run python -m reznar.pipeline ocr --work-dir data/extracted_fresh --first-page 1 --last-page 2 --force
```

Test Gemini extraction on a small page sample:

```bash
uv run python -m reznar.pipeline extract --work-dir data/extracted_fresh --page 1 --page 2 --sleep 1
uv run python -m reznar.pipeline collect --work-dir data/extracted_fresh
uv run python -m reznar.pipeline summary --work-dir data/extracted_fresh
```

Inspect:

```text
data/extracted_fresh/vlm_pages/page-001.json
data/extracted_fresh/vlm_pages/page-002.json
data/extracted_fresh/items_source.json
```

If the sample looks good, extract the full PDF into `data/extracted_fresh/`:

```bash
uv run python -m reznar.pipeline extract --work-dir data/extracted_fresh --sleep 10
uv run python -m reznar.pipeline collect --work-dir data/extracted_fresh
uv run python -m reznar.pipeline summary --work-dir data/extracted_fresh
```

If Gemini returns temporary 429/5xx errors, retry only those pages instead of
rerunning the whole PDF. Failed retries will not overwrite a previous successful
page JSON:

```bash
uv run python -m reznar.pipeline extract --work-dir data/extracted_fresh --force --sleep 10 --page 1 --page 2
uv run python -m reznar.pipeline collect --work-dir data/extracted_fresh
uv run python -m reznar.pipeline summary --work-dir data/extracted_fresh
```

To test whether OCR context is hurting a page, keep the page image but remove
OCR from the Gemini prompt:

```bash
uv run python -m reznar.pipeline extract --work-dir data/extracted_fresh --force --no-ocr-context --sleep 10 --page 1
```

If running stages manually, enrich semantic tags and load Postgres:

```bash
uv run python -m reznar.pipeline enrich --work-dir data/extracted_fresh --sleep 10
uv run python -m reznar.pipeline load --work-dir data/extracted_fresh
uv run python -m reznar.pipeline export --work-dir data/extracted_fresh
```

For a faster enrichment rerun, use:

```bash
uv run python -m reznar.pipeline enrich --work-dir data/extracted_fresh --force --workers 16 --sleep 0
```

The enrichment stage automatically retries transient Gemini timeouts for each
item before recording an enrichment error.

## 4. Generated Artifacts

Generated artifacts are written under `data/extracted_fresh/`.

```text
page_images/                 rendered PNG pages
ocr/                         Tesseract OCR helper text
vlm_pages/                   raw per-page Gemini extraction records
items_source.json            merged and Pydantic-validated source item records
items_enriched.json          final ontology records with semantic mechanics tags
magic_items.csv              flat export for spreadsheet/review workflows
magic_items.sql              standalone Postgres import for magic_item_export
magic_item_bonuses.csv       one row per extracted bonus
magic_item_defenses.csv      one row per extracted defense
magic_item_usage_limits.csv  one row per extracted usage limit
```

These files are intentionally kept outside the database so extraction can be
audited and resumed without re-running every stage.

The item `description` fields are concise non-verbatim mechanics summaries.
Complete page-level OCR text is preserved in `ocr/` artifacts and in the
`extraction_page.ocr_text` database column.

The export stage writes two reviewer-friendly result files:

- `magic_items.csv` flattens the catalog into sortable columns, adds readable
  summary columns for nested effects, and retains the full ontology record in
  `data_json`.
- `magic_item_bonuses.csv`, `magic_item_defenses.csv`, and
  `magic_item_usage_limits.csv` split nested effect records into one row per
  effect for easier spreadsheet filtering.
- `magic_items.sql` creates and populates a standalone `magic_item_export`
  table with the same flattened columns, readable summaries, and full JSONB
  data.

## 5. Database Tables

The loader creates:

```text
extraction_page       page image path, OCR text, raw VLM JSON, status, warnings
magic_item            normalized item rows, plus full ontology JSONB
extraction_error      page/enrichment errors and raw payloads
```

Quick database inspection:

```bash
uv run python - <<'PY'
import db

with db.connect() as conn, conn.cursor() as cur:
    for table in ["extraction_page", "magic_item", "extraction_error"]:
        cur.execute(f"select count(*) from {table}")
        print(table, cur.fetchone()[0])

    cur.execute("""
        select item_kind, rarity, count(*)
        from magic_item
        group by item_kind, rarity
        order by item_kind, rarity
    """)
    for row in cur.fetchall():
        print(row)
PY
```

## 6. Design Notes

Gemini vision is the primary extractor because the catalog content lives in page
images, not a useful embedded text layer. OCR preserves the generated full page
text for auditability and can be passed as noisy context, but the extractor does
not ask Gemini to reproduce full paragraphs. This avoids brittle verbatim
transcription failures while still retaining all PDF-derived page text. The
second Gemini pass is text-only: it adds semantic tags such as equipment slot,
damage types, spell grants, target creatures, and usage limits. Pydantic
validates both source extraction and semantic enrichment before anything is
inserted into Postgres.
