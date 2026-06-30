# Implementation Notes

## 2026-06-29 - Part 1 ontology and extraction pipeline

- Reviewed the assignment requirements and scoped the work to Part 1 only:
  ontology design, PDF extraction, local Postgres loading, and run instructions.
- Confirmed `data/items_combined.pdf` is image-based, so direct PDF text
  extraction is not a reliable primary source.
- Designed the ontology around four Pydantic entities:
  - `ExtractionPage` for rendered page image path, OCR text, raw VLM payload,
    status, and warnings.
  - `SourceMagicItem` for direct catalog facts from the PDF, including name,
    printed type line, item kind, rarity, attunement, source pages,
    continuation flags, confidence, warnings, and a concise mechanics summary.
  - `ItemMechanics` for queryable semantic tags such as equipment slots,
    bonuses, damage types, defenses, spells, target creatures, usage limits,
    and action economy.
  - `MagicItem` for the final enriched catalog entity stored in Postgres.
- Built the extraction pipeline as:
  `PDF -> PNG pages -> OCR provenance text -> Gemini VLM source extraction
  -> Pydantic validation -> Gemini LLM mechanics enrichment -> Pydantic
  validation -> Postgres`.
- Kept OCR as provenance and fallback evidence, while using Gemini vision as
  the primary structured extractor from the rendered page image.
- Changed Gemini item descriptions to concise non-verbatim mechanics summaries
  because asking the model to reproduce full page prose can trigger recitation
  failures. Full page OCR remains stored on `ExtractionPage`.
- Added resilience for messy source data:
  - page-level extraction records with raw model payloads;
  - no overwrite of previous successful page artifacts on API failure;
  - continuation-page merging for multi-page item entries;
  - source page normalization to actual PDF page numbers;
  - page and enrichment errors captured separately.
- Added `validate` stage to check generated artifacts and loaded Postgres data:
  item counts, page coverage, source page references, duplicate names,
  required fields, OCR provenance, table existence, and zero extraction errors.
- Added `export` stage to write reviewer-friendly `magic_items.csv` and
  `magic_items.sql` files from the validated enriched ontology artifact.
- Added readable CSV summaries plus separate bonus, defense, and usage-limit
  detail exports so nested mechanics can be reviewed without reading JSON
  arrays inside spreadsheet cells.
- Added `reznar_exports.sqlite` so the same export can be opened directly in
  DB Browser for SQLite as relational tables and readable item/effect views.
- Added optional `--workers` support for expensive stages. Rendering runs pages
  concurrently at the requested DPI, page extraction uses a parallel first pass
  plus sequential context repair for continuation/error-sensitive pages, and
  enrichment tags independent items in parallel. The fast run uses 16 workers.
- Final validated run loaded 39 page records, 80 magic items, and 0 errors into
  local Postgres.
