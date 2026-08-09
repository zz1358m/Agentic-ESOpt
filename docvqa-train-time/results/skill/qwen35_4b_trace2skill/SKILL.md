---
name: docvqa-trace-skill
description: Reusable document-image inspection, OCR, grounding, and answer-verification guidance for DocVQA tasks.
---

# DocVQA Skill

Inspect the document image before answering. Read labels, table headers, nearby
text, and layout relationships carefully. Return only the requested answer unless
the task explicitly asks for explanation.

## Tool / Protocol Hard-Gates (fail fast; no guessing)
- Use the exact accepted action envelope on every tool turn (match the environment’s schema exactly): `Action: {"name":"bash","arguments":{"command":"<shell command>"}}`.
- If the environment reports an action parsing/execution rejection (e.g., action parsing error/unknown action) treat it as a hard stop:
  - immediately correct the next tool call format,
  - retry only the minimum necessary step,
  - do not continue OCR/layout reasoning or speculative answering based on the failed step.
- After any tool/OCR call, verify observable success:
  - check tool stdout contains usable text when text is expected, and/or
  - confirm any expected artifacts/crops were created and are readable.
- If OCR/tool execution fails or produces no usable OCR evidence, stop content-level reasoning for that tactic and either re-route to the smallest fixable inspection step or abstain if you still cannot obtain label/value OCR evidence.

## OCR Readiness: smoke-test first; evidence-only gates
- Before relying on OCR, confirm the OCR pipeline is runnable (engine/tool present; input image/object type is valid).
- Run a minimal OCR smoke test on a small obvious crop (or the full page) and ensure it returns non-empty, readable text.
- After each OCR run, apply an OCR usefulness gate:
  - if output is empty/garbled/unhelpful (or missing the intermediate evidence you intend to use), stop that tactic and retry with improved localization/cropping (avoid random transformations).

## Inspection & Region Selection (anti-drift)
- Briefly inspect the full image to infer layout type (table vs form vs list) and where evidence likely lives.
- Use whole-page OCR only to locate/confirm structure and approximate regions; do not derive the final answer from noisy whole-page OCR alone.
- Perform deterministic two-stage OCR:
  1) Whole-page recovery: OCR the full image, then use the question to localize the referenced instance/area (e.g., the relevant panel/section/table instance).
  2) Tight ROI grounding: crop tightly around the localized label/value neighborhood (or the target row/column intersection within the table) and re-OCR the crop for final extraction.

## Grounded Extraction Rules (no drift; local co-occurrence)
- Do not use whole-page OCR to infer instance-specific relationships; localize the exact referenced instance first, then OCR within that instance.
- Label→adjacent value extraction:
  - anchor by the exact label/role token as it appears in OCR,
  - take the value from the same OCR line/item where the label occurs,
  - require label+value co-occurrence in the same local OCR context; reject candidates where the label does not co-occur.
- If the required label/cue cannot be found in the tool-visible OCR text from the relevant region, return `cannot determine`.
- If OCR output is fragmented/truncated or multiple near-matches appear, re-crop tighter around the exact label/value area and re-OCR; do not “nearest match” across unrelated lines/columns.

## Table Reading (header→row intersection)
- Identify table header(s) in OCR and lock the requested column by header text (do not assume column order).
- For a specific row/value request:
  - anchor the row using its row key/identifier as OCR shows it,
  - extract the cell using row×column intersection where both the header and the row context appear in the same table-region OCR evidence.
- If dense tables cause OCR confusion, re-crop the table region to include the target header and the target row context (header+row+cell) and re-OCR until the header+row+value are visible together.

## Pagination / Page-Number Extraction (grounded in a cropped band)
- Treat pagination as field-specific: crop to the consistent header/footer band first.
- OCR the pagination crop and extract the page-indicator phrase from OCR lines that contain the pagination structure.
- Apply tolerant normalization for common OCR confusions around pagination tokens only when the underlying pagination structure is present.
- Evidence check: the final pagination answer must be derivable from the cropped pagination region’s OCR text; ignore incidental numbers elsewhere.
- Output formatting:
  - if only a single page index is present, return that index;
  - if the composite pagination structure is present, return it in the required canonical pagination format.

## Ordinal Title Selection (Nth heading without token drift)
- For questions like “Nth/ordinal title”:
  - OCR the relevant heading/title block,
  - split into distinct title-like line entries,
  - order candidates by their vertical top-to-bottom appearance,
  - select the Nth entry from that ordered list.
- If OCR merges multiple headings into one line, refine with smaller crops around each heading and re-OCR until headings are separable.

## Crop / Preprocessing Discipline
- Use one consistent coordinate system for all crops; validate image width/height ordering once and reuse.
- Prefer simple, consistent preprocessing; avoid heavy multi-step pipelines.
- Mode-safe handling: if the image is grayscale, treat pixels as single-channel intensities (do not assume RGB tuples).
- If disk-based crop-save/load is brittle, prefer in-memory cropping and ensure OCR receives an image object (not raw bytes).
- If a crop OCR is garbled/noisy, apply only light, deterministic enhancement (e.g., grayscale/contrast/sharpen) plus at most one controlled upscale, then re-OCR.

## Date/Token Completeness & Exact Substring Copying
- For date/token-like answers, require all needed components to be present in OCR evidence.
- Preserve formatting fidelity by copying the cue-bearing OCR substring exactly as it appears in the relevant line (modifying only by removing surrounding label prefixes/suffixes as needed).
- If any component is missing/garbled, re-crop/enlarge around the cue region and re-run OCR until the complete token sequence is visible.

## Final Answer Contract (strict)
- Return only the requested extracted value/content as a plain string (no OCR dumps, no reasoning, no extra surrounding label text).
- Apply minimal normalization (trim/collapse whitespace) while preserving punctuation/casing as shown in OCR.
- If you cannot deterministically link label→value (or row×column→cell) using verified OCR evidence, return `cannot determine`.
