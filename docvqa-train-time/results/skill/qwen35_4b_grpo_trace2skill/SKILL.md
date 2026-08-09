---
name: docvqa-trace-skill
description: Reusable document-image inspection, OCR, grounding, and answer-verification guidance for DocVQA tasks.
---

# DocVQA Skill

Inspect the document image before answering. Read labels, table headers, nearby
text, and layout relationships carefully. Return only the requested answer unless
the task explicitly asks for explanation.

## Protocol-safe OCR + Evidence-Grounded Extraction

### 1) Tool/action protocol (hard stop before OCR)
- Always emit a controller-parseable Action wrapper for every environment/tool step (do not output free-form commands).
- If the runner reports an action parsing issue (e.g., nothing valid was parsed), stop and immediately retry with correct Action syntax.
- Do not proceed to OCR/table parsing after tool/protocol parsing failures.
- After each tool call, verify it actually produced usable evidence (expected text/artifact exists and is readable). If evidence is missing, do a minimal targeted re-run of the same step; otherwise switch to a simpler fallback approach.

### 2) Inspect + decide the plan (no answering yet)
- First visually inspect the page to determine whether the target is: a labeled field, a header/footer token (including page-number style queries), or a table.
- Plan to anchor extraction locally (fields/tables) rather than searching globally.

### 3) OCR baseline, validation, and retry escalations
- Prefer a baseline whole-page OCR pass.
- Gate progress on evidence: only continue to reasoning/extraction if OCR produced non-empty, usable text for the relevant cues.
- If baseline OCR is weak/noisy/empty:
  - Preprocess escalation: apply basic readability improvements (e.g., grayscale/contrast/light binarization) and rerun the same simple OCR call.
  - Crop escalation: crop a tight region around the visually identified (or OCR-located) area and rerun OCR on that crop.
- Dependency safety: use only OCR entrypoints that are available in this runtime.
  - If the intended OCR path fails due to missing functions/modules or unsupported arguments, immediately fall back to a known-good OCR route.
- Known-good fallback OCR: load the image with PIL and run `pytesseract.image_to_string(...)` on the full image (or on the tight crop if full-page OCR is too noisy), then extract by evidence from the OCR text.
- Treat empty/irrelevant OCR outputs as a hard extraction failure (do not answer without grounded evidence).

### 4) Evidence-only grounding rules (no mismatched nearby values)
- Anchor-and-adjacent extraction for non-table fields:
  - Locate the label/descriptor in OCR.
  - Extract the value from OCR text that is layout-proximate to that label (same line first; otherwise the immediately neighboring line/block).
  - Require OCR co-location: accept a value only if it appears in the same local region/crop that contains the matched label relationship.

- Table grounding:
  - Identify the header(s) first.
  - Locate the correct row using the question constraint and local OCR matches.
  - Read the value from the cell at the intersection of the matched row and the requested column/header.
  - If row/column alignment is ambiguous, expand the crop to include the full relevant header + the full matched row, then rerun OCR.
  - Never mix values across different rows/columns.

- Category/descriptor specificity:
  - When the question distinguishes categories/tiers/groups, extract only from the exact category-specific cell/line tied to the requested context—do not substitute adjacent summaries/averages from other categories.

### 5) Header/footer + page-number style recovery (high-impact)
- For page-number requests, search within the whole-page OCR for the closest header/footer phrase that contains the page-index pattern.
- If the token is garbled or missed, retry with crops focused on header/footer regions.
- Recover the final page-index output only when the required components can be extracted from the matching header/footer evidence; otherwise treat it as not found.

### 6) OCR matching hygiene
- Normalize OCR text for matching (case-insensitive key comparison; remove harmless punctuation/spacing differences).
- Normalize units/formatting enough to match the expected pattern without fabricating missing characters.

### 7) Final answer discipline (strict)
- Output exactly one line: `Final answer: <requested_value>`.
- If the value cannot be verified from the relevant OCR region(s), output: `Final answer: Not found`.
- Do not include explanations or multiple candidates.
