# DocVQA Skill

Inspect the document image before answering. Read labels, table headers, nearby text, and layout relationships carefully. Return only the requested answer unless the task explicitly asks for explanation.

## Evidence-first (no guessing)
- Use only visible/extracted evidence; do not infer from typical document types.
- Do not finalize unless the value/entity/phrase is supported by extracted OCR/visible text.
- If the value cannot be verified after targeted recovery, abstain (e.g., “Unable to determine from the provided image/document.”).

## Tool/action protocol (fail-fast; prevent action-parsing loops)
- Before running any tool/action, ensure the action payload matches the expected schema exactly.
- For bash-like tools, use required nesting: `Action: {"name":"bash","arguments":{"command":"<shell command>"}}`
- If you see “No valid action was parsed” (or similar), immediately fix the formatting next attempt; don’t keep retrying the same malformed call.
- Never attempt unsupported action types (e.g., a “python” action). If Python is needed, run it via the supported mechanism.
- If execution remains blocked after repeated parse failures (e.g., 2+ consecutive parse errors), stop iterative debugging and return an evidence-safe abstention.

## Extraction-first workflow (bounded retries)
1) Confirm input type (PDF vs raster image). If raster, use OCR; do not use PDF-only extraction.  
2) Preflight OCR readiness (images loadable; non-zero dimensions).  
3) Run OCR once with a stable configuration; keep output order top-to-bottom.  
4) Sanity check: OCR output must be non-empty and text-like.  
5) Stop immediately once the needed value/entity is verified.

## OCR reliability
- Pass a `PIL.Image.Image` into pytesseract (not file-path strings or raw bytes).
- Treat `image_to_string` output as text: only decode if it is bytes; if it is already `str`, don’t `.decode()`.
- Require non-empty OCR containing recognizable alphanumeric/entity-like strings relevant to the question.

## OCR grounding discipline (ROI + exact phrase)
- Prefer ROI/crops for field questions (header/footer/top/bottom/metadata blocks) over whole-page OCR.
- Grounding rule: the output must come from the OCR neighborhood of the relevant label/header/row.
- Truncation is a failure condition: if snippets look cut off (e.g., end with “...”), re-run OCR on a targeted ROI around the missing part.

## Entity lookup & snippet validation
- If the exact requested entity/string is not found in OCR, do not treat that as proof of absence.
- Re-run OCR with targeted cropping/zooming around likely regions; re-check.
- Allow near-match reconciliation for common OCR artifacts (l/1, O/0, rn/m, etc.) only when the value is verifiably present in the extracted snippet.

## Label-bound extraction (value adjacent to cue)
- Find the exact on-page label/cue.
- Extract the value adjacent to that label (same line or immediately following tokens; after `:` if present).
- If multiple similar labels/values exist, filter using the question context.
- Normalize minimally (trim punctuation/whitespace; collapse spaces). Prefer the exact OCR-captured value.

## Numeric/date recovery fallback (when OCR is garbled)
- Use date-like regex matching on OCR text (e.g., `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}`).
- Normalize digit confusions (O/0, l/1, I/1, S/5) before regex re-check.
- Only abstain after ROI-focused OCR attempts + preprocessing/parameter variation + regex+normalization fallback failure.

## Tables/spreadsheets (preserve row/column relationships)
- Use OCR table text as intermediate evidence; never jump from unrelated numbers to the answer.
- Map header → column first; then extract values from that column.
- Pair rows next: locate the unique row identifier/value from the question, then extract the paired cell in the mapped column.
- Special case (“Final X for the row where Initial Y = …”):
  - Anchor on the Initial column header.
  - Match the Initial row where Initial Y equals the stated value.
  - Extract Final from the corresponding Final column cell in the same row.
  - If pairing is unclear/noisy, do an additional OCR pass with tighter cropping around the specific rows/columns.

## Page number grounding (page X of Y)
- Do not infer from layout.
- Crop the bottom-right corner using relative coordinates so it scales (e.g., x_start ≈ 0.65*width, y_top ≈ 0.40*height).
- OCR the crop with an appropriate sparse-text `--psm` (e.g., `--psm 6`).
- Extract a standalone page-like integer only if OCR yields plausible standalone numeric tokens.
- Tolerate OCR swaps via normalization (case-insensitive matching for patterns like `PAGE/PASE/P`, `OF/GF`, `\b\d+\s*(OF|GF)\s*\d+\b`, `\b(\d+)\s*/\s*(\d+)\b`) and then re-check patterns after normalization.
- If digits can’t be validated, abstain (e.g., “Page number not found in the provided image.”).

## Category mapping rule (“Extremely” vs “Very”)
- Parse into `{product: {extremely: x, very: y}}`.
- Ensure each percentage/value is associated with both the correct product and the correct category label (“Extremely” vs “Very”).
- Only conclude insufficient data after targeted re-OCR around the table/category anchors still can’t disambiguate.

## Output discipline (format + abstain)
- Output only the requested final value/text.
- Copy values exactly as shown; avoid adding/removing units unless required.
- If normalization is required by evaluation:
  - Remove negative sign if the question/ground truth expects positive.
  - Convert digit → spelled-out forms when required.
- If grounding fails, abstain (do not guess).
