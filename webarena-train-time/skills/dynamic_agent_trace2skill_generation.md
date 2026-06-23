# WebArena Skill

## Essential Workflow for Geolocation Extraction
- Perform search input, submit the query, and parse visible content for coordinate data.
- After entering search/filter input, identify and click the correct submit button.
- Distinguish interactive elements from labels by roles, types, and context.
- Verify results or page updates before proceeding.
- Confirm elements contain relevant content before interaction.
- After data extraction, analyze and summarize results.
- Monitor interaction effects; if progress stalls, change strategy.
- Before creating milestones, check for unique titles.
- After filling fields, perform final submission.
- Monitor submission feedback; adapt or report failure if errors persist.
- Before interacting with critical controls like submit buttons or transport mode radios, check if they have identifiable element IDs.
- If no IDs exist, use alternative interaction methods such as keyboard events, indirect clicks on parent elements, or triggering form submission programmatically.
- Always set transport mode explicitly to match task requirements (e.g., walking) even if UI elements lack IDs; do not assume defaults.
- Use precise, specific location names or addresses when entering inputs into map or directions fields; avoid vague queries like "hotels near [location]" to prevent search failures.
- Use site search fields deliberately to locate relevant communities or target content before posting (see references/search-and-post-verification.md).
- After posting or submitting, verify success by checking for confirmation messages or the appearance of the new content in context (see references/search-and-post-verification.md).

<!-- Trace2Skill sanitizer truncated additional generated rules to keep SKILL.md bounded. -->
