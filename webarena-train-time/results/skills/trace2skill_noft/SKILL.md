---
name: webarena-sft-trace-skill
description: Skill instructions for WebArena agents using WebRL id actions.
---

# WebArena Skill

## WebArena workflow
- Use only visible page evidence and current WebRL ids; re-resolve elements after any navigation, refresh, rerender, or layout change.
- Start from the most direct visible path. Prefer search, filters, menus, or built-in navigation before long scrolling or guessing controls.
- Re-read the current visible page after every click, selection, text entry, scroll, or navigation step. If the page did not visibly change, do not repeat the same action; choose a different visible control or route.
- Before typing, verify the field label, current value, and input type. Clear or select all existing text when overwriting.
- For dropdowns, pickers, autocomplete fields, and multi-selects, inspect the current control state first, choose the visible option that matches the task, and confirm the selected value is shown afterward.
- Treat search results, snippets, counts, and summary links as discovery aids only. Do not conclude completion until the requested state is visibly shown on the relevant page or detail view.
- If search or filter results are empty, ambiguous, or stuck, adjust the query, broaden or narrow the filter, or switch to a different grounded view instead of repeating near-identical searches.
- Use visible filters, chips, row counts, and result scopes as the source of truth for the current view.

## Lists, pagination, and completeness
- For tasks that ask for all matching items, exhaust pagination, scrolling, and expansion until no more entries remain and the requested cardinality is confirmed.
- Use page controls and visible range indicators to move stepwise through the relevant span; do not rely on a single viewport or old row positions.
- Keep a running tally or deduplicated set while scanning across pages or scroll regions when completeness matters.
- For count, total, or range questions, verify every contributing visible row or bucket directly and compute from the source values.
- For most, least, highest, or lowest questions, confirm the global extreme across the full relevant result set or a trustworthy aggregate before selecting the target.

## Detail pages, reviews, and extraction
- Open the item detail page before answering any status, review, comment, or explanation question; confirm the title and key state from the header, badge, or metadata rather than the list row.
- For review or thread-based questions, use explicit review, comment, or reply text from the dedicated discussion view; do not infer from ratings, snippets, or summaries.
- Treat clipped snippets, ellipses, and partial text as incomplete evidence; expand or open the page that shows the full content.
- When asked to extract text, copy the source text verbatim and keep only the exact requested span.
- If a likely match appears, verify its visible details on the detail page before answering or editing.

## Forms, edits, and persistence
- Locate an explicit writable composer, dialog, or editor before typing; do not type into read-only previews or summaries.
- Fill every required field explicitly and use the page’s own save, apply, update, or submit control.
- After saving or submitting, verify the persisted record or changed state is visible on the destination page; do not rely on a generic toast alone.
- If validation, required-field, login, modal, or blank/stale-state interruptions appear, resolve the issue or back out to the last stable page and continue with a different path.
- When editing a prefilled field, clear or select all first so the new value replaces the old one cleanly.
- For validated inputs, enter a format the field accepts and confirm success through a cleared error, visible confirmation, or reread saved value.
- For create-then-configure tasks, finish and confirm the created resource before moving to follow-up settings such as members, permissions, or access controls.
- Prefer reusing an existing item when it satisfies the task; avoid creating duplicates unless creation is explicitly required.

## Reports, maps, and directions
- For report or analytics tasks, use the site’s dedicated report or analytics page first; set the requested filters before generating the report.
- Match the report subview or page type to the question, and read the rendered results directly rather than the filter form or summary widgets.
- For place or location tasks, use the site search or map view to locate the anchor place first, then ground the answer in visible map context or a detailed result panel.
- Use directions mode for travel-time or reachability tasks, keep it separate from general search, and fill both endpoints with concrete locations.
- Verify the route summary, travel time, or distance from the route output, not from surrounding map UI.
- If a destination search fails or resolves incorrectly, refine the query or switch to a broader grounded place label instead of repeating the same guess.

## Dead ends and login barriers
- If navigation lands on a login, cookie, account, or other blocking screen, treat it as a dead end: backtrack once to the last useful page and continue with a different route.
- Check for an existing signed-in session, account menu, or visible credential source before attempting login.
- Do not invent or brute-force credentials; use only credentials provided by the task or visible page evidence.
- If a login attempt fails, read the error message and switch strategy instead of repeating similar guesses.

## Completion rule
- Once the requested state, output, or persisted change is visibly confirmed, stop immediately and base the final answer only on page-visible evidence.
- Do not add extra verification after the completion condition is already satisfied.
