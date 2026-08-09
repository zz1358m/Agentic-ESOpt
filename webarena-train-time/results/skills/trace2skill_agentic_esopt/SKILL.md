---
name: webarena-sft-trace-skill
description: Skill instructions for WebArena agents using WebRL id actions.
---

# WebArena Skill

## Core workflow
- Use the most specific visible page or view that can directly show the requested signal; prefer list, detail, report, or management pages over inference.
- Use only the current visible page and DOM evidence after each navigation or state change; never reuse stale element ids.
- Before acting, re-scan the visible controls and match the requested verb to the exact UI control. Do not substitute a nearby action for the one the task asks for.
- Clear stale or conflicting filters before applying a new search or constraint, and confirm the active filters and sort order before extracting anything.
- For search-first tasks, use the page’s actual search control or Enter, then confirm the page is truly showing search results before selecting anything.
- After each click, search, filter, or sort, verify that the page visibly changed. If the same action leaves the page unchanged or produces an error, stop repeating it and switch to a different control or path.
- For list-based tasks, do not stop at a preview row. Inspect enough pages or candidates to cover the full scope, and use pagination or page-jump controls when available.
- For plural or broad tasks, verify each requested item explicitly against the task criterion before exiting.
- Open the exact matching detail view before using a record’s date, time, status, identifier, or other exact field as the answer.
- For form, edit, or dialog workflows, type only into writable controls, fill all required fields, confirm dropdown selections visibly changed, and use the explicit save, submit, or confirm control.
- After any save, submit, or other critical click, look for explicit visible success evidence and inspect a stable destination or reopened view once to confirm the persisted state matches the target.
- Do not trust transient banners, toasts, or typed text without visible page evidence that the change persisted.
- For discussion, comment, reply, or notification tasks, use the visible composer/editor and the explicit submit control from the record-level path.
- If an unexpected but related page opens, use back navigation or breadcrumbs to return to the source list or item and continue from there.
- If a click, submit, or navigation action has no visible effect, do not repeat it blindly; recover a visible state, locate the correct control, and try a different path.
- If the page becomes blank, stale, or unexpected, stop using old element ids and recover a visible state first by going back, reloading, or navigating to a known page.
- For toggle-style tasks, click the actual state-changing control, not a count, auxiliary link, or profile-like detour.
- For saved-item or wishlist-style tasks, confirm the item appears in the persistent saved view, not just in a transient confirmation message.
- Track every requested target explicitly in multi-item tasks and verify each one in the final visible state before exiting.
- Stop immediately once the requested page state, saved change, or answer is visibly confirmed; do not add extra exploration or verification after completion.
