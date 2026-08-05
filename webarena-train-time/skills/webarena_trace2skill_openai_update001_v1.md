---
name: webarena-sft-trace-skill
description: Evolved WebArena skill from Trace2Skill update_001 using OpenAI analysis.
---

# WebArena Skill

Output exactly one valid WebRL action per turn.

## Core Policy

- Use only visible element ids from the simplified HTML.
- Prefer actions that reveal evidence or complete the requested website change.
- Do not guess final answers. Use `exit(message="ANSWER")` only after the answer or completion evidence is visible.
- If an action does not change the page, choose a different visible route instead of repeating it.
- Use exact names, ids, titles, order numbers, issue names, user names, addresses, and product names from the task.
- For website edits, save or submit the change, verify confirmation or changed state, then exit with `done`.
- Before acting, identify the target entity, required output, and any constraints from the task; use these as keywords for search, filters, links, maps, or site navigation.
- Prefer direct evidence-gathering routes: search boxes, filters, internal links, infoboxes, sidebars, maps, sort controls, and page text that names the requested entity or attribute.
- Before clicking, check the element text and surrounding context. Click only if it plausibly moves toward the target, reveals evidence, applies a needed filter, or completes the requested website change.
- Track the current page, recent clicks, and extracted facts. If the same page/action repeats without new content or progress, switch strategy instead of continuing the loop.
- If a click has no visible effect, retry only when the UI appears to require confirmation; otherwise try a different element, search query, filter, or navigation path.
- For multi-part questions, solve one subtask at a time. Extract exact names, dates, numbers, addresses, prices, statuses, or other requested facts before moving to the next subtask.
- When choosing among products, posts, communities, repositories, or other candidates, compare visible names, descriptions, metadata, recency, activity, and task-specific attributes before selecting.
- For posting, form edits, purchases, cart/wishlist actions, or settings changes, verify the destination/context is correct before submitting, then check for confirmation text or the updated page state.
- For Reddit/forum posting tasks, first navigate into a relevant community or subreddit whose title/description matches the question; do not submit from an unrelated post page or generic front page.
- Do not default to broad communities such as AskReddit when the question has a location, product, domain, or topic. Search or browse for the most specific matching community first, such as the city, product, project, or topic named in the task.
- If a submit click leaves the same form visible, inspect the page for validation errors, missing required fields, or wrong destination. Do not retype the same title/body more than once; fix the missing field, choose another community, or change navigation.
- Entering a relevant community is not task completion. For posting tasks, only `exit` after the newly submitted post title, success message, or updated post page is visible in the current observation.
- Do not call `exit` until the answer is visible or the requested change is confirmed. If the task has several requirements, verify each requirement is satisfied first.

## Action Format

- Click: `do(action="Click", element="ID")`
- Type: `do(action="Type", argument="TEXT", element="ID")`
- Search: `do(action="Search", argument="TEXT", element="ID")`
- Scroll: `do(action="Scroll Down")` or `do(action="Scroll Up")`
- Finish: `exit(message="ANSWER")`
