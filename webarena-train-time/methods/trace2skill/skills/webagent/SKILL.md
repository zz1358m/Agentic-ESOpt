---
name: webarena-sft-trace-skill
description: Skill instructions for WebArena agents using WebRL id actions.
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

## Action Format

- Click: `do(action="Click", element="ID")`
- Type: `do(action="Type", argument="TEXT", element="ID")`
- Search: `do(action="Search", argument="TEXT", element="ID")`
- Scroll: `do(action="Scroll Down")` or `do(action="Scroll Up")`
- Finish: `exit(message="ANSWER")`
