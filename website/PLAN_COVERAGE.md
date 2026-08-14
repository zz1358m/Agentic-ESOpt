# Website plan coverage

This file records how the website plan maps to the public implementation and which requested views are limited by the retained research artifacts. Paper results are traceable to the Agentic-ESOpt source tree and the current manuscript; capability replays additionally use the explicitly labeled, formally accepted Stage 3 recheck archive.

## Site-wide coverage

| Plan area | Implemented evidence |
| --- | --- |
| Static site structure | Physical deep-link entry points exist for all five tasks, scaling, and paper pages. |
| Home research narrative | Paper-title hero, three motivation claims, seven-step ES loop, five environment cards, a Sudoku/WebArena/AHD capability switcher, selected results, Model-size/ES-population scaling callout, and paper metadata/citation. |
| Shared task controls | Generation controls include previous/play/next; task, case, split, generation/turn, method, baseline, capability task, and capability step selectors persist in the URL; charts show the selected value and change from their first retained point. |
| Data provenance | Every task links its source metadata; `data_audit.json` verifies source curves, manuscript tables, private-path scans, local-endpoint scans, contact-pattern scans, and published asset counts. |
| Public-data boundary | Six compact JSON payloads, two selected document images, one paper PDF, and no raw-log download are published. |

## Task coverage

| Task | Repository-backed implementation | Retention boundary shown in the UI |
| --- | --- | --- |
| Sudoku | Paper curves remain selectable for masks 5/10/15. Favorable mask-5 case `eval-000064` maximizes the three-repeat Base-to-final gain without regression and links five aggregate scores, prediction boards, and feedback. | The linked model-output replay is intentionally scoped to recheck mask 5; masks 10/15 retain paper curves and task demos but no original checkpoint boards. |
| Math | Four curated DAPO/AIME 2026 cases; full replay at generations 9, 19, 24, and 25; train and selected-dataset evaluation curves; turn-level trace controls. | Other generations expose aggregate metrics only. Reasoning and raw retained steps are opt-in. |
| DocVQA | Two selected document images; Base, 9, 19, 29, and 39 replays; zoom and drag-to-pan; train and ANLS evaluation curves. | OCR boxes are omitted because the retained observations contain no image coordinates. Only selected images are published. |
| WebArena | Six tasks across five retained site categories; four settings × three repeats for each task; same-task setting comparison; evaluation curve. Favorable task 4 links epochs 10/50/70 to its real success/failure, improves from 0% No Skill to 100% Agentic ESOpt across three repeats, and exposes the retained final answer at epoch 70. | Earlier epoch text and turn-level browser observations/actions were not retained. The UI shows an explicit missing state and never reuses the final answer. |
| AHD | All 144 original result artifacts and 72 original EoH curves remain available. Favorable ACO-TSP case `tsp-aco-sample-agentic-1000-r1` links real heuristic versions at generations 1/12/50 and improves from 6.48937 to 5.90256. | The code evolution is explicitly labeled as an execution-side PASS recheck awaiting designated final review; the original Sample artifact retains final code only. |
| Model-size & ES population scaling | Complete 4B/9B × G=8/16 matrix, Best/Final toggle, cell selection, per-axis deltas, and URL state. | G is defined as perturbation directions per ES update, not physical compute nodes; these four observed settings, each evaluated with three repeats, are not presented as a universal scaling law. |

## Reader validation pending

The implementation and automated acceptance checks are complete. The plan's five-person, one-minute comprehension study requires external participants and is not claimed as passed: recruit at least five target readers, use the three-question comprehension protocol, and record the result before public launch.

## Verification contract

Run the following before publishing:

```bash
npm run data
python -m unittest discover -s tests -p 'test_*.py'
npm test
npm run typecheck
npm run build
```

`npm run data` fails if selected source checkpoints disappear, manuscript table values drift, public payloads contain private paths/contact patterns, or required source configurations are missing.
