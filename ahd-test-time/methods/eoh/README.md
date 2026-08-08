# EoH method

EoH is the maintained AHD baseline and the heuristic-search loop used by the
Agentic ESOpt model-weight integration. This one runtime supports fixed-model
EoH, fixed-model independent sampling, EoH + Agentic ESOpt, and independent
sampling + Agentic ESOpt.

The original implementation is stored under
`ahd-test-time/methods/eoh/original/eoh`.
The public launcher is `scripts/ahd/run.sh`; callers should not depend on
internal upstream modules directly.

The former parallel `eoh_four_methods` copy has been consolidated here.
`ahd-test-time/scripts/run_eoh_ahd.py` is the canonical runner;
`run_ahd_four_methods.py` exists only to forward older scripts.
