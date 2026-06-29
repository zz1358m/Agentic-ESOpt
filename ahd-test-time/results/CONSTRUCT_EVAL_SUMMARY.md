# Construct Evaluation Summary

This summary only includes construct tasks under `ahd-test-time/results`.

## Evaluator

- Script: `ahd-test-time/results/eval_construct_results.py`
- TSP/KP results: `construct_eval_results.csv`, full 1000 test instances per setting.
- ASP results: `construct_eval_asp_results.csv`, exact ReEvo ASP evaluator.
- Ignored for ASP reporting: old sampled ASP entries in `construct_eval_smoke.*`.

## TSP Construct

Objective: minimize average tour length.

| Method | Setting | Reps | Mean of Means | Best Mean | Worst Mean |
|---|---:|---:|---:|---:|---:|
| Dynamic-EoH | N=50 | 3 | 6.529941 | 6.364714 | 6.628773 |
| Dynamic-EoH | N=100 | 3 | 9.100426 | 8.937346 | 9.248696 |
| EoH | N=50 | 3 | 6.545004 | 6.463936 | 6.628670 |
| EoH | N=100 | 3 | 9.073673 | 9.000981 | 9.180640 |

## KP Construct

Objective: maximize selected value. Capacity `W=25`.

| Method | Setting | Reps | Mean of Means | Best Mean | Worst Mean |
|---|---:|---:|---:|---:|---:|
| Dynamic-EoH | N=100,W=25 | 3 | 40.510139 | 40.513402 | 40.506540 |
| Dynamic-EoH | N=200,W=25 | 3 | 57.721725 | 57.738446 | 57.690659 |
| EoH | N=100,W=25 | 2 | 40.507598 | 40.509787 | 40.505410 |
| EoH | N=200,W=25 | 2 | 57.738305 | 57.740107 | 57.736502 |

Note: `EoH/KP_construct` currently has only two final code files, so its KP aggregation is 2 reps, not 3.

## ASP Construct

Objective: maximize expanded admissible set size. The ASP evaluator is the exact ReEvo evaluator copied into `eval_construct_results.py`: enumerate valid triple-code children, score each child with `priority(sum(TRIPLES[x]...), n, w)`, greedily prune with `get_surviving_children`, then expand with `expand_admissible_set`.

| Method | Setting | Reps | Mean Set Size | Best | Worst |
|---|---:|---:|---:|---:|---:|
| Dynamic-EoH | N=12,w=7 | 3 | 768.00 | 783 | 756 |
| Dynamic-EoH | N=15,w=10 | 3 | 2775.00 | 2808 | 2748 |
| Dynamic-EoH | N=21,w=15 | 3 | 31539.00 | 33279 | 30648 |
| EoH | N=12,w=7 | 3 | 776.00 | 786 | 765 |
| EoH | N=15,w=10 | 3 | 2760.00 | 2784 | 2742 |
| EoH | N=21,w=15 | 3 | 28465.67 | 30130 | 25904 |

## Full Commands

TSP/KP full results already generated:

```powershell
python ahd-test-time\results\eval_construct_results.py --tasks tsp,kp --tsp-sizes 50,100 --kp-settings 100:25,200:25
```

ASP exact evaluator:

```powershell
python ahd-test-time\results\eval_construct_results.py --tasks asp --asp-settings 12:7,15:10,21:15 --output ahd-test-time\results\construct_eval_asp_results.json --csv-output ahd-test-time\results\construct_eval_asp_results.csv
```
