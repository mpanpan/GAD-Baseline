# Experiment matrix and execution order

## Canonical configuration
- Model: `cola`, `anemone`, `greet` (actual author code, not GADBench built-in baselines).
- Dataset names: `amazon`, `yelp`, `elliptic`, `tfinance` from `mpanpan/GAD-datasets/dataset`. **NOT** `hetero/amazon` or `hetero/yelp`. Elliptic is a three-CSV directory.
- Trial and seed: `trial=seed=0..9`; 3×4×10=120 runs. Independent model initialization and stochastic training/inference each time.
- Goal: **node anomaly detection**, `1` anomalous, `0` normal, unknown label excluded only from evaluation. Never substitute injected anomalies.
- ANEMONE/CoLA training and GREET representation learning remain self-supervised. A GREET labeled scoring probe, if selected, is a separate supervised stage with its own training labels and label-use disclosure; the scoring protocol must be settled before formal runs.
- Validation/test node subsets: stored `val_masks[:,trial]` and `test_masks[:,trial]` for the three DGL binaries; deterministic class-stratified 10/10/80 masks for Elliptic CSV. Test metrics use known labels only. All-node score export and stable node order are required.
- Save every seed's scores; `AGENTS.md` gate must be passed before full launch.

## Suggested staging
| Stage | What | Success |
| --- | --- | --- |
| G0 | Source + provenance, GREET scoring and GAD-datasets audit | Three method SHAs; score protocol; three DGL binaries and Elliptic CSV joins/masks validated |
| G1 | CPU/single H200 smoke of graph framework | Checked CUDA kernels, versions and memory |
| G2 | Port untouched model principles + sparse engineering | Small-graph parity and score contract for all three; no dense full adjacency |
| G3 | Amazon seed 0 and Elliptic seed 0 per method | Valid all-node scores, metrics, time/memory and no label leakage |
| G4 | 120 total runs with idempotent resume | 10 complete seeds for each of 12 method-dataset pairs |
| G5 | Metrics and provenance audit | 120/120, finite scores, means/sample SD, time sums, memory maxima, logs retained |

`python scripts/dispatch.py --data-root /absolute/GAD-datasets/dataset --gpus 0 --seeds 0 --datasets amazon --models cola,anemone,greet --dry-run`

`python scripts/dispatch.py --data-root /absolute/GAD-datasets/dataset --gpus 0,1,2,3,4,5,6,7`

The second command only after G0–G3 pass and GREET scoring is approved; GPUs each run an independent training job. Eight GPUs do not mean training one model on all eight simultaneously.

## Hyperparameter fairness
Freeze original documented/harmonized settings ahead of full experiments, e.g. test rounds, augmentation/subgraph size, epochs, patience, hidden size and learning rate. Original ANEMONE alpha and CoLA defaults are dataset-specific; target four datasets are NOT part of original paper settings. The Agent must record a defensible agreed cross-dataset setting, preferably an author-default and an optional validation-tuned sensitivity study, and must never choose hyperparameters by test AUROC/AUPRC. `configs/experiment.json` retains placeholders that must be approved in `results/porting_notes.md` before G4.

## Compare with published work carefully
GADBench has supervised baselines and trial-specific masks. If GREET uses a labeled probe, its scores belong to a distinct supervision track. Do not rank it against unsupervised ANEMONE/CoLA as if label use were identical. Keep the final evaluation node set identical for the three methods within each dataset/seed, even when their internal training splits differ.

## Efficiency summary

For each run record preprocessing, training, optional scorer fitting, inference and total wall time. Record peak CPU process-plus-child RSS and peak CUDA allocated memory separately for training and inference, plus the overall run peak; record the sampling interval. For each 10-run group report time sum/mean/sample SD, memory mean/sample SD/max. Never sum memory peaks. Distinguish summed task time from elapsed parallel-job time. See `GAD_3_Baselines_AGENT_Instructions.md` for required output files and controlled efficiency benchmarking.
