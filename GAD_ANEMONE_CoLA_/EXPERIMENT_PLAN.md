# Experiment matrix and execution order

## Canonical configuration
- Model: `cola`, `anemone` (author code, not GADBench built-in baseline).
- Dataset names: `amazon`, `yelp`, `elliptic`, `tfinance` from `mpanpan/GAD-datasets/dataset`. **NOT** `hetero/amazon` or `hetero/yelp`. Elliptic is a three-CSV directory.
- Trial and seed: `trial=seed=0..9`; 2×4×10=80 runs. Independent model initialization and stochastic training/inference each time.
- Goal: **node anomaly detection**, `1` anomalous, `0` normal, unknown label excluded only from evaluation. Never substitute injected anomalies.
- Self-supervised training on entire transductive graph (as in original papers), no label-based stopping.
- Validation/test node subsets: stored `val_masks[:,trial]` and `test_masks[:,trial]` for the three DGL binaries; deterministic class-stratified 10/10/80 masks for Elliptic CSV. Test metrics use known labels only. All-node score export and stable node order are required.
- Save every seed's scores; `AGENTS.md` gate must be passed before full launch.

## Suggested staging
| Stage | What | Success |
| --- | --- | --- |
| G0 | Source + provenance, GAD-datasets audit | Three direct DGL binaries and Elliptic CSV joins/masks validated |
| G1 | CPU/single H200 smoke of graph framework | Checked CUDA kernels, versions and memory |
| G2 | Port untouched model principles + sparse engineering | Cora or small-graph parity, no dense full adjacency |
| G3 | Amazon seed 0 and Elliptic seed 0 per method | Valid all-node scores and metrics, no leakage |
| G4 | 80 total runs with idempotent resume | 10 complete seeds for each of 8 method-dataset pairs |
| G5 | Metrics and provenance audit | 80/80, numeric finite, means & sample SD, score files retained |

`python scripts/dispatch.py --data-root /absolute/GAD-datasets/dataset --gpus 0 --seeds 0 --datasets amazon --models cola`

`python scripts/dispatch.py --data-root /absolute/GAD-datasets/dataset --gpus 0,1,2,3,4,5,6,7`

The second command only after approval; GPUs each run an independent training job. Eight GPUs do NOT mean training one unsupervised model on all eight simultaneously.

## Hyperparameter fairness
Freeze original documented/harmonized settings ahead of full experiments, e.g. test rounds, augmentation/subgraph size, epochs, patience, hidden size and learning rate. Original ANEMONE alpha and CoLA defaults are dataset-specific; target four datasets are NOT part of original paper settings. The Agent must record a defensible agreed cross-dataset setting, preferably an author-default and an optional validation-tuned sensitivity study, and must never choose hyperparameters by test AUROC/AUPRC. `configs/experiment.json` retains placeholders that must be approved in `results/porting_notes.md` before G4.

## Compare with published work carefully
GADBench has supervised baselines and trial-specific masks; self-supervised ANEMONE/CoLA are different training regimes. Do not describe rankings as exact reproduction of supervised GADBench tables or a fair comparison with GREET unless splits, labels, feature normalization and evaluation protocols are actually aligned.
