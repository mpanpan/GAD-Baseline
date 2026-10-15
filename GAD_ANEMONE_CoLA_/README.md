# ANEMONE, CoLA and GREET on four GAD-datasets graphs

**Status: Agent handoff framework. All three model adapters and experiment results are absent.** The 120-run matrix is a planned experiment, not a completed result. Start with [AGENT_START_PROMPT.md](AGENT_START_PROMPT.md), then read the full [three-baseline instructions](GAD_3_Baselines_AGENT_Instructions.md) and [source audit](GREET_SOURCE_AUDIT.md).

## Published data

Source: [mpanpan/GAD-datasets](https://github.com/mpanpan/GAD-datasets/tree/main/dataset), commit `48ba3c0ce4e367c8411937cd0e5851d8a8f66e6d`. Pass the downloaded or LFS-populated `dataset/` directory as `--data-root`.

| Requested name | Relative source path | Format |
| --- | --- | --- |
| amazon | `dataset/amazon` | DGL binary |
| yelp | `dataset/yelp` | DGL binary |
| tfinance | `dataset/tfinance` | DGL binary |
| elliptic | `dataset/elliptic_bitcoin/{elliptic_txs_classes.csv,elliptic_txs_features.csv,elliptic_txs_edgelist.csv}` | CSV transaction graph |

The Elliptic CSV has no saved trial masks. `gad_handoff/data.py` maps class `1` to anomaly, `2` to normal and `unknown` to excluded, preserves transaction IDs, and generates deterministic class-stratified 10/10/80 masks for seeds 0–9. The other three files use their stored trial masks. These different split origins must be disclosed when comparing scores. Data files are not committed to the baseline repository.

## Experiment target

Three methods × four datasets × ten independent seeds = 120 successful model fits and all-node score exports. Report mean ± sample standard deviation for AUROC, AUPRC (average precision), Recall@K, Precision, Recall, F1, Macro-F1, G-Mean, Precision@K, Balanced Accuracy, Specificity, MCC and Accuracy. Record per-run training, inference and overall memory/time; the executing Agent must add phase-specific preprocessing, scorer-fit and train/infer memory measurements before final efficiency claims. `scripts/report.py` currently emits per-run, ten-run mean/SD, time sums, memory maxima and missing-run files. Incomplete groups have no numerical summaries.

ANEMONE/CoLA training and GREET representation learning receive only graph features and topology. GREET's current downstream logistic probe uses training labels, so its anomaly scoring protocol must be explicitly approved and named before a GREET run is counted. Validation labels calibrate the classification threshold; held-out test labels are for final evaluation only. Recall@K and Precision@K use the test positive count as an oracle K; see [PROTOCOL.md](PROTOCOL.md).

## Run order

1. Fetch `mpanpan/ANEMONE`, `mpanpan/CoLA` and `mpanpan/GREET`, pin commits, audit GREET scoring, then port the three adapters while preserving their mathematical objectives and anomaly score direction. See [INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md).
2. Install a compatible PyTorch/DGL GPU stack and `environment/requirements-eval.txt`; record the environment.
3. `python -m unittest discover -s tests -v`
4. `python scripts/preflight.py --data-root /absolute/path/GAD-datasets/dataset`
5. Run small-graph parity checks and Amazon/Elliptic seed-0 smoke tests. Set a model's configuration status to `APPROVED` only after its score protocol, label use and regression checks pass.
6. `python scripts/dispatch.py --data-root /absolute/path/GAD-datasets/dataset --gpus 0 --max-parallel 1`
7. `python scripts/verify_scores.py --data-root /absolute/path/GAD-datasets/dataset` and `python scripts/report.py`.

This package contains instructions and evaluation infrastructure for the executing Agent. Original model code, GPU environment and experimental results are not included. See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the execution gates.
