# ANEMONE and CoLA on four GAD-datasets graphs

**Status: dataset/evaluation integration only. Model adapters are still absent; no training results are claimed.** The two author methods must be ported and tested before the 80-run matrix can execute.

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

Two methods × four datasets × ten independent seeds = 80 successful model fits and all-node score exports. Report mean ± sample standard deviation for AUROC, AUPRC (average precision), Recall@K, Precision, Recall, F1, Macro-F1, G-Mean, Precision@K, Balanced Accuracy, Specificity, MCC, Accuracy, training time, inference time, and peak memory. Every run stores its own metrics, full scores and log. `scripts/report.py` emits per-run, ten-run mean/SD/sum and missing-run files; incomplete groups are not given numerical summaries.

Self-supervised model training receives only graph features and topology. Validation labels calibrate the classification threshold, and held-out test labels are used only for evaluation. Recall@K and Precision@K use the test positive count as an oracle K; see [PROTOCOL.md](PROTOCOL.md).

## Run order

1. Fetch the original `mpanpan/ANEMONE` and `mpanpan/CoLA` source, pin commits, then port `adapters/anemone.py` and `adapters/cola.py` while preserving their mathematical objectives and anomaly score direction. See [INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md).
2. Install a compatible PyTorch/DGL GPU stack and `environment/requirements-eval.txt`; record the environment.
3. `python -m unittest discover -s tests -v`
4. `python scripts/preflight.py --data-root /absolute/path/GAD-datasets/dataset`
5. Run small-graph parity checks and single-run smoke tests, then set each model's configuration status to `APPROVED` only after it passes.
6. `python scripts/dispatch.py --data-root /absolute/path/GAD-datasets/dataset --gpus 0 --max-parallel 1`
7. `python scripts/verify_scores.py --data-root /absolute/path/GAD-datasets/dataset` and `python scripts/report.py`.

This package contains instructions and evaluation infrastructure for the executing Agent. Original model code, GPU environment and experimental results are not included. See [AGENT_START_PROMPT.md](AGENT_START_PROMPT.md) and [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the execution and acceptance gates.
