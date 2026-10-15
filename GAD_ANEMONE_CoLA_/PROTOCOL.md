# Evaluation protocol v1 — research preregistration

**Truth:** graph.ndata['label'] encoded as 0 normal / 1 anomalous; if unknown label `-1` exists, exclude it from validation/test metrics. Assert any other label value is invalid. An anomaly score is a **higher = more anomalous** real scalar for every node, never a class prediction. Verify original formula and sign per model; no post-hoc sign flip based on test AUROC.

**Split:** For the three DGL binaries, use stored `test_masks[:,seed]` and `val_masks[:,seed]`. The published Elliptic CSV lacks masks; deterministically stratify known labels by class and seed into 10% unused, 10% validation, 80% test. Unknowns remain graph context and are excluded from calibration/evaluation. The Elliptic split construction uses labels, while self-supervised model training does not. Feature scaling, if performed, must be documented and must not use labels. Disclose that the binary masks and Elliptic masks have different origins.

## Threshold-free primary metrics
- `AUROC`: `sklearn.metrics.roc_auc_score(y_test, score_test)`.
- `AUPRC_AP`: `sklearn.metrics.average_precision_score(y_test, score_test)` (**AP**, step-interpolated, not trapezoidal PR-AUC). We label AP as AUPRC explicitly.
- `Recall@K_oracle`: rank all known-label test nodes by descending anomaly score. K equals number of anomalies in that test split; TPs / anomalies. **Test-label-derived budget, not deployable without knowing prevalence**. For operational ranking, additionally report Precision@1% and Recall@1%, Precision@5% and Recall@5% with K=ceil(ratio × known-label test nodes), independent of test label prevalence.
- `Precision@K_oracle` separate (same as Recall@K when K=anomaly count).
- `PR_AUC_trapezoid` supplemental, marked separately from AP.

## Thresholded classification metrics (validation-label-calibrated)
Fit a single score threshold on **known-label validation nodes only** maximizing positive-class F1 (fixed deterministic tie rule). Do not fit on test. Apply unchanged to held-out test to compute: `Precision`, `Recall` (TPR), `F1`, `Macro-F1`, `G-Mean = sqrt(TPR × TNR)`, Balanced Accuracy, Specificity, MCC and Accuracy. All metrics binary average with anomaly class=1; zero_division=0. These *require validation labels*, so these are not label-free deployment results. If validation has fewer than one positive or one negative, mark run BLOCKED or calibration unavailable, not fabricated.

Define threshold prediction consistently `score >= threshold`; score ties broken by node ID for ranking. All reported metrics are fractions in [0,1], not percentage. For each metric, **mean ± sample SD (`ddof=1`) over ten independent seeds**. Do not aggregate node-level scores across seeds and then compute one AUROC. Do not silently drop failed seeds; 10/10 mandatory.

**Inference stochasticity:** original CoLA/ANEMONE average anomaly scores across multiple random-subgraph test rounds (often 256). This must remain fixed for every model-dataset-seed, subject to audited original implementation. Save inference rounds and timing. `run seed` controls both training and inference RNG; scores per node must remain aligned.

**Unknown label caveat:** Elliptic commonly contains unknown transaction labels. Unknowns may remain graph context but must not be treated as normal for evaluation. If graph's actual encoding differs, document explicit mapping before running; never assume.
