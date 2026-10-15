# GREET integration audit — 2026-10-16

## Evidence and limits

The audit inspected the checked-in `GAD-Baseline/main` commit `f57965c91dd52e51821d3b3814a61504709559ec` and the local GREET source copy at `D:/260709/GAD/update/GREET`. Attempts to clone/query `mpanpan/GREET` failed because GitHub connectivity was reset/unavailable. **The local copy is not a verified Git revision of the remote fork.** The executing Agent must pin the remote GREET SHA, verify these files and compare with official `yixinliu233/GREET` before implementing or reporting a result.

Local source SHA256: `main.py` `74AD4D4F965B42B9197E40A69A5E960C1CCE3CAEE890C0E4882096C0F0BFA66B`; `model.py` `6F5BA51910C9DD5D650A48607DE2C1BD840588DE4936407939ABA89C229E537E`; `utils.py` `4BA1800C120F08DC354CA9204E21E53CDFBC907F7A9E36DA25D20998A18CC9CE`; `data_loader.py` `3DF96D807FB4E940F002A47EC2CF06ABCDDF8194C9FB74DBB483EFDC067C8C7F`; local added `evaluation.py` `E90B41F7102A414C817F0A7BD9D32E2768E0B52935045ACB771CF6B80EB72700`.

## Current project audit

- `adapters/` contains only an interface and `__init__.py`; `anemone.py`, `cola.py` and `greet.py` do not exist. `results/` is Git-ignored and has no committed experiment results. No prior model algorithm or result is present to preserve in `main`.
- `gad_handoff/data.py` selects direct non-hetero Amazon/Yelp/T-Finance DGL files and Elliptic CSV. `gad_handoff/metrics.py` already implements the requested performance metrics and validation-only threshold calibration; reuse it.
- `scripts/dispatch.py` schedules independent model/data/seed tasks and `scripts/report.py` aggregates complete ten-seed groups. `gad_handoff/run_one.py` records training/inference seconds and an **overall** process RSS/CUDA peak, but does not yet record phase-specific preprocessing/scorer-fit timing or training/inference CPU+child/GPU peaks. The executing Agent must extend this before claiming the full efficiency specification.

## GREET score and label path

- Local GREET `main.py:25–67` trains contrastive and edge-discriminator objectives without node labels in those function signatures. `main.py:73` loads labels and masks. `main.py:103–117` calls `eval_test_mode` during training and prints validation **and test** accuracy each evaluation interval; the best recorded test accuracy is chosen when validation accuracy improves.
- `utils.py:139–153` fits a `LogisticRegression`/`GridSearchCV` probe on `train_mask` labels and calls `predict` on validation/test nodes. This returns class accuracy, not a continuous node anomaly score. Test labels are read repeatedly during training, which is unsuitable for a final held-out anomaly-detection protocol.
- The local added `evaluation.py:109–149` fits a supervised probe on training labels. Its `_probabilities` at lines 199–202 provides a continuous positive-class probability, and `evaluate_test_once` at lines 263–284 uses held-out test labels after freezing. This is a **labeled downstream scorer**, not evidence that original GREET is an unsupervised anomaly detector. Its use requires the name `GREET + supervised probe`, explicit training-label budget, fit node IDs and scorer-fit time.
- The original `data_loader.py` selects PyG node-classification datasets, not the four fraud graphs. `model.py` includes dense KNN masks/similarity/adjacency paths; the executing Agent must audit exact dimensions and preserve or explicitly label approximation semantics when scaling to Elliptic/T-Finance.

## Gate for the executing Agent

Keep `greet.status=BLOCKED_SCORE_PROTOCOL` until the remote-source audit, a defensible node-score formula, label-use disclosure, leakage-safe selection and small-graph checks are complete. The existing `run_one.py` gives an adapter features/topology only; a supervised GREET scorer requires a separate, explicit training-label handoff after representation learning. Do not pass labels into that learner or simply relabel classification accuracy as AUROC. Reuse the shared final evaluator and the same test node IDs as ANEMONE/CoLA. If no approved score protocol can be established, report GREET as BLOCKED and leave its performance cells blank.
