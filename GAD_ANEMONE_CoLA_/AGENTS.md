# Mandatory Agent Instructions — Research Baseline Reproduction

## Mission and scope
Port the REAL ANEMONE and CoLA code from `mpanpan` repositories, without changing their mathematical objectives, and run 10 independent seeds on `amazon`, `yelp`, `elliptic`, `tfinance` from `mpanpan/GAD-datasets/dataset`. 80 completed experiments are mandatory for a complete report; never estimate results from code or logs.

Read README.md, PROTOCOL.md, EXPERIMENT_PLAN.md, INTEGRATION_CHECKLIST.md **before modifying code**.

## HARD RULES
1. Reject any resolved dataset path under `hetero/`. The intended source paths are `dataset/amazon`, `dataset/yelp`, `dataset/tfinance` and the three CSV files under `dataset/elliptic_bitcoin/`. Check graph fingerprints and verified stats. Never silently fall back to another file.
2. Model training is self-supervised. Ground-truth node labels / masks are forbidden inside the model adapter, for checkpoint selection, and for score construction. A labeled validation set may only be used for **calibration of classification thresholds** after training. Report that distinction.
3. ANEMONE and CoLA are **not** in vanilla GADBench's model menu. Integrate code from actual method repositories, keeping loss functions and scoring. Never claim original-baseline equivalence after changing loss/objective. All alterations (e.g., sparse engineering) need a diff and tiny-graph numerical/functional tests.
4. Never allocate a global NxN dense adjacency on Elliptic/T-Finance; use sparse graph storage and batched extraction of the same local subgraph adjacency/feature tensors. Monitor peak RAM/VRAM, runtime, NaNs.
5. All-node HIGH-is-anomaly score array is required. No final evaluation until score direction, node order, stochastic inference rounds and unique IDs are checked. Do not train a supervised substitute or directly evaluate GADBench supervised detectors.
6. Ten seeds are exactly 0..9; one separate model initialization/training per seed. Use the binary graphs' stored `test_masks[:,trial]` / `val_masks[:,trial]`; Elliptic has no stored masks and uses the deterministic stratified masks documented in `README.md`. Save seed and trial separately. No result should mix trial IDs or feature normalization variants.
7. Compute all metrics with `gad_handoff/metrics.py`. Do not optimize threshold on test labels. `Recall@K` with K = number of positives in the evaluation split is an oracle-budget ranking metric; disclose explicitly; it must NOT be used for Precision/Recall/F1. Use validation-optimized F1 threshold for point classification, disclosed as **validation-label-calibrated**, not fully label-free.
8. Source SHA, dataset path/hash, environment packages, method hyperparameters, full score NPZ, validation/test counts, score sign, raw metrics and logs mandatory. Do not upload private data or credentials to GitHub. Check resource permission before using all 8 GPUs.
9. If code/data/label rules/environment are missing, write `BLOCKED` plus exact reason. Do not invent data, commits, adapted code, metrics, or successful model runs.
10. Never modify user's source repositories in place without keeping a pinned clean copy. Use `external/` (gitignored) and local adapter integration. No network upload, external telemetry or data exfiltration without explicit authorization.

## STAGE GATES
**G0 source audit:** fetch mpanpan repos and pin SHA, verify license/version and original run outputs. If inaccessible, BLOCKED. Verify three binary graph schemas and Elliptic CSV joins, generated masks and source file list.
**G1 environment:** modern H200 compatible torch/DGL/PyG as applicable. Confirm CUDA GPU kernels, no unsupported ops. Document exact versions; don't mix original DGL 0.4 CUDA expectations with modern PyTorch.
**G2 port:** create `adapters/{cola,anemone}.py`, sparse-safe data path, exact node-score orientation; run small graph regression/original comparison and score-contract tests. Show original-to-new diff.
**G3 smoke:** 1 × amazon per method/seed0 then 1 × elliptic per method/seed0; inspect logits, labels, sample graph memory, checkpoint, output.
**G4 full runs:** first concurrency=2, expand to 4–8 GPU slots only after total host RAM, CPU and I/O headroom established. Resumable scheduler. Ten seeds × four datasets × two methods.
**G5 report:** check 80/80 success, reproduce scores-to-metrics, compare unit tests, export mean ± sample SD table plus costs/failures and code SHAs.

## Failures/changes
- Existing ANEMONE may fail `torch.save` on nonexistent checkpoints directory; create per-run `run_dir/checkpoints` before save, don't suppress save errors.
- Existing CoLA can fail DGL old `DGLGraph(nx_graph)` / NetworkX `.edges` API; replace graph construction correctly under the tested DGL version, verify edge indices and random walk equivalence; do not import/execute old DGL internals blindly.
- If full model still produces OOM, document the precise engineering limit and implement memory-preserving chunking; do NOT quietly reduce graph/node count, evaluate sample, change subgraph sizes, inference rounds, or split.

## Deliverables
`results/<model>/<dataset>/seed_<0..9>/{scores.npz,metrics.json,run.log}`; `results/summary.csv`, `results/summary.md`, `results/missing.csv`; `results/data_audit.json`; `results/environment.json`; `results/porting_notes.md` with baseline parity; Git SHAs. Preserve errors.
