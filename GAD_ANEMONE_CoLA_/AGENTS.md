# Mandatory Agent Instructions — Research Baseline Reproduction

## Mission and scope
Port the REAL ANEMONE, CoLA and GREET code from `mpanpan` repositories without changing their mathematical objectives. Use `amazon`, `yelp`, `elliptic`, `tfinance` from `mpanpan/GAD-datasets/dataset`; ten independent seeds per method/data pair make 120 experiments. This repository currently contains a handoff framework, no model adapters or completed runs. Never estimate results from code or logs.

Read README.md, PROTOCOL.md, EXPERIMENT_PLAN.md, INTEGRATION_CHECKLIST.md, GREET_SOURCE_AUDIT.md and GAD_3_Baselines_AGENT_Instructions.md **before modifying code**.

## HARD RULES
1. Reject any resolved dataset path under `hetero/`. The intended source paths are `dataset/amazon`, `dataset/yelp`, `dataset/tfinance` and the three CSV files under `dataset/elliptic_bitcoin/`. Check graph fingerprints and verified stats. Never silently fall back to another file.
2. ANEMONE/CoLA training and GREET representation learning are self-supervised: ground-truth node labels/masks are forbidden in these stages. A GREET supervised probe, if explicitly approved, is a separate labeled scoring stage and must be named `GREET + supervised probe`; record its training-label use and do not compare it as a fully unsupervised detector. Validation labels may calibrate a classification threshold; test labels never train or select a model.
3. Integrate actual ANEMONE/CoLA/GREET source, keeping loss functions and scoring semantics. GREET's original entrypoint trains embeddings and uses a labeled logistic probe; it does not itself establish an unsupervised anomaly score. No GREET AUROC/AUPRC before score protocol and label use are explicit. All engineering alterations need a diff and tiny-graph checks.
4. Never allocate a global NxN dense adjacency on Elliptic/T-Finance; use sparse graph storage and batched extraction of the same local subgraph adjacency/feature tensors. Monitor peak RAM/VRAM, runtime, NaNs.
5. All-node HIGH-is-anomaly score array is required. No final evaluation until score direction, node order, inference rounds and unique IDs are checked. A labeled GREET probe requires a separately documented score-fitting interface and supervision track; never pass labels into the representation learner.
6. Ten seeds are exactly 0..9; one separate model initialization/training per seed. Use the binary graphs' stored `test_masks[:,trial]` / `val_masks[:,trial]`; Elliptic has no stored masks and uses the deterministic stratified masks documented in `README.md`. Save seed and trial separately. No result should mix trial IDs or feature normalization variants.
7. Compute all metrics with `gad_handoff/metrics.py`. Do not optimize threshold on test labels. `Recall@K` with K = number of positives in the evaluation split is an oracle-budget ranking metric; disclose explicitly; it must NOT be used for Precision/Recall/F1. Use validation-optimized F1 threshold for point classification, disclosed as **validation-label-calibrated**, not fully label-free.
8. Source SHA, dataset path/hash, environment packages, method hyperparameters, full score NPZ, validation/test counts, score sign, raw metrics, per-phase time/memory and logs are mandatory. Ten-run time totals and memory maxima are separate; never add peak memories. Do not upload private data or credentials to GitHub. Check resource permission before using all 8 GPUs.
9. If code/data/label rules/environment are missing, write `BLOCKED` plus exact reason. Do not invent data, commits, adapted code, metrics, or successful model runs.
10. Never modify user's source repositories in place without keeping a pinned clean copy. Use `external/` (gitignored) and local adapter integration. No network upload, external telemetry or data exfiltration without explicit authorization.

## STAGE GATES
**G0 source audit:** fetch all three mpanpan method repositories, pin SHAs, verify licenses/versions and GREET scoring/label paths. If inaccessible, BLOCKED. Verify three binary graph schemas and Elliptic CSV joins/masks.
**G1 environment:** modern H200 compatible torch/DGL/PyG as applicable. Confirm CUDA GPU kernels, no unsupported ops. Document exact versions; don't mix original DGL 0.4 CUDA expectations with modern PyTorch.
**G2 port:** create `adapters/{cola,anemone,greet}.py`, sparse-safe data paths and exact node-score orientation. GREET score protocol must be resolved before approval. Run small-graph regression and score-contract tests; show original-to-new diffs.
**G3 smoke:** Amazon seed 0, then Elliptic seed 0, for all three methods whose score protocol is approved. Inspect labels, scores, memory, checkpoint and output.
**G4 full runs:** only after G0–G3 pass, start concurrency=2; expand to 4–8 GPU slots after RAM/CPU/I/O checks. Resumable scheduler. Ten seeds × four datasets × three methods.
**G5 report:** check 120/120 success, reproduce scores-to-metrics, export performance/efficiency tables with mean ± sample SD, time sums, memory maxima, failures and code SHAs.

## Failures/changes
- Existing ANEMONE may fail `torch.save` on nonexistent checkpoints directory; create per-run `run_dir/checkpoints` before save, don't suppress save errors.
- Existing CoLA can fail DGL old `DGLGraph(nx_graph)` / NetworkX `.edges` API; replace graph construction correctly under the tested DGL version, verify edge indices and random walk equivalence; do not import/execute old DGL internals blindly.
- If full model still produces OOM, document the precise engineering limit and implement memory-preserving chunking; do NOT quietly reduce graph/node count, evaluate sample, change subgraph sizes, inference rounds, or split.

## Deliverables
`results/<model>/<dataset>/seed_<0..9>/{scores.npz,metrics.json,run.log}`; performance and efficiency summaries, `results/missing.csv`, `results/data_audit.json`, `results/environment.json`, `results/porting_notes.md` with baseline parity and GREET score protocol; Git SHAs. Preserve errors. If no completed runs exist, report INCOMPLETE/BLOCKED rather than a numerical table.
