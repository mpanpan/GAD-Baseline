# Integration checklist — what the coding Agent must really implement

> No original model code bundled. User's `mpanpan` forks could differ from upstream. They must be inspected, cloned and SHA-pinned on the server before completing adapters. This package implements evaluation/distribution, not original model mathematics.

## CoLA
1. Identify `run.py`, `utils.py`, `model.py` from actual SHA; map its DGL construction, random-walk generation, sampling, contrastive loss and score calculation.
2. Port old `dgl.DGLGraph(nx_graph)` to supported `dgl.graph((src,dst),num_nodes=N)` or equivalent; inspect self-loops, edge direction, multiplicities, node order.
3. Remove **full graph** `.todense()` adjacency conversion; take small dense `(subgraph_size+1)×(subgraph_size+1)` batches only, from sparse graph, preserving subgraph local ordering exactly. Avoid preallocating `(N,...)` GPU scratch that could instead be batch-local.
4. Export one continuous score per node (full graph, original test rounds/contrast sign), verify small-graph comparison to upstream.

## ANEMONE
1. Pin `run.py`/`utils.py`/`model.py` and confirm distinct patch- and context-level objectives, `alpha` and negative sample settings; never collapse to CoLA.
2. Create checkpoint path **per run**, not shared `checkpoints/exp_1.pkl` across seeds/GPU processes.
3. Replace full adjacency with sparse-local extraction while keeping the exact stochastic random-walk and dual-loss semantics. Preserve evaluation rounds and score formula.
4. Port run control to `train_and_score` with an independent seed; do not use original `--runs 10` inside a task already scheduled for 10 seeds.

## Data
- Amazon, Yelp and T-Finance use DGL binary graphs under `GAD-datasets/dataset/`. Elliptic uses three CSVs in `elliptic_bitcoin/` joined by transaction ID; no published trial masks exist for it.
- Verify `feat`, `label`, `val_masks`, `test_masks` after loading, and `source_tx_id` for Elliptic. Check dtype/dim, mapped labels (`1` illicit, `2` licit, `unknown` excluded), nodes, edges, prevalence and content SHA256. Normalize features only as explicitly declared.
- Original CoLA/ANEMONE example code was developed for anomaly-injected Planetoid `.mat` files; DO NOT inject anomalies into real fraud labels.
- Critical: `dataset/amazon` and `dataset/yelp`, **not** any `hetero/` paths. Elliptic scores must retain the CSV transaction ID order.

## Baseline correctness
- Compare original/ported CoLA and ANEMONE on one tiny known graph with same seeds: feature statistics, sampled subgraph neighborhoods, logits, training loss trajectory, node score ordering.
- Report any residual numerical differences; keep adaptation diff and unit tests under git.
- No test-label tuning of score polarity or hyperparameters.
- Log precision mode (FP32 recommended for parity), deterministic torch flags, worker seed and all runtime configuration.

## Server
- H200 CUDA SM90 compatibility check: `torch.cuda.get_device_capability()`, torch/DGL versions, random walk ops, PyTorch/PyG extras only if actually needed, `nvidia-smi`, CPU RAM, filesystem throughput.
- Old CoLA documentation explicitly requires DGL 0.4; original ANEMONE uses DGL 0.4.1. Old Python/PyTorch/CUDA stack is not a validated H200 environment. Use modern tested compatible stack plus explicit API port; don't claim exact original binary environment.
