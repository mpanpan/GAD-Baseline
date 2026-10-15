# H200 setup — intentionally not an unverified pinned training stack

Host administrator must approve GPU use/network/dependencies. Create fresh Python >=3.10 env and install NVIDIA/H200-compatible CUDA PyTorch + DGL (and PyG only if actual code requires). Validate `torch.cuda.is_available()`, capability=(9,0), DGL CUDA graph, random-walk sampling and segment operations. Record `pip freeze`, OS, driver, torch/CUDA/DGL versions and `nvidia-smi` output in `results/environment.json` before G3.

ANEMONE original: Python 3.6 / torch 1.8 / DGL 0.4.1. CoLA original: DGL 0.4.0. These must NOT be installed blindly on Hopper H200. Functional parity after modern sparse port is required.

GREET's local README cites Python 3.9, PyTorch 1.11, PyG 2.0.4, DGL 0.8, NumPy 1.21.2, scikit-learn 1.0.2 and SciPy 1.7.3. These are source requirements, not a validated H200 lockfile. Install a compatible modern stack only after testing its CUDA kernels and documenting any API port. Do not install old packages merely to match the README. `requirements-eval.txt` remains the shared metric/resource dependency list.
