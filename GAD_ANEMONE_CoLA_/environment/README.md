# H200 setup — intentionally not an unverified pinned training stack

Host administrator must approve GPU use/network/dependencies. Create fresh Python >=3.10 env and install NVIDIA/H200-compatible CUDA PyTorch + DGL (and PyG only if actual code requires). Validate `torch.cuda.is_available()`, capability=(9,0), DGL CUDA graph, random-walk sampling and segment operations. Record `pip freeze`, OS, driver, torch/CUDA/DGL versions and `nvidia-smi` output in `results/environment.json` before G3.

ANEMONE original: Python 3.6 / torch 1.8 / DGL 0.4.1. CoLA original: DGL 0.4.0. These must NOT be installed blindly on Hopper H200. Functional parity after modern sparse port is required.
