# 给接手 Agent 的执行指令

你的任务是使用真正的 ANEMONE 与 CoLA 方法，在 [mpanpan/GAD-datasets](https://github.com/mpanpan/GAD-datasets/tree/main/dataset) 的 Amazon、Yelp、Elliptic、T-Finance 四个图数据集上各独立运行 10 次（seed 0–9），共 80 次，并提供真实的逐次结果及均值 ± 样本标准差。请持续执行到完成或记录明确的 BLOCKED 原因，不要根据代码推测实验数值。

## 必须先读

依次阅读 `AGENTS.md`、`README.md`、`PROTOCOL.md`、`EXPERIMENT_PLAN.md` 和 `INTEGRATION_CHECKLIST.md`。已有 `gad_handoff/`、`scripts/` 是数据/评测/调度框架，不包含 ANEMONE、CoLA 的模型适配器；不要把框架当作已经完成的算法复现。

## 数据与源码

1. 克隆并固定 `mpanpan/GAD-datasets` 提交 `48ba3c0ce4e367c8411937cd0e5851d8a8f66e6d`，拉取 Git LFS 对象。`--data-root` 必须是仓库的 `dataset/` 目录。核验 `amazon`、`yelp`、`tfinance` 三个 DGL 文件和 `elliptic_bitcoin/` 下三份 CSV；严禁使用 `hetero/`。记录每份源文件的 SHA256。
2. 克隆并固定 `mpanpan/ANEMONE` 与 `mpanpan/CoLA` 的实际源码提交。若这些仓库不可访问，写明 BLOCKED；不要擅自替换成 PyGOD 或其他实现。保留源码于 `external/`，不要直接改动原仓库。
3. `gad_handoff/data.py` 已按四份已发布数据处理路径：Elliptic CSV 通过 txId 对齐，类别 `1` 为异常、`2` 为正常、`unknown` 不参与评价；它没有发布的试验 mask，因此生成 seed 0–9 的确定性分层 10/10/80 mask。其余三个 DGL 文件使用自带试验 mask。通过 `scripts/preflight.py` 验证真实图统计与所有试验；发现源数据版本或字段不同，先修订加载器并记录依据。

## 模型适配与运行

4. 为 `adapters/cola.py`、`adapters/anemone.py` 实现 `train_and_score`，保持原方法的损失、随机游走/子图采样、负样本、推理轮数及异常分数方向。模型训练只能接收特征和边，不得读取标签、mask 或用测试指标调参。大图不得建立全局密集 NxN 邻接矩阵。用固定种子的小图对比原实现的子图、logit、损失与分数，记录差异到 `results/porting_notes.md`。
5. 在可用的 GPU 环境记录 PyTorch、DGL、CUDA、驱动、内存及源码 SHA；先跑 Amazon seed 0 的两个方法，再跑 Elliptic seed 0 的两个方法。验证全节点分数长度、txId 对齐、有限值、正负方向、时间及内存。确定超参数后将 `configs/experiment.json` 对应状态改为 `APPROVED`，记下选择依据。
6. 运行 `python scripts/dispatch.py --data-root /absolute/path/GAD-datasets/dataset --gpus 0 --max-parallel 1`，先控制并发，观察主机内存与显存后再调整。80 个训练必须各自重新初始化；失败必须保留日志并按原因修复后续跑，不能用缺失值凑出均值。

## 指标与交付

7. 对每次运行输出 AUROC、AUPRC (AP)、Recall@K、Precision、Recall、F1、Macro-F1、G-Mean、Precision@K、Balanced Accuracy、Specificity、MCC、Accuracy、Training Time、Inference Time、单次进程峰值 RAM 与 CUDA 峰值显存。保存逐次资源记录、10 次资源总和及平均值。完整组再计算各指标的均值 ± 样本标准差。分类阈值仅以验证标签校准；Recall@K/Precision@K 的 K 是测试集异常数，属于 oracle 预算，务必披露。
8. 运行 `python scripts/verify_scores.py --data-root /absolute/path/GAD-datasets/dataset` 和 `python scripts/report.py`。交付每次的 `scores.npz`、`metrics.json`、`run.log`，以及 `results/data_audit.json`、`environment.json`、`porting_notes.md`、`per_run.csv`、`summary.csv`、`summary.md`、`missing.csv`。只有 80/80 有效且来源一致时可标为 COMPLETE；否则明确列出所有未完成试验及原因。
