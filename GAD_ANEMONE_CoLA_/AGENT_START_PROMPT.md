# 给执行 Agent 的起始指令（ANEMONE / CoLA / GREET）

任务目标：在 `mpanpan/GAD-datasets/dataset` 的 Amazon、YelpChi、Elliptic、T-Finance 上，使用真正的 ANEMONE、CoLA、GREET 源码各独立训练/推理 seed 0–9，最终形成 3×4×10=120 次经核验的节点异常检测实验。当前仓库只有数据/评测/调度交接框架，**没有任何已实现的模型适配器或正式实验结果**。不要填写推测值。

## 先读与审计

1. 依次阅读 `AGENTS.md`、`README.md`、`GAD_3_Baselines_AGENT_Instructions.md`、`GREET_SOURCE_AUDIT.md`、`PROTOCOL.md`、`EXPERIMENT_PLAN.md`、`INTEGRATION_CHECKLIST.md`，再审计所有 `configs/`、`scripts/`、`gad_handoff/` 和 `adapters/` 代码。记录当前提交、文件清单与已有测试结果；保留已有 ANEMONE/CoLA 行为，不能假设它们已经实现。
2. 固定 `mpanpan/ANEMONE`、`mpanpan/CoLA`、`mpanpan/GREET` 和 `mpanpan/GAD-datasets` 的真实 Git SHA；核验许可与原实现。数据根目录是下载并拉取 LFS 后的 `dataset/`。Amazon/YelpChi 必须是非 `hetero/` 的普通图；Elliptic 三份 CSV 按 txId 对齐，unknown 不计入评价。执行 `python scripts/preflight.py --data-root /absolute/GAD-datasets/dataset`，记录源文件 SHA256 与所有 seed 的 mask、图统计。
3. **先解决 GREET 的评分协议。** 现有本地 GREET `main.py` 是表示学习+有标签逻辑回归评价，并没有确认的无监督异常分数。审计远端源码及官方 `yixinliu233/GREET` 差异。若只有分类探针，须明确命名 `GREET + supervised probe`，列明训练标签、拟合节点、概率分数和标签预算，并作为不同监督轨道汇报。协议未确定时保持 `configs/experiment.json` 的 `BLOCKED_SCORE_PROTOCOL`；不得自行发明异常分数，也不得以测试标签选模。

## 实现与阶段门槛

4. 保留原损失、模型、采样和评分语义。分别实现 `adapters/anemone.py`、`adapters/cola.py`、`adapters/greet.py`；GREET 表示学习不接收标签，若使用监督探针，单独实现受控的标签拟合阶段。大图不得构建全局密集 NxN 张量；任何工程近似都要具名记录并做小图对照。统一输出全部节点的 `(node_id, higher_is_anomaly_score)`，三个方法同数据/seed 的最终评价节点集合必须相同。
5. 在 H200 环境固定依赖与设备、记录 GPU/CPU/并发状态。先跑单元、原始实现对照与回归检查，再跑三个方法的 Amazon seed 0 和 Elliptic seed 0 Smoke Test。核验标签隔离、节点顺序、分数方向、有限值、计时和内存。任一门槛失败先修复或标记 BLOCKED；通过后才把相应配置状态设为 `APPROVED`。
6. 用 `python scripts/dispatch.py --data-root /absolute/GAD-datasets/dataset --gpus 0 --dry-run` 核对 120 项矩阵，再从低并发开始正式实验。每次种子必须重新初始化模型。断点续跑须核对源码 SHA、数据哈希、配置、评分协议、seed 和已有分数；失败保留日志，不补造结果。

## 输出和验收

7. 复用 `gad_handoff/metrics.py`，逐次计算 AUROC、AUPRC(AP)、Recall@K、Precision@K、Precision、Recall、F1、Macro-F1、G-Mean、Balanced Accuracy、Specificity、MCC、Accuracy。分类阈值只用验证标签校准；K=测试异常数属于 oracle 预算。保存逐次原始分数、节点 ID、标签/可验证标签引用、指标、配置、环境和日志，再独立复算对账。
8. 逐次记录预处理、训练、可选评分器拟合、推理、总墙钟时间，训练/推理阶段峰值 CPU 进程及子进程 RSS、CUDA 已分配显存峰值、采样间隔。10 次时间统计 sum/mean/sample SD；内存统计 mean/sample SD/max，绝不对峰值内存求和。吞吐型并行运行与受控的论文效率测量分开标注。
9. 交付完整的性能汇总表、效率表、120 条逐次记录、原始日志、失败清单、数据审计、环境清单、方法移植差异及可复现实验报告。只有 120/120 均通过协议与完整性检查时才能写 COMPLETE。若源代码、评分协议、GPU 或实验数据缺失，写出具体 BLOCKED 原因和仍缺的运行，不得给出虚构均值。
