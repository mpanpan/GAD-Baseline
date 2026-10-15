# GAD Baselines — ANEMONE / CoLA / GREET: Agent 实验执行指令（v2）

## 0. 角色与目标

你是负责科研实验复现的 Coding Agent。此次任务是在 8×NVIDIA H200 服务器上，完成 **ANEMONE、CoLA、GREET** 三个 Baseline 的四数据集图异常检测实验，用于与我们提出的新方法比较。任务是最小修改的 Baseline 适配，而不是开发新方法。不得编造结果，不得以测试标签调参。

模型：
- https://github.com/mpanpan/ANEMONE
- https://github.com/mpanpan/CoLA
- https://github.com/mpanpan/GREET

数据：
- https://github.com/mpanpan/GAD-datasets/tree/main/dataset

实验管理：
- https://github.com/mpanpan/GAD-Baseline/tree/main/GAD_ANEMONE_CoLA_ （审计后扩展命名或目录，使其包括 GREET）

**正式任务量：3 模型 × 4 数据集 × 10 随机种子 = 120 次独立训练和推理**。随机种子统一 `0..9`。单个任务未通过验证时标记 `BLOCKED`，不得伪装已完成。

## 1. 保持原方法，统一外部评价

1. 保留三个 Baseline 的原始网络结构、目标函数、正负样本构造、采样/随机游走、表示学习过程和异常评分（如存在）；保留训练流程与合理适用的作者配置。
2. 可以进行必要的数据加载、GPU/DGL/PyG 兼容、数值等价的稀疏化和日志工具修改。严禁未经验证的算法替换。如大图需要采样近似、近似负例或随机游走探针，记录为 `adapted/approximate` 实现，不可称为严格原始复现。
3. 原方法没有监督训练时不得擅自增加监督训练。不得为了统一形式强制修改各自内部的数据划分；但是**同一数据集的最终检测指标必须基于相同的评价节点集合和真实标签**。
4. 对新增数据集缺少作者给定超参数的情形，采用来源可追溯且事前固定的默认配置；任何使用验证标签进行选择的过程都要记录。严禁使用测试标签选择 checkpoint、阈值、超参数或下游评分器。
5. 每个随机种子必须重新初始化、独立训练和推理；不可将 10 次推理当作 10 次独立训练。

## 2. 数据集审计与加载

目标数据集：`Amazon`、`YelpChi`、`Elliptic`、`T-Finance`。

- **Amazon、YelpChi 必须使用 `dataset` 目录下 `hetero` 文件夹之外的普通图版本**。大小写、文件实际名称及后缀以 GitHub 仓库为准；发现歧义时停止并输出候选文件路径与哈希，禁止猜测。
- 核对节点数、边数、特征维度、标签分布、未知标签、图方向性、自环、重复边、节点顺序、mask、特征缺失/NaN 等。
- 禁止重新注入合成异常；不利用异常标签参与无监督训练。
- 记录每个源文件 SHA256、数据处理脚本版本。各模型使用同一数据版本；若预处理为原方法所必需，保留不同模型自身的合法预处理并说明差异。
- 处理 Elliptic 中未知或未标注类别时，排除这部分节点的**指标计算**，但若原方法是全图无监督训练，可按其原有规则保留这些节点参与图结构计算。禁止把未知标签直接当正常标签。

## 3. GREET 特殊审计（正式训练前必做）

**重要：原版 GREET 主要是无监督图表示学习 + 下游节点分类评价，并不像 ANEMONE/CoLA 一样天然输出图异常分数。**

在提供任何 GREET 的 AUROC/AUPRC 前必须完成：

1. 审查 https://github.com/mpanpan/GREET 的入口、loss、结构编码、kNN 构建、推理和下游评价逻辑；对照官方 https://github.com/yixinliu233/GREET 的原始实现，记录每处差异与使用的 Git commit。
2. 明确最终输出是 `(node_id, anomaly_score)`，且分数越大越异常；记录 `score_method`、是否需要监督标签、是否拟合分类器、分数计算公式、使用哪些节点拟合、是否有单独的阈值校准。
3. **若此 GREET 版本已有经过确认的图异常检测评分器**，沿用并记录其监督信息使用情况，不擅自更换。
4. **若 GREET 仅有节点嵌入和有标签的分类探针**，不要自动把分类概率冒充无监督异常分数：将它命名为 `GREET + supervised probe` 并在结果中单独标识标签使用；不能把该结果与无监督 ANEMONE/CoLA 作为同监督协议直接横向排序。向项目负责人明确报告分歧，评估协议未确定前标记该项 `BLOCKED`，不得自行发明异常评分函数。
5. 若采用已有 `GREET_sparse_sampled`（工程近似适配），必须标注与原始 GREET 在负例、排序边、随机游走/结构编码等方面的差异，精确命名，必要时做小图一致性验证。
6. 对 GREET 的表示学习、结构编码/kNN 预计算、下游评分器拟合、推理分别计时。确认是否在训练/评价过程中使用任何标签决定最优轮次或 checkpoint，避免测试标签泄漏。

## 4. 正式实验矩阵

| Method | Amazon | YelpChi | Elliptic | T-Finance | Total |
|---|---:|---:|---:|---:|---:|
| ANEMONE | 10 | 10 | 10 | 10 | 40 |
| CoLA | 10 | 10 | 10 | 10 | 40 |
| GREET（注明具体评分协议） | 10 | 10 | 10 | 10 | 40 |
| **Total** | **30** | **30** | **30** | **30** | **120** |

种子 `0,1,2,3,4,5,6,7,8,9`。必须记录每次结果是否 `PASS / FAIL / BLOCKED`，只有通过协议和完整性检查才能计入 120 个正式结果。

## 5. 检测性能指标

- 主指标：AUROC、AUPRC（Average Precision）、Recall@K、Precision@K。
- 分类指标：Precision、Recall、F1、Macro-F1、G-Mean。
- 补充：Accuracy、Balanced Accuracy、Specificity、MCC。
- 统一正类为 `anomaly=1`；过滤未知/未标注标签；确认异常分数朝向。一个结果只有单一类别时，AUROC 等不适用的指标应记 `N/A` 并注明原因，不能伪造。
- K 必须事前固定定义，报告数值与含义；若 K=评价集的真实异常数，标记 `oracle-budget`，不能表述为完全无标签的预算选择。可另外报告 P@1%、R@1%、P@5%、R@5%。
- 分类阈值规则须独立于测试标签、事前固定；如果用验证标签拟合阈值，应明确注明标签协议与校准节点范围。不得在测试集最大化 F1。
- 各方法统一使用评估器，不改变原始连续 anomaly score 的计算逻辑。

## 6. 效率指标（每次和 10 次汇总）

每个 `(method, dataset, seed)` 记录：

- `preprocessing_time_sec`：数据预处理、结构编码/kNN 等任务相关预计算；分清一次性缓存和每次种子重新生成的部分。
- `training_time_sec`：模型正式训练墙钟秒数（包含训练中采样、前向、反向、更新）；同步 CUDA。
- `scorer_fit_time_sec`：如果 GREET 或其他方法确实拟合下游评分器，单独计时（可设 0 或 N/A，但不能漏算总耗时）。
- `inference_time_sec`：从可用模型/评分器开始直到目标节点异常分数全部生成；包含原方法要求的多次采样或分数聚合；同步 CUDA。
- `total_runtime_sec`：任务级数据加载、预处理、训练、评分器拟合、推理和保存结果的总墙钟时间；不包括事先的软件环境安装。
- `inference_ms_per_node`：根据实际推理节点数计算。
- `peak_gpu_memory_train_gib`、`peak_gpu_memory_infer_gib`：CUDA 已分配峰值；若用 NVML，另列 `peak_gpu_process_memory_gib`，不能混为一谈。
- `peak_cpu_ram_train_gib`、`peak_cpu_ram_infer_gib`：进程+子进程 RSS 采样峰值，并写出采样间隔。
- 如可测，单独统计预处理与评分器拟合阶段峰值内存，并保留 GPU、CPU 环境及并发状态。

对每个方法-数据集 10 个有效种子分别输出：

- 所有 10 条**逐次值**；
- 时间：`sum`（10 次累计）、`mean`、样本 `std(ddof=1)`；
- 内存：`mean`、样本 `std(ddof=1)`、`max`；**不对 10 个峰值内存相加**；
- 区分单任务墙钟时间的累计和整个并行作业的实际总历时，不能混用。

资源公平性：论文时间和峰值内存比较优先在同 GPU 类型、相近环境、无 GPU 共享任务、受控 CPU/内存资源条件下采集。吞吐优先的 8 卡并行实验与用于论文效率表的受控效率实验应明确区分。若重测，请使用相同种子和完整算法流程。

## 7. H200 工程环境

- 原版 GREET README 使用旧版 PyTorch/DGL/PyG，H200 上应新建可兼容环境，不可假定原依赖版本直接可用。
- 兼容代码只能做等价替换，并提供变更记录及必要的小图数值/统计验证。
- 对大图严禁直接把稀疏全图邻接强制转换成 N×N 稠密矩阵；若等价优化无法实现或资源仍不足，标记 `BLOCKED`，严禁通过静默更改损失、采样或模型结构解决。
- 安装和运行不向外上传私有日志、服务器密钥、数据及凭据。

## 8. 目录与输出格式

建议：

```text
GAD-Baseline/
  AGENTS.md
  EXPERIMENT_PLAN.md
  configs/
  scripts/
  adapters/
    anemone/
    cola/
    greet/
  results/
    raw/{method}/{dataset}/seed_{seed}/
    summary/
```

每次保存 `metrics.json`、`efficiency.json`、`anomaly_scores.npy`、`node_ids.npy`、`labels.npy`（或校验可读取的标签引用）、`config.json`、`environment.json`、`train.log`。`config.json` 包含标签使用方式、原模型/适配版本、原仓库 commit、数据哈希、评估节点范围和 score_method。

最终汇总文件：
- `results/summary/detection_summary.csv`
- `results/summary/efficiency_summary.csv`
- `results/summary/all_runs.csv`
- `results/summary/detection_table.md`
- `results/summary/efficiency_table.md`
- `results/summary/experiment_report.md`
- `results/summary/failures.md`

必须校验 3×4×10 完整性，缺失不能用重复/均值填充。汇总按模型、数据集分组报告 mean±std，记录 total/maximum 和有效数量。

## 9. Agent 执行与验收门槛

1. **Repository audit**：读取 3 模型、数据和项目仓库，记录 commit、依赖、运行入口，核验新数据格式；特别定位 GREET 异常评分协议。
2. **Data adapter**：仅将四数据集转成模型要求的特征、图、标签/评估对象，核查节点映射；非 hetero Amazon/YelpChi。
3. **Compatibility tests**：H200 GPU 环境、CPU 小图测试、数值等价性及原实现对照。
4. **Smoke tests**：三个模型 Amazon seed=0，再测试 Elliptic seed=0；确认输出、指标、计时、内存正常。GREET 评分协议未确认则不得进入 GREET 正式实验。
5. **Formal runs**：每个模型×数据集 10 次独立训练；合理调度 8 GPU，保留日志与失败记录，支持断点续跑（必须同时验证 commit/config/data hash）。
6. **Efficiency benchmarking**：资源受控条件复核时间和内存，必要时单 GPU 串行重测。
7. **Aggregation & independent verification**：由保存的原始分数重新计算指标，与 `metrics.json` 对账，生成论文表格；列出所有协议差异。

**最终 PASS**：四个正确数据集、3 个模型、120 次通过审计的有效独立运行；10 次均值±标准差及每次原始数据；完整的性能/效率报告；准确披露所有监督下游步骤与工程近似。

**严禁**：造结果、以测试标签优化超参数/阈值、把有监督 GREET 输出标为无监督结果、把工程近似标为完全原始 GREET、自动改损失/采样逻辑、用训练失败的重复结果凑满 10 个种子。
