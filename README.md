# 多视图聚类课程设计

本仓库按课程任务书实现不完整多视图聚类与多视图流式聚类，并整理 HSACC、DSVC 的论文复现入口。

**当前进度：课程A/B统一主实验已完成，消融正在运行，噪声与隔离效率随后执行。** A含3种缺失模式×3种比例×3种子，B含4种方法×2种漂移条件×3种子×6视图。每完成一个阶段同步GitHub；仍在运行的实验以目录内进度文件为准，不能视为最终结果。报告在实验核验后补齐，PPT按用户要求最后制作。

## 实验目录

每个实验都有独立的 `README.md`、`run.py`、`config.json`。公共的数据、模型、训练、评估与绘图代码放在根目录对应模块中，避免各实验复制出不同的评测实现。

| 目录 | 实验内容 | 状态 | 结果 |
| --- | --- | --- | --- |
| [01_complete_baselines](experiments/01_complete_baselines/) | 4个数据集的完整特征拼接和各单视图 K-means | 已运行，3个种子 | 63条评测记录 |
| [02_incomplete_multiview](experiments/02_incomplete_multiview/) | 随机、单视图集中、类别相关缺失；零/均值填充；完整数据参考 | 基线已运行，3个种子 | 54条缺失实验与9条匹配视图参考 |
| [03_streaming_baselines](experiments/03_streaming_baselines/) | warm-start、滑动窗口、全量重训练；无漂移、均值漂移、视图噪声 | 基线已运行，3个种子 | 270条批次评测记录 |
| [04_hsacc_reproduction](experiments/04_hsacc_reproduction/) | HSACC，NeurIPS 2025，不完整视图 | 已完成单配置500轮、3个数据种子 | 3条最终指标、邻域分析和图表 |
| [05_dsvc_reproduction](experiments/05_dsvc_reproduction/) | DSVC，ICML 2025，视图流 | 已完成单数据集6视图、200轮/视图、3种子 | 18条视图阶段指标与曲线 |
| [06_course_comparison](experiments/06_course_comparison/) | 同一数据、掩码或视图流下的课程主比较 | A/B已完成 | A有81条缺失质量记录、9条完整参考；B有144条批次记录 |
| [07_ablations_robustness](experiments/07_ablations_robustness/) | 去一致性、去补全、等权、单视图加噪与独立进程效率 | 配置冻结，排队执行 | 仅完成后计入结论 |

基线合计396条评测记录，均来自实际运行。流式实验的一条记录表示一个种子、条件、方法下的一个批次，不应当作一次独立的完整流实验。

另有21条完整预算作者代码记录：HSACC固定最后一轮ACC为71.88%±8.26个百分点；DSVC最终视图ACC为83.53%±11.06个百分点。两者使用不同数据和协议，不能互相比较高低，也不能直接与课程基线表作公平性能比较。短流程证据另存 `validation/`；最终指标已经逐条与本地预测文件核对。

## 复用已有Conda环境

本项目已切换到已有的 `Micro` 环境，使用其中的PyTorch，**不重新下载或安装PyTorch，也不修改原Conda环境**。已验证Python 3.10.20、PyTorch 2.11.0+cu128与RTX 5070能够完成CUDA前向/反向计算，两篇论文短流程均已跑通。

在PowerShell中指定自己的解释器，以下路径是本项目的实际环境示例：

```powershell
$env:MVC_PYTHON = 'D:\Anconda\envs\Micro\python.exe'
python scripts/run_in_environment.py --show-runtime
python scripts/run_in_environment.py -- -m unittest discover -s tests -v
python scripts/run_in_environment.py experiments/04_hsacc_reproduction/run.py --smoke
python scripts/run_in_environment.py experiments/05_dsvc_reproduction/run.py --smoke
```

启动器只借用所指定的解释器，将 `TEMP`、`TMP`、pip、PyTorch、CUDA和Matplotlib缓存等重定向到项目的 `.cache/`。本项目位于D盘，因此临时文件和缓存也位于D盘；不修改系统全局环境变量。可在本地忽略的 `.cache/runtime.json` 保存解释器路径，`MVC_PYTHON`或 `--python`可覆盖它。

缺少的小型附加包可安装到项目目录，不改动Conda环境。例如本机仅缺少 `munkres`，已从本地wheel安装到 `.cache/python_deps/`，启动器会自动加入该目录。另一台机器缺少该包时可运行：

```powershell
python scripts/run_in_environment.py -- -m pip install --no-deps --target .cache/python_deps munkres==1.1.4
```

这条命令只补充 `munkres`，不安装PyTorch。**不要在已有环境中直接批量安装旧版依赖清单**，以免覆盖可用的PyTorch或其他依赖。

## 原始CPU基线环境与快速开始

已验证的基线环境为 Windows 11、Python 3.12.10，具体依赖固定在 [requirements.txt](requirements.txt)。以下是 PowerShell 命令；Linux/macOS 将 `.venv\Scripts\python.exe` 换成 `.venv/bin/python`。

```powershell
git clone https://github.com/Younglbos/The-machine-learning-group-reports.git
cd The-machine-learning-group-reports
python -m venv .venv
python scripts/run_in_environment.py --python .venv/Scripts/python.exe -- -m pip install -r requirements.txt
python scripts/run_in_environment.py --python .venv/Scripts/python.exe scripts/download_datasets.py
python scripts/run_in_environment.py --python .venv/Scripts/python.exe scripts/inspect_datasets.py
python scripts/run_in_environment.py --python .venv/Scripts/python.exe -- -m unittest discover -s tests -v
```

按实验分别运行：

```powershell
.venv\Scripts\python.exe experiments/01_complete_baselines/run.py
.venv\Scripts\python.exe experiments/02_incomplete_multiview/run.py
.venv\Scripts\python.exe experiments/03_streaming_baselines/run.py
```

每个入口从自己的 `config.json` 读取设置，将 CSV、图表、日志、最终标签等保存到本实验的 `results/`。重复运行会更新对应结果文件；日志会追加。若要保留多次完整运行，请修改配置中的 `result_dir`。

本仓库提供已运行的 CSV 和图表快照。原始运行配置保存在各结果目录的 `run_config.json`；实验根目录的 `config.json` 使用拆分后的输出位置。`provenance.json` 记录两者的关系，不会把新配置冒充为旧结果的运行配置。

## 论文资源与启动命令

两篇论文均使用已有环境中的PyTorch。下载固定版本的作者源码后，通过上面的启动器运行：

```powershell
python scripts/run_in_environment.py scripts/download_references.py
python scripts/run_in_environment.py experiments/04_hsacc_reproduction/run.py --smoke
python scripts/run_in_environment.py experiments/05_dsvc_reproduction/run.py --smoke
```

本机已有PyTorch构建支持CUDA 12.8和RTX 5070。HSACC入口也允许CPU运行；DSVC作者实现直接调用 `.cuda()`，需要可用的CUDA设备。

`--smoke` 只检验短流程，不构成论文指标复现。完整命令、参数及协议差异见各论文实验目录 README。原始CPU基线与现用Conda环境版本不同，各次运行的环境记录分别保存，不能把两者混写。

## 数据与协议

| 数据集 | 样本数 | 视图数 | 类别数 | 各视图维度 |
| --- | ---: | ---: | ---: | --- |
| BBCSport | 544 | 2 | 5 | 3183 / 3203 |
| BBC4view | 685 | 4 | 5 | 4659 / 4633 / 4665 / 4684 |
| Caltech101-20 | 2386 | 6 | 20 | 48 / 40 / 254 / 1984 / 512 / 928 |
| NUS-WIDE子集 | 2400 | 5 | 12 | 64 / 144 / 73 / 128 / 225 |

下载的是公开多视图特征与标签，原文件不随仓库上传。逐文件来源、大小与Git blob校验值见 [data/dataset_sources.json](data/dataset_sources.json)，读取注意事项见 [data/README.txt](data/README.txt)。程序保留来源文件中的样本顺序；当前验证了各视图行数与标签长度，原文件未提供独立对象ID，因此没有额外证明语义对应关系。

场景A课程实验使用 Caltech101-20 的 HOG/GIST 双视图（Python索引3、4）。**课程缺失率定义为“不完整样本比例”：选取 `floor(N*r)` 个样本，每个删除一个完整视图，始终保留另一个视图。** 对双视图，缺失块比例约为 `r/2`。所有方法共享同一掩码，标准化参数只从已观测行估计。该定义与HSACC作者生成器参数分别记录，不能直接比较同名 `missing_rate`。

实验03的场景B基线使用 NUS-WIDE 的10个样本批次，预处理只在首批拟合并冻结；漂移从第5批开始。实验06B另外将所有方法统一到HandWritten六视图流，并冻结90秒单批预算，详细历史约束和DSVC课程适配见该目录README。不同流式协议分开分析。

## 文件用途

```text
experiments/           每个实验的入口、配置、README和已验证结果
data/                  数据来源、下载后检查、缺失生成、预处理和读取
models/                K-means、流式基线及作者代码导入辅助
train_or_update/       基线与论文的训练/更新逻辑
evaluation/            匹配后的ACC、NMI、ARI、邻域、计时与内存记录
visualization/         从CSV生成带随机种子标准差的图表
configs/               拆分前的统一基线配置，保留用于原始结果溯源
scripts/               数据与固定版本论文资源下载工具
tests/                 协议测试，使用Python unittest
references/            公开论文/源码下载清单，不提交PDF或源码压缩包
requirements/          课程验收进度清单
third_party/           运行下载脚本后生成，Git忽略
```

## 评价与复现边界

- ACC用匈牙利匹配处理簇编号置换；NMI和ARI按原预测标签计算。
- 基线使用种子1、2、3，K-means为10次初始化、最多300轮、`tol=1e-4`；统计标准差用 `ddof=1`。
- 基线计时不含指标、Silhouette和文件保存；流式时间包含当前批预处理和聚类更新。神经网络作者训练包含周期评估，另列时间字段，不与纯K-means计时直接等同。
- `peak_rss_mb` 是20毫秒采样的整个进程工作集峰值，包含共享数据和库，未隔离成各算法独占内存。`model_state_mb` 单独统计流式聚类器保留的数组，不包含共享编码器、解释器等开销。
- 真实类别只用于评估和明确标注的类别相关缺失模拟；不用于学习中心或选择课程主结果的最佳轮次。
- 课程A/B统一对照已实际运行；消融、鲁棒性和隔离效率正在依次执行。阶段分析见 [reports](reports/)。最终报告和验收清单在结果核验后更新，PPT最后制作。作者协议只覆盖选定数据集与配置，不代表复现整篇论文的全部表格。

## 第三方来源

作者源码、数据与论文通过原地址获取，来源及固定提交见 [THIRD_PARTY.md](THIRD_PARTY.md)。不将本项目的适配入口说成作者原始实现，也不把原论文表格数字写成本地实验结果。
