# 实验04 HSACC论文复现

**状态：作者源码已按固定提交获取，适配入口已编写；完整训练与GPU运行验证尚未完成，没有本地论文复现指标。**

## 论文与源码

- 论文：Ding等，Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion，NeurIPS 2025。
- [正式论文页面](https://papers.nips.cc/paper_files/paper/2025/hash/4b2a917e30e1bb1aff055b4d8c6c081c-Abstract-Conference.html)。
- [作者代码](https://github.com/XiaojianDing/2025-NeurIPS-HSACC)，固定提交 `eb53c217a1e1ac7f8550978981e762f33cce4c97`。

## 文件与命令

| 文件 | 用途 |
| --- | --- |
| `run.py` | 本实验入口，读取配置或启动短流程检查 |
| `config.json` | 原协议首次验证设置：Caltech101-20、参数0.5、500轮、3个数据种子 |
| `../../train_or_update/run_hsacc.py` | 作者模型调用、兼容处理、结果与表示导出 |
| `../../models/author_import.py` | 从忽略的第三方源码目录导入模型 |
| `../../evaluation/neighborhood.py` | 仅用于离线真实/补全/共识邻域分析 |

先按根README安装基线和深度学习依赖、下载数据及作者资源，再运行：

```powershell
python experiments/04_hsacc_reproduction/run.py --smoke
python experiments/04_hsacc_reproduction/run.py
```

短流程为5轮、1个种子，保存在 `smoke_results/`；完整配置为500轮、3个种子，保存在 `results/`。完整结果出现前，不报告论文方法已复现。

## 协议与适配说明

采用原始6视图Caltech文件中的HOG/GIST（索引3、4），与作者双视图配置一致。`author`模式保留作者的逐视图全局MinMax、原模型结构、损失、Adam、训练循环及缺失生成器。`np.int` 的兼容替换只作用于内存副本，不改动作者原文件。

作者 `missing_rate=0.5` 是生成器参数，不是实际缺失块比例；入口会保存实际比例与掩码。作者实现用真实ACC记录周期评估中的最好结果，入口将这些数值保存为 `upstream_best_*`，同时单独导出固定最后一轮表示的ACC/NMI/ARI，不将二者混合。

固定最后一轮的独立K-means重新设置了种子，与作者内部周期K-means的随机状态可能不同。需要分别核对，不能期待两组指标完全相同。

`course`模式另提供只用已观测数据估计标准化和统一掩码的接口；这是课程扩展，不称为原协议复现。去一致性、去补全、等权重变体目前也只是待验证接口。

预期输出包括 `runs.csv`、`summary.csv`、每次运行的配置、源码校验值、`outputs.npz`、最终参数、周期评估轨迹和邻域分析JSON。完整缺失真值只在训练完成后用于离线诊断。
