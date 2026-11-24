# 实验05 DSVC论文复现

**状态：作者源码已按固定提交获取，适配入口已编写；尚未完成训练验证，没有本地论文复现指标。**

## 论文与源码

- 论文：Yuan等，Deep Streaming View Clustering，ICML 2025，PMLR 267:73526–73540。
- [正式论文页面](https://proceedings.mlr.press/v267/yuan25d.html)。
- [作者代码](https://github.com/hl-yuan/DSVC)，固定提交 `0fb55660b2418b20d9ac7cd6ed0118522a324878`。

## 文件与命令

| 文件 | 用途 |
| --- | --- |
| `run.py` | 读取本实验配置并启动运行 |
| `config.json` | HandWritten六视图、200轮/视图、3个种子 |
| `../../train_or_update/run_dsvc.py` | 调用作者网络和逐视图训练函数，保存每个视图的结果 |
| `../../scripts/download_references.py` | 下载固定提交作者代码及其随附数据 |

本实验需要可用的CUDA GPU。先完成根README中的依赖与作者资源下载，再运行：

```powershell
python experiments/05_dsvc_reproduction/run.py --smoke
python experiments/05_dsvc_reproduction/run.py
```

短流程为2轮/视图、1个种子，保存到 `smoke_results/`；原预算为200轮/视图，使用作者种子列表中的前3个11、21、22。

## 协议与边界

这是同一批对象的6个视图依次到达，不是10批新样本到达。使用作者仓库随附的HandWritten数据，按作者数据加载顺序开放视图。保留作者网络、损失、原型缓冲和历史样本嵌入拼接方式。

作者 `Streaming_train.py` 默认只在第1轮和每200轮导出特征；短流程检查在内存副本中增加“最终一轮也导出”的条件，完整200轮配置不改变导出时点。作者原文件不被修改。

当前入口保留已安装scikit-learn版本的K-means默认 `n_init` 并记录其值；这与作者旧依赖的默认值可能不同，是正式复现前必须核对的兼容差异。

预期输出 `runs.csv`、`summary.csv`、环境与配置、每个视图的标签和新视图表示、时间、GPU峰值分配、历史嵌入与原型缓冲大小。作者训练包含周期评估，耗时单独标注。

该作者原协议不能直接替代实验03的课程样本流对照；要完成场景B，还需在一致数据与协议下比较在线改进方法、加入显式漂移条件，并审查历史访问、固定时间预算和遗忘分析。
