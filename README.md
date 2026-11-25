# 多视图聚类课程设计

实现不完整多视图聚类和多视图流式聚类，包含传统基线、HSACC与DSVC选定配置的复现、课程统一对照及消融分析。

## 报告与结果

- [实验报告 Word](reports/多视图聚类课程设计_实验报告.docx)：25页，含方法、实验、误差与效率分析。
- [文献综述 Word](reports/多视图聚类课程设计_文献综述.docx)：3页，含论文来源及复现选择。
- [结果表](reports/tables/) · [分析图](reports/figures/) · [核验证据](reports/verification/) · [课程验收清单](requirements/course_checklist.json)。

课程实验、消融、单视图噪声、独立进程效率与报告已完成；252条课程预测已重算ACC/NMI/ARI。

## 实验入口

各实验分别保存 `README.md`、`config.json`、`run.py` 和正式结果，详细参数与命令见对应目录。

| 目录 | 内容 | 已保存结果 |
| --- | --- | --- |
| [01_complete_baselines](experiments/01_complete_baselines/) | 四个数据集的完整拼接与单视图K-means | 63条记录 |
| [02_incomplete_multiview](experiments/02_incomplete_multiview/) | 三种缺失模式下的零/均值填充 | 54条记录及9条完整参考 |
| [03_streaming_baselines](experiments/03_streaming_baselines/) | 十批样本流及漂移基线 | 270条批次记录 |
| [04_hsacc_reproduction](experiments/04_hsacc_reproduction/) | HSACC作者协议，Caltech HOG/GIST | 500轮、3种子 |
| [05_dsvc_reproduction](experiments/05_dsvc_reproduction/) | DSVC作者协议，HandWritten六视图 | 200轮/视图、3种子 |
| [06_course_comparison](experiments/06_course_comparison/) | 同一观测条件下的课程A/B主比较 | A 81条缺失记录及9条参考；B 144条批次记录 |
| [07_ablations_robustness](experiments/07_ablations_robustness/) | 去一致性、去补全、等权、加噪与隔离效率 | 各条件3种子 |

## 使用已有环境运行

以下为PowerShell示例。将解释器路径替换为已有环境；启动器将临时文件和缓存放入项目 `.cache/`，不自动安装PyTorch。项目放在D盘时，这些文件也位于D盘。

```powershell
$env:MVC_PYTHON = 'D:\Anconda\envs\Micro\python.exe'
python scripts/run_in_environment.py --show-runtime
python scripts/run_in_environment.py scripts/download_datasets.py
python scripts/run_in_environment.py scripts/download_references.py
python scripts/run_in_environment.py experiments/06_course_comparison/run.py --stage all
```

已验证的课程环境为Python 3.10.20、PyTorch 2.11.0+cu128、RTX 5070。DSVC需要CUDA GPU。CPU基线依赖见 [requirements.txt](requirements.txt)，GPU运行记录中的附加依赖见 [requirements-deep.txt](requirements-deep.txt)；使用现有环境时只补缺少的包，避免覆盖可用依赖。

无需重新训练即可核对仓库快照：

```powershell
python scripts/run_in_environment.py -- -m evaluation.verify_course_outputs --public-only
python scripts/run_in_environment.py scripts/validate_repository.py --include-paper-results --include-course-results
python scripts/run_in_environment.py -- -m unittest discover -s tests -v
```

演示与Word重建见 [报告目录说明](reports/README.md)。

## 数据与实验边界

公开数据为BBCSport、BBC4view、Caltech101-20、NUS-WIDE子集；HandWritten来自固定版本DSVC作者仓库。样本数、视图维度、下载地址和校验值见 [数据说明](data/README.txt) 与 [来源清单](data/dataset_sources.json)。原始数据、论文PDF、作者源码包和训练权重不进入仓库，通过脚本获取或重新运行生成。

课程A的不完整率为“缺少一个视图的样本比例”；双视图的缺失块比例约为其一半。课程B主比较为六视图流，与实验03的十批样本流分开分析。作者协议与课程适配分别报告；两篇论文只覆盖选定数据集/配置，未复现原文全部表格。ACC经匈牙利匹配，统计使用3种子的均值与样本标准差，不挑选最好一次。

## 仓库结构与结果保留

```text
data/             数据来源、读取、预处理、缺失与漂移生成
models/           聚类基线、历史摘要与课程模型
train_or_update/  训练和流式更新
evaluation/       指标、误差、核验与分析
visualization/    论文及基线绘图
experiments/      七个独立实验的入口、配置、正式CSV、日志和环境
reports/          两份Word文档；tables/、figures/、verification/、sources/
scripts/          下载、环境启动、演示、Word导出与仓库检查
tests/            12项协议测试
configs/          默认基线配置
requirements/     教师任务验收清单
references/       固定论文与作者源码下载清单
```

每个方法保留逐次CSV、汇总、运行环境及必要日志。重复的种子配置、源码校验与训练轨迹合并到 `reproduction_details.json`；将各条 `configuration_overrides` 递归覆盖到 `common_configuration`，可还原每次配置。短流程产物、每个病例的重复文件和过程进度不再发布；重新训练仍会在本地生成它们，供断点续跑和诊断。

第三方来源和固定提交见 [THIRD_PARTY.md](THIRD_PARTY.md)。
