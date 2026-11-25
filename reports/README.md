# 课程报告与核验材料

正式报告为 `多视图聚类课程设计_实验报告.docx`，文献综述为 `多视图聚类课程设计_文献综述.docx`。两个文件均为可编辑的Word文档，图片、表格、公式和引用已嵌入，可单独下载。正式交付不使用HTML，PPT仍按用户安排最后制作。

当前主实验、消融、噪声、独立效率和报告均已完成并核验；252条课程预测已重算ACC/NMI/ARI。PPT仍待后续制作。

| 文件 | 用途 |
| --- | --- |
| `多视图聚类课程设计_文献综述.docx` | 教师8条论文的逐项核验、方法脉络、复现选择与可点击来源链接 |
| `多视图聚类课程设计_实验报告.docx` | 问题、算法、统一实验、消融、误差、漂移、效率、复现边界 |
| `sources/*.json` | 文档内容源，供导出工具重建Word，不是提交给教师的文档格式 |
| `word_export_manifest.json` | Word内容来源、文件校验值、表格和图表数量、最终分页检查记录 |
| `scenario_a_*.csv` | 场景A逐次指标、3种子汇总与相对均值填充的配对收益 |
| `geometry_*.csv` | 真实原始视图、均值填充、解码补全和共识的10-NN诊断 |
| `scenario_b_*.csv` | 场景B质量、历史查询、累计时间、噪声配对影响与恢复 |
| `ablation_*.csv` | 三项消融与复用完整方法的统计，不重复计算控制种子 |
| `robustness_*.csv` | 观测单视图噪声及配对无噪声控制 |
| `isolated_efficiency*.csv` | 新进程中的代表性A条件成本，不加入主质量样本数 |
| `analysis_verification.json` | 主实验条数、配对输入与预算核对 |
| `prediction_verification.json` | 252条课程预测的指标重算与本地来源文件校验值 |
| `course_predictions.jsonl.gz` | 压缩的公开标签证据，仅含条件、真实与预测标签，不含特征或权重 |
| `report_manifest.json` | 报告生成状态和来源核验摘要 |

未出现的文件表示该阶段尚未生成，不应把此文件清单当作完成证明。阶段同步可能先有A/B分析，待消融、噪声和效率完成后才生成最终主报告。

## 按顺序生成

```powershell
python scripts/run_in_environment.py -- -m evaluation.course_geometry
python scripts/run_in_environment.py -- -m evaluation.build_course_analysis
python scripts/run_in_environment.py -- -m evaluation.verify_course_outputs
python scripts/run_in_environment.py -- -m evaluation.build_report
# 使用包含python-docx、lxml、NumPy、Pillow的文档环境运行
python scripts/export_word_reports.py
```

Codex中使用Documents工作区自带Python生成Word，不需要改变原Conda训练环境。命令行用户可在单独文档环境安装 `requirements-documents.txt`，不包含PyTorch。`evaluation.finalize_course` 支持 `--document-python` 指定该解释器。生成后须用Word/WPS或文档渲染器检查全部分页，不能仅凭文件存在认定排版通过。

只下载仓库即可核对公开预测，执行：

```powershell
python scripts/run_in_environment.py -- -m evaluation.verify_course_outputs --public-only
```

这不需要原始数据、作者模型或训练权重；需要已有NumPy/scikit-learn依赖。完整训练仍需要下载数据和固定作者源码。所有数字均来自本地运行，原论文表格没有混入本地结果。

## 演示入口

```powershell
python scripts/run_in_environment.py scripts/demo_course.py --mode replay --scenario a
python scripts/run_in_environment.py scripts/demo_course.py --mode replay --scenario b
python scripts/run_in_environment.py scripts/demo_course.py --mode live --scenario a
python scripts/run_in_environment.py scripts/demo_course.py --mode live --scenario b
```

`replay`明确回放保存的3种子统计；`live`现场执行一个种子的指定K-means基线，演示掩码或六视图更新，现场数字不覆盖正式结果。神经网络完整训练通过实验04—07入口运行。PPT按用户要求最后制作，目前不列为已交付。
