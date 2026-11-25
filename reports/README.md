# 报告与结果

## 正式文档

- [实验报告 Word](多视图聚类课程设计_实验报告.docx)：25页，含14张表和全部10组分析图。
- [文献综述 Word](多视图聚类课程设计_文献综述.docx)：3页，含教师论文核验、方法脉络及来源链接。

两个文件均可编辑，图表和公式已嵌入，已逐页检查。PPT仍待后续制作。

## 文件分类

| 目录 | 内容 |
| --- | --- |
| [tables/](tables/) | A/B逐次指标、3种子汇总、配对收益、消融、噪声、补全误差和效率CSV |
| [figures/](figures/) | 报告中的10组分析图 |
| [verification/](verification/) | 252条预测的压缩标签、重算核验、配对输入与预算检查、Word校验及分页记录 |
| [sources/](sources/) | 可重建Word的文档内容源 |

原论文数值未混入本地结果；原作者运行与课程适配分开统计。复用控制和效率重跑不增加独立随机种子数。

## 核验与演示

只核对已发布预测，无需下载数据或训练模型：

```powershell
python scripts/run_in_environment.py -- -m evaluation.verify_course_outputs --public-only
```

回放保存结果，或现场运行一个种子的K-means基线：

```powershell
python scripts/run_in_environment.py scripts/demo_course.py --mode replay --scenario a
python scripts/run_in_environment.py scripts/demo_course.py --mode replay --scenario b
python scripts/run_in_environment.py scripts/demo_course.py --mode live --scenario a
python scripts/run_in_environment.py scripts/demo_course.py --mode live --scenario b
```

`replay`不重新训练神经网络；`live`需要数据，现场输出不覆盖正式结果。

## 重建分析与Word

从已保存的正式CSV生成分析图表及报告内容：

```powershell
python scripts/run_in_environment.py -- -m evaluation.build_course_analysis
python scripts/run_in_environment.py -- -m evaluation.build_report
```

Word导出在单独的文档环境运行，依赖见 [requirements-documents.txt](../requirements-documents.txt)。Codex中使用Documents工作区自带Python，不修改训练环境：

```powershell
python scripts/export_word_reports.py --sources-only
```

该命令直接使用已发布的文档内容源。若要采用刚重新计算的报告内容，去掉 `--sources-only`。重新生成后需检查分页；新文件不会沿用旧文件的排版通过标记。

完整训练会在本地另存标签、模型权重和逐次诊断；重新生成原始特征几何分析需先完成HSACC训练，再运行 `evaluation.course_geometry`。新一轮完整训练结束并生成本地完成标记后，可使用 `evaluation.finalize_course --document-python <文档解释器路径>` 依次核验并导出。
