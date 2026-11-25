多视图聚类课程设计 数据集说明
整理日期：2026-11-24

一、已下载的主数据集

原始文件位于 data/raw/，均为已提取特征的 MATLAB .mat 文件，含多视图特征和评估标签。
保留下载原文件，不做标准化、补全、降维、样本重排或标签重编码。

文件                  样本数  视图数  类别数  各视图特征维度（按文件中的顺序）
BBCSport.mat          544     2       5       3183, 3203
BBC4view.mat          685     4       5       4659, 4633, 4665, 4684
Caltech101-20.mat     2386    6       20      48, 40, 254, 1984, 512, 928
NUS-WIDE.mat          2400    5       12      64, 144, 73, 128, 225

NUS-WIDE 是2400样本的小规模特征子集，不是原始全量图像数据集。
维度与类别数来自本地文件实际读取，尤其是 NUS-WIDE 第三个视图为73维。
标签仅用于离线评估；不能直接作为输入视图，也不能用于拟合补全模型或调优视图权重。
文件中的原始标签值未改为从0开始，读取函数会将标签展平为一维。

二、来源

BBCSport / BBC4view / NUS-WIDE 的公开特征整理仓库：
https://github.com/ChuanbinZhang/Multi-view-datasets

Caltech101-20 的公开特征文件来自 COMPLETER 论文代码仓库：
https://github.com/XLearning-SCU/2021-CVPR-Completer/tree/main/data

逐文件下载地址、来源页、文件字节数、Git blob SHA-1、MATLAB变量名见 dataset_sources.json。
下载后的 SHA-256 与完整性校验结果见 download_report.json。
各类别样本数、每视图类型、零向量行数等检查结果见 dataset_summary.json。
使用数据时请按原始数据集及实际使用的特征版本注明来源。

三、读取

在项目根目录运行 Python，依赖 NumPy 和 SciPy：

    from data.load_dataset import load_dataset
    views, labels = load_dataset('BBCSport')
    print([view.shape for view in views], labels.shape)

其他名称：BBC4view、Caltech101-20、NUS-WIDE，也可使用带 .mat 后缀的文件名。
每个 view 均为 样本数 × 特征维度。

BBC4view 来源文件在 X、y 两个有效变量后附加了983字节。
不限定变量的 scipy.io.loadmat 读取会报错；提供的函数通过
loadmat(path, variable_names=['X', 'y']) 正常读取完整685个样本。
原始文件未改写，且其 Git blob 校验值与下载来源一致。
Caltech101-20 标签变量名为大写 Y，其他三个数据集为小写 y；读取函数已处理。

四、重新下载与验证

    python scripts/download_datasets.py
    python scripts/download_datasets.py --verify-only
    python scripts/inspect_datasets.py

下载脚本只使用 Python 标准库；检查及读取还需 NumPy、SciPy。
下载脚本会校验文件大小与 Git blob SHA-1，已有且通过校验的文件会直接复用。
已完成：4个主文件完整性校验；所有视图样本数与标签长度一致；特征和标签均无NaN/Inf。
该检查只验证结构与数值，不额外证明原始数据的语义标注或跨视图对应关系。

