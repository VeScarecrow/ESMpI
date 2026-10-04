# ESMpI

基于蛋白质语言模型的等电点（pI）预测。将 IPC2 九参数物理基线与 ESM-2 150M 残差 SVR 修正结合，
仅使用氨基酸序列——不需要三维结构。

**核心结果**（581 条独立测试集）：

| 方法 | RMSE | MAE | R² |
|------|------|-----|-----|
| ESMpI（本文） | **0.8104** | 0.5573 | 0.6478 |
| IPC2 SVR | 0.8552 | 0.5907 | 0.6077 |

在 61 个高置信实验结构蛋白子集上，ESMpI（RMSE 0.6064）与最佳结构方法 DeepKa（0.6058）持平。

---

## 模型架构

| 组件 | 详情 |
|------|------|
| **编码器** | ESM-2 150M（`esm2_t30_150M_UR50D`），冻结权重，640 维逐残基嵌入 |
| **池化** | 7 类残基加权平均（D/E/H/K/C/Y/R），权重 = `[0.0156, 0.0156, 0.0062, 0.5, 0.05, 0.0062, 1.0]` |
| **回归器** | RBF-SVR（C=0.5, epsilon=0.05, gamma=scale），带 StandardScaler |
| **训练目标** | 残差 r = 实验_pI − 物理基线_pI |
| **样本权重** | w = 1 + max(abs(y) − 9, 0)，强调极端 pI 蛋白 |
| **交叉验证** | 5 折，5 个随机种子；OOF 预测用于模型选择 |
| **物理基线** | IPC2 发表 9-pKa 值，H-H 二分求根（容差 1e-8） |

### GBMS pKa 优化

本工作的一个核心贡献是在 1743 条训练蛋白上重新优化了 9-pKa 向量。
该协议（称为 **GBMS**）使用与 IPC2 相同的目标函数（pI 残差平方和），
但将原始优化器替换为现代局部求解器：

| 方面 | IPC2 (2021) | GBMS（本工作） |
|------|-------------|----------------|
| 优化器 | Differential Evolution | 多起点 Trust Region Reflective (TRF) |
| 起点数 | 种群进化（默认 10×9=90 个体/代） | 13 起点（Thurlkill + IPC1 + IPC2 + 10 LHS） |
| 边界 | [0, 14] | [0, 14] |
| 训练集 RMSE | 0.8255 | **0.8253** |
| 测试集 RMSE | 0.8660 | **0.8674** |
| 实测速度（1743 蛋白） | 50 代 × 90 个体 ≈ **95 秒** | 13 起点 ≈ **13 秒**；单起点 ≈ **0.4 秒** |

*测试环境：Intel Core i5-9400F（6 核，2.90 GHz），16 GB 内存，Windows 10，Python 3.13。*

GBMS pKa 值与 IPC2 几乎一致（最大差异 0.274，酪氨酸），证实 9 参数
pI 目标函数在最优点附近 landscape 平坦。实际优势在于**可复现性和速度**：
TRF 从 13 个起点中的任意一个确定性收敛，单起点约 0.4 秒，13 起点约 13 秒，
使得大规模 bootstrap 和置换分析（4 核 70 分钟完成 1000 次迭代）成为可能。

完整优化实现见 `scripts/07_permutation_pka.py`。

### 逐蛋白预测结果

581 条测试集的逐蛋白预测（实验 pI、IPC2 基线、GBMS 基线、IPC2.svr.19、ESMpI）
已整理为可读 CSV：
[data/reference_results/per_protein_predictions.csv](data/reference_results/per_protein_predictions.csv)

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- Git（可选，用于克隆仓库）
- Conda（推荐）或任意 Python 3.10+ 环境
- [Git LFS](https://git-lfs.com)——嵌入文件（.npz）与结构压缩包（.zip）通过 Git LFS 存储。新版 Git for Windows 及官方 macOS/Linux 安装包已自带；可用 `git lfs version` 检查（Ubuntu/Debian：`sudo apt install git-lfs`）。未安装 LFS 时 clone 得到的是指针文件而非真实数据。

### 安装（Conda，推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/VeScarecrow/ESMpI.git
cd ESMpI

# 2. 创建 conda 环境
conda create -n esmpi python=3.11 -y
conda activate esmpi

# 3. 安装核心依赖
pip install -r requirements.txt

# 4.（可选）完整 ESMpI 预测需要 ESM-2 嵌入
pip install torch transformers
```

### 安装（venv，备选）

```bash
# 1. 克隆并进入
git clone https://github.com/VeScarecrow/ESMpI.git
cd ESMpI

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. 安装核心依赖
pip install -r requirements.txt

# 4.（可选）完整 ESMpI 预测需要 ESM-2 嵌入
pip install torch transformers
```

核心 `requirements.txt` 涵盖表格复现和图表生成
（numpy、pandas、scipy、scikit-learn、matplotlib）。`torch` 和
`transformers` 仅在用完整 ESM-2 模型预测新序列时需要。

### 预测新序列的 pI

最常用的场景——预测自己的蛋白质序列：

```bash
# 仅物理基线（无需 GPU，秒出结果）
python predict.py -s "MKKFFDSRREQQKFLDAVAEHGRPDQVNPTQFIKVDSSAYNGLTEFLVFDRYLDGFNLDFEGTRTTAHQKLIEEAIDAFIKHGNTNLTIADALKDKGYRVEGYLKGYVDGNLSTTAQFNQAFKEKVNRLPDGQVVDHLAQGQPVVTAEQYAANEKRQAFDQVTGLPGYTHPQTAAPRTL" --baseline-only

# 完整 ESMpI 预测（需要 torch + transformers，自动下载 ESM-2 150M）
python predict.py -s "MKKFF..." --device cpu

# 从 FASTA 文件批量预测
python predict.py -i my_proteins.fasta -o predictions.csv

# 使用本地 ESM-2 模型（避免网络下载）
set ESM2_MODEL_PATH=C:\models\esm2_t30_150M_UR50D
python predict.py -i my_proteins.fasta
```

输出列：`id`、`sequence`、`length`、`pI_IPC2_baseline`、`pI_GBMS_baseline`、`pI_ESMpI`（仅完整模式）。

---

## 项目结构

```
ESMpI/
├── esmpi/                    # 可导入的 Python 包（共享引擎 + 工具）
│   ├── pka_engine.py         #   Henderson-Hasselbalch pI 二分求根 + 解析雅可比
│   ├── dataio.py             #   数据加载、路径管理（相对路径）
│   ├── pooling.py            #   ESM-2 七类残基加权池化（W_ELEGANT）
│   ├── models.py             #   SVR 工厂、5 折交叉验证、全量训练
│   ├── structure.py          #   基于结构的 pI 计算（逐位点 pKa，FULL/DEHK 协议）
│   └── metrics.py            #   RMSE、MAE、R²、离群率、分段 RMSE
│
├── predict.py                # 预测用户序列的 pI（命令行工具）
│
├── benchmark/                # 自包含的一条命令模型指标复现
│   ├── run_benchmark.py      #   全量重训：5 折 OOF + 测试集只评一次，含冻结锚点校验
│   ├── inputs/               #   复制的输入（嵌入走 Git LFS、特征表、IPC 原始划分）
│   ├── expected_outputs/     #   冻结预测与论文发表指标
│   └── outputs/              #   重新生成的指标/预测（gitignore 忽略）
│
├── IPC_protein/              # 原始 IPC2 训练/测试数据集（发表原貌）
│   ├── IPC_protein_75.csv    #   训练集：1743 条蛋白（exp_pI, sequence）
│   └── IPC_protein_25.csv    #   测试集：581 条蛋白（exp_pI, sequence）
│
├── scripts/                  # 流水线脚本（按 01 → 10 顺序运行）
│   ├── 01_generate_predictions.py   # ESMpI 预测 → data/predictions/z166_preds.npz
│   ├── 02_reproduce_benchmark.py    # 性能基准表格（581 全集 + 61 蛋白 PDB 子集）
│   ├── 03_bootstrap_pka.py          # B=1000 pKa bootstrap 样本
│   ├── 04_bootstrap_intermethod.py  # B=2000 配对 bootstrap
│   ├── 05_make_figures.py            # 散点图、效应图、bootstrap 图、SI 图
│   ├── 06_stratified_analysis.py     # 按来源/长度/pI 分层 RMSE
│   ├── 07_permutation_pka.py         # 1000 次置换重优化 9-pKa
│   ├── 08_ipc2_protocol.py           # IPC2 评估口径复核
│   ├── 09_prepare_structures.py      # 下载/修复实验 PDB 结构
│   ├── 10_structure_pI.py            # 从逐位点 pKa 计算结构 pI
│   └── 11_benchmark_inference_e2e.py  # 端到端推理时间基准测试（GPU + CPU）
│
├── data/                     # 全部输入数据（已提交，共约 28 MB）
│   ├── embeddings/           #   ESM-2 150M 七类嵌入（22 MB）
│   ├── features/             #   物理特征（z81，4 MB）
│   ├── predictions/          #   预计算的 ESMpI 预测（61 KB，第一层锚点）
│   ├── mapping/              #   seq_no → npz 行号映射 + 数据源标签
│   ├── benchmark/            #   基准对比输入 CSV（8 个文件）
│   ├── full_sources/         #   两个全量数据源 FASTA（SI Fig S3）
│   ├── structures/           #   PDB 结构压缩包（见数据清单）
│   └── reference_results/    #   冻结的预期输出（供验证）
│
├── figures/                   # 静态图源
│   ├── figure1_framework.png  #   框架图（手绘，非代码生成）
│   └── figure1_framework.pptx
│
├── results/                   # 运行时输出（gitignore 忽略）
│   ├── tables/                #   基准指标 CSV
│   ├── figures/               #   生成的图片（PNG/PDF）
│   └── bootstrap/            #   bootstrap 结果
│
├── requirements.txt
├── README_EN.md
├── README_CN.md
└── .gitignore
```

---

## 数据清单

> 每个数据文件的列名与含义详见 [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md)。

| 文件 | 位置 | 大小 | 说明 |
|------|------|------|------|
| `IPC_protein_75.csv` | `IPC_protein/` | 667 KB | 原始 IPC2 训练集，1743 条蛋白（`exp_pI`、`sequence`）；4 条序列含 U/Z，训练前已清洗（见数据字典）|
| `IPC_protein_25.csv` | `IPC_protein/` | 225 KB | 原始 IPC2 测试集，581 条蛋白（`exp_pI`、`sequence`）；3 条序列含 U/B，评估前已清洗（见数据字典）|
| `z82_150m_pertype_train.npz` | `data/embeddings/` | 16.3 MB | ESM-2 150M 七类嵌入（训练集，1743×7×640）|
| `z82_150m_pertype_test.npz` | `data/embeddings/` | 5.5 MB | ESM-2 150M 七类嵌入（测试集，581×7×640）|
| `z81_features_train.csv` | `data/features/` | 3.1 MB | 物理特征 + 19 套 pKa 标度 pI（训练集）|
| `z81_features_test.csv` | `data/features/` | 1.0 MB | 物理特征 + 19 套 pKa 标度 pI（测试集）|
| `z166_preds.npz` | `data/predictions/` | 61 KB | 预计算的 ESMpI + IPC2-SVR 测试预测（第一层锚点）|
| `z164_seqno2zidx.csv` | `data/mapping/` | 5 KB | seq_no → npz 行号映射 |
| `z46d_source_labels.csv` | `data/mapping/` | 81 KB | 全量集数据源标签（PIP-DB=1380 / SWISS-2DPAGE=944，已修正版）|
| `pip_db_normal.fasta` | `data/full_sources/` | 1.0 MB | PIP-DB 全量 2427 条（FASTA 头含实验 pI）|
| `ch2d19_2_1st_isoform.fasta` | `data/full_sources/` | 432 KB | SWISS-2DPAGE 全量 1054 条首个同工酶 |
| `IPC_protein_25.csv` | `data/benchmark/` | 225 KB | 581 条测试序列 + 实验 pI |
| `reg_581.csv` | `data/benchmark/` | 155 KB | 主注册表：结构来源、PDB 路径、置信等级 |
| `pi_fallback.csv` | `data/benchmark/` | 35 KB | 7 种方法的逐蛋白 pI（实验结构输入）|
| `pi_af.csv` | `data/benchmark/` | 132 KB | AF2 结构方法的逐蛋白 pI |
| `pkalm_piprott.csv` | `data/benchmark/` | 14 KB | pKALM 官方 PIPROT 预测 |
| `pypka_exp_z44.csv` | `data/benchmark/` | 18 KB | PypKa 实验结构预测 |
| `perm_pka_samples.csv` / `perm_pka_summary.csv` | `data/reference_results/` | 184 KB | 冻结的 1000 次置换 pKa 样本与汇总 |
| `bench_inference_time_e2e.csv` | `data/reference_results/` | 52 KB | 冻结的逐蛋白端到端推理时间（IPC2、IPC2.svr.19、ESMpI GPU/CPU）|
| `per_protein_predictions.csv` | `data/reference_results/` | 254 KB | 全部 581 条测试蛋白的逐蛋白预测（可读 CSV，含 UniProt 编号）|
| `int8_deployment_description.txt` | `data/reference_results/` | 0.6 KB | INT8 部署推理描述与基准测试环境 |
| `table_ipc2_protocol.csv` | `data/reference_results/` | 2 KB | 冻结的 IPC2 协议口径表 |
| `alphafold2_structures.zip` | `data/structures/` | 28.9 MB | 581 条 AF2 预测结构（按 seq_no 命名）|
| `experimental_pdb_A_high_confidence.zip` | `data/structures/` | 2.4 MB | 61 条高置信实验 PDB（A61 子集，Table 2）|
| `experimental_pdb_B_trusted.zip` | `data/structures/` | 3.5 MB | 73 条可信实验 PDB |
| `experimental_pdb_C_limited.zip` | `data/structures/` | 3.3 MB | 71 条有限置信度实验 PDB |
| `experimental_pdb_D_rejected.zip` | `data/structures/` | 1.2 MB | 16 条已拒绝的实验 PDB |
| `structure_patches.zip` | `data/structures/` | 3.4 MB | 33 条补丁结构（23 条 ESM-2/3 尾部补丁 + 10 条 tail-10 补丁）|
| `patch_seq111.zip` | `data/structures/` | 0.1 MB | seq_no 111 特殊处理结构 |

**实验结构置信度分级**（逐蛋白映射见 `reg_581.csv`）：

| 等级 | 数量 | 标准 | 用途 |
|------|------|------|------|
| A — 高置信 | 61 | X-ray 分辨率 < 3.0 Å，完整链，无缺口 | Table 2（A61 子集）|
| B — 可信 | 73 | X-ray/NMR，轻微缺口或较低分辨率 | 补充分析 |
| C — 有限 | 71 | 显著缺口、低分辨率或部分链 | 不用于基准测试 |
| D — 已拒绝 | 16 | 严重质量问题 | 排除分析 |
| P — 仅预测 | 360 | 无实验 PDB | 使用 AF2 结构替代 |

---

## 方法对照表（Table 1 & 2）

| 论文名称 | 类型 | CSV 列 / 来源 |
|----------|------|---------------|
| ESMpI | 序列 | `z166_preds.npz` → `pt_ours`（IPC2 基线 + ESM-2 150M SVR 残差）|
| IPC2.protein.svr.19 | 序列 | F19 特征 → SVR(C=1.0, epsilon=0.12) |
| IPC2_protein | 序列 | IPC2 发表 9-pKa H-H pI |
| pKALM | 序列 | `pkalm_piprott.csv`（官方服务器）|
| PropKa_AF2 | 结构（AF2）| PropKa 逐残基 pKa → H-H pI |
| PypKa_AF2 | 结构（AF2）| PypKa Poisson-Boltzmann → H-H pI |
| DeepKa_AF2 | 结构（AF2）| DeepKa ML pKa → H-H pI |
| PropKa_PDB | 结构（实验 PDB）| PropKa 对实验 PDB |
| PypKa_PDB | 结构（实验 PDB）| PypKa 对实验 PDB |
| DeepKa_PDB | 结构（实验 PDB）| DeepKa 对实验 PDB |

Table 2（A61 子集）中结构方法使用**实验测得 PDB**，Table 1（全集 581）使用 AlphaFold2 预测结构。
缺失的可滴定位点统一以 Thurlkill (2006) 模型化合物 pKa 回填。

---

## 复现

### 最快验证——自包含模型基准（约 20 秒，无需 GPU）

仅使用 `benchmark/` 下复制的输入文件，从零重建论文发表的 ESMpI 模型
（1743 条训练蛋白五 seed 五折 OOF + 一次全量重训 + 581 条测试蛋白仅评估一次），
并将重算预测与冻结锚点逐位比对：

```bash
python benchmark/run_benchmark.py
```

预期输出：训练 OOF RMSE 0.8002，测试 RMSE **0.8104**（MAE 0.5573，R² 0.6478），
末行打印 `ALL CHECKS PASSED`。详见 [benchmark/README.md](benchmark/README.md)。

### 第一层——复现表格（1 分钟，无需 GPU）

```bash
python scripts/02_reproduce_benchmark.py
```

产出 Table 1（581 全集）和 Table 2（61 蛋白 PDB 子集），输出到 `results/tables/`。
可用 `data/reference_results/` 中的冻结结果做一致性校验。

### 第二层——完整流水线（重新生成预测、bootstrap、图）

> 步骤 1 在 i5-9400F（6 核）上实测约 12 分钟（19 套 pKa 标度 × 5 折 CV）；
> 步骤 3/4/7 涉及上千次 bootstrap/置换，单核耗时数小时，可适当调小 `B` 先验证流程。

```bash
# 步骤 1：从 ESM-2 150M 嵌入重新生成 ESMpI 预测（≈12 min）
python scripts/01_generate_predictions.py

# 步骤 2：复现基准表格（第一层也可独立运行）
python scripts/02_reproduce_benchmark.py

# 步骤 3：pKa 参数 bootstrap（B=1000，Figure 4 数据）
python scripts/03_bootstrap_pka.py

# 步骤 4：方法间配对 bootstrap（B=2000，ESMpI vs IPC2-SVR）
python scripts/04_bootstrap_intermethod.py

# 步骤 5：生成论文图片（正文 Figure 2/3/4 + SI Figure S1–S4）
python scripts/05_make_figures.py

# 步骤 6（可选）：分层 RMSE 分析
python scripts/06_stratified_analysis.py

# 步骤 7（可选，约 70 分钟）：置换稳健性实验（1000 次）
#   可用环境变量快速冒烟测试：NITER=3 NPROC=1 python scripts/07_permutation_pka.py
python scripts/07_permutation_pka.py

# 步骤 8（可选）：IPC2 评估口径复核
python scripts/08_ipc2_protocol.py
```

> 1000 次置换样本已在 `data/reference_results/` 提供冻结副本，
> 不运行步骤 7 也可出图（05 自动回退）；bootstrap 图需先运行步骤 3 生成样本。

### 论文图表与补充分析

| 输出 | 生成脚本 | 内容 |
|------|----------|------|
| `paper_fig2_scatter` | 05 | ESMpI vs 实验 pI 散点：黑色全线与橙色离群点拟合线（误差>0.5）|
| `paper_fig3_effects` | 05 | 四面板对比——GBMS vs ESMpI 密度散点、酸/碱 pI、数据来源、蛋白长度 |
| `paper_fig4_bootstrap` | 05 | 9 参数 pKa 的 B=1000 bootstrap 分布 |
| `si_figS1_pI_distribution` | 05 | 训练/测试集实验 pI 分布 |
| `si_figS2_pairwise` | 05 | 实验 / GBMS / IPC2.svr.19 / ESMpI 两两相关（n=581）|
| `si_figS3_source_distribution` | 05 | PIP-DB（2427）与 SWISS-2DPAGE（1054）全量数据源 pI 分布 |
| `si_figS4_perm_pka` | 05 | 1000 次置换实验的 9-pKa 分布 |
| `table_ipc2_protocol.csv` | 08 | 16 种方法的 pooled / 10 折 CV / IPC2 论文印刷值对比 |
| `perm_pka_*.csv` | 07 | 置换实验逐次 pKa 与均值/中位数/95% CI |

**IPC2 评估口径（08）**：IPC2 论文 Table 2 报告的是随机 10 折
逐折指标的平均（并非全集 pooled RMSE），二者存在约 0.005 pH 的系统性差异
（Jensen 不等式）。08 脚本以 20 个种子 × 10 折 = 200 折消除切分随机性，
IPC_protein 的 200 折 RMSE = 0.8675，与论文印刷值 0.8677 仅差 0.0002。

**置换稳健性实验（07）**：每轮从 1743 条训练集随机
移除 581 条、以全部 581 条测试蛋白填充，再按 GBMS 协议重优化 9-pKa，
共 1000 轮（种子 20260927，4 进程约 70 分钟，Intel Core i5-9400F）。本仓库移植版与冻结参考结果
逐位一致；支持 `NITER`/`NPROC` 环境变量做快速冒烟测试。

---

## 许可证

本项目采用 **MIT 许可证**——详见 [LICENSE](LICENSE) 文件。

第三方组件：
- [ESM-2](https://github.com/facebookresearch/esm)（Meta AI）—— MIT 许可证
- PIP-DB 和 SWISS-2DPAGE 数据源——请参阅原始文献的使用条款

## 引用

如果使用本代码，请引用：

```bibtex
@article{luo2026esmpi,
  title={Protein language model-boosted prediction of isoelectric point},
  author={Luo, Fangfang and Lu, Xiangxiang and Cai, Zhitao and Wu, Riting and Su, Shubin and Huang, Yandong},
  journal={Journal of Molecular Biology},
  year={2026}
}
```
