### EvalNetv4

轻量化的基于 MLP 的四层命中点反向轨迹重建与效率评估管线。项目入口为 `pipeline_v5.py`，负责：

- 读取已拆分好的事件候选集合（`workdir/Eval/BackPre`）
- 对每个候选组合做评分与最优命中点挑选（反向重建）
- 统计效率、幽灵率并绘图

模型权重会从 `workdir/Model` 目录中自动选择一个文件加载（按目录列出的顺序取最后一个）。

---

### 安装

- 建议环境：Python 3.10/3.11，CUDA 11.8（可 CPU 运行但速度较慢）

1) 克隆与进入目录

```bash
git clone <your-repo-url>
cd EvalNetv4
```

2) 创建虚拟环境（可选但推荐）

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
```

3) 安装 PyTorch（CUDA 11.8）

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

4) 安装其余依赖

```bash
pip install -r requirements.txt
```

提示：Matplotlib 使用 `tkAgg` 后端，如遇到绘图显示问题，请确保本机已安装 Tk（Windows 自带，一般无需额外操作）。

---

### 配置工作目录与模型

1) 打开并编辑 `utils/PARAM.py`，设置你的工作目录 `workdir`，例如：

```python
workdir = r"D:\your\path\to\EvalWorkdir"
```

首次运行时，程序会在 `workdir` 下自动创建所需子目录：

```
RawData/
PreProcess/csv_with_hits/
PreProcess/csv_with_tracks/
Eval/BackPre/
Eval/BackResult/
Eval/BackResultConvert/
Model/
Plot/
```

2) 模型权重：将过滤模型权重放入 `workdir/Model/`，程序会自动选择该目录中“顺序最后”的一个文件。为避免误选，建议该目录仅保留目标权重文件。

- 本仓库提供了一个备份权重，可复制过去：`backup/filtering_model_back.pth` → `workdir/Model/filtering_model_back.pth`

3) 可选参数：`utils/PARAM.py` 中还包含关键控制项，可按需调整：

- `EVT_NUM`：事件上限
- `ITER_TIME`、`ITER_MULTI`、`JUDGE_PRESERVE_RATE`：迭代筛选策略
- `EVAL_PROCESSOR`、`CHECKER_PROCESSOR`：并行进程数

---

### 准备数据

首先，将一个 `.root` 文件放入 `workdir/RawData/`。

`pipeline_v5.py` 已内置并会“按需自动执行”以下预处理步骤，且会在检测到已有产物时自动跳过：

- ROOT → CSV：`root2csv.root2csv`
- 按事件拆分 CSV：`root2csv.split_csv`
- 生成反向重建候选：`process_hit.backward_process.split_hits`

因此，通常只需将一个 `.root`（推荐） 放入 `workdir/RawData/`，直接运行 `python pipeline_v5.py` 即可。

---

### 运行（pipeline_v5.py）

当 `workdir/RawData` 中已有数据，且 `workdir/Model` 下有权重文件后，直接执行：

```bash
python pipeline_v5.py
```

该脚本将依次执行：

- 反向重建：`EvalTrack/BackwardEval/BackwardEval.py`
- 效率统计：`Check_Eff/Check_Efficiency.py`
- 可视化绘图：`Check_Eff/Plot_Efficiency.py`

生成的核心产物：

- `workdir/Eval/BackResult/*.npy`：每个事件的重建结果
- `workdir/Eval/Efficiency.csv`：逐轨迹统计数据
- `workdir/Eval/Efficiency-table.csv`：分动量段的效率/幽灵率表
- `workdir/Eval/Efficiency.png`：效率与幽灵率曲线（附总量曲线）

---

### 常见问题

- 第一次运行直接退出：为了创建 `workdir` 与子目录，程序首次导入 `utils` 会创建目录并退出。请将数据放入 `workdir/RawData/` 后再运行一次。
- CUDA/驱动不匹配：请确保本机 CUDA 版本与安装的 `torch==2.7.0`（cu118 轮子）兼容；或改装 CPU 版本，例如：`pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu`。
- 找不到模型：请将权重文件放入 `workdir/Model/`，程序会自动加载该目录中最新的一个。
- `RawData` 无文件：`pipeline_v5.py` 会报错并退出；请先完成“准备数据”流程。

---

### 最小示例流程（从零开始）

```bash
# 0) 设置 utils/PARAM.py 的 workdir

# 1) 准备目录（可通过任意一次运行自动创建）
python -c "import utils"  # 如首次创建后退出属于正常

# 2) 放数据（优先使用 ROOT 文件；若是 CSV 也可直接放入）
copy your_data.root  %workdir%/RawData/

# 3) 放模型
copy backup/filtering_model_back.pth  %workdir%/Model/

# 4) 运行主管线
python pipeline_v5.py
```

