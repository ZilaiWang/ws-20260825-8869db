# XH-202625 遥感小目标检测

## 项目目标

不均衡小样本遥感影像检测舰船、飞机和车辆。

| 指标 | 要求 |
|---|---:|
| Overall Recall | ≥ 0.85 |
| Overall FDR | ≤ 0.20 |
| 10000×10000 端到端推理 | ≤ 20 秒（RTX 3090） |
| IoU | 舰船/飞机 0.50，车辆 0.35 |

预测按分数降序贪心匹配，只有相同细类 ID 才能匹配；每个 GT 最多匹配一次，
重复框计为 FP。匹配后按舰船、飞机、车辆汇总指标，最终提交保留 25 个细类 ID 的
COCO detection JSON。

## 当前进度

- 数据契约、切片坐标、官方评估指标已建立并通过测试。
- 数据审计完成：4481 张训练图、20933 个框，25 个细类已归并为三大评测类。
- 统一数据加载器 `XHDataset` 已集成，含 YOLO 标签解析、PyTorch 适配、COCO 导出和数据自检。详见 [`reports/data/数据集详细说明.md`](reports/data/数据集详细说明.md)。
- 类别映射和 IoU 阈值配置在 [`configs/project.yaml`](configs/project.yaml)，为唯一配置源。

```text
数据审计 → 数据划分 → 模型训练 → 大图切片 → tile 推理
→ 坐标恢复 → 跨切片融合 → 分数校准 → 官方评估 → COCO JSON
```

## 目录

```
configs/     配置文件（通用配置；个人路径写 local.yaml，已忽略）
src/rsdet/   核心代码（契约、切片、模型接口、评估、工具）
scripts/     命令行入口（数据审计、训练、推理、评估、测速）
tests/       测试
docs/        文档
reports/     报告和实验汇总
data/        本地数据（不提交）
outputs/     实验输出（不提交）
```

## 开始

```bash
# 1. 安装
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp configs/local.example.yaml configs/local.yaml

# 2. 验证
python -m pytest -q                        # 70 passed

# 3. 数据检查与导出
python scripts/check_dataset.py --data-root /path/to/data --official-train
python scripts/export_coco.py --data-root /path/to/data --output outputs/train_coco.json

# 4. 评估
python scripts/evaluate.py --gt gt.json --pred pred.json --output outputs/metrics.json

# 5. 训练 / 推理
python scripts/train.py --config configs/train.example.yaml
python scripts/infer.py --config configs/infer.example.yaml
python scripts/benchmark.py --config configs/infer.example.yaml
```

## 使用 Gitee 协作

远端仓库：`https://gitee.com/zilai-wang/xh-202625`

**本机操作**：

```bash
# 每次开始工作前，拉取最新代码
git pull origin master

# 开分支干活
git checkout -b feat/xxx
# ... 改代码 ...
git add -A
git commit -m "feat(模块): 做了什么"

# 推到远端，然后在 Gitee 网页上创建 Pull Request
git push origin feat/xxx
```

**PR 审核、合并**：

打开仓库页面 → Pull Requests → "新建 Pull Request" → 选分支（改了什么 ）→ 提交。

**合并后同步**：

```bash
git checkout master
git pull origin master        # 拿到合并后的最新代码
```

> **不要直接 push master。所有改动通过分支 + PR 合入。**

阅读顺序：`README.md` → `docs/PROJECT_PLAN.md` → `docs/DEVELOPMENT_WORKFLOW.md`

## 注意事项

- 原始数据、测试集、模型权重、密钥和大型日志**不提交 Git**。
- 开发期间用 `logging` 记录关键步骤和耗时；每个正式实验必须记录 Recall、FDR、端到端耗时、显存等。
- 实验必须固定随机种子，确保可复现。配置中写死 seed 值，每次都从配置文件读入。
