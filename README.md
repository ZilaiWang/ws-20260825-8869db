# XH-202625 小样本遥感图像目标检测

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
- MAR20 飞机图已生成 60 个机场代理视觉组；单次开发划分
  [`dev_v2_airport_proxy_k60`](data/splits/dev_v2_airport_proxy_k60.json) 和正式三折
  [`cv3_airport_proxy_k60_v2`](data/splits/cv3_airport_proxy_k60_v2.json) 均已冻结。
  两者用途、SHA 和完整索引见
  [`DATA_SPLITS_MASTER_INDEX_v1.md`](reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)。
- 统一数据加载器 `XHDataset` 已集成，含 YOLO 标签解析、PyTorch 适配、COCO 导出和数据自检。详见 [`reports/data/数据集详细说明.md`](reports/data/数据集详细说明.md)。
- 协作契约和官方评估已冻结版本；类别映射、IoU 阈值和版本号均以 [`configs/project.yaml`](configs/project.yaml) 为唯一配置源。
- 全局阈值扫描和三个固定工作点已可用；正式实验统一登记到 [`reports/experiments/leaderboard.csv`](reports/experiments/leaderboard.csv)。
- 正式 CV3 v2 的公共输入、检测数据字节锁、P03/P04 复验、M1/M3 OOF、
  模型资产锁与 10K 工程代码/任务单已实现，尚待服务器运行；执行顺序与
  证据边界统一见
  [`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](reports/experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)。

当前项目状态、分工、P 系列结论和待解锁实验统一从
[`docs/hub/README.md`](docs/hub/README.md) 进入。旧计划和历史讨论不得替代该状态导航。

## 优先复用的公共能力

新增同类代码前先检查下表；已有能力应直接调用或在原实现上扩展，避免重新写一套后
产生格式、坐标或评分差异。

| 需求 | 已有入口 | 当前状态 |
|---|---|---|
| 数据读取、自检和 COCO 导出 | [`XHDataset`](src/rsdet/data/xh_dataset.py)、[`check_dataset.py`](scripts/check_dataset.py)、[`export_coco.py`](scripts/export_coco.py) | 可用 |
| 预测契约、批量调用和交付校验 | [`contracts.py`](src/rsdet/contracts.py)、[`predictor.py`](src/rsdet/engine/predictor.py)、[`validate_predictions.py`](scripts/validate_predictions.py) | 可用 |
| 滑窗位置和 tile/原图坐标转换 | [`slicer.py`](src/rsdet/tiling/slicer.py)、[`coordinates.py`](src/rsdet/tiling/coordinates.py) | 可用 |
| 官方 Recall/FDR 评估 | [`official_metric.py`](src/rsdet/evaluation/official_metric.py)、[`evaluate.py`](scripts/evaluate.py) | 可用，禁止另写评分逻辑 |
| 全局阈值扫描与工作点选择 | [`calibration.py`](src/rsdet/postprocess/calibration.py)、[`sweep_thresholds.py`](scripts/sweep_thresholds.py) | 可用 |
| 实验结果登记 | [`leaderboard.csv`](reports/experiments/leaderboard.csv) | 可用 |
| 公共训练、完整推理和跨 tile 融合 | [`train.py`](scripts/train.py)、[`infer.py`](scripts/infer.py)、[`tile_fusion.py`](src/rsdet/postprocess/tile_fusion.py) | 仍是骨架，不要误当成已完成流水线 |

上述公共能力只规定模块交接格式，不限制成员使用的模型、训练框架或内部实现。调用约定
见 [`docs/INTEGRATION_CONTRACT.md`](docs/INTEGRATION_CONTRACT.md)，实验规则见
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)。

```text
数据审计 → 数据划分 → 模型训练 → 大图切片 → tile 推理
→ 坐标恢复 → 跨切片融合 → 阈值选择 → 官方评估 → COCO JSON
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
python -m pytest -q                        # 应全部通过

# 3. 数据检查与导出
python scripts/check_dataset.py --data-root /path/to/data --official-train
python scripts/export_coco.py --data-root /path/to/data --output outputs/train_coco.json

# 4. 评估
python scripts/evaluate.py --gt gt.json --pred pred.json --output outputs/metrics.json

# 5. 扫描全局阈值（默认 0.00~1.00，步长 0.01）
python scripts/sweep_thresholds.py --gt gt.json --pred pred.json \
  --output-dir outputs/实验ID/threshold_sweep

# 6. 模型接入
# 训练和完整推理入口仍在接入真实基线；模型成员可先使用原生框架训练，
# 按 docs/INTEGRATION_CONTRACT.md 交付标准 COCO prediction JSON。
python scripts/validate_predictions.py --pred pred.json --gt gt.json
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

阅读顺序：`README.md` → `docs/INTEGRATION_CONTRACT.md` → `docs/EXPERIMENT_PROTOCOL.md` → `docs/DEVELOPMENT_WORKFLOW.md`

## 注意事项

- 原始数据、测试集、模型权重、密钥和大型日志**不提交 Git**。
- 开发期间用 `logging` 记录关键步骤和耗时；每个正式实验必须记录 Recall、FDR、端到端耗时、显存等。
- 实验必须固定随机种子，确保可复现。配置中写死 seed 值，每次都从配置文件读入。
