# XH-202625 遥感小目标检测

## 项目目标

用不均衡小样本遥感影像检测舰船、飞机和车辆，并满足比赛硬指标：

| 指标 | 要求 |
|---|---:|
| Overall Recall | ≥ 0.85 |
| Overall FDR | ≤ 0.20 |
| 10000×10000 图像端到端推理 | ≤ 20 秒，参考 RTX 3090 |
| IoU | 舰船/飞机 0.50，车辆 0.35 |

预测按分数降序贪心匹配，每个 GT 只能匹配一次，重复框计为 FP。最终提交标准 COCO detection JSON。

## 当前进度

- 仓库、数据契约、切片坐标和官方指标骨架已建立。
- 数据审计已完成；4481 张训练图、20933 个框、官方验证集为空。见 [`reports/data/DATASET_BRIEF.md`](reports/data/DATASET_BRIEF.md)。
- 25 个训练细类已在 [`configs/project.yaml`](configs/project.yaml) 中归并为三大评测类。
- 基线模型、训练器、推理流水线和完整测速尚未接入。`DummyDetector` 只用于接口测试，不是基线。

主流程：

```text
数据审计 → 数据划分 → 模型训练 → 大图切片 → tile 推理
→ 坐标恢复 → 跨切片融合 → 分数校准 → 官方评估 → COCO JSON
```

## 目录

| 路径 | 用途 |
|---|---|
| `configs/` | 可提交的通用配置；个人路径写入忽略的 `local.yaml` |
| `src/rsdet/` | 数据契约、切片、模型接口、评估与工具代码 |
| `scripts/` | 数据审计、训练、推理、评估、测速入口 |
| `tests/` | 无 GPU、无原始数据也能运行的测试 |
| `docs/` | 计划、分工、数据和实验规范 |
| `reports/` | 脱敏数据报告和小型实验汇总 |
| `data/`、`outputs/` | 本地数据和实验产物，默认不提交 |

## 开始使用

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
cp configs/local.example.yaml configs/local.yaml
python -m pytest -q
```

新成员依次阅读：`README.md` → `AGENTS.md` → `docs/PROJECT_PLAN.md` → `docs/TEAM_ROLES.md` → `docs/DEVELOPMENT_WORKFLOW.md`。

## 常用命令

```bash
# 可用：基础数据统计
python scripts/analyze_dataset.py --data-root /path/to/data --output-dir outputs/audit

# 接口已预留：接入基线前会明确返回“未实现”，不会生成假结果
python scripts/train.py --config configs/train.example.yaml
python scripts/infer.py --config configs/infer.example.yaml
python scripts/benchmark.py --config configs/infer.example.yaml

# 可用：COCO GT + 标准 COCO detection 列表
python scripts/evaluate.py --gt gt.json --pred pred.json --output outputs/metrics.json
```

## 协作规则

1. 从 `master` 拉短期分支，一个分支只做一件事。
2. 不直接 push `master`；通过 PR 合并，至少一人审核。
3. 正式实验必须绑定 commit、配置、数据版本、划分版本和随机种子。
4. 记录 Recall、FDR、端到端耗时、显存和失败结论；只报 mAP 不足以决策。
5. 原始数据、测试集、权重、密钥、个人路径和大型日志不得提交。

分工见 [`docs/TEAM_ROLES.md`](docs/TEAM_ROLES.md)，开发细则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
