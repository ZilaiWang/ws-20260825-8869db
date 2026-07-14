# 基于不均衡小样本学习的遥感陆上小目标检测系统

## 赛题背景

揭榜挂帅 XH-202625：基于不均衡小样本学习的光学遥感卫星陆上目标检测识别。

## 检测目标

| 类别 | 英文名 | IoU 阈值 |
|------|--------|----------|
| 舰船 | ship | 0.50 |
| 飞机 | aircraft | 0.50 |
| 车辆 | vehicle | 0.35 |

## 核心指标

- **Overall Recall ≥ 0.85**, **FDR ≤ 0.20**
- **10000×10000 推理 ≤ 20s**（RTX 3090）
- 预测按 score 降序，greedy matching，每 GT 匹配一次
- 重复检测框计为 FP，最终提交标准 COCO JSON

## 当前阶段

**Phase 0 — 仓库和评估基础设施**（已完成）

后续：Phase 1 数据审计 → Phase 2 基线检测器 → Phase 3 错误归因 → Phase 4 创新 → Phase 5 速度优化 → Phase 6 交付

## 总体流程

```
数据审计 → 数据划分 → 模型训练 → 大图切片 → tile推理
→ 坐标恢复 → 跨切片融合 → 分数校准 → 官方评估 → COCO JSON
```

## 目录

```
├── README.md / AGENTS.md / CONTRIBUTING.md
├── configs/        YAML 配置
├── src/rsdet/      核心 Python 库
├── scripts/        CLI 脚本
├── tests/          单元测试
├── docs/           项目文档
├── data/           本地数据（不入 Git）
└── outputs/        实验输出（不入 Git）
```

## 新成员阅读顺序

README → AGENTS.md → docs/PROJECT_PLAN.md → docs/TEAM_ROLES.md → docs/DEVELOPMENT_WORKFLOW.md → 自己模块

## 环境安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements-dev.txt
cp configs/local.example.yaml configs/local.yaml  # 填写本地路径
```

## 常用命令

```bash
python scripts/analyze_dataset.py --data-root /path/to/data
python scripts/train.py --config configs/train.example.yaml
python scripts/infer.py --config configs/infer.example.yaml
python scripts/evaluate.py --gt gt.json --pred pred.json
python scripts/benchmark.py --config configs/infer.example.yaml
python -m pytest -q
```

## 数据和权重

不入 Git。通过 `configs/local.yaml` 配置本地路径。

## 实验记录

每个实验必记：experiment_id / owner / date / git_commit / config_path / 各类别 Recall+FDR / overall_recall+fdr / runtime_total+p95 / peak_vram / notes

规则：无 git commit 不进入正式结果；只报 mAP 不能用于方案决策；失败实验保留结论。

## 未确定

- 最终检测器、切片尺寸、增强策略、创新模块
- 是否使用 GroundingDINO / CAPR / 伪标签

## 团队

王子莱（架构/报告）、吴晓宇（评估/切片/推理）、蔡婕（数据/不均衡学习）、潘扬东杰（检测器/训练）、吴事凡（可视化/文档）

详见 `docs/TEAM_ROLES.md`

> **不要直接修改 master 分支。所有修改通过分支 + PR。**
