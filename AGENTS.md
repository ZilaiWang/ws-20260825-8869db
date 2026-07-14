# AGENTS.md — AI 助手指令

## 项目目标

基于不均衡小样本学习的光学遥感卫星陆上小目标检测。三类目标: ship, aircraft, vehicle。

## 核心指标

- Overall Recall ≥ 0.85, FDR ≤ 0.20
- 10K×10K 推理时间 ≤ 20s (RTX 3090)
- vehicle IoU=0.35, ship/aircraft IoU=0.50

## 目录职责

| 目录 | 用途 |
|------|------|
| `src/rsdet/` | 核心 Python 库 |
| `scripts/` | CLI 入口脚本 |
| `configs/` | YAML 配置文件 |
| `tests/` | 单元测试 |
| `docs/` | 项目文档 |
| `data/` | 本地数据（不入 Git） |
| `outputs/` | 实验输出（不入 Git） |

## 禁止事项

- 不改变 bbox 格式（内部统一 xyxy）
- 不改变 category_id 映射
- 不写死个人路径
- 不将数据/权重加入 Git
- 不删除失败实验记录
- 不将普通 AP 当成 Recall/FDR
- 不将 model forward 时间当成完整推理时间

## 代码规则

- 路径使用 `pathlib.Path`
- 日志使用 `logging`，不用 `print`
- 公开函数加 docstring 和类型标注
- 未实现功能抛出 `NotImplementedError`
- 配置用 YAML + argparse

## 必读

修改公共接口前先更新 `docs/DECISION_LOG.md`。
