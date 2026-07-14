# 团队角色

模块责任制 ≠ 代码所有权，所有成员均可通过 PR 参与。

## 王子莱

- 第一责任: 总体架构、技术路线决策、核心创新
- 第二责任: 模型集成、报告总负责
- 审核: `src/rsdet/models/`, `docs/PROJECT_PLAN.md`

## 吴晓宇

- 第一责任: 官方评估、大图切片、全图推理
- 第二责任: 跨切片融合、速度测试和部署
- 审核: `src/rsdet/evaluation/`, `src/rsdet/tiling/`

## 蔡婕

- 第一责任: 数据分析、数据划分、不均衡学习
- 第二责任: 半监督和伪标签、统计验证
- 审核: `src/rsdet/data/`, `src/rsdet/postprocess/calibration.py`

## 潘扬东杰

- 第一责任: 检测器基线、训练框架
- 第二责任: 定位和模型改进、蒸馏与模型压缩
- 审核: `src/rsdet/models/`, `src/rsdet/engine/`

## 吴事凡

- 第一责任: 可视化、实验图表
- 第二责任: README 和文档、PPT、复现检查和基础测试
- 审核: `src/rsdet/visualization/`, `docs/`, `tests/`
