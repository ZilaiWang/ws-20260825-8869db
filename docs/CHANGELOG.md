# CHANGELOG

## Unreleased

### Added
- 官方评分方案 V1.6 排名口径聚合 `evaluate_ranking_metrics`：大类指标 = 大类内
  细类指标的简单平均（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1 型即 FSC），
  0-GT 细类不参与平均；`evaluate.py` 默认输出 `official_ranking` 块
- `leaderboard.csv` 增加 `*_macro_*` 列并登记 M1 正式工作点（双口径基线）
- `contract_v1` / `official_eval_v1` 协议版本及其评估产物传播
- 全局置信度阈值扫描、官方/内部/Recall 上限三个固定工作点
- `reports/experiments/leaderboard.csv` 正式实验总表
- 最小跨成员集成契约：模型可先用原生框架训练并交付 COCO prediction JSON
- 框架无关的 `InferenceSample`、预测校验、COCO 序列化和批量推理编排
- `validate_predictions.py` 预测交付校验入口
- 项目仓库初始化
- 数据契约定义 (contracts.py)
- 官方评估指标实现 (official_metric.py)
- 大图滑窗切片器 (slicer.py)
- 坐标转换工具 (coordinates.py)
- 模型注册表和 DummyDetector
- 推理计时框架 (runtime.py)
- CLI 脚本框架 (train, infer, evaluate, benchmark, analyze_dataset)
- 项目文档框架 (6 个阶段规划)
- 协作规范
- 标准 COCO detection 列表解析测试
- 可提交的数据划分 manifest 目录

### Changed

- README、实验协议、项目状态与团队任务文档对齐评分方案 V1.6：所有正式实验
  同时报告 pooled（门槛校验）与官方 macro（排名优化）双口径；内部目标
  （FDR≤0.17 等）以官方 macro 口径计
- M1 报告新增 5.3 节官方排名口径基线（舰船 macro FDR 0.52 为最大排名风险，
  LQS/HM/TU-160/F-22 为优先靶点）
- 评估和阈值扫描共用 COCO 读取、配置解析与官方匹配实现
- 实验记录规范与 README 增加可直接执行的阈值扫描入口
- README 增加公共能力索引，明确可复用实现与尚未完成的骨架
- 推理示例配置区分候选保留门槛与扫描后正式阈值
- 模型 adapter 不再被要求实现统一训练步骤，只统一推理输出
- DummyDetector 保留输入 image_id，便于在无真实模型时验证完整接口
- 官方指标按配置把 25 个细类归并为舰船、飞机、车辆
- 滑窗最后一行和最后一列改为完整 tile 贴边
- README 和 Gitee 模板改为可直接执行的简明版本

### Fixed

- 修复同一大类内错误细类仍被计为 TP 的问题；现按官方 QA 先细类匹配再汇总
- 修复车辆细类误用 0.50 IoU 的问题
- 修复评估脚本无法读取标准 COCO prediction list 的问题
- 未实现的训练、推理和测速不再返回成功或生成假结果
- 远端地址不再包含访问令牌

### Removed

- 删除 Gitee 自动生成且未填写的英文 README
- 删除不属于代码仓库的组织信息
