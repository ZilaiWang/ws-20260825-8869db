# CHANGELOG

## Unreleased

### Added

- R1-5 飞机双视图一致性训练、R1-7 与固定飞机后 NMS 的组合对照，
  以及未通过的 R1-8 等权概率融合
- 基于 R1-6 最终候选链的 N0-4 v3 FP_BG 盲审包、解封编译器与去重
  `clear_background` 白名单门禁
- R1-6 飞机对象头后重排 NMS：固定官方飞机 IoU=0.50，舰船/车辆精确旁路，
  外层三折阈值稳定性审计、守恒错误分解和确定性实现
- 冻结 R1 条件的守恒错误分解入口，以及与原 evaluator JSON 完全等价的快速 cross-fit
  阈值扫描；后者将 31 点扫描部分缩短至约 0.49 秒
- R1-4 飞机物理属性辅助监督正式结果与停止结论
- 论文 arXiv:2512.24074v1 的 Decoupled Queries、BHCL、分层 EMA 原型及逐 decoder 层监督
- 仅使用 `bhcdetr` 的活动训练、推理、checkpoint、滑窗融合与 10K 端到端测速链路
- 双视图翻转/平移数据管线、冻结 manifest/fold 读取和水平框 Hungarian 损失
- `docs/BHCDETR_IMPLEMENTATION.md` 详细复现边界与可执行运行手册
- `configs/bhcdetr.smoke.yaml` 软件链路冒烟配置
- 官方评分方案 V1.6 排名口径聚合 `evaluate_ranking_metrics`：大类指标 = 大类内
  细类指标的简单平均（船 4 型各 1/4、飞机 20 型各 1/20、车辆 1 型即 FSC），
  正式模式固定覆盖配置中的完整 25 类；`evaluate.py` 默认输出 `official_ranking` 块
- `ranking_version=official_ranking_v1_6`、partial-taxonomy 诊断开关和不完整队伍排名保护
- `leaderboard.csv` 增加 `*_macro_*` 列并登记 M1 正式工作点（双口径基线）
- `contract_v1` / `official_eval_v1` 协议版本及其评估产物传播
- 全局置信度阈值扫描、官方/内部/Recall 上限三个固定工作点
- `reports/experiments/leaderboard.csv` 正式实验总表
- 基于正式错误分解与本地开源论文清单的 YOLO 改进优先级、实验顺序和停止条件
- Gitee Release 中的正式 M1 OOF 证据包与关机恢复审计包登记
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

- 在创新阶段前对 `scripts/`、`src/`和 `tests/` 执行全仓 Ruff 格式化与静态清理；
  不改变模型参数或实验结论
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

- 修复 N0-4 盲重复一致性只统计重复卡、未与对应原卡比较而伪造
  100% 一致的问题
- 修复 N0 对象证据将 oracle 命中错误传播给同图同预测类候选的问题；
  新合同使用 `source_prediction_index` 做候选级对齐
- 修复 N2 的 oracle 标签、未审核背景、score 碰撞对齐、模式语义和 held-out 选模泄漏
- 修复 V1.6 macro 在缺类子集上被误当成正式 4/20/1 排名值的问题
- 修复同一大类内错误细类仍被计为 TP 的问题；现按官方 QA 先细类匹配再汇总
- 修复车辆细类误用 0.50 IoU 的问题
- 修复评估脚本无法读取标准 COCO prediction list 的问题
- 未实现的训练、推理和测速不再返回成功或生成假结果
- 远端地址不再包含访问令牌

### Removed

- 删除 Gitee 自动生成且未填写的英文 README
- 删除不属于代码仓库的组织信息
