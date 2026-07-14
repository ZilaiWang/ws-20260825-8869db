# CHANGELOG

## Unreleased

### Added
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

- 官方指标按配置把 25 个细类归并为舰船、飞机、车辆
- 滑窗最后一行和最后一列改为完整 tile 贴边
- README 和 Gitee 模板改为可直接执行的简明版本

### Fixed

- 修复车辆细类误用 0.50 IoU 的问题
- 修复评估脚本无法读取标准 COCO prediction list 的问题
- 未实现的训练、推理和测速不再返回成功或生成假结果
- 远端地址不再包含访问令牌

### Removed

- 删除 Gitee 自动生成且未填写的英文 README
- 删除不属于代码仓库的组织信息
