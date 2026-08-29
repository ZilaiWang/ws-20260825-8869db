# 最终交付物

当前统一入口见 [`docs/SUBMISSION_RUNBOOK_20260828.md`](../../docs/SUBMISSION_RUNBOOK_20260828.md)。
模型身份、Y5 checkpoint 兼容性和正式冻结缺口见
[`MODEL_AND_WEIGHT_FREEZE_AUDIT_20260829.md`](MODEL_AND_WEIGHT_FREEZE_AUDIT_20260829.md)。

## 平台两类提交物

1. Docker 镜像：包含运行代码、Linux x86_64 环境与推理权重；平台追加
   `--input /input --output /output`，程序写 `/output/result.json`。
2. 源代码与报告 ZIP：只含源码和报告，不含权重、训练图、缓存或敏感信息，页面上限
   500MB。

## 仓库内对应入口

- Docker 模板：`submission/docker/`
- 官方入口实现：`src/rsdet/submission/competition.py`
- 交付目录生成：`scripts/build_submission_delivery.py`
- 结果校验：`scripts/validate_submission_result.py`
- 源码报告 ZIP：`scripts/package_source_report.py`

权重不纳入 Git；由构建脚本在本地生成的 `dist/detector-docker-delivery/models/model.pt`
只进入 Docker build context。最终研究报告尚未冻结前，不生成正式上传 ZIP。
