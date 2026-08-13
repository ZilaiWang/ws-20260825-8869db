# R1-4 服务器任务：飞机物理属性辅助监督

本任务在 R1-2 完成后串行执行，复用同一服务器、环境、P03 三折 checkpoint、R1-1
bundles、正式 OOF aggregate、冻结 Y1/C2 和 proposal manifest。入口：

```bash
bash scripts/server/run_r1_aircraft_structured_attribute.sh
```

## 冻结约束

- R1-2 status 必须为 `complete`，避免两个 CPU cross-fit evaluator 竞争；
- 三折固定五轮，不使用 held-out 指标选 checkpoint；
- 仅改训练期 auxiliary supervision；
- 属性 YAML 必须通过 SHA、覆盖率、共享性和成对区分审计；
- checkpoint 只保存原 ConvNeXt-T，属性头不得序列化；
- ship/vehicle 完全旁路；
- 不下载或使用 PSP/MAR20 bridge 图像、外部模型或伪标签；
- 结果无论正负均回传，不因失败门禁重跑超参数。

## 验收产物

- `audit/aircraft_data_audit.json`，含 attribute taxonomy audit；
- smoke summary；
- 3 个训练 summary、3 个 D4 bundle 与 runtime；
- 三条件正式 cross-fit 结果、condition summary、decision；
- `FINAL_GATE_PASS`、checkpoint SHA 清单；
- 无 checkpoint 回传包及 SHA256。

## 科学判定

主条件 `structured_attribute_identity` 对比 `ce_identity`。附加条件
`structured_attribute_d4` 必须再与 R1-1 `ce_d4` 横向比较。若 identity 主门禁失败，停止
当前多头属性版本；不扫描 loss weight。若 identity 通过但 D4 仍低于 CE+D4，则属性监督不
进入当前最强工作点。
