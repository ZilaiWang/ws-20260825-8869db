# M1-CV3-OOF 三折训练回传审计 v1

> **历史诊断记录，已被正式结果取代。** 本文审计的是误命名为
> `yolo26s.pt` 的 YOLOv8-s 三折诊断运行，不是最终 M1。正确 YOLO26-s
> 正式 OOF、关机续跑边界、官方指标与后续决策统一见
> [`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`](M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md)。

日期：2026-07-24  
审计对象：`outputs/M1-CV3-OOF-results.tar.gz`  
回传包 SHA-256：
`29ba0cbdbd991b17acce0979166834cdf56964091cbc5b495bda54456f9e7f58`

## 1. 结论

本回传包只能验收为：

```text
三折训练过程完成；
实际模型为误命名成 yolo26s.pt 的 YOLOv8-s；
尚未执行或至少尚未交付正式 held-out OOF 推理与聚合；
不能验收为预注册的 M1 YOLO26-s 正式结果。
```

建议把现有三折登记为 `M1D-YOLOv8s-CV3` 诊断运行并保留，不删除；
正式 M1 仍需使用冻结的 YOLO26-s 资产重新执行。

## 2. 已通过的训练层证据

- fold 0/1/2 均完成固定 160 epoch；
- 三折均为 `seed=42`、`imgsz=1024`、`batch=12`、AdamW；
- `resume=false`、`val=false`、`patience=0`；
- 三折训练/held-out 图像数分别为：
  - fold 0：2974 / 1507；
  - fold 1：2868 / 1613；
  - fold 2：3120 / 1361；
- 三份 `results.csv` 均为 160 行且全部数值有限；
- 未发现 Traceback、OOM、CUDA error 或非有限 loss；
- 三折训练耗时分别约 1.659、1.587、1.733 小时，总计约 4.98 小时；
- 三个 `last.pt` 均存在且彼此不同：
  - fold 0：`38438fc4433f711cc5968ad93e56d13608d22d976f93c497b26c162a55d986e0`
  - fold 1：`2cea6ddfbe819edf55bfe90bdf13dc167d51f131c734c0364be95f48284ef3a4`
  - fold 2：`8b7c4845ebd5c0207a97622cc9cb52901fa9083f02a4b6d9c4b711c3fe2cfc89`

## 3. 关键阻断：实际训练的不是 YOLO26-s

冻结正式合同要求：

```text
asset: yolo26s.pt
size: 20,422,725 bytes
sha256: 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
```

回传的 `oof_run_plan.json` 实际记录：

```text
sha256: 1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a
```

该 SHA 对应常见的 `yolov8s.pt`。训练日志也给出相互独立的结构证据：

```text
130 layers
11,145,275 parameters（改为 25 类后）
普通 Detect 头
reg_max=16 / DFL
Transferred 349/355 items
```

这些特征与 YOLOv8-s 一致；YOLO26-s 应为不同的 C3k2/C2PSA、双头、
DFL-free 架构，参数和可迁移项数量也不同。因此这不是单纯的文件序列化
差异，而是模型家族用错。

可能原因包括资产被错误覆盖、错误文件被命名成 `yolo26s.pt`，或服务器
实际训练资产路径未绑定 A00 已验证路径。正式重跑前必须同时检查：

1. 实际训练文件的绝对路径、大小和 SHA；
2. A00 lock 与 verification；
3. 模型 YAML/结构签名、参数量、head 类型和 COCO-80 label namespace；
4. 三折训练计划中写入的实际资产 SHA。

## 4. 当前内部读出

下表来自 Ultralytics 训练结束后的框架内部验证。它使用框架自己的评估
口径，并且日志自动验证 `best.pt`，不是本项目规定的 `last.pt` 低阈值 OOF，
也不是比赛官方 Recall/FDR，只能用于确认模型没有训练崩溃。

| fold | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | 0.8468 | 0.8151 | 0.8500 | 0.6390 |
| 1 | 0.8627 | 0.8396 | 0.8691 | 0.6621 |
| 2 | 0.8410 | 0.7898 | 0.8251 | 0.6307 |
| 简单均值 | 0.8502 | 0.8149 | 0.8481 | 0.6439 |

诊断性信号：

- fold 2 整体最难，Recall 比 fold 1 低约 5 个百分点；
- FSC、HM、LQS、TU-160 等低样本或跨来源困难类波动明显；
- TU-160 的极端机场组不均衡已经真实反映在 held-out 结果中；
- 这支持正式 OOF 后优先做逐类、来源组和错误类型分解，但不能据此直接
  决定 P05/P06。

## 5. OOF 层尚未完成

回传包只有训练配置、日志、CSV 和 checkpoint，缺少正式合同要求的：

- 每折 `resolved_infer.yaml`；
- 每折 `predictions_low.json` 与 runtime；
- 每折 `fold_metadata.json`；
- 每折数据锁验证和环境记录；
- aggregate：
  - `oof_metadata.json`
  - `oof_images.csv`
  - `oof_proposals.csv`
  - `predictions_oof_low.json`

因此当前无法验收：

- 4481 张图是否恰好各有一次 held-out 预测；
- 低阈值候选召回；
- 官方 Recall/FDR；
- 逐类、头中尾、来源组和边界错误；
- `FP_BG / FP_DUP / FP_FINE_CLS / FP_LOC`；
- `FN_FINE_CLS / FN_LOC / FN_MISSING`；
- P05 或 P06 的正式准入。

## 6. 数据异常登记

两张车辆图各有一条完全重复的 YOLO 标签：

- `fsc_TG-N22.33-E120.62-lv20-Google_crop0002.txt`：第 2、3 行相同；
- `fsc_TG-N25.20-E121.42-lv20-Google_crop0001.txt`：第 10、11 行相同。

D00 原始计数为 20,933，Ultralytics 自动去重后为 20,931。总影响只有
2 个框，约为全体 GT 的 0.0096%，不改变主要判断，但正式评估应登记原始
口径与去重敏感性，避免将其误判成模型漏检。

## 7. 最小且实用的后续顺序

1. **立即修复资产门禁**：验证正确 YOLO26-s 文件及结构签名；M3 也增加
   同类双门禁。
2. **保留现有三折**：可补跑低阈值 held-out 推理，形成
   `M1D-YOLOv8s-CV3` 诊断 OOF，快速获得第一份真实错误分布。
3. **正式重跑 M1**：正确 YOLO26-s、三折独立初始化、固定 160 epoch、
   `last.pt` 推理和 aggregate。
4. **正式 OOF 到达后再分流**：
   - 背景 FP 主导：进入 P05；
   - 门槛附近定位 FN/边界误差主导：进入 P06-REAL；
   - 细类混淆主导：优先 P03/P04 与对象级精分类；
   - M1 仍有明显结构性短板：再决定是否支付 M3 的长训练成本。
5. **M3 与 E-10K 暂缓合理**：
   - M3 先等正式 M1 错误分解决定异构模型价值；
   - E-10K 等 E 的大图链路、最终模型和阈值冻结后再做；
   - 4080 SUPER 结果仅作工程基线，最终 20 秒结论在 3090 或公认等效设备
     上复测。

## 8. 关联文档

- `docs/server/M1_CV3_OOF_TASK.md`
- `docs/server/M1_CV3_OOF_RECOVERY_R1.md`
- `docs/server/CV3_OOF_COMMON_CONTRACT.md`
- `configs/experiments/cv3_model_asset_env.json`
- `reports/experiments/DEFERRED_WORK_REGISTER.md`
- `reports/experiments/M1_M3_CV3_OOF_POSTPROCESS_ANALYSIS_PLAN_v1.md`
