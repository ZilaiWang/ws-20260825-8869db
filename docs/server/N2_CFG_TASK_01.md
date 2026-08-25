# N2-CFG 服务器任务：粗类条件式前景门控 三折训练 + S0/S1/S2 快筛

## 1. 任务边界

一键运行 `scripts/server/run_n2_cfg.sh`。任务包含：环境 preflight + 代码 SHA 门禁、
输入资产 SHA 校验、前景门控 manifest 构建、三折 `freeze_backbone` 固定 5 epoch 训练、
三折前景 logit 推理、S0/S1/S2 leave-one-fold-out cross-fit 快筛、门禁判定与
无 checkpoint 回传包。

只做 `background_reject`：不改框、不改细类、不恢复候选。舰船/车辆正式门控，
飞机 shadow 旁路（输出逐条不变）。

**严禁**：修改 epoch/LR/batch/sampler/seed/freeze 策略、用 held-out fold 选
checkpoint、把未确认 FP_BG 自动当背景、启动其他正式实验。

## 2. 前置（本地完成，B 回传 CSV 后）

1. 本地执行 `compile_fp_bg_review.py`，一致率 >=0.85 编译出
   `clear_background_whitelist.csv`（0.90 + κ>=0.75 为科学放行门槛）；
2. 把白名单上传到服务器
   `$PROJECT_ROOT/outputs/N0-FP-BG-AUDIT-R1-6-V3/compiled/clear_background_whitelist.csv`；
3. 确认 `DATA_ROOT` 指向 4,481 张源图目录。

若白名单缺失，脚本在第 2 步以 exit 2 退出并提示等待，不会误跑。

## 3. 外部路径（可用环境变量覆盖）

```bash
export PROJECT_ROOT=/workspace/xh-202625
export PYTHON_BIN=/workspace/venvs/p03-cu121/bin/python   # torch 2.5.1 + torchvision 0.20.1
export DATA_ROOT=/workspace/formal-detection-data        # 4,481 源图（需现场确认）
export RESULTS_ROOT=/workspace/results
```

输入资产均相对 `$PROJECT_ROOT`，SHA 与 `n2_cfg_background_gate_v1.yaml` 合同一致：

| 资产 | SHA（前 8 位） |
|---|---|
| selected_predictions_xyxy.json | d07f43e5 |
| formal_crop_manifest.csv | a3bed44f |
| oof_images.csv | fc2aa7ca |
| convnext_tiny_imagenet1k_v1.pth | 983f1562 |

## 4. 预计资源

- GPU：RTX 3090；
- 训练：3 个轻量 run（freeze_backbone，仅 shared/coarse head 可训练，5 epoch ×
  200 batch × 64），预计总计 < 30 分钟；
- 推理：19,470 deployable + 白名单负样本 的 context_1.25 crop 单视图，3090 上
  ~1,270–1,350 crops/s，预计 < 5 分钟；
- 评估：S0/S1/S2 纯 CPU 校准拟合，分钟级。

若实测外推明显超过两小时，保留现场并报告，不自行减 epoch 或缩折。

## 5. 门禁（《改进方案 1》3.2 / 3.4）

脚本只输出 `admission.json` 做初步判定（S2 是否在相同 Recall 约束下优于 S0、
零 TP 损失）。正式 pooled/逐类/来源稳健性门禁由本地基于 `evaluate_S{0,1,2}.json`
复算：

- pooled `FP_BG` 减少 >=10%（>=154），舰船 >=15%、车辆 >=10%；
- Overall Recall 下降 <=0.2pp，任一粗类 <=0.5pp；
- 车辆 / HM / LQS 零 TP 损失；舰船 TP 损失 <=4 且不集中单类；
- 飞机 TP/FP/FN 逐条一致；
- >=2/3 fold 同向；S2 在相同 Recall 约束下显著优于 S0。

任一停止条件触发即停，不继续搜网络/loss/阈值。

## 6. 回传

```text
/workspace/results/N2-CFG-BACKGROUND-GATE-V1-return-no-checkpoints.tar.gz
```

回传包排除 `bg_gate_fold*_final.pt`，含 manifest、train_result、fg_logits、
evaluate_S{0,1,2}.json、admission.json、日志与 CHECKPOINTS_SHA256.txt。
