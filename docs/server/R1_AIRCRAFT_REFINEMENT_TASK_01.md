# R1-1 服务器任务：飞机候选框提议域适配与 D4 消融

## 1. 任务边界

一键运行 `scripts/server/run_r1_aircraft_refinement.sh`。任务包含：数据审计、CE/KD 双
smoke、两种方法三折固定 5 epoch 训练、P03/CE/KD 三组 D4 OOF 推理、六条件 CPU
cross-fit、冻结 C2 增量分析和无 checkpoint 回传包。

不得修改 epoch/LR/KD 权重、不得使用 held-out 指标选 checkpoint、不得把背景拒识或
bbox 修正并入本任务、不得启动其他正式实验。

## 2. 预计资源

- GPU：3090/4080S；
- 训练：6 个短 run，预计总计 30–90 分钟；
- 推理：32,062 个 aircraft proposal × 3 模型 × D4，预计 10–30 分钟；
- 输出包含 9 个概率 bundle；回传包排除 6 个 adapted checkpoint。

如一个小时后的实测外推明显超过两小时，保留现场并报告，不自行减 epoch 或缩折。

## 3. 外部路径

应复用 R1-0/P03/Y1 同一服务器资产。典型设置：

```bash
export PROJECT_ROOT=/workspace/xh-202625
export PYTHON_BIN=/workspace/venvs/p06-cu121/bin/python
export DATA_ROOT=/workspace/formal-detection-data
export RESULTS_ROOT=/workspace/results
export TRAINING_MANIFEST=/workspace/results/N2-PROPO-CROP-v2/proposal_crop_manifest.csv
export INFERENCE_MANIFEST=/workspace/results/R1-0-P03-TEACHER-M1-OOF/prepare/proposal_inference_manifest.csv
export BASE_LOGITS_DIR=/workspace/results/R1-0-P03-TEACHER-M1-OOF/logits
export AGGREGATE_DIR=/workspace/results/M1-CV3-OOF-aggregate
export FORMAL_CROP_MANIFEST=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
export Y1_CALIBRATION_RESULT=/workspace/results/Y1-CROSSFIT-CALIBRATION-V1/calibration_result.json
export CONVNEXT_WEIGHTS=/workspace/p04-assets/convnext_tiny-983f1562.pth
export P03_FOLD0_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold0/final_checkpoint.pt
export P03_FOLD1_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold1/final_checkpoint.pt
export P03_FOLD2_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold2/final_checkpoint.pt
```

路径不存在时只查找精确 SHA 对应资产，不用近似文件替代。N2-v2 manifest 若服务器缺失，
从仓库同名本地产物或已上传附件补齐并验证 SHA：
`bf1bc8fe3db193dfaac6900ec35f0c1c9df16be25ed782535e13490de148e5a4`。

## 4. 启动

```bash
cd /workspace/xh-202625
git pull --ff-only origin master
bash scripts/server/run_r1_aircraft_refinement.sh 2>&1 \
  | tee /workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT.driver.log
```

脚本有独占锁和非覆盖门禁。失败后先保留 `status`、driver log 和已生成目录，不重复启动。

## 5. 必须验收

技术：

- training rows 17,948；fold 6,325/6,220/5,403；20 类齐全；
- inference aircraft proposals 32,062；fold 11,090/11,142/9,830；
- 6/6 训练完成且 embedded config 为 fixed epoch、held-out 未加载；
- 9/9 bundle 完成，UID exact coverage，identity `25D`、D4 `20D`，无 NaN/Inf；
- 六个 condition result 和 `decision.json` 完整；
- ship/vehicle structural parity 最大绝对误差 ≤ `1e-12`；
- `FINAL_GATE_PASS` 内容为 `R1_1_TASK_PASS`。

科学回报：

1. 六条件 frozen-C2 Recall/FDR、aircraft macro Recall/FDR；
2. 相对 `p03_identity` 的 delta；
3. 主条件与 reference 的 new/broken/retained/net TP、FP/FN delta；
4. 每折选择的 gate、每折方向；
5. CE 与 selective KD 的训练曲线、teacher anchor fraction；
6. 训练和三组 D4 推理的耗时/显存；
7. `primary_gate_passed`、`next_action`；
8. 所有失败、恢复和人工调整。

## 6. 回传

- `/workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT-return-no-checkpoints.tar.gz`
- 同名 `.sha256`
- `/workspace/results/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT.status`

6 个 checkpoint 和历史资产保留服务器，直到本地校验完成。
