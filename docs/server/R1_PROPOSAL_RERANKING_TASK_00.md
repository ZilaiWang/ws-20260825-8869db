# R1-PROPOSAL-RERANKING-TASK-00 服务器任务单

## 1. 目标与停止边界

任务只执行 R1-0：用三份 fold-specific P03-F ConvNeXt-T checkpoint 为 M1 正式 OOF
的 55,548 个候选框提取 25 类 logits，再执行 CPU outer cross-fit 重排和冻结 Y1-C2
增量评估。

禁止事项：

- 不训练/微调任何模型；
- 不启动 M3、P03-F 重训、P04-F、E-10K、Y2/Y3；
- 不使用 best checkpoint 替代 P03-F fixed epoch30 checkpoint；
- checkpoint SHA 不匹配时不得“选择一个接近的”继续；
- 不在结果出来后扩大方法网格；
- 不覆盖已有结果目录，不断点拼接不同输入。

## 2. 预计资源

- GPU：RTX 3090、4080S 或同等级均可，本任务不是速度准入实验；
- 显存：预计 1–2 GiB 级；
- 正式推理：55,548 个 224 crop，预计数分钟；
- CPU cross-fit：本地使用伪 logits 全链路实测约 1 分钟；
- 回传包预计几十 MB，不含任何 checkpoint。

## 3. 资产定位

代码仓库中必须存在：

- `configs/experiments/r1_proposal_reranking_v1.yaml`；
- `scripts/r1_proposal_reranking.py`；
- `scripts/server/run_r1_proposal_reranking.sh`；
- `src/rsdet/analysis/proposal_reranking.py`；
- `docs/server/R1_PROPOSAL_RERANKING_TASK_00_CODE_SHA256.txt`。

外部/历史资产：

| 资产 | 必须 SHA256 |
|---|---|
| ConvNeXt-T ImageNet | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` |
| P03-F fold0 | `243d96481f4c6b8c0048c72484b5f395cd2f0c5ba55d103c9b6b665ac4981f54` |
| P03-F fold1 | `bd56974d6598d4f104f11d75fa38b5771e2c20acd55858d1becc26c7a66a7fb7` |
| P03-F fold2 | `eb0a13b3f97c42098ff5ba4bcef5deefcf8f0f5604b346e2c9289fa4246a1c10` |
| formal crop | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |
| Y1 result | `8b9e7a8618b77d167a40bdc2eaf01a9463e9cf8489fdfca7aed1b0059b7b3c12` |

先搜索 checkpoint：

```bash
find /workspace -type f -name 'final_checkpoint.pt' -print0 \
  | xargs -0 -r sha256sum \
  | grep -E '243d9648|bd56974d|eb0a13b3'
```

如果三份 P03-F checkpoint 缺任意一份，状态写为
`blocked_missing_exact_p03_checkpoint` 并停止。不得在本任务里自动重训；重训会产生新的
资产身份，需要单独的 replay amendment 和等价审核。

## 4. 环境

冻结版本：Python 3.10.12、torch 2.5.1+cu121、torchvision 0.20.1+cu121、
numpy 1.26.4、Pillow 10.4.0、PyYAML 6.0.2。可复用 P03/P06 的 `cu121` 环境。

数据根目录必须使 `oof_images.csv` 中的相对路径可直接解析，例如：

```text
${DATA_ROOT}/images/train/MAR20_....jpg
```

## 5. 执行

根据服务器实际路径设置变量；下面只给出结构，不允许把不存在的示例路径直接执行：

```bash
export PROJECT_ROOT=/workspace/xh-202625
export PYTHON_BIN=/workspace/venvs/p06-cu121/bin/python
export DATA_ROOT=/workspace/formal-detection-data
export RESULTS_ROOT=/workspace/results
export AGGREGATE_DIR=/workspace/results/M1-CV3-OOF-aggregate
export FORMAL_CROP_MANIFEST=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
export Y1_CALIBRATION_RESULT=/workspace/results/Y1-CROSSFIT-CALIBRATION-V1/calibration_result.json
export CONVNEXT_WEIGHTS=/workspace/p04-assets/convnext_tiny-983f1562.pth
export P03_FOLD0_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold0/final_checkpoint.pt
export P03_FOLD1_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold1/final_checkpoint.pt
export P03_FOLD2_CHECKPOINT=/workspace/results/P03-FORMAL-CV3-V2/ft-tight-224-fold2/final_checkpoint.pt

bash scripts/server/run_r1_proposal_reranking.sh 2>&1 \
  | tee /workspace/results/R1-0-P03-TEACHER-M1-OOF.driver.log
```

脚本内部固定顺序：

1. code SHA、专项 pytest、ruff；
2. 55,548 行 prepare 与源 aggregate SHA 审计；
3. 每折 32 crop smoke；
4. fold0→fold1→fold2 全量推理；
5. exact UID coverage；
6. outer cross-fit 方法/阈值选择；
7. 冻结 C2 重建 parity（四项最大绝对差≤`1e-12`）；
8. 自动决策、最终门禁、打包。

## 6. 必须验收

技术完整性：

- prepare：4,481 图、55,548 候选；折数为 20,115 / 18,437 / 16,996；
- manifest SHA 应为 `48747c3bba75ec5226e52fff5b488bb92eaa17d9722e377ec393a9cadeafdab0`；
- 三个 logits 的行数分别等于三折 proposal 数，维度均为 25；
- UID 55,548 个恰好一次，无 extra/missing；
- logits 无 NaN/Inf；
- checkpoint SHA 和 embedded config（tight/224/fine_tune/natural/seed42/fixed-last/epoch30）
  全部通过；
- 冻结 C2 重建必须精确复现 Recall 0.9134858835、FDR 0.1592877556；
- `FINAL_GATE_PASS` 存在，status 为 `complete`。

科学回报必须抄录：

1. 三折所选 variant 和 threshold；
2. D0、R1、C2、C2+R1 的 Recall/FDR/macro Recall/macro FDR；
3. 两组 delta；
4. 两组 paired transition（new/broken/retained/net TP、FP/FN delta）；
5. 每折速度、峰值显存；
6. `decision.json` 的两个 signal 和 `next_action`；
7. 所有失败、重试和人工调整；若无则明确写“无”。

## 7. 回传

脚本生成：

- `/workspace/results/R1-0-P03-TEACHER-M1-OOF-return.tar.gz`；
- 同名 `.sha256`；
- `/workspace/results/R1-0-P03-TEACHER-M1-OOF.status`。

回传包含 manifest、smoke、三折 logits/runtime、完整评估和决策；不含 P03 checkpoint、
ImageNet 权重、M1 checkpoint 或原始图像。服务器资产在本地验收前不得删除。
