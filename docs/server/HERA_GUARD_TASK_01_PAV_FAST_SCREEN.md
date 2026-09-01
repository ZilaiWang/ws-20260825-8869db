# HERA-GUARD-TASK-01：PAV fold0 快筛运行合同

## 目标

在已冻结的 corrected OER OOF 基线上，只训练 formal CV3 fold0 的 Proposal-Aligned Verifier，并执行固定融合快筛。禁止启动三折、MAR 搜参、M3、10K 测速或修改训练超参数。

## 冻结输入

- PAV manifest：`outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v1/hera_pav_manifest.csv`
- manifest SHA256：`d259ed6f5b88d96d890de0d2843c66b256d28dc7ebab5767df3f60cb0acc9156`
- corrected predictions：`outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_predictions.json`
- formal manifest：`outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv`
- formal SHA256：`a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128`
- 原图根：包含 `images/train` 与 `images/val` 的 data root
- ConvNeXt-T ImageNet 权重：项目冻结 SHA `983f1562…`，必须使用完整 SHA 门禁

## 环境与静态门禁

推荐复用 P04/P03 的 `torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / Python 3.10` 环境。

```bash
cd /workspace/xh-202625
ruff check src/rsdet/hera_guard scripts/build_corrected_oer_oof.py scripts/build_hera_pav_manifest.py scripts/train_hera_pav.py scripts/evaluate_hera_pav_fast_screen.py
pytest -q tests/test_hera_guard.py tests/test_hera_manifest.py tests/test_official_frontier.py tests/test_workpoint_labels.py tests/test_grouped_oof.py tests/test_oer_labels.py
```

必须检查 manifest summary 为：65,301 行、foreground 20,391、protected_tp 19,742、active_fp 2,687、inactive_tail 42,872。

## Smoke（必须先跑）

使用正式 manifest 临时截取训练/验证各 128 行，`epochs=1`、`samples-per-epoch=128`、`batch-size=8`、`num-workers=0`。必须确认：

- 双视图 shape 一致；
- 五个 logits shape 正确；
- loss 有限、反向无 NaN；
- checkpoint 与 NPZ 可写；
- held-out candidate ID 唯一且覆盖 smoke validation。

Smoke 产物不得计入科学比较。

## 正式 fold0 命令

服务器实际路径在执行前按资产审计替换，超参数不得变：

```bash
python scripts/train_hera_pav.py \
  --manifest outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v1/hera_pav_manifest.csv \
  --data-root /workspace/data \
  --convnext-weights /workspace/p04-assets/convnext_tiny-983f1562.pth \
  --output-dir /workspace/results/HERA-PAV-FAST-FOLD0-V1 \
  --held-out-fold 0 \
  --freeze freeze_first_stages \
  --resolution 224 \
  --hidden-dim 512 \
  --epochs 4 \
  --batch-size 48 \
  --samples-per-epoch 24000 \
  --head-learning-rate 0.0003 \
  --backbone-learning-rate 0.00001 \
  --weight-decay 0.05 \
  --num-workers 6 \
  --seed 202625 \
  --device cuda \
  --verify-weight-sha256
```

随后：

```bash
python scripts/evaluate_hera_pav_fast_screen.py \
  --predictions outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/corrected_oer_oof_predictions.json \
  --manifest outputs/HERA-GUARD-PRECHECK/pav-manifest-oer-v1/hera_pav_manifest.csv \
  --pav-logits /workspace/results/HERA-PAV-FAST-FOLD0-V1/pav_fold0_oof_logits.npz \
  --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
  --project-config configs/project.yaml \
  --held-out-fold 0 \
  --output /workspace/results/HERA-PAV-FAST-FOLD0-V1/fast_screen_result.json
```

## 停止条件

- 任一 SHA、group overlap、candidate coverage 失败；
- NaN/Inf/OOM；
- checkpoint 加载非 strict；
- 训练行不是恰好 4 epoch；
- 试图在 held-out fold 搜阈值、挑 epoch 或增加训练时长。

## 回传

回传（可不含 checkpoint）：训练 result、history、OOF logits、fast_screen_result、完整日志、环境版本、GPU/峰值显存、全部 SHA。只有 `next_action=expand_pav_to_three_outer_folds` 才进入 TASK-02。

