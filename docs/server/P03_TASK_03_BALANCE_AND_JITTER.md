# P03 服务器任务 03：224 类别均衡与 jitter 配对评估

## 0. 任务边界

你是服务器执行 AI。P03-TASK-02 已通过本地独立复算，并冻结 `tight-224` 为唯一 clean 工作点。本任务严格按以下顺序执行：

1. 同步并核对本任务的训练代码；
2. 核对 P03-TASK-02 的 3 个 `tight-224/natural` checkpoint；
3. 通过 `sqrt_inverse` 全量微调 smoke 门禁；
4. 串行训练 `tight-224/sqrt_inverse` 三折；
5. 对 natural 和 sqrt-inverse 的共 6 个 checkpoint，分别在同 fold `jitter_light-224` 上 eval-only；
6. 汇总、打包小型结果，并单独打包已入选的 3 个 natural-224 checkpoint 供本地保存。

禁止擅自执行：

- 336 或 `context_1p25`；
- 在 `jitter_light` 上训练；
- 更换模型、权重、增强、优化器、学习率、epoch、seed 或 batch size；
- MixUp/CutMix、focal loss、class-weighted loss 或任何额外不均衡方法；
- 并行启动多个 GPU 训练进程；
- 删除 P03-TASK-01/02 的任何 checkpoint。

`sqrt_inverse` 是本次唯一训练变量；它与 natural 基线都从同一份 ImageNet-1K V1 权重重新初始化，不从对方 checkpoint 继续训练。

## 1. 决策依据

P03-TASK-02 的 224/336 三折 mean macro recall 为 0.9703/0.9753，但同对象 pooled macro recall 差只有 +0.00394，聚类 bootstrap 95% 区间包含 0，而 336 吞吐约低 30%。因此本任务不再继续分辨率搜索。

本任务回答两个新问题：

1. 平方根反频率采样能否改善真实小样本/尾类，且不损害头类和官方相关大类；
2. clean GT crop 的高分在轻度 proposal 几何扰动下会损失多少，类别均衡是否改变该鲁棒性。

## 2. 冻结环境、输入与路径

优先在完成 P03-TASK-02 的同一服务器环境执行：

```text
GPU metadata: NVIDIA GeForce RTX 4080 SUPER, 33,796,456,448 bytes
Python: 3.10.12
torch: 2.5.1+cu121
torchvision: 0.20.1+cu121
PyTorch CUDA runtime: 12.1
cuDNN: 9.1.0
```

```text
/workspace/xh-202625/
/workspace/venvs/p03-cu121/
/workspace/data/
/workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv
/workspace/pretrained/convnext_tiny-983f1562.pth
/workspace/results/P03-TASK-02/
/workspace/results/P03-TASK-03/
```

冻结输入 SHA-256：

```text
crop_manifest.csv
f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e

convnext_tiny-983f1562.pth
983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

P03-TASK-02 的 natural-224 checkpoint 必须为：

```text
fold0  66aaa514420e4c1454222e8e80f3e0f614541e5e0a512b543d0135dbd367248e
fold1  79d70665ce4777a4e54bce117393fb635d39ead5b241a01a4644430d1c8b565a
fold2  2e395a2291b4c21566d7af6c3eb7e1f970be3cb2773a1292b7d029aa7d80b9d1
```

每个 checkpoint 期望大小为 111,426,262 bytes。哈希或大小任一不一致就停止，不从其他同名目录猜测替代。

## 3. 代码同步与强哈希门禁

先将委托方本地最新 P03 代码同步到 `/workspace/xh-202625/`。本任务不能只看 Git commit，因为当前 P03 文件尚在 dirty worktree；必须核对下列文件内容：

```text
5490a1d0fdb76f0486131b32aa8b5d0bec4c39225fabfdc5ba5d2544ba9d693a  configs/experiments/p03_convnext_tiny.yaml
c18163e44712d269555cb2ecf91685c78fa3e879a9d230df1b9cb7606bdef8b0  requirements-p03.txt
b273ea795707426c59abe2d3ad8ae5c6ec11d6f729e1cb700804e93145d80ab9  scripts/train_crop_classifier.py
e1c166749700a758b7e07ac03e4a4cbebe44d666b50459f3d4181c733be5fcb8  scripts/check_p03_environment.py
f2a7f781a8e7b4ef9df787606246b71d88d002deb6e6e668d7838c51c246900a  scripts/summarize_p03_runs.py
beb476acf8dbb6b4bf0ef2e3467b3a810f190ea55bd4934002d74fef9aafed53  src/rsdet/data/crop_classification.py
e60dc764eb068a9106c14787b6930eb69cc1e9236b303f4fa49dde658d089984  src/rsdet/data/xh_dataset.py
9c85cfa660a9a6c41f52856ca2b263699d699aab9cb549b49c08ebf46652243f  src/rsdet/evaluation/classification.py
aab30e7011dbc845184fbd39211f6da4b03ab7153abf8a2a21e93275775a22a7  src/rsdet/models/crop_classifier.py
363c1e2c102abf4c228986ae478a74fd5e076f6fadfabe4363aac720281fdb00  src/rsdet/utils/config.py
d9819d717bb06e5798587294bd88c3cd47588b8d9fc7de07a63e8361359fef23  tests/test_classification_metrics.py
592c13cbf7c5eefbc340cee8459ee41367af42845b52930dcb49006a7d1dff5c  tests/test_crop_classification_data.py
bf61f84186c8b7da101d8b4dcad405baa965be66a9e98c4c9024b13646b9d88a  tests/test_p03_summary_cli.py
22c89a10d55fd87d579ebc0eaeb6c97809c3216a9d01e66b9caedc69f07d13bc  tests/test_p03_training_utils.py
```

建立结果目录并保留实际代码清单：

```bash
cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
set -o pipefail
mkdir -p /workspace/results/P03-TASK-03/logs

sha256sum \
  configs/experiments/p03_convnext_tiny.yaml \
  requirements-p03.txt \
  scripts/train_crop_classifier.py \
  scripts/check_p03_environment.py \
  scripts/summarize_p03_runs.py \
  src/rsdet/data/crop_classification.py \
  src/rsdet/data/xh_dataset.py \
  src/rsdet/evaluation/classification.py \
  src/rsdet/models/crop_classifier.py \
  src/rsdet/utils/config.py \
  tests/test_classification_metrics.py \
  tests/test_crop_classification_data.py \
  tests/test_p03_summary_cli.py \
  tests/test_p03_training_utils.py \
  | tee /workspace/results/P03-TASK-03/CODE_SHA256.txt
```

将输出与上方预期值逐行比较，14/14 相同才可继续。

静态检查：

```bash
PYTHONPATH=src pytest -q \
  tests/test_classification_metrics.py \
  tests/test_crop_classification_data.py \
  tests/test_p03_summary_cli.py \
  tests/test_p03_training_utils.py \
  2>&1 | tee /workspace/results/P03-TASK-03/logs/pytest.log

ruff check \
  scripts/train_crop_classifier.py \
  scripts/check_p03_environment.py \
  scripts/summarize_p03_runs.py \
  src/rsdet/data/crop_classification.py \
  src/rsdet/evaluation/classification.py \
  src/rsdet/models/crop_classifier.py \
  tests/test_classification_metrics.py \
  tests/test_crop_classification_data.py \
  tests/test_p03_summary_cli.py \
  tests/test_p03_training_utils.py \
  2>&1 | tee /workspace/results/P03-TASK-03/logs/ruff.log
```

任一哈希或检查失败立即停止，不在服务器临时修代码。

## 4. 环境、输入与 checkpoint 复检

```bash
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); p=torch.cuda.get_device_properties(0); print(p.name, p.total_memory)"
  df -h /workspace
} 2>&1 | tee /workspace/results/P03-TASK-03/system_preflight.txt

python scripts/check_p03_environment.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output /workspace/results/P03-TASK-03/environment_check.json \
  --verify-source-count 32 \
  2>&1 | tee /workspace/results/P03-TASK-03/logs/environment-check.log

for FOLD in 0 1 2; do
  stat -c '%s %n' "/workspace/results/P03-TASK-02/ft-tight-224-fold${FOLD}/best_checkpoint.pt"
  sha256sum "/workspace/results/P03-TASK-02/ft-tight-224-fold${FOLD}/best_checkpoint.pt"
done | tee /workspace/results/P03-TASK-03/BASELINE_CHECKPOINTS_SHA256_AND_SIZE.txt
```

停止条件：

- GPU 与 TASK-02 不是同一逻辑设备，或有不明训练进程；
- torch/torchvision/PyTorch CUDA runtime 版本变化；
- 可用磁盘小于 15 GB；
- `environment_check.json` 不是 `status: pass`；
- 三个 natural checkpoint 的大小或 SHA-256 不一致。

## 5. `sqrt_inverse` smoke 与 sampler 门禁

```bash
python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir /workspace/results/P03-TASK-03/smoke-sqrtinv-tight-224 \
  --fold 0 \
  --policy tight \
  --resolution 224 \
  --regime fine_tune \
  --sampler sqrt_inverse \
  --seed 42 \
  --smoke \
  --overwrite \
  2>&1 | tee /workspace/results/P03-TASK-03/logs/smoke-sqrtinv-tight-224.log
```

必须确认：

- 10 项训练产物齐全，`smoke=true`，无 NaN/Inf/OOM；
- 27,839,353 个参数全部可训练；
- `resolved_config.yaml` 中 `sampler=sqrt_inverse`；
- `sampler_audit.replacement=true`、`num_samples_per_epoch=n_train`；
- `class_weights` 的最小值为 1，最大值不超过 10；
- `expected_class_probability` 之和在数值精度内为 1；
- 上述 probability 是理论抽样质量，不得回报为某个 epoch 的实际抽样计数。

smoke 分数不进入比较。

## 6. 串行训练 3 个均衡 run

使用冻结配置：ConvNeXt-Tiny、ImageNet-1K V1 初始化、tight-224、fine-tune、seed=42、30 epoch 上限、minimum epoch=12、patience=8、backbone/head LR=1e-4/5e-4、AdamW、label smoothing=0.1、batch=96。

```bash
for FOLD in 0 1 2; do
  RUN="/workspace/results/P03-TASK-03/ft-sqrtinv-tight-224-fold${FOLD}"
  python scripts/train_crop_classifier.py \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --data-root /workspace/data \
    --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
    --output-dir "${RUN}" \
    --fold "${FOLD}" \
    --policy tight \
    --resolution 224 \
    --regime fine_tune \
    --sampler sqrt_inverse \
    --seed 42 \
    2>&1 | tee "/workspace/results/P03-TASK-03/logs/ft-sqrtinv-tight-224-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
```

必须严格串行。断点续跑仅允许跳过已完整、条件与哈希都匹配的 run；单个 run 中断后不得从残留 epoch 擅自恢复。

三折完成后：

```bash
python scripts/summarize_p03_runs.py \
  --runs-root /workspace/results/P03-TASK-03 \
  --output-dir /workspace/results/P03-TASK-03/aggregate-sqrtinv \
  --regime fine_tune \
  --sampler sqrt_inverse \
  --seed 42 \
  2>&1 | tee /workspace/results/P03-TASK-03/logs/aggregate-sqrtinv.log
```

`aggregate.csv` 必须恰有 1 行，`policy=tight`、`resolution=224`、`n_folds=3`。服务器不根据该表自行宣布 sampler 入选。

## 7. 同 checkpoint 的 `jitter_light` eval-only

只在上述 3 个均衡 run 全部完成后执行。每个评估 run 必须加 `--eval-only --checkpoint`，不产生新训练 checkpoint。

### 7.1 natural checkpoint → jitter

```bash
for FOLD in 0 1 2; do
  CKPT="/workspace/results/P03-TASK-02/ft-tight-224-fold${FOLD}/best_checkpoint.pt"
  RUN="/workspace/results/P03-TASK-03/eval-jitter-natural-tight-224-fold${FOLD}"
  python scripts/train_crop_classifier.py \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --data-root /workspace/data \
    --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
    --output-dir "${RUN}" \
    --fold "${FOLD}" \
    --policy jitter_light \
    --resolution 224 \
    --regime fine_tune \
    --sampler natural \
    --seed 42 \
    --checkpoint "${CKPT}" \
    --eval-only \
    2>&1 | tee "/workspace/results/P03-TASK-03/logs/eval-jitter-natural-tight-224-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
```

### 7.2 sqrt-inverse checkpoint → jitter

```bash
for FOLD in 0 1 2; do
  CKPT="/workspace/results/P03-TASK-03/ft-sqrtinv-tight-224-fold${FOLD}/best_checkpoint.pt"
  RUN="/workspace/results/P03-TASK-03/eval-jitter-sqrtinv-tight-224-fold${FOLD}"
  python scripts/train_crop_classifier.py \
    --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
    --data-root /workspace/data \
    --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
    --output-dir "${RUN}" \
    --fold "${FOLD}" \
    --policy jitter_light \
    --resolution 224 \
    --regime fine_tune \
    --sampler sqrt_inverse \
    --seed 42 \
    --checkpoint "${CKPT}" \
    --eval-only \
    2>&1 | tee "/workspace/results/P03-TASK-03/logs/eval-jitter-sqrtinv-tight-224-fold${FOLD}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
```

对 6 个 eval-only run 检查：

- `condition.eval_only=true`、`policy=jitter_light`、fold/resolution/regime 与 checkpoint 一致；
- `checkpoint_source_condition.policy=tight`，且 sampler 分别为 natural/sqrt_inverse；
- `meta.loaded_checkpoint_sha256` 与对应 checkpoint 现算值一致；
- 每折 `n_val` 与 clean 对应折一致，预测中 `annotation_uid` 唯一；
- 有 logits、predictions、confusion、metrics 等 8 项评估产物，没有伪造 history 或新 checkpoint。

这一步只做验证推理，不允许根据 jitter 结果重新训练或调参。

## 8. 产物、checkpoint 保全与打包

新的 3 个 sqrt-inverse checkpoint 仍保留在服务器，本地分析选出 sampler 前不删除：

```bash
find /workspace/results/P03-TASK-03/ft-sqrtinv-*/best_checkpoint.pt -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > /workspace/results/P03-TASK-03/CHECKPOINTS_SHA256.txt

find /workspace/results/P03-TASK-03/ft-sqrtinv-*/best_checkpoint.pt -type f -printf '%s %p\n' \
  | sort > /workspace/results/P03-TASK-03/CHECKPOINTS_SIZES.txt
```

回传小型结果包，不含任何 checkpoint：

```bash
cd /workspace/results
find P03-TASK-03 -type f \
  -not -name 'best_checkpoint.pt' \
  -not -name 'RETURN_FILES_SHA256.txt' \
  -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/P03-TASK-03_RETURN_FILES_SHA256.txt
mv /tmp/P03-TASK-03_RETURN_FILES_SHA256.txt P03-TASK-03/RETURN_FILES_SHA256.txt

tar --exclude='best_checkpoint.pt' \
  -czf P03-TASK-03-results-no-checkpoints.tar.gz P03-TASK-03
sha256sum P03-TASK-03-results-no-checkpoints.tar.gz
```

分辨率已冻结，因此将 P03-TASK-02 的 3 个 natural-224 checkpoint 单独打包下载回本地保全：

```bash
cd /workspace/results
tar -cf P03-TASK-02-tight-224-natural-checkpoints.tar \
  P03-TASK-02/ft-tight-224-fold0/best_checkpoint.pt \
  P03-TASK-02/ft-tight-224-fold1/best_checkpoint.pt \
  P03-TASK-02/ft-tight-224-fold2/best_checkpoint.pt \
  P03-TASK-03/BASELINE_CHECKPOINTS_SHA256_AND_SIZE.txt
sha256sum P03-TASK-02-tight-224-natural-checkpoints.tar
```

两个包均下载后，在本地重算 SHA-256 并解包小型结果。本地确认收到前，服务器上所有 checkpoint 继续保留。

## 9. 最终回报格式

1. 状态：`complete` 或 `blocked_at_<gate>`；
2. Git commit/dirty，`CODE_SHA256.txt` 14/14 是否匹配；
3. GPU/driver/Python/PyTorch/torchvision/PyTorch CUDA runtime、空闲显存和磁盘；
4. manifest/权重/三个 natural checkpoint 哈希与大小门禁；
5. pytest/ruff 和 sqrt-inverse smoke；
6. sampler audit：每折最大/最小 class weight，HM/LQS 理论抽样概率；
7. 3/3 均衡训练结果：每折最优 epoch、完成 epoch、是否早停，macro R/F1、accuracy、aircraft20、显存、吞吐和耗时；
8. sqrt-inverse 三折 mean±std，但不自行与 natural 作最终取舍；
9. 6/6 jitter eval-only 成功数，分 natural/sqrt-inverse 报三折汇总；
10. 3 个新 checkpoint 的路径/大小/SHA-256；
11. 小型回传包和 natural-224 checkpoint 包的路径/大小/SHA-256，以及本地下载校验结果；
12. 任何失败时，给出完整命令、traceback、日志路径和当时资源，不自行改参重跑。
