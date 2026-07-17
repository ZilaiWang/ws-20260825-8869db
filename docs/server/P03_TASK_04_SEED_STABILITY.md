# P03 服务器任务 04：最终 crop 基线的随机种子稳定性

## 0. 任务边界

你是服务器执行 AI。P03-TASK-03 已通过本地独立验收，当前唯一工作点冻结为：

```text
model = ConvNeXt-Tiny
initialization = ImageNet-1K V1
crop = tight
resolution = 224
sampler = natural
regime = fine_tune
```

本任务只回答一个问题：这个工作点在 `seed=42` 之外是否保持稳定。严格按顺序执行：

1. 同步并核对冻结训练代码、manifest 和 ImageNet 权重；
2. 做一次 `seed=3407/fold0` 的单 epoch smoke；
3. 串行训练 `seed=3407` 和 `seed=202625` 的各 3 fold，共 6 个正式 run；
4. 每个 best checkpoint 在相同 fold 的固定 `jitter_light-224` manifest 上做 eval-only，共 6 个评估 run；
5. 分 seed 汇总、核验产物、打包不含 checkpoint 的小型结果并下载回本地；
6. 6 个新 checkpoint 留在服务器，等待本地完成 seed 配对分析后再决定是否下载或删除。

禁止擅自执行：

- 重跑 `seed=42`；
- 336、`context_1p25`、`sqrt_inverse` 或任何新条件；
- 在 `jitter_light` 上训练；
- 更换模型、权重、增强、优化器、学习率、epoch、batch size 或早停规则；
- MixUp/CutMix、focal loss、class-weighted loss、合成数据或其他新方法；
- 多进程并行训练，或同时占用多张 GPU；
- 从 seed=42 或其他 seed checkpoint 继续训练；
- 删除 P03-TASK-01/02/03 的 checkpoint。

两个新 seed 的正式 run 都必须从同一份 ImageNet-1K V1 权重独立初始化。服务器只执行和汇总，不根据结果自行改变基线。

## 1. 决策依据与统计边界

seed=42 的 `tight-224/natural` 三折 mean macro recall 为 `0.9703 ± 0.0078`。本地 pooled OOF 复算为 macro recall `0.97080`、accuracy `0.97970`。`sqrt_inverse` 没有稳定改善尾类或主指标，已经作为负向消融停止。

本任务新增两个 seed，而不是继续扩大超参数网格。完成后本地会把 3 个 seed × 3 fold 的 9 个 clean run 与 9 个对应 jitter 条件统一分析，区分：

- seed 内三折差异；
- 同 fold 跨 seed 差异；
- 9 run 的总体均值、seed 间和 fold 间波动；
- 同一对象跨 seed 的预测稳定性；
- fold0 是否持续较难；
- clean → jitter 损失是否依赖 seed。

服务器不得把 6 个新 run 当作 6 个相互独立的数据划分，也不得把跨对象 bootstrap 当作训练 seed 方差。

`jitter_light` 的 crop 坐标已经冻结在 manifest 中，验证 transform 不含随机增强。因此每个 seed 的 jitter eval 使用相同对象和相同几何扰动，能够与对应 clean 结果以及其他 seed 逐对象配对。

## 2. 冻结环境、输入与路径

优先在完成 P03-TASK-02/03 的同一服务器环境执行：

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
/workspace/results/P03-TASK-04/
```

冻结输入 SHA-256：

```text
crop_manifest.csv
f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e

convnext_tiny-983f1562.pth
983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
```

本任务不以任何旧 checkpoint 为训练输入。若服务器是克隆实例，可从已验证数据盘或原服务器同步 manifest、数据集和预训练权重，但必须按上述哈希重新验证，不能凭文件名推断相同。

## 3. 代码同步与强哈希门禁

当前 P03 文件仍可能位于 dirty worktree，不能只看 Git commit。先将委托方本地最新冻结训练代码同步到 `/workspace/xh-202625/`，再逐项核对：

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

执行：

```bash
cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
set -o pipefail
mkdir -p /workspace/results/P03-TASK-04/logs

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
  | tee /workspace/results/P03-TASK-04/CODE_SHA256.txt
```

必须与上方 14 行逐行相同。随后执行：

```bash
PYTHONPATH=src pytest -q \
  tests/test_classification_metrics.py \
  tests/test_crop_classification_data.py \
  tests/test_p03_summary_cli.py \
  tests/test_p03_training_utils.py \
  2>&1 | tee /workspace/results/P03-TASK-04/logs/pytest.log

ruff check \
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
  2>&1 | tee /workspace/results/P03-TASK-04/logs/ruff.log
```

任一哈希或检查失败立即停止，不在服务器临时修代码。

## 4. 系统、环境和数据门禁

```bash
{
  date -Is
  git rev-parse HEAD
  git status --short
  nvidia-smi
  python --version
  python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); p=torch.cuda.get_device_properties(0); print(p.name, p.total_memory)"
  df -h /workspace
} 2>&1 | tee /workspace/results/P03-TASK-04/system_preflight.txt

python scripts/check_p03_environment.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output /workspace/results/P03-TASK-04/environment_check.json \
  --verify-source-count 32 \
  2>&1 | tee /workspace/results/P03-TASK-04/logs/environment-check.log
```

停止条件：

- GPU 不是已批准的 RTX 4080 SUPER 32GB，或存在不明训练进程；
- torch/torchvision/PyTorch CUDA runtime 与冻结环境不同；
- 可用磁盘小于 15 GB；
- manifest 或权重 SHA-256 不一致；
- `environment_check.json` 不是 `status: pass`；
- fold 隔离、样本数量、数据尺寸或抽查 checksum 任一失败。

## 5. 单次 smoke

只做一次 smoke，验证新任务目录、seed 参数和完整训练通路。smoke 分数不进入比较。

```bash
python scripts/train_crop_classifier.py \
  --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --data-root /workspace/data \
  --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
  --output-dir /workspace/results/P03-TASK-04/smoke-natural-tight-224-seed3407-fold0 \
  --fold 0 \
  --policy tight \
  --resolution 224 \
  --regime fine_tune \
  --sampler natural \
  --seed 3407 \
  --smoke \
  --overwrite \
  2>&1 | tee /workspace/results/P03-TASK-04/logs/smoke-natural-tight-224-seed3407-fold0.log
```

必须确认：

- 10 项训练产物齐全，`smoke=true`，无 NaN/Inf/OOM；
- 27,839,353 个参数全部可训练；
- `policy=tight`、`resolution=224`、`sampler=natural`、`seed=3407`、`fold=0`；
- validation 为自然分布，`n_train`/`n_val` 是确定性 smoke 子集；
- checkpoint 能被重新加载并完成最终验证。

## 6. 串行训练 6 个正式 run

冻结配置：30 epoch 上限、minimum epoch=12、patience=8、backbone/head LR=`1e-4/5e-4`、AdamW、label smoothing=0.1、batch=96、基础旋转和翻转增强。除了 seed，必须与 TASK-02 的 natural-224 完全相同。

```bash
for SEED in 3407 202625; do
  for FOLD in 0 1 2; do
    RUN="/workspace/results/P03-TASK-04/ft-natural-tight-224-seed${SEED}-fold${FOLD}"
    python scripts/train_crop_classifier.py \
      --manifest /workspace/artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv \
      --data-root /workspace/data \
      --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
      --output-dir "${RUN}" \
      --fold "${FOLD}" \
      --policy tight \
      --resolution 224 \
      --regime fine_tune \
      --sampler natural \
      --seed "${SEED}" \
      2>&1 | tee "/workspace/results/P03-TASK-04/logs/ft-natural-tight-224-seed${SEED}-fold${FOLD}.log"
    test "${PIPESTATUS[0]}" -eq 0 || exit 1
  done
done
```

必须严格串行。断点续跑时，只允许跳过以下条件全部成立的完整 run：

- artifact contract 10 项齐全；
- `run_summary.json` 和 `resolved_config.yaml` 条件与目录名一致；
- `meta.json` 的 manifest/权重哈希正确；
- checkpoint SHA 与 `run_summary.json` 一致；
- logits、predictions、confusion 和 metrics 可读取，`n_val` 与对应 fold 一致。

单个 run 中断后不得从残留 epoch 擅自续训；删除该不完整 run 目录并用相同命令从 ImageNet 权重重新开始。禁止改 seed 或超参数“补救”。

每个 seed 完成三折后分别汇总：

```bash
for SEED in 3407 202625; do
  python scripts/summarize_p03_runs.py \
    --runs-root /workspace/results/P03-TASK-04 \
    --output-dir "/workspace/results/P03-TASK-04/aggregate-seed${SEED}" \
    --regime fine_tune \
    --sampler natural \
    --seed "${SEED}" \
    2>&1 | tee "/workspace/results/P03-TASK-04/logs/aggregate-seed${SEED}.log"
done
```

每份 `aggregate.csv` 必须恰有 1 行，且为 `tight/224/fine_tune/natural`、对应 seed、fold `0;1;2`、`n_folds=3`。服务器只报数，不以某个 seed 最好为由挑选或重跑。

## 7. 6 个固定 jitter eval-only

6 个训练全部完成后再执行。每个评估 run 必须加载同 seed、同 fold 的 best checkpoint。`jitter_light` 只改变 manifest 中冻结的 crop 几何，不训练、不早停、不产生新 checkpoint。

```bash
for SEED in 3407 202625; do
  for FOLD in 0 1 2; do
    CKPT="/workspace/results/P03-TASK-04/ft-natural-tight-224-seed${SEED}-fold${FOLD}/best_checkpoint.pt"
    RUN="/workspace/results/P03-TASK-04/eval-jitter-natural-tight-224-seed${SEED}-fold${FOLD}"
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
      --seed "${SEED}" \
      --checkpoint "${CKPT}" \
      --eval-only \
      2>&1 | tee "/workspace/results/P03-TASK-04/logs/eval-jitter-natural-tight-224-seed${SEED}-fold${FOLD}.log"
    test "${PIPESTATUS[0]}" -eq 0 || exit 1
  done
done
```

逐个确认：

- `condition.eval_only=true`、`policy=jitter_light`、sampler natural；
- condition 的 seed/fold/resolution/regime 与目录名一致；
- `checkpoint_source_condition` 为同 seed、同 fold、`policy=tight`、224、fine-tune、natural；
- `meta.loaded_checkpoint_sha256` 与加载文件现算值一致；
- `n_val` 与同 fold clean run 一致，且预测的 `annotation_uid` 唯一；
- 8 项评估产物齐全，没有 history 或新 checkpoint；
- 没有根据 jitter 结果调参或重新训练。

## 8. checkpoint 清单与回传包

6 个正式训练 checkpoint 全部留在服务器：

```bash
find /workspace/results/P03-TASK-04/ft-natural-*/best_checkpoint.pt -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > /workspace/results/P03-TASK-04/CHECKPOINTS_SHA256.txt

find /workspace/results/P03-TASK-04/ft-natural-*/best_checkpoint.pt -type f -printf '%s %p\n' \
  | sort > /workspace/results/P03-TASK-04/CHECKPOINTS_SIZES.txt
```

生成不含 checkpoint 的小型结果包。清单不得包含清单自身：

```bash
cd /workspace/results
find P03-TASK-04 -type f \
  -not -name 'best_checkpoint.pt' \
  -not -name 'RETURN_FILES_SHA256.txt' \
  -print0 | sort -z | xargs -0 sha256sum \
  > /tmp/P03-TASK-04_RETURN_FILES_SHA256.txt
mv /tmp/P03-TASK-04_RETURN_FILES_SHA256.txt P03-TASK-04/RETURN_FILES_SHA256.txt

tar --exclude='best_checkpoint.pt' \
  -czf P03-TASK-04-results-no-checkpoints.tar.gz P03-TASK-04
sha256sum P03-TASK-04-results-no-checkpoints.tar.gz
```

下载后在本地重算压缩包 SHA-256，再解压到 `outputs/P03-TASK-04/`。服务器 AI 必须确认本地包哈希一致且 `RETURN_FILES_SHA256.txt` 全部通过。完成本地验收前，不删除 6 个新 checkpoint。

本任务不要求再次打包 seed=42 checkpoint，也不要求下载 6 个新 checkpoint 本体。

## 9. 最终回报格式

1. 状态：`complete` 或 `blocked_at_<gate>`；
2. Git commit/dirty，`CODE_SHA256.txt` 14/14 是否匹配；
3. GPU/driver/Python/PyTorch/torchvision/PyTorch CUDA runtime、磁盘；
4. manifest/权重门禁、环境抽查、pytest/ruff；
5. smoke 结果和正式配置核验；
6. 6/6 训练成功数，按 seed/fold 报 best epoch、completed epoch、early-stop、macro R、macro F1、accuracy、aircraft20 R；
7. 每个 seed 的三折 mean ± sample std；
8. 6/6 jitter eval-only 成功数，按 seed/fold 报 macro R/F1/accuracy，并确认 checkpoint source；
9. 显存、吞吐、每 run 耗时和总耗时；
10. 6 个新 checkpoint 的路径、大小、SHA-256；
11. 小型回传包路径、大小、SHA-256、本地下载与包内清单校验；
12. 任何失败时，给出完整命令、traceback、日志路径和当时资源，不自行改参重跑。

不要自行宣告“seed 稳定”或根据最好 seed 选择模型；最终判断由本地对 3 seed × 3 fold 的统一配对分析完成。
