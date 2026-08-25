# M1-CV3-OOF-R1：训练入口导入冲突修复后的重跑说明

日期：2026-07-24  
适用故障：`train() got an unexpected keyword argument 'resume'`

## 1. 根因与修复边界

主仓库和模型仓库都包含顶层包 `rsdet`。失败进程误导入了主仓库中的
`rsdet.engine.trainer.train(config)`，而不是模型仓库的训练编排器；同时旧入口
在 fresh run 中仍无条件透传 `resume=None`，使导入错误立即暴露。

本次只修复入口合同：

1. `train.py` 和 `infer.py` 强制优先导入各自同级模型仓库的 `src`；
2. 未显式传 `--resume` 时，训练入口不再向下传递 `resume`；
3. 显式 `--resume` 的通用 CLI 能力仍保留；
4. M1/M3 正式任务继续禁止 `--resume`、跨折 checkpoint 和失败目录续跑。

模型、数据、epoch、batch、优化器、阈值和三折协议均未改变。

## 2. 必须同步的文件

```text
/workspace/xh-202625-model/scripts/train.py
/workspace/xh-202625-model/scripts/infer.py
/workspace/xh-202625-model/tests/test_trainer_contract.py
/workspace/xh-202625-model/tests/test_infer_evaluation.py
/workspace/xh-202625/docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt
```

冻结 SHA-256：

```text
8525a2fa5e956b4f53ec5affe7ef7e39d9ae428b7474147420f40d6acd302ad1  scripts/train.py
71556f4926d3fc622452a277f2c4b711c7b6967482699477af8c8869507b98c9  scripts/infer.py
110f0e5d0f47466db365f0b34f4783ae6c8a3fdb9498912859a9d6d3b6ef172e  tests/test_trainer_contract.py
23a4e66694b82d4ec7ebc607eefb401b1173a7fb41b49b17bad2dd6e2daa7b4e  tests/test_infer_evaluation.py
```

中央模型锁自身 SHA-256：

```text
869eb6fed492fc85dd12a399379a9e4d46e42c5568754a746ff389fc7ecde047
```

## 3. 同步后的强制门禁

```bash
set -euo pipefail

cd /workspace/xh-202625
echo \
  "869eb6fed492fc85dd12a399379a9e4d46e42c5568754a746ff389fc7ecde047  docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt" \
  | sha256sum -c -
sha256sum -c docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt

cd /workspace/xh-202625-model
source /workspace/venvs/cv3-model-cu121/bin/activate
export PYTHONNOUSERSITE=1

PYTHONPATH=src python -m pytest -q \
  tests/test_trainer_contract.py \
  tests/test_infer_evaluation.py \
  tests/test_ultralytics_adapter.py \
  tests/test_inference_pipeline.py \
  tests/test_tile_fusion.py
PYTHONPATH=src python -m ruff check \
  scripts/train.py scripts/infer.py \
  tests/test_trainer_contract.py tests/test_infer_evaluation.py

# 故意把主仓 src 放入 PYTHONPATH，验证两个入口仍会固定到模型仓源码。
PYTHONPATH=/workspace/xh-202625/src python scripts/train.py --help >/dev/null
PYTHONPATH=/workspace/xh-202625/src python scripts/infer.py --help >/dev/null
echo "M1_ENTRYPOINT_IMPORT_CONTRACT_PASS"
```

任一项失败都停止，不得临时修改服务器代码。

## 4. 保留失败现场并恢复规范路径

失败发生在首次 `--dry-run` 的函数调用边界，不能在已有目录中续跑。保留原
目录和日志，以原子改名释放规范结果路径：

```bash
set -euo pipefail

FAILED_ROOT=/workspace/results/M1-CV3-OOF
ARCHIVE_ROOT=/workspace/results/M1-CV3-OOF-FAILED-20260724-IMPORT-CONTRACT
FAILED_AGG=/workspace/results/M1-CV3-OOF-aggregate
ARCHIVE_AGG=/workspace/results/M1-CV3-OOF-FAILED-20260724-IMPORT-CONTRACT-aggregate

if test -e "$FAILED_ROOT"; then
  test ! -e "$ARCHIVE_ROOT"
  mv "$FAILED_ROOT" "$ARCHIVE_ROOT"
fi
if test -e "$FAILED_AGG"; then
  test ! -e "$ARCHIVE_AGG"
  mv "$FAILED_AGG" "$ARCHIVE_AGG"
fi

test ! -e "$FAILED_ROOT"
test ! -e "$FAILED_AGG"
```

这不是删除或覆盖：失败现场完整保留在带原因和日期的归档路径。

## 5. 从头重跑

完成以上步骤后，从
`docs/server/M1_CV3_OOF_TASK.md` 第 3 节开始原样重跑：

- 重新生成三折计划和 split view；
- fold 0/1/2 均从冻结的官方 `yolo26s.pt` 独立开始；
- dry-run 和正式训练均重新执行；
- 不添加 `--resume`；
- 不复用失败目录中的任何 checkpoint、summary 或 prepared data；
- 仍写规范路径 `M1-CV3-OOF` 与 `M1-CV3-OOF-aggregate`，下游合同无需修改。

最终回报中额外列出：

1. 归档失败目录及其日志 SHA；
2. `M1_ENTRYPOINT_IMPORT_CONTRACT_PASS`；
3. 新模型锁验证结果；
4. 三折均为 `resume_used=false`；
5. 新 aggregate 四件套的路径和 SHA。

