# CV3-FORMAL-STAGE：服务器统一调度任务单

对应科学总纲：
`reports/experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`

## 0. 服务器 AI 的职责

本任务单只负责任务调度，不授权服务器 AI 修改科学配置、补写缺失结果或
自行放宽门禁。子任务已有完整命令和停止条件，应逐份原样执行。

统一仓库：

```text
主仓库: /workspace/xh-202625
模型仓库: /workspace/xh-202625-model
结果根: /workspace/results
```

本阶段路径合同统一冻结为 `/workspace`。宿主机若使用其他存储根，应先建立
挂载或符号链接，使上述绝对路径成立；不得逐份任务改根前缀。

## 1. 先做一次总预检

```bash
set -euo pipefail
cd /workspace/xh-202625

mkdir -p /workspace/results/CV3-FORMAL-STAGE/logs
{
  date -Is
  git rev-parse HEAD
  git status --short
  python --version
  df -h /workspace
  free -h || true
  nvidia-smi
} 2>&1 | tee /workspace/results/CV3-FORMAL-STAGE/system_preflight.txt

test "$(sha256sum data/splits/cv3_airport_proxy_k60_v2.json | cut -d' ' -f1)" = \
  "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt
sha256sum -c docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt
```

不得因工作树 dirty 自动清理或 reset；只记录状态并执行任务单中的代码 SHA
门禁。服务器上的 P03/P04 cache 与旧 checkpoint 不得删除或改写。

## 2. 执行顺序

### 2.1 F00：CPU 公共输入

先完整执行：

```text
docs/server/FORMAL_CV3_CROP_TASK_01_CPU.md
```

期望状态：

```text
formal_cv3_and_crop_v2_pass
formal crop SHA:
a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
```

将 F00 验收副本的绝对路径记录到：

```text
/workspace/results/CV3-FORMAL-STAGE/formal_crop_path.txt
```

后续任务若预设路径不同，只允许把任务中的 `FORMAL` 变量指向这份精确 SHA
文件；不得重新生成一个内容不同的替代物。

### 2.2 D00：正式检测数据字节锁

执行：

```text
docs/server/CV3_DETECTION_DATA_LOCK_TASK_00.md
```

D00 必须在 F00 之后完成，得到：

```text
/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
SHA256 03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
```

该锁冻结 4,481 图像、4,481 标签及 20,933 GT。M1/M3 每一折开训前必须
按 D00 第 6 节重新全量 verify；仅在总调度时验证一次不够。

### 2.3 A00：M1/M3 共用环境与官方权重冻结

执行：

```text
docs/server/CV3_MODEL_ASSET_ENV_TASK_00.md
```

本任务只执行一次，得到：

```text
/workspace/venvs/cv3-model-cu121
/workspace/cv3-model-assets/yolo26s.pt
/workspace/cv3-model-assets/rtdetr-l.pt
/workspace/cv3-model-assets/MODEL_ASSET_ENV_LOCK.json
```

M1、M3 和 E-10K 都会在各自启动前独立重算此锁；任何环境、GPU、权重大小
或 SHA 漂移都会停止。禁止在 M1/M3 任务内临时下载或替换同名权重。

A00 不依赖 F00/D00，可与 CPU 数据链并行；但 M1/M3 启动前，A00 与 D00
必须都通过。

### 2.4 P04-F：低成本正式教师读出

执行：

```text
docs/server/P04_FORMAL_CV3_V2_REPLAY.md
```

原因：三个全量 D4 cache 已在同一服务器，复用门禁通过后只需运行 18 个轻量
probe。任一 cache 复用门禁失败即停止整个 P04 正式矩阵并回传诊断；不运行
不完整的教师子集，也不自动重新提取数小时特征。这样保持六条件完全配对。

### 2.5 P03-F：正式 tight-224 上限

执行：

```text
docs/server/P03_FORMAL_CV3_V2_REPLAY.md
```

只能有三个非 smoke run。formal crop 若已由 F00 生成，应验证 SHA 后直接
使用，不能再次写入已验收目录。

### 2.6 M1-OOF：核心检测关键路径

依次阅读并执行：

```text
docs/server/CV3_OOF_COMMON_CONTRACT.md
docs/server/M1_CV3_OOF_TASK.md
```

首次启动若命中已登记的同名包导入故障
`train() got an unexpected keyword argument 'resume'`，先执行
`docs/server/M1_CV3_OOF_RECOVERY_R1.md`，归档失败现场并通过新模型代码锁
后，再从 M1 三折计划起点重跑。不得在失败目录 resume。

三折全部结束并完成 `scripts/audit_cv3_oof.py` 后，状态才可写
`complete_downstream_ready` 且 `downstream_admission=true`。只有训练完成、
但 aggregate 未通过，不能向下游交付。

### 2.7 E-10K：M1 工程基线

当 M1 三折均完成、正式 aggregate 为
`complete_downstream_ready`，且选定折的 checkpoint 已被
`fold_metadata.json` 与 `oof_metadata.json` 双重验收后执行：

```text
docs/server/E_10K_PIPELINE_TASK.md
```

本轮选定的 CV3 某折 checkpoint 只作工程测速，结果必须明确标记
`engineering_checkpoint_only=true`。synthetic、stitched 或 project proxy
图不得写成官方时延通过。当前官方 10K manifest 注册表为空，
且使用的是折 checkpoint，因此本轮 E 无论速度如何都只形成工程
证据。GPU 型号必须记录但不作为代码门禁；不同 GPU 结果不直接
横向归因。官方声明需另立“注册官方 manifest + 最终 checkpoint +
独占 GPU”的任务。

### 2.8 M3-OOF：异构检测器

依次阅读并执行：

```text
docs/server/CV3_OOF_COMMON_CONTRACT.md
docs/server/M3_CV3_OOF_TASK.md
```

M3 完成后只形成同协议 OOF，不在服务器上自行宣告集成入选。

### 2.9 M1/M3 正式 OOF 后处理分析

M1 与 M3 的 aggregate 均为 `complete_downstream_ready` 后，执行：

```text
docs/server/M1_M3_CV3_OOF_ANALYSIS_TASK_01.md
```

该 CPU 任务统一完成官方 Recall/FDR、阈值曲线、计数守恒错误分解、
paired TP/FN 和 oracle-union；服务器只能回报预注册结果，不自行宣布最终
模型入选。

## 3. 单 GPU 的资源原则

- GPU 训练和 10K 测速串行；
- CPU 的 F00→D00、打包、SHA 和轻量汇总可与空闲 GPU 阶段错开；
- 测速时必须确认无其他 GPU 进程；
- 不在正式测速期间运行 P03/P04/M1/M3；
- 任一正式训练或测速出现 OOM，保留日志并停止当前任务；不得在同一任务 ID
  下变更 batch、模型、分辨率或 epoch 后继续；
- 不允许用 fold 0 checkpoint 继续训练 fold 1/2。

若优先需要真实 OOF 解锁 P05/P06，可把 M1 提到 P03/P04 之前。调整排队顺序
不允许调整任何冻结配置。

## 4. 每个子任务完成后的统一回报

服务器 AI 每完成一个子任务，按以下顺序回报：

1. 科学状态与技术状态分开；
2. Git commit/dirty、代码 SHA 门禁和测试数量；
3. 数据、权重、cache、checkpoint 的完整 SHA；
4. 实际环境和 GPU；
5. smoke 状态；
6. 正式 run 完成数、失败与重试；
7. 主要指标或工程读出；
8. 停止条件是否触发；
9. 大型资产的服务器保留路径；
10. 小型回传包路径、大小和 SHA。

不得只发训练日志中的最后一行，也不得把 smoke 指标混入正式平均。

## 5. 回传包

每个子任务单独打包，不把大型 cache/checkpoint 重复放入压缩包。总调度目录
只保存：

```text
CV3-FORMAL-STAGE/
├── system_preflight.txt
├── formal_crop_path.txt
├── child_task_register.csv
├── checkpoint_register.csv
├── return_package_register.csv
└── final_stage_status.json
```

登记表至少包含 task ID、状态、使用的 manifest SHA、代码 commit、结果路径、
checkpoint/cache SHA、回传包 SHA 和本地验收状态。

两份代码锁的职责不同：

- `CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt`：主仓库本阶段的数据、实验、审计
  与测试实现；
- `XH_MODEL_INTEGRATION_CODE_SHA256.txt`：同级模型仓库真实训练、推理、
  adapter、切片与融合实现。

任一不匹配均应停止并回传差异；不得在服务器临时修改后继续同一任务 ID。

## 6. 当前不执行

以下内容即使服务器尚有空闲 GPU，也不能在本阶段自行启动：

- P05 正式 hard negative；
- P06 真实框修正或 bbox diffusion；
- DINO/CleanDIFT 微调、LoRA 或新层搜索；
- M1/M3 集成阈值搜索；
- rare-rebalance、HPR、M2 多因素网格；
- 使用全 OOF 拟合阈值后在同一 OOF 宣称无偏成绩。

它们都等待 M1/M3 OOF 的正式错误分析或后续独立任务单。
