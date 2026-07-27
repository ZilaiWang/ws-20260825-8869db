# CV3 正式实验执行总纲 v1

更新日期：2026-07-23  
状态：`implemented_ready_for_server_validation`

> 历史执行合同说明（2026-07-25）：F00/D00/A00 与正确 YOLO26-s 的 M1
> 正式三折 OOF 已完成。本文件继续作为冻结配置、依赖和 lineage 合同；M1
> 结果之后的科学优先级和实际执行顺序，以
> [`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)
> 为当前入口。

## 1. 本轮要解决什么

正式分组 `cv3_airport_proxy_k60_v2` 已冻结，本轮不再继续搜索划分或扩大
模型网格，而是把已有探索结论转化为来源隔离、可复现、可追溯的正式证据。

本轮包含五项实验、一个公共输入任务、一个正式检测数据字节锁任务，以及一个
共用模型资产/环境冻结任务：

| ID | 内容 | 主要问题 | 当前代码状态 |
|---|---|---|---|
| F00 | 正式 CV3 消费层与 crop v2 | 所有支线是否读取完全相同的 fold/group | 已实现，本地门禁通过 |
| D00 | 正式检测数据字节锁 | 训练读取的 4,481 图像/标签字节与 20,933 GT 是否完全冻结 | 已实现，本地门禁通过 |
| A00 | M1/M3 共用环境与官方权重冻结 | 训练/推理是否使用同一已登记环境和权重字节 | 已实现，待服务器创建不可变锁 |
| P03-F | tight-224 ConvNeXt 三折复验 | 给定对象区域的细分类上限是否仍成立 | 已实现，待服务器 GPU |
| P04-F | 三教师 frozen-feature 三折复验 | DINOv2、ConvNeXt、CleanDIFT 正式排序 | 已实现，待服务器 cache/GPU |
| M1-OOF | YOLO26-s/1024 三折检测 OOF | 快速主检测器的正式 Recall/FDR 与错误来源 | 已实现，待服务器 GPU |
| M3-OOF | RT-DETR-L/1024 三折检测 OOF | 异构检测器是否提供独立 TP | 已实现，待服务器 GPU |
| E-10K | 10K 切片、恢复、融合与分段测速 | 完整系统能否满足 20 秒工程约束 | 已实现合同与审计，待真实运行 |

这里的“已实现”表示代码、冻结配置、自动门禁、测试和服务器任务单均已写好，
不表示已经产生正式实验结果。

## 2. 为什么本轮收缩为这些工作点

此前探索实验已提供足够的方向证据：

- P03：ConvNeXt-Tiny 全量微调的 tight-224 macro recall 约
  `0.9703±0.0078`；336 的收益很小，context 更弱；
- natural 与 sqrt-inverse 基本并列，后者方差更大；
- 三个 seed 均值差小于 0.003，因此正式复验不再遍历 seed；
- P04：DINOv2-B CLS+patch `0.9098`、ConvNeXt train-RMS `0.8797`、
  CleanDIFT map0 `0.8293`，DINO-S 与 CleanDIFT map6/map9 没有继续扩大的
  依据；
- P05 的易背景抽样未通过人工纯背景门禁；
- P06 只在合成误差上证明了任务可学，`real_system_admission=false`；
- P07 的 SD1.5 背景融合达到停止条件。

因此本轮的原则是：

1. 不重开分辨率、sampler、seed、扩散层、模型规模等大网格；
2. 先建立真实 OOF，再决定 P05/P06 是否值得继续；
3. 扩散模型只保留为 P04 正式对照，不因名称新颖而自动入选；
4. 10K 时延从一开始按完整流水线计量，不以 model-only 时间代替。

## 3. 唯一正式数据合同

### 3.1 正式 CV3

```text
version: cv3_airport_proxy_k60_v2
manifest: data/splits/cv3_airport_proxy_k60_v2.json
sha256: 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
images: 4481
objects: 20933
source groups: 255
held-out images: 1507 / 1613 / 1361
held-out objects: 7350 / 7179 / 6404
held-out groups: 82 / 95 / 78
```

一次实验中，`fold == held_out_fold` 只作训练后评估或 OOF 推理，其余两个
fold 只作训练。同一 `group_id` 不得跨训练/评估；held-out fold 不参与逐轮
验证、early stop 或 checkpoint 选择。

机场代理组是经过视觉证据构造的来源域代理，不是真实机场标签。任何报告都应写
“airport-proxy/source-group isolated”，不能写成已获得真实 airport-disjoint
ground truth。

### 3.2 正式 crop

```text
version: formal_crop_manifest_v2
sha256: a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
rows: 62799
annotations: 20933
source images: 4481
policies: tight / context_1p25 / jitter_light
```

它由原 P0-2 manifest 纯 metadata 重挂得到：

- `crop_id`、`annotation_uid`、框、裁剪几何及 canonical 输入不变；
- active `fold/group_id/leakage_group_id` 只来自正式 CV3；
- 旧探索字段仅保留为 `historical_p02_*`；
- 构建时不读取任何图像像素。

### 3.3 两种消费视图的关系

F00 输出六个框架无关 CSV，便于审计和普通分析。M1/M3 因 C 的 loader
只识别 `split=train/val`，还会从同一正式 manifest 逐折生成 JSON view。

两者不是两个数据源：

```text
同一 CV3 manifest
├── formal_cv3_fold{0,1,2}_{train,val}.csv   审计/通用消费
└── M1/M3 fold_{0,1,2}/split_view.json       C 模型 loader 适配
```

JSON view 只增加 `split` 与 `source_fold`，不得改变 image ID、路径、
group 或 fold。

## 4. 实验依赖与执行图

```text
F00 正式公共输入（CPU） ─┬─→ P03-F：对象 crop 全量微调
                         ├─→ P04-F：冻结教师特征读出
                         └─→ D00 正式检测数据字节锁
                                  ├─→ M1-OOF
                                  └─→ M3-OOF

A00 共用模型环境与官方资产冻结 ─┬─→ M1-OOF：YOLO26-s 三折
                              ├─→ M3-OOF：RT-DETR-L 三折
                              └─→ E-10K

M1-OOF
├── OOF 完整性审计
├── 官方 Recall/FDR 与真实错误分解
├── P05/P06 准入判断
└── E-10K 的首个模型工程基线

M3-OOF
└── 与 M1 配对及 oracle-union

M1/M3 正式证据汇合
└── cross-fit 阈值、候选召回曲线、最终模块选择
      └── 最终模型冻结后再次运行 E-10K
```

F00→D00 是数据依赖链；A00 与这条 CPU 链彼此独立，可并行准备。M1/M3
同时以 D00 与 A00 为前置。P03/P04 与 M1/M3 在科学上可并行；只有一张
GPU 时按任务单串行即可。
P05/P06 不在本轮无条件启动。

A00 只建立一次：冻结 Python 3.10.12、PyTorch 2.5.1+cu121、
Ultralytics 8.4.103、批准的 RTX 4080 SUPER，以及两个官方初始化权重的
URL、字节数和 SHA-256。M1、M3、E-10K 每次运行前都必须重算并通过同一
`MODEL_ASSET_ENV_LOCK.json`，不能在子任务中用“对当前文件现场取 SHA”
替代预先登记的可信常量。

D00 也只创建一次，其唯一锁为：

```text
/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
SHA256 03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
```

它绑定 4,481 图像、4,481 YOLO 标签、20,933 GT 及 formal crop 等价性。
M1/M3 的每一折开训前都必须重新做全量 verify，并把报告纳入 fold metadata。

## 5. F00：正式公共输入

执行文件：

- `docs/server/FORMAL_CV3_CROP_TASK_01_CPU.md`
- `configs/analysis/formal_cv3.yaml`
- `configs/analysis/formal_crop_manifest.yaml`
- `scripts/build_formal_cv3_views.py`
- `scripts/build_formal_crop_manifest.py`

本地已得到确定性参考：

```text
formal crop SHA:
a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128

formal crop bytes:
71,995,981

fold train/val images:
2974/1507, 2868/1613, 3120/1361
```

服务器必须用两个全新目录独立生成并逐字节比较正式 CSV。F00 的任务终点是
数据合同通过，不是训练分数。

## 6. P03-F：正式对象 crop 分类上限

冻结工作点：

```text
model: torchvision ConvNeXt-Tiny
initialization: ImageNet-1K V1
input: tight-224
regime: full fine-tune
sampler: natural
seed: 42
epochs: fixed 30
checkpoint selection: final epoch
held-out use: no per-epoch selection; one formal external evaluation after training
folds: 0/1/2
```

只运行三次。每折从同一 ImageNet 权重独立初始化，不复用上一折 checkpoint。

主要读出：

- 每折 macro recall/F1、accuracy、aircraft20 recall；
- mean±sample std；
- 20,933 对象 pooled OOF；
- 25 类、三大类和固定 9/8/8 support tier；
- TU-160 的 `train=9 / val=352` 压力折必须单列。

P03 回答的是“给定真实对象区域后能分多好”，不能作为端到端检测成绩。

任务单：`docs/server/P03_FORMAL_CV3_V2_REPLAY.md`。

## 7. P04-F：正式教师表征比较

六个条件、每个三折，共 18 个 probe：

| 教师 | native | 容量控制 |
|---|---:|---:|
| ConvNeXt-T `convnext_gap` | 768D | PCA384 |
| DINOv2-B `dino_cls_patchmean` | 1536D | PCA384 |
| CleanDIFT `clean_map0` | 1280D | PCA384 |

所有条件共享：

- 同一 20,933 对象和 canonical224；
- 同一 D4 八视图；
- `train_rms`；
- `p04_default` 线性头；
- seed 42；
- 固定跑满 15 epoch，使用 final-epoch checkpoint；
- held-out fold 不参与逐轮验证、early stop 或 checkpoint 选择；
- PCA、RMS 与分类头均逐折只在训练侧拟合；
- 验证只使用 r0。

cache 复用门禁必须同时核验 UID、crop ID、canonical224 SHA、八视图完整性、
shard 与有限性。只比较对象数量不能证明 cache 可复用。

native 与 PCA384 分开报告。CleanDIFT 只有在正式三折对尾类或稳定困难对象
提供 DINOv2 之外的一致价值时才保留。

任务单：`docs/server/P04_FORMAL_CV3_V2_REPLAY.md`。

## 8. M1-OOF：正式快速主检测器

冻结工作点：

```text
model: YOLO26-s
input: 1024
foundation epochs: 160
checkpoint selection: last
training validation: disabled
seed: 42
low candidate threshold: 0.001
max detections: 500
rare rebalance/HPR: disabled
```

三个 fold 必须：

1. 从同一个实际预训练权重文件及 SHA 独立开始；
2. 禁止 `resume` 和跨折 checkpoint；
3. 训练与低阈值推理指向同一 fold view；
4. 输出本折所有图，零预测图也必须在 image ledger 中出现；
5. 最终 4,481 张图各有且仅有一次 held-out 归属。

held-out fold 不参与逐 epoch 验证、early stop、checkpoint 选择或训练期
调参；每折固定跑满 160 epoch，选择 `foundation/weights/last.pt` 后才做
一次正式外部 OOF 推理。Ultralytics 可能在最终 epoch 额外运行一次框架内部
终局验证；该辅助读出不能参与任何决策，也不能作为本项目正式指标。

标准汇总产物：

```text
oof_metadata.json
oof_images.csv
oof_proposals.csv
predictions_oof_low.json
```

该四件套与 formal crop 合并后，可解锁真实 P06 输入。训练日志中的
Ultralytics mAP 不能代替官方 Recall/FDR。

公共合同：`docs/server/CV3_OOF_COMMON_CONTRACT.md`。  
任务单：`docs/server/M1_CV3_OOF_TASK.md`。

## 9. M3-OOF：正式异构对照

冻结工作点：

```text
model: RT-DETR-L
input: 1024
foundation epochs: 120
checkpoint selection: last
training validation: disabled
seed: 42
low candidate threshold: 0.001
max detections: 300
```

数据、初始化、低阈值输出及 OOF 完整性合同与 M1 相同。M3 的去留不能只看
单模 mAP，应在同一 GT 上报告：

- 共同 TP、仅 M1 TP、仅 M3 TP、共同 FN；
- 对同一 GT 的定位 IoU 与细类分歧；
- 背景 FP 的交集与并集；
- oracle-union 上限；
- 三大类、尺寸、边界、tier 与 TU-160。

M3 若较慢但提供稳定独立 TP，优先研究困难对象门控，而不是默认全量双模型。
每折固定跑满 120 epoch；held-out fold 不参与逐轮验证、early stop、选模或
训练期调参，使用固定 `last.pt` 生成一次正式外部 OOF。若框架在最终 epoch
产生辅助终局验证读出，只允许留档，不得据此改变模型或报告正式成绩。

任务单：`docs/server/M3_CV3_OOF_TASK.md`。

## 10. E-10K：完整流水线工程验证

冻结首轮几何：

```text
image: 10000x10000
tile: 1280
overlap: 256
stride: 1024
tile count: 100
fine NMS: 0.55
coarse duplicate NMS: 0.85
candidate threshold: 0.001
```

分段必须包含：

```text
image_read
tiling
preprocess
model
tile_postprocess
coordinate_restore
fusion
serialization
```

每次 GPU 计时前后执行 CUDA synchronize；三次预热不计入统计，正式至少
十次。报告各阶段、model-only、`total_after_read` 和包含读盘的 wall
p50/p95/max。

官方硬门槛的审计口径是：

```text
每一个 measured run 的 total_after_read <= 20.0 秒
```

不是均值或 p95 通过即可。synthetic/stitched/proxy 图只能形成工程证据。
即使填写 `real_official`，也必须同时满足：图像 manifest 已进入代码内官方
注册表、硬件是 RTX 3090、GPU 无其他计算进程、使用最终冻结 checkpoint 且
`engineering_checkpoint_only=false`，才允许标记为官方时延声明候选。当前
4080 SUPER、fold checkpoint 和模板默认值只能形成工程证据。

任务单：`docs/server/E_10K_PIPELINE_TASK.md`。

## 11. 同一服务器的推荐顺序

只有一张 GPU 时，推荐：

1. **F00（CPU）**：先冻结所有后续共同输入；
2. **A00（一次）**：建立 M1/M3/E 共用环境与不可变资产锁；
3. **D00（CPU）**：在 F00 后冻结正式图像、标签与 GT 字节；
4. **P04-F**：复用现有 cache，运行成本最低，也能尽早发现 cache 是否可复用；
5. **P03-F**：三折 tight-224，完成对象级正式上限；
6. **M1-OOF**：进入核心检测关键路径；
7. **M1 OOF 完整性审计**；
8. **E-10K M1 工程基线**：用明确标记的可用 M1 checkpoint；
9. **M3-OOF**；
10. **M1/M3 配对**；
11. 最终模型与阈值冻结后，再做一次 **E-10K 最终复测**。

若团队正在等待真实错误样本，可将 M1 提前到 P03/P04 之前；这不改变科学
合同，只改变 GPU 排队顺序。

## 12. 全局停止条件

以下任一情况不得自行放宽：

- 正式 CV3、formal crop 或预训练权重 SHA 不匹配；
- D00 数据锁、任一图像/标签字节或 GT 等价性不匹配；
- 来源组跨训练/验证；
- 任一 fold 续训其他 fold checkpoint；
- 训练和推理使用不同 split view；
- P04 的 PCA/RMS/head 读取验证侧；
- OOF 未覆盖全部 4,481 图或一图被覆盖多次；
- 低阈值候选被隐藏阈值提前删除；
- 10K 图像来源、模型、配置、checkpoint 或计时方法无法追溯；
- OOM 后擅自修改 batch、图像尺寸、epoch、模型或科学超参数。

任一正式 run OOM 时保留日志并停止该任务。若确需改变 batch，必须另立任务
ID、更新合同并重新执行整套配对矩阵，不能静默重跑。

## 13. 结果回来后的决策顺序

M1 OOF 回来后依次进行：

1. OOF 完整性与官方评估；
2. 阈值无关候选召回曲线；
3. cross-fit 阈值，禁止在同一 OOF 全集拟合又宣称无偏；
4. 计数守恒的 `FP_BG/FP_DUP/FP_FINE_CLS/FP_LOC` 与
   `FN_FINE_CLS/FN_LOC/FN_MISSING`；
5. `R_loc@oracle-class`、`Acc_fine@localized`、边界、尺寸、tier；
6. 再决定是否放行 P05 或 P06-REAL。

M3 OOF 到达后再做 paired/oracle-union。P06-DIFF 仍需等待真实定位错误和
确定性 refiner 强基线，不能因为当前扩散代码可写就提前训练。

上述官方评估、错误分解与 M1/M3 配对已经实现为
`M1-M3-OOF-ANALYSIS-TASK-01`；输入未到达时只允许保持 waiting，不另写一套
临时统计逻辑。

## 14. 文件索引

### 公共数据层

- `docs/server/FORMAL_CV3_CROP_TASK_01_CPU.md`
- `docs/server/CV3_DETECTION_DATA_LOCK_TASK_00.md`
- `configs/experiments/formal_detection_data_lock.json`
- `src/rsdet/data/formal_cv3.py`
- `src/rsdet/analysis/formal_crop.py`
- `src/rsdet/experiments/detection_data_lock.py`
- `scripts/build_formal_cv3_views.py`
- `scripts/build_formal_crop_manifest.py`
- `scripts/lock_formal_detection_data.py`

### P03/P04

- `reports/experiments/P03-P04-FORMAL-CV3-V2-REPLAY-PLAN.md`
- `docs/server/P03_FORMAL_CV3_V2_REPLAY.md`
- `docs/server/P04_FORMAL_CV3_V2_REPLAY.md`
- `configs/experiments/p03_formal_cv3_v2.yaml`
- `configs/experiments/p04_formal_cv3_v2.yaml`
- `src/rsdet/analysis/formal_replay.py`
- `scripts/audit_p03_p04_formal_inputs.py`
- `scripts/freeze_p03_formal_config.py`
- `scripts/train_crop_classifier.py`
- `scripts/train_p04_feature_probe.py`
- `scripts/summarize_p03_p04_formal.py`

### M1/M3 OOF

- `docs/server/CV3_MODEL_ASSET_ENV_TASK_00.md`
- `configs/experiments/cv3_model_asset_env.json`
- `src/rsdet/experiments/model_asset_env_lock.py`
- `scripts/lock_cv3_model_assets.py`
- `docs/server/CV3_OOF_COMMON_CONTRACT.md`
- `docs/server/M1_CV3_OOF_TASK.md`
- `docs/server/M3_CV3_OOF_TASK.md`
- `src/rsdet/experiments/cv3_oof.py`
- `scripts/prepare_cv3_oof.py`
- `scripts/materialize_cv3_oof_config.py`
- `scripts/finalize_cv3_oof_fold.py`
- `scripts/audit_cv3_oof.py`
- `configs/experiments/m1_yolo26s_1024_cv3_oof.template.yaml`
- `configs/experiments/m1_yolo26s_1024_cv3_oof_infer.template.yaml`
- `configs/experiments/m3_rtdetr_l_1024_cv3_oof.template.yaml`
- `configs/experiments/m3_rtdetr_l_1024_cv3_oof_infer.template.yaml`

### 10K

- `docs/server/E_10K_PIPELINE_TASK.md`
- `configs/experiments/e_10k_pipeline_cv3.template.yaml`
- `configs/experiments/e_10k_benchmark_contract.template.json`
- `src/rsdet/experiments/runtime_10k.py`
- `scripts/benchmark_10k_pipeline.py`
- `scripts/audit_10k_runtime.py`

### OOF 官方评估、错误分解与模型配对

- `reports/experiments/M1_M3_CV3_OOF_POSTPROCESS_ANALYSIS_PLAN_v1.md`
- `docs/server/M1_M3_CV3_OOF_ANALYSIS_TASK_01.md`
- `configs/experiments/m1_m3_cv3_oof_analysis_v1.yaml`
- `src/rsdet/analysis/oof_detection.py`
- `scripts/analyze_cv3_oof_models.py`
- `tests/test_oof_detection_analysis.py`

### 代码与跨仓库集成锁

- `docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt`
- `docs/server/XH_MODEL_INTEGRATION_CODE_SHA256.txt`

### 自动测试

- `tests/test_formal_cv3.py`
- `tests/test_formal_crop.py`
- `tests/test_detection_data_lock.py`
- `tests/test_p03_p04_formal_replay.py`
- `tests/test_model_asset_env_lock.py`
- `tests/test_cv3_oof.py`
- `tests/test_runtime_10k.py`

### 同级模型仓库生产实现

- `../xh-202625-model/scripts/train.py`
- `../xh-202625-model/scripts/infer.py`
- `../xh-202625-model/src/rsdet/engine/trainer.py`
- `../xh-202625-model/src/rsdet/engine/predictor.py`
- `../xh-202625-model/src/rsdet/models/ultralytics_adapter.py`
- `../xh-202625-model/src/rsdet/tiling/`
- `../xh-202625-model/src/rsdet/postprocess/`
- `../xh-202625-model/tests/test_ultralytics_adapter.py`
- `../xh-202625-model/tests/test_inference_pipeline.py`
- `../xh-202625-model/tests/test_tile_fusion.py`
- `../xh-202625-model/tests/test_trainer_contract.py`

## 15. 本地交付验收记录

本轮代码与任务单完成后，已在 2026-07-23 做最终本地验收：

- 主仓库全量测试：`258 passed, 4 skipped`；
- 同级模型仓库全量测试：`124 passed`；
- 两仓 Ruff：全部通过；
- 主仓与模型仓中央 SHA-256 锁：全部通过；
- F00、D00、M1/M3 OOF 分析三个专项 SHA-256 锁：全部通过；
- 11 份核心服务器任务单中的 60 个 Bash 代码块：`bash -n` 全部通过；
- 数据、实验和服务器文档共 73 份 Markdown：本地链接 0 个断链；
- `SERVER_ARTIFACT_REGISTER.csv`：17 行、12 列，字段数和 CSV 语法通过；
- D00 使用完整 4,481 图像、4,481 标签和 20,933 GT 重新构建并独立
  verify，锁 SHA-256 仍为
  `03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`。

这些结果证明当前交付物在本机代码与数据条件下自洽；不替代服务器上的
CUDA、真实权重加载、正式三折训练、正式 OOF 和真实 10K 测速。服务器仍须
逐项执行任务单，不得把本地测试通过写成正式实验完成。
