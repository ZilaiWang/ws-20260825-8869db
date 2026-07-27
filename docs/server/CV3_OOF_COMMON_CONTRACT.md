# M1/M3 正式 CV3 OOF 公共合同

状态：可执行  
适用模型：M1 YOLO26-s/1024、M3 RT-DETR-L/1024  
唯一正式划分：`cv3_airport_proxy_k60_v2`

## 1. 冻结输入

```text
manifest:
  data/splits/cv3_airport_proxy_k60_v2.json
sha256:
  27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
images:
  4481
folds:
  3
```

每次选择一个 `held_out_fold`：

- `fold == held_out_fold`：不参与训练期逐轮验证、early stop、checkpoint
  选择或调参；固定 epoch 训练完成后用 `last.pt` 做一次正式外部 OOF
  推理。框架若在最终 epoch 产生辅助终局验证，只能留档，不能进入任何
  决策或正式指标；
- 其余两个 fold：只作训练；
- 同一 `group_id` 不跨折；
- 三次运行后，每张图必须恰好产生一次 OOF 归属。

旧 `cv3_airport_proxy_k60_v1`、`dev_v1` 和 `dev_v2` 均不得混入正式
OOF。

正式训练还必须消费 D00 生成的唯一数据字节锁：

```text
path:
  /workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
sha256:
  03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
```

该锁逐字节绑定 4,481 图像、4,481 标签和 20,933 个 GT。仅 manifest 分组
正确不足以开训；每折训练前都必须按
`CV3_DETECTION_DATA_LOCK_TASK_00.md` 第 6 节重新做全量 verify。

## 2. 为什么生成 split view

C 的同级模型仓库 `xh-202625-model` 当前 loader 只读取
`sample.split=train/val`，而正式 CV3 使用 `sample.fold=0/1/2`。
本任务不修改 C 的大段训练代码，而是由主仓库为每折生成只读视图：

```text
CV3 fold manifest
  → fold_0/split_view.json
  → fold_1/split_view.json
  → fold_2/split_view.json
```

视图只增加 `split` 和 `source_fold`，不移动图像、不改 `image_id`、
`relative_path`、`group_id` 或标签。训练/推理 resolved config 必须指向
同一份视图，工具会按 SHA 检查，指错 fold 会立即失败。

## 3. 每折必须独立初始化

三折可以顺序运行，但不得连续微调：

1. 三折使用同一个官方原预训练权重文件；
2. 计划生成时即核验该文件的真实 SHA256；
3. 每折训练配置必须再次指向该绝对路径并通过下列预登记可信常量复核；
4. 禁止 `--resume`；
5. 禁止把 fold 0 的任何 checkpoint 作为 fold 1/2 初始权重；
6. 只运行冻结的 `foundation` 阶段；rare-rebalance 和 HPR 不进入本轮。

```text
M1 yolo26s.pt
  path: /workspace/cv3-model-assets/yolo26s.pt
  bytes: 20422725
  sha256: 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b

M3 rtdetr-l.pt
  path: /workspace/cv3-model-assets/rtdetr-l.pt
  bytes: 66511432
  sha256: 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
```

这些值同时受 A00 的 `MODEL_ASSET_ENV_LOCK.json` 约束。不得以
`PRETRAINED_SHA="$(sha256sum 当前文件)"` 把未知文件的现场摘要当作可信
期望值。

每折可使用相同 seed 42。这里比较的是不同 held-out group，而不是把 fold
当作随机种子。

## 4. 低阈值预测合同

所有 OOF 候选使用：

```text
confidence = 0.001
ship/aircraft/vehicle output threshold = 0.001
fine_score_thresholds = {}
coordinates = original-image absolute COCO xywh
category_id = 0..24
```

M1 `max_detections=500`；M3 受 object query 数约束，
`max_detections=300`。不得先应用正式工作点阈值、25 类阈值、HPR 或
二阶段精修。零预测图仍必须在 `oof_images.csv` 中保留。

## 5. 标准执行骨架

先在主仓库生成计划。以下变量由服务器实际路径替换：

```bash
cd /workspace/xh-202625
export PYTHONPATH=src

MANIFEST=data/splits/cv3_airport_proxy_k60_v2.json
MANIFEST_SHA=27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
# M1 使用以下两行；M3 换成上文登记的 rtdetr-l 常量。
PRETRAINED=/workspace/cv3-model-assets/yolo26s.pt
PRETRAINED_SHA=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
DATA_LOCK=/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
DATA_LOCK_SHA=03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
test "$(stat -c %s "$PRETRAINED")" = 20422725
echo "$PRETRAINED_SHA  $PRETRAINED" | sha256sum -c -
echo "$DATA_LOCK_SHA  $DATA_LOCK" | sha256sum -c -

python scripts/prepare_cv3_oof.py \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA" \
  --output-dir /workspace/results/模型键-CV3-OOF \
  --model-key M1或M3 \
  --model-family yolo或rtdetr \
  --model-name 模型名 \
  --seed 42 \
  --input-size 1024 \
  --foundation-epochs 160或120 \
  --low-score-threshold 0.001 \
  --max-detections 500或300 \
  --pretrained-weight "$PRETRAINED" \
  --pretrained-weight-sha256 "$PRETRAINED_SHA" \
  --detection-data-lock "$DATA_LOCK" \
  --detection-data-lock-sha256 "$DATA_LOCK_SHA"
```

每折用 `scripts/materialize_cv3_oof_config.py` 解析已提交的训练/推理模板，
不得手工编辑 resolved config。训练完成后执行：

```bash
python scripts/finalize_cv3_oof_fold.py \
  --plan /workspace/results/模型键-CV3-OOF/oof_run_plan.json \
  --fold "$FOLD" \
  --train-config "$FOLD_DIR/training/resolved_config.yaml" \
  --train-summary "$FOLD_DIR/training/train_summary.json" \
  --infer-config "$FOLD_DIR/resolved_infer.yaml" \
  --environment "$FOLD_DIR/environment.txt" \
  --checkpoint "$FOLD_DIR/training/runs/foundation/weights/last.pt" \
  --predictions "$FOLD_DIR/predictions_low.json" \
  --runtime "$FOLD_DIR/predictions_low.runtime.json" \
  --data-lock-verification \
    "$FOLD_DIR/input-gates/detection_data_lock_verification.json" \
  --output "$FOLD_DIR/fold_metadata.json"
```

`finalize` 会检查：

- train/infer config 指向本折 split view；
- seed、输入尺寸、foundation epoch 和模型族正确；
- `checkpoint_selection=last`、`val=false`、`patience=0`，held-out fold
  未参与逐轮验证、early stop、选模或训练期调参；若 Ultralytics 在最终
  epoch 仍执行框架内部终局验证，该读出只能留档，不能改变 checkpoint 或
  充当正式结果；
- 本折训练前的 D00 全量验证状态为 `pass`，且锁 SHA、两个 fingerprint
  被 fold metadata 闭环记录；
- runtime 的图像数等于该折全部验证图像数；
- runtime 使用 `rsdet_inference_runtime_v2`，并在其内部把实际 config、
  checkpoint、predictions 的绝对路径与内容 SHA256 闭环绑定；
- 预测 image ID 只能来自本折；
- `train_summary.json` 必须为 `dry_run=false`、只含 foundation，且
  `initial_weights`、stage `input_weights`、stage `last` 与
  `selected_checkpoint` 分别与冻结预训练权重和传入 checkpoint 闭环一致；
- infer config 的 `model.checkpoint`、`output_json` 必须分别就是传入的
  checkpoint 和 predictions 实体；
- 每图候选数不得超过模型计划上限；
- checkpoint/config/environment/prediction/runtime 均记录 SHA。

三折均完成后：

```bash
python scripts/audit_cv3_oof.py \
  --manifest "$MANIFEST" \
  --manifest-sha256 "$MANIFEST_SHA" \
  --plan /workspace/results/模型键-CV3-OOF/oof_run_plan.json \
  --run-root /workspace/results/模型键-CV3-OOF \
  --output-dir /workspace/results/模型键-CV3-OOF-aggregate \
  --formal-crop-manifest \
    /workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
```

正式 aggregate 必须一次性绑定唯一冻结
`formal_crop_manifest_v2`，其 SHA256 必须为
`a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128`。
没有该文件时不得在正式 aggregate 目录运行。若只需排错，必须改用全新的
`*-diagnostic-no-formal-crop` 目录并显式传
`--diagnostic-without-formal-crop`；该结果会写
`downstream_admission=false`，不得交给 P05/P06。

## 6. 汇总产物

```text
模型键-CV3-OOF-aggregate/
├── oof_metadata.json
├── oof_images.csv
├── oof_proposals.csv
└── predictions_oof_low.json
```

这是“某一个冻结检测器”的三折 OOF 四件套；与上游
`formal_crop_manifest_v2.csv` 合起来，才构成该模型的一份 P06 候选输入。
M1 与 M3 必须分别生成、分别审计，不能混成一个四件套。大型 checkpoint
不打入回传包，但服务器保留，回传其路径、大小和 SHA。

## 7. 停止条件

以下任一情况不得自行放宽：

- manifest、预训练权重或 split view SHA 不匹配；
- D00 数据锁、任一图像/标签字节或本折 verify 报告不匹配；
- 任一 fold 从其他 checkpoint resume；
- 训练/推理 config 指向不同 fold；
- 任一预测 image ID 不属于本折验证集；
- 三折合并后 4481 张图不是恰好覆盖一次；
- 输出使用高于 0.001 的隐藏阈值；
- NaN、越界框、类别越界或 OOM 后擅自改变科学配置。

任一正式 run 出现 OOM，保留日志并停止当前任务。若后续确需改变 batch，必须
由负责人登记新任务 ID 和新合同，不能在本任务内降级续跑。
