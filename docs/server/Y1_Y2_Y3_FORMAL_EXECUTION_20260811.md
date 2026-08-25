# Y1-Y3 正式执行单

## 1. 执行顺序

1. Y1：本地 CPU 严格 cross-fit 校准，已完成；
2. Y2：先执行 `YOLO_FAST_SCREEN_PIPELINE_20260811.md` 的配对快筛；仅快筛入围后，
   才运行真正的 `yolo26s-p2.yaml` 三折 160 epoch/fixed-last 正式 OOF；
3. Y3：只在 `Y2-P2-DECISION.json` 准入后，训练 P2 颈部单对 IBS 采样变体。

Y2 和 Y3 不允许同时启动。Y3 入口会机器校验 Y2 决策；未准入时必须停止，不训练。

## 2. 已发现并修正的命名问题

Ultralytics 8.4.103 的 P2 s 级模型名是 `yolo26s-p2.yaml`。
历史脚本写成 `yolo26-p2s.yaml`，实际会警告“no model scale
passed”并回退到 n 级：

| 名称 | 实际规模 | 参数量 |
|---|---|---:|
| `yolo26-p2s.yaml` | n（错误回退） | 2,662,400 |
| `yolo26s-p2.yaml` | s（Y2 冻结） | 9,765,856 |

因此历史 P2 仍只是机理证据，Y2 才是与 M1 YOLO26-s 容量对齐的单因素对照。

## 3. 服务器前置

- Python 3.10.12；
- torch 2.5.1+cu121；
- torchvision 0.20.1+cu121；
- **ultralytics 8.4.103**；
- GPU 与正式 CV3 训练环境一致；
- 正式 CV3、P0-2、formal crop、detection data lock 与 `yolo26s.pt`均保留在服务器。

所有折从同一份 `yolo26s.pt` 独立开始；禁止 resume、禁止 best checkpoint、禁止用 held-out 折选 epoch。

## 4. 执行 Y2

先设置路径，所有变量必须指向已存在的真实资产：

```bash
export PROJECT_ROOT=/workspace/xh-202625
export DATA_ROOT=/workspace/data
export RESULTS_ROOT=/workspace/results
export PYTHON_BIN=/workspace/venvs/cv3-model-cu121/bin/python
export PRETRAINED_WEIGHT=/workspace/cv3-model-assets/yolo26s.pt
export DATA_LOCK=/workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00/formal_detection_data_lock.json
export P02_MANIFEST=/workspace/path/to/P0-2/exploration_crop_manifest.csv
export FORMAL_CROP=/workspace/path/to/formal_crop_manifest.csv
export M1_AGGREGATE_ROOT=/workspace/results/M1-CV3-OOF-aggregate

bash scripts/server/run_y2_y3_formal_cv3.sh y2
```

若已上传本地 Y1 完整结果，也可以不设
`M1_AGGREGATE_ROOT`，改设
`M1_CALIBRATION_RESULT=/path/to/calibration_result.json`。两者都存在时优先使用已给定的结果文件。

脚本按 fold0→fold1→fold2 串行，每折顺序为：数据锁复验→配置解析→构图门禁→训练→低阈值推理→fold 冻结。三折后才做 aggregate、Y1 同协议评估和 Y2 准入。

## 5. Y2 验收

必须同时满足：

- 三个 `results.csv` 均 160 行；
- 三个 `last.pt`，且都从冻结 `yolo26s.pt` 独立开始；
- `architecture_audit.json` 为 9,765,856 参数、Detect stride 4/8/16/32、四输入；
- 4481 图一次且仅一次 held-out OOF 覆盖；
- aggregate `status=complete_downstream_ready`；
- Y2 C0 与 M1 C0 比较，校准不参与 P2 结构因果判断。

Y2 预注册准入要求：

- 官方 Recall/FDR 硬门槛通过；
- pooled Recall 相对 M1 下降不超过 0.005；
- pooled FDR 增加不超过 0.01；
- macro Recall 下降不超过 0.005；
- vehicle pooled Recall 提升至少 0.02，且至少 2/3 折同方向。

## 6. 执行 Y3

只有 `Y2-P2-DECISION.json` 中
`p2_structure_admission=true` 且 `quality_stage_admission=true` 时：

```bash
export Y2_DECISION=/workspace/results/Y2-P2-DECISION.json
export P2_CALIBRATION_RESULT=/workspace/results/Y2-P2-CALIBRATION/calibration_result.json
bash scripts/server/run_y2_y3_formal_cv3.sh y3
```

Y3 唯一变量是：

- layer17：P3→P2 nearest upsample 改为 IBS-U；
- layer20：P2→P3 stride convolution 改为 IBS-D。

不同时加 SFRCF，不更换 backbone，不改 loss，不改训练时长或数据增强。Y3 未准入时删除该分支，不继续在同一分支堆 SFRCF。

## 7. 回传

脚本不打包 checkpoint，只回传配置、审计、OOF 预测、指标、决策和日志：

- Y2：`P2-FORMAL-CV3-return-no-checkpoints.tar.gz`；
- Y3：`Y3-FORMAL-CV3-return-no-checkpoints.tar.gz`；
- 同名 `.sha256`。

三折 checkpoint 留服务器，本地准入审核完成前不删除。
