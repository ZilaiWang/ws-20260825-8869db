# 实验记录规范

模型训练框架可以不同，但正式结果必须使用同一评估协议、阈值规则和实验总表。
预测交接格式见 [`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md)。

## 1. 协议版本

当前版本以 [`configs/project.yaml`](../configs/project.yaml) 为唯一配置源：

- `contract_version: contract_v1`：预测、模型 adapter 和跨模块接口；
- `eval_version: official_eval_v1`：细类匹配、三大类汇总、IoU、Recall/FDR 计算规则。

接口语义变化时升级 `contract_version`，评估规则变化时升级 `eval_version`。包版本
`rsdet.__version__` 与两者独立。历史实验不改写；不同协议版本的结果不得直接横向比较。
`evaluate.py`、阈值扫描产物和正式实验总表都必须记录这两个版本。

## 2. 实验总表

总表为 [`reports/experiments/leaderboard.csv`](../reports/experiments/leaderboard.csv)，每个
正式工作点占一行。字段按以下五组填写：

- 身份：`experiment_id,date,status,git_commit,contract_version,eval_version`；
- 数据与模型：`dataset_version,split_version,seed,model_name,config_path,pretrained_weight,checkpoint_checksum`；
- 推理设置：`evaluation_scope,input_size,tile_size,tile_overlap,operating_point,score_threshold,threshold_stage`；
- 结果：`overall_recall,overall_fdr` 及 ship、aircraft、vehicle 的 Recall/FDR；
- 资源与追溯：`latency_p50,latency_p95,peak_vram,artifact_ref,notes`。

`status` 仅用 `exploratory/complete/failed/invalid`。完整正式实验必须有 commit、配置、
数据与划分版本、随机种子、预测和指标产物；只报 mAP 或 model forward 时间不能进入
正式比较。延迟统一用秒、显存统一用 GiB、权重校验值统一用 SHA-256。失败实验保留
一行简短结论，避免重复踩坑。

当前基线编号：`M1` 为主线快速 one-stage，`M2` 为同系列高容量或高分辨率版本，
`M3` 为 RT-DETR 类备选。实验 ID 示例：`E-M1-model-1024-devv1-seed42`。

## 3. 全局阈值扫描

先在足够低的候选门槛下交付原始 `score`，再在固定验证集上运行：

```bash
PYTHONPATH=src python scripts/sweep_thresholds.py \
  --gt outputs/dev_v1_gt.json \
  --pred outputs/实验ID/predictions.json \
  --output-dir outputs/实验ID/threshold_sweep
```

默认用同一个全局阈值从 0.00 扫到 1.00，步长 0.01，保留 `score >= threshold`。
正式大图结果在跨 tile 融合后扫描（`post_fusion`）；需要研究融合前行为时显式传
`--threshold-stage pre_fusion`，两者不得混记。

固定输出三个工作点：

- `official_best`：FDR ≤ 0.20 时 Recall 最高，门槛为 Recall ≥ 0.85；
- `internal_best`：FDR ≤ 0.17 时 Recall 最高，内部目标为 Recall ≥ 0.88；
- `recall_ceiling`：不限制 FDR 的 Recall 上限，只用于判断召回瓶颈。

并列时依次选择 FDR 更低、阈值更高的点。扫描直接复用官方评测器，不另写匹配逻辑。
输出包括完整曲线 `threshold_sweep.csv`、选择结果 `selected_thresholds.yaml` 和三个工作点
的完整指标 `metrics_at_selected_thresholds.json`。将使用的工作点、阈值、阶段和产物
目录写入 leaderboard；暂不支持 25 类独立阈值或学习式校准。

## 4. 实验产物

```text
outputs/YYYYMMDD-task-model-tag/
├── config.yaml
├── meta.json
├── metrics.json
├── runtime.json
├── predictions.json
├── train.log
├── threshold_sweep/
└── error_cases/
```

`outputs/` 不提交 Git；总表只记录小型结果和 `artifact_ref`。模型权重、大型日志和原始
数据不得提交。
