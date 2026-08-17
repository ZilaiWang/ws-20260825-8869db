# E 正式测速任务（E-FORMAL-BENCHMARK）

- 日期：2026-08-18
- 依赖：服务器（RTX 3090，独占 GPU）+ M1 正式 last.pt（已找回）+ 合成 10K 图（已生成 10 张）
- 一键执行：`bash scripts/server/run_e_benchmark.sh`
- 状态：**就绪，等开 GPU**

## 1. 目标与口径

把「10K ≤ 20 秒」从 E 成员的工程 smoke（1 warmup + 5 measured，best.pt）升级为
**正式采集**：M1 正式 last.pt（fold0，SHA d403ca0d…，engineering_checkpoint_only=false）
+ 3 warmup + 10 measured + 完整冻结合同。

**口径声明**：image_source_type=`synthetic`（官方未提供 real 10K 图）。
因此即使通过 20 秒，结论为「**工程正式化 smoke 通过**」；写「官方时延通过」
必须等 real_official 图（任务单 E_10K_PIPELINE_TASK.md 第 4 节约束）。

## 2. 前置资产（需先 scp 到服务器）

| 资产 | 服务器路径 | SHA256（必须匹配） |
|---|---|---|
| M1 fold0 last.pt | `/workspace/cv3-model-assets/m1_fold0_last.pt` | `d403ca0d…e19501d` |
| fold_0/fold_metadata.json | `/workspace/cv3-model-assets/fold_0_fold_metadata.json` | `b2bd717d…26af81` |
| oof_metadata.json | `/workspace/cv3-model-assets/oof_metadata.json` | `53b35f2c…bf4c6` |
| 合成 10K 图（≥10 张） | `/workspace/data/10k`（已生成，seed 42..51 内容互异） | manifest 校验 |

## 3. 流程（run_e_benchmark.sh，9 步）

1. 环境 preflight（venv + torch/ultralytics/yaml/PIL）；
2. 输入资产 SHA 门禁（上述 3 个 SHA 逐项校验，不符即退出）；
3. GPU 独占检查（存在其他 compute-app 即退出）+ 合成图数量检查；
4. 生成 `resolved_config.yaml`（模板替换）+ `image_manifest.json`（e_10k_image_manifest_v1）
   + `checkpoint_provenance.json`（checkpoint_provenance_v1，lineage 已验证）；
5. 现场采集 `hardware.json`（nvidia-smi + torch/cuda 版本 + 独占确认）；
6. 生成 `benchmark_contract.json`（runtime_10k_benchmark_v1，各 SHA 冻结）；
7. 正式采集：3 warmup + 10 measured（`benchmark_10k_pipeline.py`，全程 torch
   cuda synchronize 计时）；
8. `audit_10k_runtime.py` 审计（p50/p95/max、after-read ≤ 20s、10 measured 内容互异）；
9. 打回传包（不含 predictions，含全部合同文件 + JSONL + audit + 日志）。

## 4. 验收门禁（audit.json）

- measured_runs ≥ 10（十张图内容互异，SHA 校验）；
- after-read p50/p95/max ≤ 20s；
- 全链路（read/tiling/infer/fuse/export）逐 phase 记录；
- 合同各 SHA 与输入文件实际 SHA 一致。

## 5. 回传包内容

`E-FORMAL-BENCHMARK-return.tar.gz`：resolved_config / image_manifest /
checkpoint_provenance / hardware / benchmark_contract / runtime_samples.jsonl /
audit.json / logs / status.txt。

## 6. 后续（M3 组合测速）

任务单 E_10K_PIPELINE_TASK.md 第 8 节还要求 M1/M3/组合分别测速。M1 完成后，
M3（RT-DETR-L）同流程换 checkpoint + family=rtdetr 再跑一遍（M3 正式三折
last.pt 已在服务器 `/workspace/results/M3-CV3-OOF/fold_0/training/runs/foundation/weights/last.pt`）。
