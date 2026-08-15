# E 主线 3 评审整改响应（材料 37）

日期：2026-08-15
响应人：E（dashuaiguo）
对应评审：`E_MAINLINE3_REVIEW_20260810.md`（A 评）

## 0. 结论摘要

| 评审缺陷 | 状态 | 说明 |
|---|---|---|
| 缺陷1：评估脚本缺失，不可复现 | ✅ 已闭环 | A 已补 `scripts/eval_wp3_global_aggregation.py`；本地跑通，结果精确一致 |
| 缺陷2：FDR(score≥0.25)/聚合前基线无存档 | ✅ 已补 | 新增 `scripts/eval_wp3_threshold_baseline.py`，产出存档 |
| 缺陷3：macro 口径未说明 | ✅ 已补 | 对齐官方 V1.6 口径，macro_recall=0.8462 |
| 主线3核心场景（跨 tile） | ✅ 已验证 | 新增 `scripts/validate_cross_tile_merge.py`，合成 10K 图 100% 归并 |

## 1. 缺陷1：评估脚本可复现（已闭环）

`scripts/eval_wp3_global_aggregation.py`（A 提供）本地执行结果：

| 字段 | A 声明 | 实测 | 一致 |
|---|---|---|---|
| n_objects | 31,092 | 31,092 | ✅ |
| dedup_rate | 0.4403 | 0.44027 | ✅ |
| recall（官方细类） | 0.9008 | 0.90078 | ✅ |
| class_agnostic_recall | 0.9705 | 0.97053 | ✅ |
| macro_recall（V1.6） | — | 0.84623 | — |

输入 SHA 已记录：proposals `abc93445...`、formal_crop_manifest `a3bed44f...`。

**口径澄清**：评审正确指出我原始存档 `recall 0.9694` 是 **class-agnostic 定位召回**（框覆盖即算，不校验细类）。官方细类口径（框匹配 + 细类正确）下 recall = **0.9008**。两者之差 `fp_cls = 1460`（框位置正确但细类预测错误）。

## 2. 缺陷2：score 阈值扫描 + 聚合前基线（已补）

新增 `scripts/eval_wp3_threshold_baseline.py`，对同一份 proposals 按「聚合前/聚合后」两条链路在 score∈{0, 0.25, 0.5} 下评估（官方细类口径）：

| 链路 | score≥ | 框数 | Recall | FDR |
|---|---|---|---|---|
| 聚合前 | 0 | 55,548 | 0.9039 | 0.6254 |
| 聚合后 | 0 | 31,092 | 0.9008 | 0.3935 |
| 聚合前 | 0.25 | 21,557 | 0.9039 | 0.1222 |
| 聚合后 | 0.25 | 20,632 | **0.8848** | **0.1023** |
| 聚合前 | 0.5 | 20,116 | 0.8807 | 0.0836 |
| 聚合后 | 0.5 | 19,759 | 0.8699 | **0.0784** |

结论：
- **聚合在每个 score 阈值下都降低 FDR**（0.25 档 10.2% vs 12.2%；0.5 档 7.8% vs 8.4%）——聚合在去重的同时减少误检；
- 我之前汇报的 "FDR(score≥0.25)=3.83%" 是 **class-agnostic 口径**，官方细类口径下为 **10.2%**（仍 ≤17% 目标）。此数字以本次存档为准。
- 聚合后 Recall 略降（0.90→0.88 @0.25），为去重的正常代价。

存档：`outputs/e_wp3/wp3_threshold_baseline.json`（含输入 SHA）。

## 3. 缺陷3：macro 口径（已补）

- 我的原始存档 `macro_recall 0.9550` 为**自算口径**（25 细类平均、class-agnostic），**与官方 V1.6 不一致**；
- 官方 V1.6 口径（`evaluate_ranking_metrics`：细类级 Recall/FDR 先算，再按大类内细类简单平均，船4/飞机20/车辆1，要求完整类目）下 **macro_recall = 0.8462**，macro_fdr = 0.3734。
- 后续统一以官方 V1.6 口径汇报。

## 4. 主线3核心场景：跨 tile 冲突归并（已验证）

新增 `scripts/validate_cross_tile_merge.py`。用 `generate_synthetic_scene` 生成 10,000×10,000 合成图（144 tiles，32 目标，其中 22 个跨 tile），mock 检测器对跨 tile 目标按 50% 概率漂移细类（模拟真实跨 tile 细类冲突），跑 `run_pipeline(fusion="global", collect_objects=True)`：

| 指标 | 结果 |
|---|---|
| 跨 tile 目标归并为单对象 | **22 / 22（100%）** |
| 归并错误（被拆开） | **0** |
| evidence = 出现 tile 数 | 22 / 22 |
| 触发多细类投票对象 | 13 |
| 单 tile 目标不误合并 | 10 / 10 |
| 端到端耗时 | 8.45s（10K 图 / 144 tiles / mock） |

样例（跨 tile 细类投票）：对象 `cat=10` votes=`{9:0.87, 10:2.40}` evidence=4 tiles=[26,27,38,39]——同一目标在 4 个 tile 被报成细类 9/10，加权投票选 10。

说明：投票"细类=真值"13/22 是在 50% 漂移概率下（一半 tile 被强制漂走）的结果，属测试强度上限；真实模型漂移远低于此。跨 tile 归并本身 100% 正确。

## 5. WP4 10K 端到端实测（真实 M1 + 3090，工程 smoke）

权重来源：Gitee Release `v0.1-m1-weights`（best.pt，仅工程联调），SHA 校验通过：
`fold_0/1/2_best.pt` = `4ad6d8c8…` / `8503ad6c…` / `c7d43159…`，`yolo26s.pt` = `646f8bc3…`（与登记一致）。

实测（`scripts/eval_wp4_end2end_10k_3090.py`，服务器 `/root/autodl-tmp/e2e/run_e2e_10k.py` 同源）：

| 项 | 值 |
|---|---|
| 图像 | 10,000×10,000 合成图（官方不提供真实 10K，`image_source_type=synthetic`） |
| 冻结几何 | tile 1280 / overlap 256 / stride 1024 → **恰好 100 tiles** / batch 8 |
| 模型 | M1 YOLO26s fold_0 best.pt @ RTX 3090，conf=0.001 / iou=0.7 / max_det=500 |
| 计时 | `perf_counter + torch.cuda.synchronize`，1 warmup + 5 measured |
| **total_after_read** | **p50 = 1.35s，p95 = 1.41s，max = 1.42s，6/6 ≤ 20.0s 硬门禁 ✅** |
| 拆分 | model-only ≈1.17s / tiling ≈0.13s / fusion ≈0.00s |
| 峰值显存 | 0.26 GiB |
| 输出对象 | M1 合成图检出 17 个 → 聚合 17 个 GlobalObject |

存档：`outputs/e_wp3/e2e_result_fold0_3090.json`（含权重 SHA、GPU、逐 run 明细）。

**口径说明（重要）**：
- 该结论只证明**管道工程可跑通 + 20s 预算充足（13× 余量）**，属"工程 smoke 通过"；
- 因输入为合成图、权重为工程 best.pt，**不能宣称"官方时延通过"**。官方结论需
  `real_official` 10K 图（官方未提供）+ 正式 `last.pt` + 独占 GPU，见
  `docs/server/E_10K_PIPELINE_TASK.md`（当前 `real_official` 注册表为空，本任务不满足条件）；
- 合成图为纯色块，M1 检出 17 个对象属工程环境下的正常假阳性，不用于精度结论
  （精度结论以第 1-3 节真实 OOF 数据为准）。

**真实大图就绪**：`scripts/eval_wp4_end2end_10k_3090.py` 已加 `--image-path` / `--image-source-type`
支持——给定真实大图（PNG/TIF/JPG）时加载真实图并记录 `image_sha256` / `image_width` / `image_height`，
`image_source_type=real_project_proxy`、`n_gt=None`；无图则退回首选项合成图。该分支已在本地
（mock + 2048×2048 测试 PNG）与服务器 `/root/autodl-tmp/e2e/run_e2e_10k.py`（同源同步、SHA 双向核对）
双端验证通过。拿到团队真实 10K 图（C 服务器 `real_10000x10000.png`，需 C 授权 / 跨实例传输）后即可直接：

```bash
cd /root/autodl-tmp/e2e
PYTHONPATH=/root/autodl-tmp/xh-202625/src /root/miniconda3/bin/python run_e2e_10k.py \
    --weights /root/autodl-tmp/M1/fold_0_best.pt \
    --image-path /path/to/real_10000x10000.png \
    --output /root/autodl-tmp/e2e/e2e_result_real_10k.json
```

报真实检测数 + 跨 tile 归并对象数 + 耗时（真实代理图无 GT，只做工程口径，不宣称精度/官方时延）。

**切片参数对比（准确率档 / 速度档）**：E 分工要求比较一组准确率档和一组速度档的切片参数，
真实 M1 + 3090 + 合成 10K 图实测（每档 1 warmup + 5 measured）：

| 档 | tile | overlap | stride | batch | 切片数 | p50 | p95 | max | 显存 | 对象数 | ≤20s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 速度 | 1280 | 128 | 1152 | 16 | 81 | 1.03s | 1.14s | 1.15s | 0.26GiB | 27 | ✅ |
| 冻结 | 1280 | 256 | 1024 | 8 | 100 | 1.28s | 1.40s | 1.41s | 0.26GiB | 17 | ✅ |
| 准确率 | 1024 | 384 | 640 | 8 | 256 | 2.34s | 2.48s | 2.50s | 0.26GiB | 13 | ✅ |

- 计算量（切片数 81→100→256）与耗时（p50 1.03→1.28→2.34s）正相关，速度档比准确率档快约 **2.3×**；
- 三档全部 ≤20s（最差 max=2.50s，余量 8×），即使选准确率档也远低于门槛；
- 显存三档恒为 0.26GiB——峰值由模型自身占用主导，切片参数不抬升；
- 合成图纯色块场景下对象数（27/17/13）为假阳，无精度含义；**真正的精度档选择需真实大图/真实数据评估**（见已知问题）。

存档：`outputs/e_wp3/param_{speed,freeze,accuracy}.json`。

**失败样例分析**：
- 合成图为纯色块 + 简单形状，M1 检出的 17 个对象全部为假阳性（场景无真实目标纹理，GT 32 个均未被真实检出）；
  该结果只验证流程与测速，不用于精度判断（精度以第 1-3 节真实 OOF 数据为准）；
- 跨 tile 归并验证（`scripts/validate_cross_tile_merge.py`）在 mock 50% 细类漂移的强扰动下 22/22 正确归并、
  0 误合并，属测试强度上限；真实模型漂移率远低于此。

**已知问题、风险与优化建议**：
- 已知问题：
  1. 合成图无法让 M1 检出真实目标，精度结论必须回到 dev_v1 真实数据（A/C/D 环节）；
  2. 官方真实 10K 图未提供（`real_official` 注册表为空），官方时延结论结构性不可做，只能工程 smoke；
  3. 真实模型在真实 10K 图上的跨 tile 归并行为尚未实测（缺图）。
- 风险：
  1. 若官方评测工具按细类匹配（非纯 IoU），FDR 会从字面口径 ~0.04 回到细类口径 ~0.10——仍满足 ≤0.20 门槛；
  2. 3090 为临时借用实例，正式测速需评分方案认可的独占 GPU。
- 优化建议：
  1. 拿到团队真实 10K 图（C 服务器 `real_10000x10000.png`）后，直接用 `--image-path` 跑真实图端到端（脚本已就绪）；
  2. 当前 20s 预算余量充足（三档 max 1.15-2.50s），后续为提升精度增加的后处理复杂度仍有充足空间；
  3. E 分工要求的切片参数"准确率档/速度档"对比已完成（见上表），三档均远低于 20s 门槛。

**代码同步**：评审期间 Gitee master 新增 M3 提交 `8d13b2a`（CV3 OOF 训练/推理引擎，纯新增 4 文件），
已 fast-forward 本地 master 并入分支（`4ff3f6a`），E 测试 58 passed 无回归。
