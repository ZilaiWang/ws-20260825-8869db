# B 阶段来源审计与低分候选重排最终报告

日期：2026-08-14
状态：`complete_negative_ablation`
结论范围：`exploratory`、`cross_fit`、`fixed_per_image_budget`、`not_deployment_final`

## 1. 最终结论与适用范围

B0 来源/尺寸审计与 B1 固定预算重排已经完成。简单来源族分位数在逐图固定预算下只是单调变换，与原始分数完全等价；来源族+尺寸分位数在三个 held-out fold 均显著降低 Recall、提高 FDR。因此该简单规则未通过准入，不进入正式 outer-fold replay，也不作为部署规则。

这里的“来源族”严格指脚本中的三个粗粒度类别：MAR20 飞机、舰船场景和
车辆场景。60 个 `mar20-airport-proxy-*` 组在本实验中全部折叠为同一个
`aircraft_source_family`。因此本结论不能外推为“逐机场代理组校准无效”；
它只否定当前三粗来源族及其尺寸分箱规则。

B2 曾在本地生成 709 条带 proposal 身份和坐标的复核表，但最新主线已经提供更严格的 N0-4 v3 正式盲审包：322 张盲化卡片、密封映射、重复一致性门槛和人工决策表。为避免非盲信息泄漏及重复审阅，本 PR 不提交 709 条旧表，后续人工复核统一使用主线 N0-4 v3 包。

## 2. 与最新主线的关系

本报告在最新 `master` 提交 `f27a868` 上重新复跑并核对：4,481 张图、55,548 个候选、全部指标与原结果一致。

- Y1-C2 已是主线正式 cross-fit 校准分支；本实验不能替代 Y1-C2。
- R1 已使用 proposal crop 教师模型做更强的类别/分数重排；本实验只记录无新模型的简单来源/尺寸规则为何失败。
- N0-4 已建立正式 `FP_BG` 盲审流程；本地 709 条坐标表不再作为正式 B2 交付物。

因此本 PR 的价值是补齐“来源族/尺寸分位数简单规则”的可复现负向证据，避免后续重复尝试，不与当前主线准入模块竞争。

## 3. 输入与身份

| 项目 | 值 |
|---|---|
| 图像 | 4,481 |
| 唯一 GT | 20,933 |
| OOF 候选 | 55,548，`score >= 0.001` |
| split | `cv3_airport_proxy_k60_v2` |
| split SHA-256 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| formal crop SHA-256 | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |
| proposals SHA-256 | `abc93445693d05ba5454388900c634a63f52e58bdf56713e56e36f9ce249d2e0` |

脚本复用项目 `load_oof_aggregate`、`load_formal_ground_truth` 和官方 evaluator，没有实现第二套匹配规则。

## 4. B0 审计

探索工作点 `score >= 0.051` 的 Overall Recall 为 `0.917164`，Overall FDR 为 `0.195685`。错误分解为 `FP_BG=3303`、`FP_CLS=1115`、`FP_DUP=187`、`FP_LOC=66`。

31,678 个候选低于 `0.051`，占全部候选 57.0%。短边 `<32 px` 的候选共有 21,326 个，仅 1,268 个达到 `0.051`。这证明小框分数集中在低端，但不能证明这些框是真实目标。

在候选下限 `0.001` 的官方匹配轨迹中：

| 来源族 | 候选 | TP | FP_BG | FP_CLS | FP_DUP | FP_LOC |
|---|---:|---:|---:|---:|---:|---:|
| aircraft | 34,306 | 16,768 | 16,512 | 992 | 28 | 6 |
| ship | 18,865 | 2,412 | 15,577 | 115 | 692 | 69 |
| vehicle | 2,377 | 321 | 1,733 | 0 | 313 | 10 |

其中 aircraft `<32 px` 有 12,531 个候选、12,487 个 `FP_BG`；ship `<32 px` 有 7,094 个候选、6,840 个 `FP_BG`；vehicle `<32 px` 有 1,701 个候选、1,405 个 `FP_BG`。这解释了统一抬升小框为何会迅速挤占候选预算。

## 5. B1 逐图固定预算 cross-fit

每张 held-out 图像保留与原始 `score >= 0.051` 完全相同的候选数量。每个 held-out fold 的分位数统计只使用另外两个 fold。

| 方法 | 三折 Recall 均值 | 三折 FDR 均值 | 官方 macro Recall 均值 | 官方 macro FDR 均值 |
|---|---:|---:|---:|---:|
| raw_score | 0.9172 | 0.1958 | 0.8782 | 0.2446 |
| source_quantile | 0.9172 | 0.1958 | 0.8782 | 0.2446 |
| source_size_quantile | 0.7486 | 0.3437 | 0.7125 | 0.3678 |

`source_quantile` 在单张图像内保持原顺序，所以结果与 raw score 完全一致。`source_size_quantile` 三折全部恶化，不满足“至少 2/3 fold 同方向、pooled FDR 不恶化”的准入条件。

## 6. B2 与人工复核边界

`FP_BG` 只表示经过重复、类别冲突和定位规则后仍未归因的预测，不等于人工确认的纯背景。本阶段不把 `FP_BG` 自动转换为训练负样本。

正式人工复核应使用最新主线：

- `reports/experiments/N0_FP_BG_VISUAL_REVIEW_PACKAGE_20260814.md`
- `scripts/render_fp_bg_review.py`
- `src/rsdet/analysis/fp_bg_review.py`

在 N0-4 的 324 张卡片全部标注、盲重复一致率达到 0.85、映射解封并冻结 SHA 前，不产生背景训练白名单。

## 7. 复现与产物

```text
python scripts/b0_source_score_audit.py --root <M1_OOF_ROOT> --output-dir <B0_OUTPUT>
python scripts/b_stage_source_rerank.py --root <M1_OOF_ROOT> --output-dir <B1_OUTPUT>
```

Git 提交保留脚本和汇总报告。复跑生成的 `candidate_audit.csv`、逐折 JSON、
原始 OOF、模型权重和压缩包属于可再生产物，不在本 PR 中提交；报告中的冻结
输入 SHA、三折指标和停止结论构成当前小型审计记录。

最终决策：**simple source/size reranking = stop / not admitted**。
