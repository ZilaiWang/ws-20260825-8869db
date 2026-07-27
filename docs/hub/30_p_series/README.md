# P 系列实验总览

更新日期：2026-07-25  
状态：`current_after_M1_formal_OOF`

本页补足当前 Git 中缺失的 P04 后续、P05、P06、P07 结果索引。数值来自已验收的服务器最终回报；正式报告和回传包仍需按服务器资产登记恢复。

| 阶段 | 关键结果 | 证据等级 | 决策 |
|---|---|---|---|
| P0-1 token 可见性 | 20,933 个对象；整 tile 下车辆 `ViT/14 < 4 token` 风险约 98.5%，对象重裁显著改善 | 几何事实 | 长期保留，支持“先检测再重裁” |
| P0-2 crop manifest | 20,933 对象、3 种 crop、62,799 行，可追溯并通过几何/哈希审计 | crop 几何可靠，旧 fold 探索性 | CV3 后只重挂 fold |
| P03 普通 crop 分类 | ConvNeXt-T 全量微调：tight-224 macro R `0.9703±0.0078`；三 seed 均值约 0.9686；context 较差，sqrt-inverse 无收益，jitter 仅小幅下降 | GT-crop 探索上限 | 冻结 tight-224 + natural，不再重开大网格 |
| P04 教师探针 | DINOv2-B CLS+patch `0.9098`；ConvNeXt train-RMS `0.8797`；DINO-S `0.8629`；CleanDIFT map0 `0.8293` | 探索性教师排序 | DINOv2-B 为首选教师；CleanDIFT 只作对照 |
| P05 真实背景拒识 | 旧易背景 pilot 被阻断；M1 工作点现有 `FP_BG=3303/4671` | 正式 OOF 已给出准入方向，但语义仍待人工审计 | 先构建 Pred-OOF hard-negative manifest 并抽检，随后做 cross-fit 背景头 |
| P06 合成框修正 | 合成任务可学；M1 正式工作点只有 `FN_LOC=66/1734` | 真实定位收益容量过小 | `deferred_low_localization_evidence`，不占近期 GPU |
| P07 扩散背景融合 | 保护区工程门禁通过；SD1.5 仅 1/24 优于最佳传统融合，48 个输出中 43 个有 halo | 有充分停止证据 | 扩散融合停止；传统 Copy-Paste 可保留 |

## 对总体路线的影响

1. 对象 crop 确实有很高的细类可分上限，但当前是干净 GT crop，不等于真实检测框表现。
2. DINOv2 比 CleanDIFT 更适合作为第一教师；扩散特征尚未证明独立价值。
3. 正式 OOF 已证明真实系统最需要 Pred-OOF crop 和真实难负样本；真实框残差
   暂无同等优先级。
4. P06 的确定性视觉 refiner 是框扩散之前必须击败的强基线。
5. P07 已停止，除非换成有新证据的新任务定义，否则不再扩大 SD1.5 实验。
6. P03/P04/P05 的工程入口应合并为同一个对象学生和同一份对象证据 manifest，
   但必须保留独立消融，不能把多项变化一次性打包。

## CV3 v2 完成后的正式复跑链

```text
冻结 CV3 SHA
├─ P0-2 按 annotation_uid 重挂 fold → P03/P04 关键工作点
├─ M1 三折低阈值 OOF
├─ M3 三折低阈值 OOF
└─ E 的 10K 工程基线
          ↓
OOF 审计、cross-fit 阈值与真实错误分解
          ↓
按背景 FP / 定位错误证据有条件放行 P05 / P06
```

M1 已完成，因此 P03/P04/P05 已解锁；P06 因真实定位容量不足主动暂缓。
当前顺序是：N0 证据层 → P04/P03 正式复验 → 真实对象学生与 P05 背景头。
M3 与 E-10K 可作为独立支线并行。

M1 后的当前顺序、准入条件和资源队列统一见
[`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](../../../reports/experiments/NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)。
原 [`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](../../../reports/experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)
保留为 M1 完成前的冻结执行合同。

## 当前已跟踪入口

- [`reports/experiments/README.md`](../../../reports/experiments/README.md)
- [`X-CROP-00-token-visibility.md`](../../../reports/experiments/X-CROP-00-token-visibility.md)
- [`P0-2-exploratory-crop-manifest.md`](../../../reports/experiments/P0-2-exploratory-crop-manifest.md)
- [`P03-00-crop-classification-master-plan.md`](../../../reports/experiments/P03-00-crop-classification-master-plan.md)
- [`P03-04-seed-stability-results.md`](../../../reports/experiments/P03-04-seed-stability-results.md)
- [`P04-00-teacher-feature-probe-master-plan.md`](../../../reports/experiments/P04-00-teacher-feature-probe-master-plan.md)

P04 总纲已在文首区分“探索阶段已完成”和“正式 CV3 v2 复验待执行”；
探索数值不得替代正式三折结果。
