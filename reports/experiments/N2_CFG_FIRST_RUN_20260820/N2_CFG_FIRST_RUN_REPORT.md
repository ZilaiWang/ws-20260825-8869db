# N2-CFG 首次完整运行报告(2026-08-20 00:42-01:17)

## 0. 触发

成员B回传 13 冲突对重裁决定 → 合并重编译 → 一致率 100%(54/54) → 白名单签发 **33 个
clear_background**(fold0=15/fold1=7/fold2=11;ship 13/aircraft 9/vehicle 11)。
全套 N2-CFG 首次在服务器跑通(代码锁→资产SHA→manifest→三折训练→推理→S0/S1/S2→门禁)。

## 1. 运行结果(全流程成功, 门禁未过)

- 三折训练 loss: fold0 0.0085 / fold1 0.0156 / fold2 0.0119(收敛良好);
- 回传包已打: `N2-CFG-BACKGROUND-GATE-V1-return-no-checkpoints.tar.gz`;
- **门禁 overall FAIL**: 4/7 过、2/7 挂、1 报数。

## 2. 门禁明细

| 门禁 | 结果 | 数据 |
|---|---|---|
| g1 pooled FP_BG reduction | ✅ PASS | S2 删 186 vs S0 删 19, ratio 0.12 ≥ 0.10 |
| g2 coarse reduction | ❌ FAIL | ship 0.0014(需0.15)/ vehicle 0.0259(需0.10)/ aircraft 0.3065 |
| g3 fold consistency | ✅ PASS | - |
| g4 source concentration | ✅ PASS | max share 0.0968 ≤ 0.4 |
| g5 paired bootstrap | ✅ PASS | delta 167, 95%CI [117,228] 下界>0, 2000 次 |
| g6 focus classes | 报数 | HM/LQS/TU-160 removed 0, F-22 removed 1 |
| g7 zero_tp_loss | ❌ FAIL | applied_removed_tp=3(recall_budget=0 不允许删 TP) |

## 3. 根因诊断(关键)

**S0 基线已经几乎删光所有 FP_BG, 门控无显著增量空间。**

- S0(纯分数阈值基线)在 fold0/1/2 只保留 **6/1/12 个 FP_BG**(1,539 个 heldout
  FP_BG 中 S0 删掉 1,520 个,ratio 达 0.99);
- S2(门控)保留 64/58/64 个 → 相对 S0 增量仅 167 个(即 g5 delta);
- g2 ship: S0 删 7 vs S2 删 1——**S0 在 ship 上删得比门控还多**,ratio 0.0014
  完全无法达标;
- 本质: S0 拟合出的 `tau_drop≈0.0006`(极低分数阈值)已把低分 FP_BG 清光,
  门控(fg logit)找不到可再删的增量;
- g7: S2 应用时误删 3 个 TP(recall_budget=0),触发 zero-tp-loss 门禁。

## 4. 与白名单量的关系

- 33 个负样本中 ship 13/vehicle 11/aircraft 9——ship/vehicle 的负样本**能学到
  门控, 但 S0 基线本身太强**, 门控无法"显著超越";
- 扩审第 2 批(566 卡, 已发成员B)若回传, 白名单到 ~98, 可重新训练验证
  g2 是否改善; 但**核心问题在基线口径(S0 过强), 非单纯负样本量**。

## 5. 下一步选项

1. **复查 S0 基线定义**: 检查 `evaluate_bg_gate.py` 中 S0 的拟合约束——是否
   `tau_drop` 应受最小保留数/recall 预算约束(当前 recall_budget=0 却允许
   S0 删 99% FP_BG, 疑似 S0 拟合未应用 recall 约束);
2. **g7 修复**: S2 应用时须保证 removed_tp=0(已有 recall_budget 参数, 检查
   applied 阶段是否强制);
3. 扩审第 2 批回传后重训(白名单 98)验证 g2 改善;
4. 若 S0 基线确有缺陷, 修正后重跑(30 分钟级, 成本低)。
