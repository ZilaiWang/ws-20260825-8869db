# E12: 验证漏斗 + Attack 组合定义(2026-08-20)

## 四级验证漏斗(L0-L3)

| 级别 | 内容 | 成本 | 通过条件 |
|---|---|---|---|
| **L0 机制检查** | 单折 40ep 短训 + 候选量/Recall/FDR/错误分解 | ~90min GPU | 候选量↑或 Recall↑, 无灾难性 FDR |
| **L1 单折完整链** | fold0 + R3/NMS/SoftRisk 全链, 与 Y5 同口径对比 | 复用本地 | 链后 Recall 净增, FDR 不显著恶化 |
| **L2 三折 OOF** | 完整三折推理 + 合并链评估 | ~4h GPU | 三折一致, 与 fold0 趋势相同 |
| **L3 sentinel** | 冻结 23 组 555 图(12.4%)只评最终版本 | 本地 | 收益不是对旧错误清单的记忆 |

## Attack 组合定义(SparseZoom 停止后修订)

原 Attack = Balanced + SparseZoom-KD + risk-triggered fallback。
SparseZoom 已按停止条件终止(E10), Attack 修订为:

```
Attack = COPH 候选 + R3 融合 + 全类 NMS
       + SoftRisk(更激进: beta=0.7, clip=2.0)
       + 学习式 E5(改类决策器, 若时间允许)
       + E7 困难课程(训练期难例)
```

与 Balanced 的差异: 更激进的 SoftRisk 参数 + 学习式改类模块。

## 已落地组件

| 组件 | 状态 |
|---|---|
| paired_delta_ledger.py(对象转移账本) | ✅ 已验证(Y5 vs E1E2: FP-6597/TP 不变) |
| PROSPECTIVE_SENTINEL(23 组 555 图冻结) | ✅ 已冻结 |
| run_safe_chain.py(R3+NMS+SoftRisk 一键) | ✅ 已升级(--nms-all/单折CV/图域限制) |
| e8_coph_softrisk_verify.py(公平对比) | ✅ 已验证 |
| L0/L1(COPH fold0) | ✅ 已通过(fold0 完整链 R=0.9423) |
| L2(COPH 三折) | ⏳ 训练中(fold1/2) |
| L3(sentinel 评估) | ⏳ 等最终版本 |
