# HERA-Field 方案5 GPU 批次启动 + 离线验证总结 — 2026-08-21(下半场)

> 新服务器(RTX 3090)已配置, F2/D3/D4 三折训练已排队启动, D2/D6/V3/V4 离线验证完成。
> 本报告记录实现细节、离线结论、训练队列状态与剩余规划。

## 一、服务器与代码同步

- 新服务器: `ssh -p 17879 root@connect.nmb2.seetacloud.com`(RTX 3090 24GB)
- 已配免密; 环境 torch 2.5.1+cu121 / ultralytics 8.4.103 / venv p06-cu121
- 代码经 gitee 同步到最新(服务器有旧 R1 未提交改动 → 备份后 reset --hard 对齐)
- Y5 权重: `/workspace/results/Y5-ROT90-CV3-OOF/fold_{0,1,2}/training/runs/foundation/weights/last.pt`

## 二、已实现的三个"重改动"(GPU 队列, 各三折 40ep)

### F2 — family 三层辅助损失(方案 §10.1)

**目的**: 针对 FP_CLS 841 兄弟机型混淆(TU-22↔TU-160 347 / MS↔QHS 288 / SU-35/34/24 277 /
E-8↔KC-135 173 / F-15/16/22 174)。

**实现**: 定义 9 个机型家族, 在检测损失上叠加 family 中间层辅助分类(类权重分解近似
`w_c = w_family + w_coarse + δ_c`), 让兄弟类共享家族表征、家族内细判, 缓解平坦 25 类
softmax 对尾类(HM 17 / LQS 30 / TU-160 361)过拟合。

- `src/rsdet/innovation/coarse.py`: 加 `FAMILY_MAPPING`(25→9)
- `src/rsdet/innovation/family_loss.py`: `FamilyHierarchicalLoss`(fine+family+coarse 三层)
- `scripts/train_cv3_oof.py`: `--innovation family --family-gain 0.5`

**9 家族**: ship-large(HM/LQS) / ship-combat(QHS/MS) / su-jet(SU-35/34/24) /
fighter(F-16/15/22/FA-18) / tupolev(TU-160/22/95) / us-bomber(B-52/B-1B) /
transport(C-130/17/C-5) / awacs-refuel(E-3/8/P-3C/KC-135/KC-10) / vehicle(FSC)。

**状态**: 三折训练中(fold0 已到 epoch 18/40, cls_loss 130→7.7 正常下降)。

### D3 — group-balanced 采样(方案 §十二 批次4)

**目的**: D1 发现 12 高误检域贡献 30% FP_BG、20 低召回域 Recall=0.382, 头部机场主导表征。

**实现**: `d3_worst_group_curriculum.py` 复用 D1 的 OER 工作点 + 每 group 错误统计,
选 worst-group(FP_BG top-20 ∪ Recall bottom-20 = 40 group), 映射到 1410 张 hard images,
用 E7 hard-curriculum 机制(训练图重复 1 次 = 2x 采样)。

**状态**: 排队中(等 F2 完成)。

### D4 — worst-group loss(方案 §十二 批次4)

**目的**: 在 loss 层对 worst-group 样本重加权(比 D3 采样层更精细的梯度控制)。

**实现**: `worst_group_loss.py` 继承 `HierarchicalCoarseLoss`, 按 `batch["im_file"]`
匹配 worst-group 路径, 对 cls 损失乘 1.5x。`split_view.json` 直接含 `group_id` 字段
简化了 path→group 映射。

**状态**: 排队中(D3 完成后)。

**训练队列**: F2(三折, 约 1.2h) → D3(三折, 约 1.75h) → D4(三折, 约 1.75h), 全串行。

## 三、离线验证结论(已完成)

### D2/D6 — latent domain 聚类 + leave-cluster-out(方案 §十二)

KMeans 按 group 画像聚类 4 域, leave-one-domain-out 训练 OER 评估跨域泛化:

| 域 | group | GT | 域内 R | 跨域 R | gap |
|---|---:|---:|---:|---:|---:|
| 0 | 17 | 31 | 1.000 | 1.000 | 0.000 |
| 1(困难) | 101 | 3381 | 0.710 | 0.608 | **+0.101** |
| 2(大头) | 53 | 17176 | 0.991 | 0.988 | +0.002 |
| 3 | 84 | 345 | 0.841 | 0.841 | 0.000 |

**结论**: 困难域(域1, tp_rate 0.123)有 **10pp 跨域泛化损失**, 印证 D3/D4 域平衡训练
的必要性, 也为 D5(域 adapter)提供了上限依据。

### V3/V4 — 车辆受控恢复(方案 §九 批次3)

支持面 gating(region 特征聚类环境原型判断"车辆支持面") + density top-K(每图车辆预算):

| 配置 | R@FDR=.12 | Δ |
|---|---:|---:|
| OER 14 特征 | 0.9616 | — |
| +支持面分 | 0.9619 | +0.0002 |
| +gating+topK | 0.9614 | −0.0003 |

车辆支持面判别力: TP 0.016 vs FP_BG 0.011(几乎为 0)。**结论**: 车辆太稀疏
(每图平均 <1 辆), 环境原型/密度无法有效区分车辆支持面, V3/V4 证伪"受控车辆恢复"
在现有候选上的价值。V 系列(V2 频域种子 / V6 蒸馏)前景不明。

## 四、剩余未做(需 GPU)

| 实验 | 内容 | 优先级 |
|---|---|---|
| V2 | 车辆中心种子图 + 频域增强(SET) | 中(V1/V3/V4 已证伪大部分车辆方向) |
| V6 | DFD→seed head 蒸馏 | 中 |
| F3 | counterfactual attribute negatives | 中(需属性伪标注) |
| F4 | observability mask | 中 |
| F6 | tail prototype residual | 中(F2 family 的延伸) |
| 批次5 | D4蒸馏 / crop稀疏激活 / FP16 / 10K 时延 | 收尾 |

## 五、关键结论

1. **F2/D3/D4 是方案5剩余里最有依据的三个"重改动"**, 分别针对 FP_CLS 841(兄弟混淆)
   和域问题(12 高误检域 + 20 低召回域), 已全部实现并启动训练。
2. **D2/D6 首次量化跨域泛化损失**: 困难域 10pp gap, 域平衡训练方向正确。
3. **V3/V4 证伪"受控车辆恢复"**: 车辆太稀疏, 环境/密度无判别力; V 系列前景收窄到
   V2 频域(唯一可能增强车辆特征的方向)。
4. F2 三折训练中, D3/D4 排队; 训练完成后需推理 + 三折 OER 评估 + deploy 口径验证。

## 六、产物

- 代码: family_loss.py / worst_group_loss.py / d3_worst_group_curriculum.py /
  d2_d6_domain_cluster.py / v3_v4_vehicle_gating.py / coarse.py(FAMILY_MAPPING) /
  train_cv3_oof.py(--innovation family/worstgroup)
- config: f2_family_fold{0,1,2} / d3_worstgroup_fold{0,1,2} / d4_worstgrouploss_fold{0,1,2}
- 数据: d3-worst-group-curriculum.json(1410 hard images)
- 结果: d2-d6-domain-cluster.json / v3-v4-vehicle-gating.json
