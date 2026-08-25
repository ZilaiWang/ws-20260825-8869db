# R1-5 飞机双视图一致性训练结果（2026-08-14）

## 1. 结论

R1-5 三折训练、identity/full-D4 推理与严格 outer cross-fit 评估已完整
结束。训练期从同一飞机 proposal 独立抽取两个不同 D4 视图，在原 CE 上
加入对称 KL 一致性损失；只训练 5 epoch，不搜损失权重或 checkpoint。

相对同设定 CE 基线：

- identity：净增 `78 TP`，FP `-8`，pooled Recall `+0.00373`；
- full D4：净增 `50 TP`，FP `+32`，pooled Recall `+0.00239`；
- 舰船/车辆全部指标严格不变；
- 预注册主、次门禁都通过，但仍属同一 OOF 上的迭代开发，
  `formal_admission=false`。

它提供了独立的召回改善证据，但 D4 工作点的 FP 也略增，因此不单独
替换 R1-1，需与 R1-6 飞机后 NMS 做组合对照。

## 2. 完整性

- 三折各固定 5 epoch，三个 checkpoint 与三个 D4 bundle 齐全；
- fold2 首次训练后推理进程遭外部中断，恢复脚本只重跑 fold2 推理，
  没有重训、resume 或改参数；
- 最终服务器状态 `complete`，回传包 SHA256：
  `78b09a85d8bbfaf0aff211a79e1d8b0871b93147267670ab410230ff8c3e59af`；
- 视图一致性训练损失从约 `0.007` 降至 `0.001--0.0017`，无 NaN/OOM。

## 3. Identity 结果

| 指标 | CE identity | consistency identity | 差值 |
|---|---:|---:|---:|
| TP / FP / FN | 19394 / 3437 / 1539 | 19472 / 3429 / 1461 | **+78 / -8 / -78** |
| pooled Recall | 0.926480 | 0.930206 | **+0.003726** |
| pooled FDR | 0.150541 | 0.149731 | **-0.000809** |
| macro Recall | 0.886262 | 0.891501 | **+0.005240** |
| macro FDR | 0.203530 | 0.202691 | **-0.000838** |
| aircraft macro Recall | 0.934442 | 0.940992 | **+0.006549** |
| aircraft macro FDR | 0.132540 | 0.131492 | **-0.001048** |

配对转移为 `new_tp=149, broken_tp=71, net_tp=78`，说明改善不是由单个网格
或极少样本偶然触发。

## 4. Full-D4 结果

| 指标 | CE + D4 | consistency + D4 | 差值 |
|---|---:|---:|---:|
| TP / FP / FN | 19470 / 3329 / 1463 | 19520 / 3361 / 1413 | **+50 / +32 / -50** |
| pooled Recall | 0.930110 | 0.932499 | **+0.002389** |
| pooled FDR | 0.146015 | 0.146890 | +0.000875 |
| macro Recall | 0.891217 | 0.891803 | **+0.000586** |
| macro FDR | 0.200975 | 0.200817 | -0.000157 |
| aircraft macro Recall | 0.940637 | 0.941369 | **+0.000732** |
| aircraft macro FDR | 0.129347 | 0.129150 | -0.000197 |

`new_tp=132, broken_tp=82`，但新增的候选也带来 32 个 FP。因此它是高召回
候选，不是对 CE+D4 的全面支配。

## 5. 后续决策

1. 已固定把 consistency+D4 接入同一个飞机 `IoU=0.50` 后 NMS，不重搜
   NMS 阈值；
2. 再做一次固定 `0.5 CE + 0.5 consistency` 概率融合，权重不搜索；
3. 若不能同时优于 R1-6 的召回和 FDR，则保留 R1-6 为主工作点，
   consistency 只作召回备选；
4. 不再搜一致性损失权重、温度或 epoch。

## 6. 产物索引

- 冻结配置：`configs/experiments/r1_aircraft_view_consistency_v1.yaml`
- 执行入口：`scripts/r1_aircraft_refinement.py`
- 恢复脚本：`scripts/server/recover_r1_aircraft_view_consistency_fold2.sh`
- 解压产物：`outputs/R1-5-AIRCRAFT-VIEW-CONSISTENCY/`
- 权威决策：`outputs/R1-5-AIRCRAFT-VIEW-CONSISTENCY/evaluation/decision.json`
