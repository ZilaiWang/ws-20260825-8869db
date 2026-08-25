# R1-1 飞机 proposal-domain 精识别与 D4 正式结果

日期：2026-08-14  
状态：`complete_iterative_oof_development`  
技术门禁：`pass`  
最终正式准入：`false`（同一正式 OOF 上的迭代开发，需最终系统确认）

## 1. 结论先行

本实验得到了一条明确有效、且可以解释的飞机精识别路线：

> P03 对象分类器先用真实 detector proposal crop 做短程域适配，再对对象的 D4
> 视图概率求平均，最后只在 detector 已判为 aircraft 的路径内做 cross-fit 重排。

最强条件是 **CE + D4**，而不是预注册主条件 selective-anchor KD + D4。
相对 `P03 identity` 对象参考，冻结 C2 后：

| 条件 | Overall Recall | Overall FDR | Macro Recall | Macro FDR |
|---|---:|---:|---:|---:|
| P03 identity | 0.91994 | 0.15387 | 0.87942 | 0.20409 |
| P03 D4 | 0.92185 | 0.15193 | 0.88069 | 0.20515 |
| CE identity | 0.92648 | 0.15054 | 0.88626 | 0.20353 |
| **CE D4** | **0.93011** | **0.14602** | **0.89122** | **0.20098** |
| selective KD identity | 0.92758 | 0.15080 | 0.88635 | 0.20316 |
| selective KD D4 | 0.92892 | 0.14801 | 0.88672 | 0.20105 |

`CE D4 - P03 identity`：

- Overall Recall `+0.01018`；
- Overall FDR `-0.00786`；
- Macro Recall `+0.01179`；
- Macro FDR `-0.00312`；
- paired `new_tp=608 / broken_tp=260 / net_tp=+348`；
- `fp_delta=-294 / fn_delta=-348`；
- ship/vehicle 指标严格零变化。

这不是 detector 本身的 1024 瓦片级提升；它是 C2 后的飞机条件式对象层收益。

## 2. 飞机专项结果

| 条件 | Aircraft macro R | Aircraft macro FDR | Aircraft pooled R | Aircraft pooled FDR |
|---|---:|---:|---:|---:|
| P03 identity | 0.92590 | 0.13325 | 0.93944 | 0.11956 |
| P03 D4 | 0.92748 | 0.13456 | 0.94168 | 0.11723 |
| CE identity | 0.93444 | 0.13254 | 0.94711 | 0.11571 |
| **CE D4** | **0.94064** | **0.12935** | **0.95137** | **0.11024** |
| selective KD identity | 0.93456 | 0.13207 | 0.94840 | 0.11608 |
| selective KD D4 | 0.93501 | 0.12945 | 0.94997 | 0.11267 |

CE D4 在 macro 与 pooled 两个口径上同时优于其余条件。尤其相对 CE identity，
D4 仍提供 Overall Recall `+0.00363`、FDR `-0.00453` 和 Macro Recall
`+0.00496`，说明多视图并非只是在补偿较弱的 P03 模型。

## 3. 因果解释

### 3.1 最主要增益来自 proposal-domain 适配

训练数据由 GT tight crop 转成 detector 实际 proposal crop；模型学习了部署阶段真实的
框偏移、上下文、截断和 resize 分布。CE identity 相对 P03 identity 已经取得
Recall `+0.00655`、FDR `-0.00333`，幅度大于 P03 上单独 D4。

训练准确率很快达到约 `99.9%`，因此后续问题不是继续拟合训练样本，而是如何改善
来源隔离折上的细类边界与结构不变性。

### 3.2 D4 是稳定的正交增益

CE D4 相对 CE identity 的 paired 结果更好；三个 held-out fold 相对原 detector
基线的 net TP 分别为 `+204/+43/+106`，没有负折。D4 的价值是降低朝向扰动造成的
细类不稳定，不是提高框定位。

完整 D4 bundle 推理在 RTX 3090 上每折约 38–43 秒，三折 32,062 个飞机候选、
每候选 8 视图，总吞吐约 2,050 view/s，峰值显存约 1.05 GiB。这个数字是对象模型
吞吐，不是 10K 端到端时延。

### 3.3 当前 selective-anchor KD 没有独立价值

教师正确 anchor 比例约 `99.7%–99.9%`，同视图 KD 几乎覆盖全部容易样本。
selective KD identity 相对 CE identity：Recall `+0.00110`，但 FDR `+0.00026`，
Macro Recall 仅 `+0.00009`；KD D4 也低于 CE D4。

因此停止调当前 KD 的 temperature/weight。后续蒸馏若启动，目标应改为
**D4 ensemble distribution → single-view student**，使教师真正提供视图不变性信息。

## 4. 重要限制

1. `decision.json` 的预注册主条件是 `selective_anchor_kd_d4`，其门禁通过；但全量
   消融显示 `ce_d4` 更强，科学结论必须以六条件比较为准，不能只复述主门禁；
2. 此实验在正式 OOF 语料上进行了多次迭代开发，不是最终独立确认；
3. R1 只允许 detector-aircraft 路径内换 20 类，无法找回被 detector 错分成 ship/
   vehicle 的飞机，也不处理背景 FP；
4. fold1 的 FDR 相对原 detector 路径仍有轻微上升，最终系统要保留逐折和 source
   group 压力检查；
5. D4 计算是否进入最终系统必须等 E 的 10K 对象数量和端到端时延证据。

## 5. 对下一步的约束

执行顺序更新为：

1. `R1-2`：用 CE identity 作公平参考，验证训练期类中心约束是否有独立增量；
2. 纯 CPU 自适应 D4：固定单视图置信度门控，回答能否只对困难对象运行其余视图；
3. 若全量 D4 成本不可接受且收益成立，做 D4 ensemble-to-single-view 蒸馏；
4. 若 R1-2 无增益，再考虑 ExpertDet/SLIP-RS 启发的少量结构属性辅助头；
5. 不引入 PSP.Plane 图像或 checkpoint。PSP.Plane 明确包含 MAR20，存在与本项目
   held-out 图重叠风险；只能借鉴 taxonomy/attribute 方法，不能绕过 CV3 使用其数据；
6. FP_BG 背景拒识继续等待盲审白名单；视觉抽查已确认大量 FP_BG 可能是未标目标。

## 6. 资产与复现

- 配置：`configs/experiments/r1_aircraft_refinement_v1.yaml`；
- 实现：`scripts/r1_aircraft_refinement.py`；
- 服务器驱动：`scripts/server/run_r1_aircraft_refinement.sh`；
- 本地回传：`outputs/R1-1-AIRCRAFT-PROPOSAL-REFINEMENT-return/`；
- 回传包 SHA256：
  `4e217ffb88b53dbe2ec039a25d3724b19a3081409fb18e4ae5380c85e16430de`；
- 正式选择与六条件详情：回传目录下 `evaluation/decision.json`、
  `condition_summary.json` 与六个 `*_result.json`。

本实验回答了“目前为什么有的尝试不涨”：模型规模和通用教师不是主瓶颈；将训练
分布对齐到真实 proposal，并显式处理俯视目标的旋转不稳定，才直接对应当前错误。
