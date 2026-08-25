# R1-3 困难对象自适应 D4 快筛结果

日期：2026-08-14  
状态：`complete_stopped`  
正式准入：`false`

## 1. 假设

CE 单视图与 D4 类别不同的候选大多具有较低单视图置信度，因此尝试固定规则：

> 当 `max p(class|aircraft) < 0.80` 时使用 8-view 平均；否则只使用 identity。

`0.80` 来自无标签的 identity-vs-D4 变化覆盖诊断，未根据 GT 网格选择。除这个
门控外，重排 variant、cross-fit 阈值、C2 和全部输入均与 R1-1 相同。

## 2. 结果

| 指标 | CE identity | Adaptive D4 | Full CE D4 |
|---|---:|---:|---:|
| Overall Recall | 0.92648 | 0.92715 | 0.93011 |
| Overall FDR | 0.15054 | 0.15011 | 0.14602 |
| Macro Recall | 0.88626 | 0.88682 | 0.89122 |
| Macro FDR | 0.20353 | 0.20309 | 0.20098 |

计算审计：

- 飞机候选 `32,062`；
- 启用 D4 `9,449`，占 `29.47%`；
- full-view equivalent `98,205`；
- 理论视图计算为全量 D4 的 `38.29%`。

相对 full CE D4：Recall `-0.00296`、FDR `+0.00410`、Macro Recall
`-0.00439`、Macro FDR `+0.00212`，超过预注册容忍；门禁失败。

## 3. 解释与停止

低置信候选虽然覆盖大多数 identity-vs-D4 类别变化，但“发生变化”不等于“形成
最终 TP 净收益”。一些高置信、朝向敏感的错误仍由 D4 修复；同时 cross-fit variant
与阈值会随概率分布改变。单一 max-probability 不能作为可靠的不变性风险量。

为避免在同一 OOF 上继续搜索第二阈值、entropy/margin 组合，本方向停止：

- 不做更多置信门控网格；
- full D4 先进入真实 10K 对象数与时延评估；
- 若成本不可接受，改做 D4 ensemble-to-single-view distillation；
- 若蒸馏不能保留收益，则最终系统在“全量 D4”与“CE identity”之间按真实预算选取。

资产：服务器 `/workspace/results/R1-3-ADAPTIVE-D4/decision.json`；实现
`scripts/r1_adaptive_d4.py`，配置 `configs/experiments/r1_adaptive_d4_v1.yaml`。
