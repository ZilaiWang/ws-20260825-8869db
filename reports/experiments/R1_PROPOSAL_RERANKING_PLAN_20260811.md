# R1-0：P03-F 教师对 M1 OOF 候选框的推理期重排

日期：2026-08-11
状态：`preregistered_waiting_for_server`
正式准入：`false`

## 1. 为什么下一步先做 R1-0

当前正式 M1 cross-fit 基线为 Recall 0.9176、FDR 0.1990；已通过门禁的 Y1-C2
把 FDR 降至 0.1593，同时 Recall 保持 0.9135。M1 的主要错误不是普通定位误差：

- FP_BG 3,303，占 FP 的 70.7%；
- FP_CLS 1,115，占 FP 的 23.9%；
- FP_LOC 仅 66；
- FN_CLS 1,115，占 FN 的 64.3%；
- FN_MISS 553。

P03-F 的 GT tight crop ConvNeXt-T 在正式 CV3 上取得 pooled macro Recall 0.9287、
accuracy 0.9593，证明“给定对象后，25 细类仍有很强的可分上限”。相反，Y2 完整
P2 快筛新增 36.1% 低分候选，却没有提高车辆候选下限 Recall，最终 Recall 明显下降。

因此当前最便宜且信息量最高的问题不是再改检测层，而是：

> 已有 M1 框是否包含足够对象信息，使 P03-F 教师在不改框的情况下纠正细类、改善
> 排序，并且对已经采用的 C2 工作点仍有增量？

## 2. 本实验回答和不回答什么

R1-0 是一次推理期域迁移快筛：

1. 输入 M1 正式 OOF 的全部 55,548 个 conf≥0.001 候选，而不是 N2-v2 中仅有 GT
   对齐标签的 20,628 个正样本；
2. fold0 候选只使用 P03-F fold0 checkpoint，后者训练时没有见过 fold0；fold1/2
   同理；
3. 只使用候选框 tight crop、224 分辨率；不改 bbox；
4. 分类结果只允许在原检测大类内部变化：船 4 类、飞机 20 类，车辆保持 FSC；
5. 方法、门控参数和阈值只用另外两折选择，再应用一次到 held-out 折；
6. 最后用 Y1 已冻结的 C2 参数重放，C2 不在 R1 输出上重新拟合。

本实验不证明：proposal-domain 微调有效、背景拒识已解决、10K 延迟达标、或模块可以
正式部署。P03-F 没有背景类，因此 R1-0 对 FP_BG 的作用只能来自低置信度分数融合，
不能把它解释为真正的背景分类器。

## 3. 输入锁

| 输入 | 冻结内容 |
|---|---|
| M1 aggregate | 4,481 图、55,548 候选、candidate floor 0.001 |
| formal crop | 20,933 GT，SHA `a3bed44f…` |
| P03-F teacher | ConvNeXt-T, tight-224, natural, seed42, fixed epoch30 |
| P03 checkpoint | fold0 `243d9648…`; fold1 `bd56974d…`; fold2 `eb0a13b…` |
| ImageNet weight | `983f1562…` |
| Y1-C2 | 原始正式结果 SHA `8b9e7a86…`，只重放冻结参数 |

prepare 阶段把 `oof_images.csv` 与 `oof_proposals.csv` 严格连接，输出
`proposal_inference_manifest.csv`。本地真实输入审计已经复现：20,115 / 18,437 /
16,996 候选，合计 55,548，4,481 图，manifest SHA `48747c3b…`。

## 4. 候选方法

所有候选共享相同框和 detector score，仅改变细类或分数：

| 方法 | 作用 |
|---|---|
| D0 | 完全保留 detector 输出 |
| R1 | 同大类内 hard argmax 重分类，分数不变 |
| R2 | crop top probability 与 top1-top2 margin 同时过门才重分类 |
| R3 | 类别不变，将 detector score 与原类别 crop probability 几何融合 |
| R4 | R2 重分类后，再融合最终类别的 crop probability |

预注册网格为：top probability `{0.75, 0.90}`，margin `{0.15, 0.35}`，R3 alpha
`{0.20, 0.40}`，R4 alpha 固定 `0.30`。连同 D0/R1 共 12 个变体。没有运行后新增
网格。

## 5. 外层 cross-fit 选择

对 held-out fold h：

1. 在另外两折上为每个变体扫描 0.001–0.301、步长 0.01 的阈值；
2. 候选须满足官方 Recall/FDR 门槛、相对 D0 Recall 下降不超过 0.005、FDR 不增加；
3. 合格候选按 macro Recall、macro FDR、pooled Recall、pooled FDR 排序；完全并列优先
   D0；
4. 选择结果和阈值原样应用于 h；
5. 三折 held-out 输出合并成唯一 R1 OOF 结果。

这不是把三折 logits 合并后再选一套最优参数；后者会让 held-out 标签参与自身决策。

## 6. 两层验收

第一层比较 `R1_selected` 与原始 `D0_detector`，回答 crop 教师是否有独立信号。

第二层把 D0 和 R1 分别送入原 Y1 每折已经冻结的 C2 参数，比较
`C2_frozen_after_r1` 与 `C2_frozen_original`。运行前必须先精确复现原 C2 的 Recall、
FDR、macro Recall、macro FDR，容差 `1e-12`；否则属于输入/实现不等价，停止。

除四类官方指标外，记录逐 GT 配对转移：

- `new_tp`：原工作点未命中、候选命中；
- `broken_tp`：原工作点命中、候选破坏；
- `retained_tp`；
- `net_tp`、FP/FN delta；
- 每折最终选中的变体及阈值；
- 各变体重分类数和被门控保留数。

## 7. 自动决策

推理期直接入围必须同时满足：

- C2+R1 仍满足 Recall≥0.85、FDR≤0.20；
- 相对冻结 C2：Recall 下降≤0.005，FDR 不增加；
- macro Recall 增益≥0.002；
- `new_tp >= broken_tp`。

结果分三种：

1. `admit_r1_inference_and_run_short_proposal_domain_finetune`：R1 对 C2 有稳定增量；
2. `run_short_proposal_domain_finetune_before_operational_admission`：只对原始 D0 有信号，
   说明 crop 方法可能有价值，但 GT-crop→proposal-crop 域差需修正；
3. `stop_r1_and_prioritize_background_rejection`：推理期信号不足，停止重分类分支，把资源
   转向带显式背景负样本的 rejector。

无论哪一种，R1-0 本身保持 `formal_admission=false`；正式准入至少还需要 proposal-domain
训练和新的完整 cross-fit 对照。

## 8. 产物

- `prepare/proposal_inference_manifest.csv` 与审计；
- `logits/fold_{0,1,2}_logits.npz` 与每折 runtime；
- `evaluation/reranking_result.json`；
- `evaluation/decision.json`；
- `evaluation/selected_crossfit_raw_predictions.json`；
- `evaluation/summary.json`；
- 全量回传包和 SHA256。

代码入口为 `scripts/r1_proposal_reranking.py`，服务器一键入口为
`scripts/server/run_r1_proposal_reranking.sh`，操作合同见
`docs/server/R1_PROPOSAL_RERANKING_TASK_00.md`。
