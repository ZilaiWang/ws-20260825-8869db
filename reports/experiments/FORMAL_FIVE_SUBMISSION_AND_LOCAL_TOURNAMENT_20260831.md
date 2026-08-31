# 正式评测五次提交与本地锦标赛合同（2026-08-31）

## 1. 当前唯一基准

正式阶段的 incumbent 固定为预测评 `trial-v2.0` 的完全同源镜像：full Y5-S、identity
单视图、safe-1024、统一阈值 0.15。官方结果为：

| 粗类 | Recall | FDR |
|---|---:|---:|
| ship | 0.942287 | 0.126937 |
| aircraft | 0.999246 | 0.024300 |
| vehicle | 0.946309 | 0.237838 |

综合分为 86.2274，平均时延 2.704833 秒。`trial-v3.0` 已被正式否决，不得作为回退：
它虽然改善 ship Recall 和 vehicle FDR，但 vehicle Recall 降至 0.906040、ship FDR 升至
0.165644、时延升至 4.888833 秒，综合分降至 85.0018。

## 2. 只保留两套本地判定口径

### 2.1 方法选择账本

所有需要学习参数或改变候选排序的方法必须同时经过：

1. Normal-CV3：检查正常域、25 类 macro 和三粗类不退化；
2. Hard10K-CV3：检查复杂背景、低分 TP 与高分 FP 的排序；
3. source-disjoint sentinel：只做一次冻结外推，不参与选参数。

准入条件固定为：

```text
Normal pooled Recall 降幅 <= 0.3pp
任一粗类 Recall 降幅 <= 0.5pp
Hard10K @ FDR<=0.15 Recall 增益 >= 0.5pp
sentinel 与 Normal/Hard 同方向
aircraft 基本不退化
vehicle Recall 与 FDR不能以明显交换换取表面改善
```

### 2.2 提交前部署账本

方法通过上面的三域门后，才使用：

1. full Y5-S 或候选 full checkpoint；
2. 与 Docker 完全相同的切片、阈值、NMS、融合和 TTA；
3. 冻结 trial-mix V1.6 macro 审计；
4. Linux/amd64 离线容器；
5. RTX 3090 真实 10K 输入测速、结果 schema、有限性与确定性复跑。

该账本只预测提交方向和风险，不用于反复选阈值。预测评已证明它能准确预测六项指标的
变化方向，并较准确预测 ship FDR、aircraft Recall、vehicle FDR；高 vehicle 阈值下的
Recall 会明显偏乐观，必须额外施加 1.5--3.0pp 的域偏移风险预算。

## 3. 2026-08-31 本地锦标赛顺序

### L0：v2 精确复现

重新验收正式镜像 SHA、权重 SHA、配置、10K 输出与时延。任何新候选都必须与这一镜像
配对比较，禁止把历史 fold0、错误 M1 或 v3 配置当基线。

### L1：vehicle-only 中间工作点（无训练备用）

保持 ship=0.15、aircraft=0.15、identity 和全部融合逻辑不变，只比较固定的 vehicle
工作点。全量部署回看结果：

| vehicle 阈值 | pooled R/FDR | vehicle R/FDR |
|---:|---:|---:|
| 0.15 | 0.961075 / 0.144036 | 0.945652 / 0.236842 |
| 0.18 | 0.959222 / 0.141435 | 0.923913 / 0.212963 |
| 0.20 | 0.959222 / 0.139651 | 0.923913 / 0.194313 |
| 0.22 | 0.959222 / 0.137500 | 0.923913 / 0.170732 |
| 0.24 | 0.957831 / 0.137312 | 0.907609 / 0.169154 |

0.22 是唯一保留的风险备用点：FDR 大幅下降且时延不变，但本地 vehicle Recall 已下降
2.174pp，隐藏域可能进一步下降。因此它不是“全面升级”，只在需要押注 vehicle FDR
排名且没有更强质量模型时考虑；0.18/0.20/0.24 不形成额外提交，避免浪费机会。

### L2：Q0 归因后改为 crop-only 质量头

完整 Q0 在 Normal-CV3 FDR15 上把 Recall 从 0.916830 提升到 0.922801
（+0.597pp），是当前唯一达到正式增益门且三粗类均未下降的学习模块。但其 65 维特征
包含 crop 分类、D4 和 OTO，当前 Docker 不会生成这些证据。

34 维 Q0-lite 三折复验已完成，FDR15 Recall 为 0.916400、FDR 为 0.152807，低于
detector 0.916830/0.151022，因此停止。后续证据归因证明完整 Q0 的正增益来自 crop，
而不是 D4/OTO。新的唯一 L2 候选固定为 63 维 `base + crop`：

```text
34D base metadata
crop top-1 probability / margin / entropy / detector agreement
25-way crop class one-hot
```

严格三折结果为 Recall 0.923613 / FDR 0.152686，相对 detector +0.678pp；D4/OTO
各自为负，把任一项加回 crop 都会回落。下一步先建立 full crop 证据和 Docker 部署等价，
再建立 Hard 与 sentinel 同 schema 缓存。不得扩大 hidden dim、epoch 或 residual limit，
也不得恢复 D4/OTO。完整审计见
`reports/experiments/HERA_GUARD_V5_REMAINING_DIRECTIONS_AUDIT_20260831.md`。

### L3：proposal-domain 三类开放拒绝（固定组合）

双视图/旧专家 support 的新增复验均未达到独立准入线，L3 不再堆叠轻证据。L2 完成
部署等价后，只允许一个结构升级：在 held-out Y5 proposal 域训练
`foreground / structured_background / ordinary_background` 三类 crop verifier，输入固定
`tight 1.0x + context 1.25x`，输出只进入 crop-only 质量头，不直接创建框、DROP 或改类。

### L4：第二随机种子 Y5-S 一致性专家（中期主攻）

现有结果显示旋转一致性具有排序价值，但视图域偏移会伤害 vehicle；D-FINE vehicle
specialist 又存在严重折间不稳定。下一条更稳健的候选是同架构、不同初始化/数据顺序的
第二 Y5-S：

1. 先做来源分组 CV3 配对训练；
2. 不无条件并集 novel 框；
3. 对 v2 identity 候选提取第二 seed 的 same-fine IoU/score support；
4. aircraft 保持 v2；ship/vehicle 只做一致性质量重排；
5. 通过三域后才训练 full seed2。

其目标是以约 2 倍检测计算换取 vehicle/ship FDR 下降而不损失 Recall，预计时延仍接近
5 秒。若 CV3 任一粗类下降超过 0.5pp，停止 full 训练。

### L5：外部 coarse/objectness 预训练（最终机会候选）

只有来源、许可、类别映射和数据完整性可在 8 月 31 日锁定时才启动。外部数据只训练
aircraft/ship/vehicle/other_remote_object 四粗类，随后重建官方 25 类头，不把外部细类
伪装成官方型号。先做 fold0 40 epoch 快筛，通过候选地板、Hard 固定风险和 macro 门后
再扩三折/full。该路线可能提高候选形成上限，但不应阻塞 L0--L3。

## 4. 五次正式机会的使用顺序

| 正式机会 | 候选 | 使用条件 |
|---:|---|---|
| 1 | v2 exact | 正式阶段建立可复现锚点；不带任何实验改动 |
| 2 | 当日最强低风险候选 | 优先 Q0-lite；未通过则仅在明确接受 Recall/FDR 交换时使用 vehicle 0.22 |
| 3 | Q0-lite + dual consistency | 三域和 full 部署审计均优于机会 2；否则保留 |
| 4 | seed2 consistency expert | 完成 CV3、full 模型、容器和 10K 验收后使用 |
| 5 | 最终 Balanced/Attack | 只给截至 9 月 4 日的全局 Pareto 最优；保留到最后，不提前消耗 |

正式系统取最高分，因此第 2--5 次没有通过本地门时可以不提交。一次官方回传只用于判断
预注册候选是否符合预期，不根据隐藏结果继续细扫阈值或融合权重。

## 5. 明确停止的路线

本轮不再投入正式机会或 GPU 时间到：全量 rot90 TTA、D-FINE vehicle specialist、
DEIM、Y5-L、M3 单模型/无条件并集、普通 crop verifier、FPN-Q1/Q2、SAHI/P2、困难背景
微调、全局 NMS/阈值网格、DINO/CleanDIFT 直接拼接。它们已有完整负向或不稳定证据。

## 6. 当前执行状态

- vehicle-only 0.15--0.24 全量部署回看：完成；0.22 仅保留为备用；
- Q0-lite 34D 三折：已在 CPU 服务器启动；
- GPU 服务器：空闲，待 Q0-lite 门结果或 seed2 正式合同锁定；
- 正式提交：尚未由本合同自动触发。
