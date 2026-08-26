# 创新阶段前实验收尾与可信基线

日期：2026-08-10
状态：`current_authority`
用途：创新实验立项、评审和数字引用的统一入口

> 本文档只收录已通过当前代码、数据血缘与评估口径审计的结论。
> 与本文冲突的旧聊天、旧服务器回报或历史报告统一降级。

## 1. 官方口径已冻结

### 1.1 刚性门槛

- 25 细类正确才能匹配；
- score 降序、一对一贪心匹配，重复框计 FP；
- 车辆 IoU=0.35，舰船/飞机 IoU=0.50；
- 三类合并 pooled Recall ≥ 0.85，pooled FDR ≤ 0.20；
- 10K 单图 ≤ 20 s，不含图像磁盘读取，包含读取后的完整算法链。

### 1.2 V1.6 排名

- 舰船 4 类、飞机 20 类、车辆 1 类分别做细类简单平均；
- 舰船/飞机/车辆的 Recall/FDR 加时延，共 7 项排名；
- 正式 macro 必须使用完整税表，缺类 fold/子集只是诊断；
- 缺任一排名项的内部对手模拟不参与排名，避免不全报告获利。

唯一配置源为 `configs/project.yaml`：

```text
contract_version = contract_v1
eval_version = official_eval_v1
ranking_version = official_ranking_v1_6
```

## 2. 可信的共同基础

| 对象 | 冻结结论 |
|---|---|
| 原始数据 | 4,481 图，20,933 GT，25 细类 |
| 分组 | MAR20 60 个机场代理视觉组；是 proxy，不宣称真实机场真值 |
| 正式 CV | `cv3_airport_proxy_k60_v2`，255 不可拆组，4,481 图 OOF 各一次 |
| M1 | YOLO26-s / 1024 / seed42 / 160 fixed epochs / 三折 `last.pt` |
| M1 原始 OOF | 55,548 低阈值候选，输入与 D00 数据锁已验收 |
| 对象学生 | ConvNeXt-T / tight-224 / ImageNet 初始化 |
| 对象教师 | DINOv2-B CLS+patch 作训练期教师候选；CleanDIFT 不单独入选 |

## 3. 当前可引用数字

### 3.1 M1 检测基线

| 口径 | Recall | FDR | 证据等级 |
|---|---:|---:|---|
| 同 OOF 描述工作点 `t=0.051` | 0.9172 | 0.1957 | 描述性，不是无偏阈值结论 |
| cross-fit 合并 held-out | 0.9176 | 0.1990 | 当前 pooled 无偏基线 |

cross-fit 仅以约 0.001 的 FDR 余量过线，且 fold0/2 为 0.2259/0.2294；
创新实验不得以“总体已稳定过线”为前提。

cross-fit 合并 held-out 的 V1.6 macro：

| 大类 | macro Recall | macro FDR |
|---|---:|---:|
| 舰船 | 0.7162 | 0.5389 |
| 飞机 | 0.9080 | 0.1589 |
| 车辆 | 0.6119 | 0.6239 |
| 25 细类 overall diagnostic | 0.8654 | 0.2383 |

舰船和车辆是排名短板；飞机总体较强，但 TU-160、F-22 等细类仍是结构性风险。

### 3.2 错误机制

- 正式、计数守恒的工作点分解：`FP_BG=3303`、`FP_CLS=1115`、
  `FP_DUP=187`、`FP_LOC=66`；
- `R_loc@oracle-class=0.9705`；
- `Acc_fine@localized=0.9297`；
- <32 px 小目标和 FSC 车辆的 oracle 定位召回显著低，说明候选缺失与小目标
  定位不能靠对象重分类解决。

`N0-EVIDENCE-M1-v2` 中的 FP 子类是 nearest-overlap 研究标签，不与上述正式
计数混用。

### 3.3 对象表征上限

| 实验 | 当前正式结论 |
|---|---|
| P03-F | GT tight-224 ConvNeXt-T 微调 macro Recall 0.9287，是理想 crop 学生上限 |
| P04-F | frozen probe：DINOv2-B 0.8294 > ConvNeXt 0.7815 > CleanDIFT 0.7036 |
| 教师决策 | 若做蒸馏，DINOv2-B 是第一对照；扩散特征只可作互补消融 |

## 4. 已作废或降级的结论

| 项目 | 状态 | 原因 |
|---|---|---|
| N0-3 v1 及其 N0-4 抽样 | `superseded_invalid` | oracle 命中按同图+同预测类错误传播 |
| N2 v1 对象学生成绩 | `superseded_invalid` | 错误标签、自动背景、score 键碰撞、held-out 选模 |
| P2 “正式超越 M1” | `invalid_formal_claim` | 60 vs 160 epoch、best vs last、评估器非官方一对一 |
| P2 通道/注意力响应 | `mechanism_diagnostic` | 可证明模块发生响应，不证明端到端改善 |
| Gitee `M1-CV3-OOF-fold012-best.tar.gz` | `engineering_only` | 是 best.pt，正式 M1 使用三折 last.pt |
| CleanDIFT 单教师 | `stopped_as_sole_teacher` | 正式 probe 弱于 DINOv2-B 和 ConvNeXt |
| P07 SD1.5 融合 | `stopped` | 人工盲评与伪影达停止条件 |

## 5. 已修复的工程合同

1. 官方评估：增加 `ranking_version`、固定 25 类 macro、partial-taxonomy
   显式诊断开关、不完整队伍排除规则。
2. 阈值扫描：pooled 门槛与 V1.6 macro 内部目标同时输出。
3. N0：候选级 `source_prediction_index` 全链路稳定对齐。
4. N2：真值标签、人工背景、三种模式语义和固定 epoch 外层评估已修复。
5. P2：新代码冻结 fixed-last 契约，机制评估器显式标注非官方。
6. 大文件：跨成员交付只认 Gitee Release/附件 + SHA256，服务器路径仅作个人
   运行日志。

## 6. 创新实验的统一验收门槛

任何 A—E 创新点都以 M1 fixed-last CV3 OOF 为出发点，且至少报告：

1. 相同 `cv3_airport_proxy_k60_v2`、同一预训练权重、固定 epoch、三折原始低阈值 OOF；
2. pooled Recall/FDR 门槛和完整 4/20/1 macro 排名口径；
3. 至少 2/3 folds 与基线同方向，不只看三折平均；
4. 错误转移：TP 新增、TP 被破坏、FP_BG/CLS/DUP/LOC 变化；
5. 头/中/尾类、TU-160 压力组、小目标和三大类；
6. 阈值由 cross-fit 或独立校准决定，不在同一 OOF 上选点又宣称无偏；
7. 模块入选后再记录 10K 完整算法链延迟，不用 model-only 时间代替。

当前首要优化顺序：

1. 车辆/小目标候选召回与质量估计；
2. 舰船背景虚警和少样本细类；
3. 飞机 TU-160/F-22 等结构性混淆；
4. 全局对象唯一化与跨 tile 重复；
5. 在上述收益成立后，再做蒸馏、困难门控和时延压缩。

## 7. 待办与非阻塞项

| 事项 | 状态 | 对创新立项是否阻塞 |
|---|---|---|
| M1 正式 OOF/恢复审计包发布到 Gitee 附件并核 SHA | complete | 两份无 checkpoint 证据包已在 `v0.1-m1-weights` 发布 |
| 三个 M1 `last.pt` 取回并发布到 Gitee 附件 | pending_source_retrieval | 当前本地同名文件是已作废 YOLOv8s 诊断权重且 SHA 不符；阻塞其他成员正式复现 M1 |
| N0-4 v2 背景人工盲审 | pending_on_reopen | 不阻塞其他创新；阻塞 N2 背景拒识 |
| N2 v2 外层纯净重放 | optional | 不阻塞；只在重分类/拒识被立项时执行 |
| M3 异构 OOF | member line | 不阻塞 M1 单模型创新；阻塞异构 oracle-union 结论 |
| 10K 正式时延 | engineering line | 不阻塞训练期创新；阻塞最终落地入选 |

## 8. 索引

- 官方口径：`docs/hub/01_scoring_standard/README.md`
- 实验契约：`docs/EXPERIMENT_PROTOCOL.md`
- 正式 M1：`reports/experiments/M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`
- N0：`reports/experiments/N0_EVIDENCE_COMPLETE_20260809.md`
- N1：`reports/experiments/N1_FORMAL_P03_P04_EXECUTION_20260809.md`
- N2 v2：`reports/experiments/N2_V2_CONTRACT_REPAIR_20260810.md`
- P2 定位：`reports/experiments/A_MAINLINE1_P2_TRIPLE_VERIFICATION_20260810.md`
- 大文件登记：`reports/experiments/ARTIFACT_RELEASE_REGISTER.csv`
- YOLO 改进筛选：`reports/experiments/YOLO_INNOVATION_DIRECTIONS_20260811.md`
- 正式总表：`reports/experiments/leaderboard.csv`
- 延期/停止：`reports/experiments/DEFERRED_WORK_REGISTER.md`
