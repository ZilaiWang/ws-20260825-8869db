# GPT 讨论包 — README 索引

> 用途: 给外部 AI(GPT)深度讨论"如何设计创新点、把指标推到最好"的完整输入包。
> 建议阅读顺序: **00 主状态 → 01 赛题方案 → 02-10 创新评估 → 11-15 N2-CFG → 16-18 M3 → 19-21 时延 → 22-33 基线与历史 → 34-37 数据**。
> 核心问题集: 见 `00_MAIN_STATUS_AND_PROBLEMS_20260820.md` 第 3/4/6 节。

## 文件清单(39 个)

### 状态与赛题
| # | 文件 | 内容 |
|---|---|---|
| 00 | MAIN_STATUS_AND_PROBLEMS_20260820.md | **主文档**: 进度/最好成绩/未解问题/激进目标/讨论弹药(必读) |
| 01 | 改进方案1.md | 我们的核心方法方案(改进方案 1 全文) |
| - | 比赛评分方案-V1.5.pdf | 官方评分细则(Recall/FDR/时延门禁 + 排名口径) |

### 创新评估(三创新 × M1)
| # | 文件 | 内容 |
|---|---|---|
| 02 | Y5_DEEP_EVAL.md | Y5-ROT90 深度评估(最终链基座, t=0.1 双占优) |
| 03 | Y4_DEEP_EVAL.md | Y4-AFSS 深度评估(达标但全弱于 Y5) |
| 04 | Y3_DEEP_EVAL.md | Y3-HIER 深度评估(少数类塌陷, 教训) |
| 05 | Y5_M1_PAIRED.md | Y5 vs M1 互补分析(净增 888 GT) |
| 06 | Y5_M1_FUSION.md | Y5+M1 融合(+0.62pp, FDR 不劣化) |
| 07 | TRIPLE_MISS.md | 三模型全漏 91 个(真残余短板) |
| 08 | FP_CLS_CONFUSION.md | M1 细类混淆基线(兄弟机型) |
| 09 | COMPARISON_ALL.md | 三创新 vs M1 并排表(人类可读) |
| 10 | T4_DECISION.json | 门控推理决策(2/3 门禁过) |

### N2-CFG(背景门控)
| # | 文件 | 内容 |
|---|---|---|
| 11 | N2_CFG_PLAN.md | 门控方案计划(2026-08-14 冻结) |
| 12 | N2_NEGATIVE_SUFFICIENCY.md | 负样本量评估(33 个白名单不足) |
| 13 | N2_LINK_PREFLIGHT.md | 链路预检(锁修复/报错路径) |
| 14 | N2_FIRST_RUN.md | 首次完整运行(门禁 4/7 过) |
| 15 | N2_DIAGNOSIS.md | **根因诊断**(ship/vehicle 负样本不足) |

### M3 教师证据
| # | 文件 | 内容 |
|---|---|---|
| 16 | M3_TEACHER_EVIDENCE.md | 教师证据交付说明(给 C) |
| 17 | M3_HARD_POSITIVES.csv | 1,313 个 hard positives 明细 |
| 18 | M3_STRATIFIED.json | 分层统计 |

### E 时延
| # | 文件 | 内容 |
|---|---|---|
| 19 | E_COMBINED_FORMAL.md | 组合正式实测(p50 14.28s ≤20s) |
| 20 | E_COMBINED_REHEARSAL.md | 预演(wall 逼近风险) |
| 21 | E_TASK_GAP.md | E 侧任务清点 |

### 基线与历史(背景)
| # | 文件 | 内容 |
|---|---|---|
| 22 | M1_FORMAL_RESULT.md | M1 正式 OOF 结果与恢复审计 |
| 23 | YOLO_INNOVATION_DIRECTIONS.md | 创新方向头脑风暴(早期) |
| 24 | NEXT_STAGE_MASTER.md | 团队创新执行总纲 |
| 25 | PRE_INNOVATION_CLOSURE.md | 创新前收口审计 |
| 26 | Y2Y3_IMPLEMENTATION.md | Y2/Y3 正式实现记录 |
| 27 | R1_POSTRERANK_NMS.md | R1-6 后处理链(候选/阈值) |
| 28 | R1_RERANK.md | 排序重排结果 |
| 29 | R1_SHIP_VEHICLE_NMS.md | 舰船/车辆后处理 |
| 30 | N0_REVIEW_GUIDE.md | 背景盲审指南 |
| 31 | INNOVATION_RUNBOOK.md | 创新对比运行手册 |
| 32 | E_SUMMARY.md | E 成员工作总结 |
| 33 | M1M3_POSTPROCESS_PLAN.md | M1/M3 后处理分析计划 |
| 36 | D_STAGE2_PLAN.md | D 第二阶段规划(教师发现) |
| 37 | N0_FINAL_CHAIN.md | 背景证据链终审 |

### 数据(GPT 可直接消费)
| # | 文件 | 内容 |
|---|---|---|
| 34 | Y3_DEEP_EVAL_DATA.json | Y3/M1 阈值曲线数据 |
| 35 | COMPARISON_ALL_DATA.json | 三创新对比结构化数据 |
