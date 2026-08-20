# A5: OTM/OTO 零训练诊断(2026-08-20)

> HERA-YOLO 方案4 §四.1: YOLO26 一对多/一对一互补使用, OTO 作 precision/uniqueness 证据。

## 诊断设计

- Y5 fold0 last.pt 同时含 one2many(OTM)与 one2one(OTO)head(end2end=True);
- 对 fold0 1507 图分别跑 OTM(高召回+NMS)与 OTO(高精度无NMS)推理;
- OTM 23,822 候选, OTO 24,725 候选(0.001 阈值)。

## 核心结果: OTO 支持率 vs 错误类型

| 类型 | 数量 | OTO 支持 | 支持率 |
|---|---:|---:|---:|
| TP | 6,858 | 6,832 | **0.996** |
| FP_BG | 15,259 | 9,884 | **0.648** |
| FP_CLS | 1,623 | 1,557 | 0.959 |
| FP_DUP | 82 | 80 | 0.976 |

## 结论

1. **OTO 是 FP_BG 的强判别信号**: TP 几乎都有 OTO 支持(99.6%), 而 35.2% 的
   FP_BG 无 OTO 支持——"has_oto_support" 可直接作为 OER 的 precision 特征;
2. OTO 对 FP_CLS(位置对类错)几乎不区分(95.9% 支持)——因为 OTO 也是细类分类,
   位置对了但类错的, OTO 也会报同类框; 这正是"细类纠错"应由 crop 教师(A3)负责,
   而非 OTO;
3. OTO 对 FP_DUP 也不区分(97.6%)——重复框在 OTO 里同样存在;
4. **分工清晰**: OTO → 压 FP_BG(背景), crop 路由器(A3)→ 纠 FP_CLS(细类), 
   edge/NMS → 压 FP_DUP(重复)。三类错误各有对应证据, 全部并入 OER 联合裁决;
5. OTO 独立精度 FDR=0.71(0.001), 不能直接替换 OTM, 只能作 support 特征。

## 下一步

- fold1/2 OTO 诊断跑完后, 把 has_oto_support 加入 OER node_validity 特征,
  全量三折重训, 看 Recall@FDR=0.12 能否进一步突破 0.9584。
