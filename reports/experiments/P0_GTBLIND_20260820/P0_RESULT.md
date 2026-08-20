# P0-1: GT-blind 改类审计(冻结真实可部署基线)

> HERA-Field 方案第二章: 区分 oracle relabel upper bound vs 可部署主提交。

## 关键发现: 0.9620 是 oracle, 不是可部署结果

当前主提交 0.9620 的改类部分依赖 `gt_fine`(位置匹配 GT 的细类集合)判断"yolo 是否错",
是 **oracle relabel upper bound**。测试时没有 GT, 无法知道 yolo_wrong。

## 审计结果

| 改类方式 | R@FDR=.12 | 改类数 | corrected | broken |
|---|---:|---:|---:|---:|
| **oracle(GT 依赖)** | **0.9620** | 10,398 | 8,227 | **0** |
| deploy 路由器(thr=0.8) | **0.9405** | 9,747 | 7,043 | **538** |
| deploy 路由器(thr=0.7) | 0.9383 | 10,821 | 7,384 | 654 |
| deploy 路由器(thr=0.5) | 0.9340 | 12,748 | 7,741 | 805 |

路由器(预测 yolo_wrong)AUC = 0.9601(纯预测特征三折 cross-fit)。

## 核心结论

1. **oracle 改类 broken=0**(只改"yolo 错"的, 天然不 broken), corrected 8,227;
2. **deploy 路由器 broken=538**(把 yolo 对的误判成错, 改类后变错), corrected 7,043;
3. **oracle 上界 0.9620 vs 可部署 0.9405, gap = 2.15pp**;
4. broken 538 是 deploy 的核心代价——这正是 HERA-Field 批次 2(F2-F5 属性识别 +
   可观测性路由)要收复的空间: 更强的细类证据(属性组合 + 可观测性掩码)能让路由器
   更准确区分"该改/不该改", 减少 broken。

## 后续所有目标以 GT-blind 基线为准

- oracle 上界: 0.9620(改类依赖 GT);
- **可部署主提交(真实起点): 0.9405**(纯预测路由器);
- HERA-Field 的 F5(纯预测改类路由器) + F2-F4(属性组合识别)正是为收复这 2.15pp gap。

## 产物

- scripts/p0_1_gtblind_audit.py
- outputs/Y5-OER-RESTORE/p0-1-gtblind.json
