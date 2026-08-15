# 创新实验手册：尝试方向 + 启动入口 + 分析接口

本手册汇总"要尝试的创新方案"的**启动入口**与**分析接口**，让服务器空闲时
能直接启动实验，跑完直接得到可对比结果、并能定位"哪里出了问题"继续改。

冻结基线：M1 YOLO26-s / 1024 / CV3 v2 / seed42 / 160 epoch / last.pt。
官方口径：pooled Recall≥0.85、FDR≤0.20、10K≤20s；V1.6 七项排名（大类 macro）。

## 1. 统一评估 + 错误诊断（所有实验复用）

```bash
# 统一评估：产出 pooled/macro/fold 分层/分层错误/专项(HM/LQS/TU-160/F-22)
PYTHONPATH=src python scripts/evaluate_experiment.py \
  --predictions <oof_predictions_list.json> --model-key <NAME> \
  --output <evaluate_NAME.json>

# 错误诊断：定位"哪里出了问题"（按类/尺寸/fold/source-group 聚合）
PYTHONPATH=src python scripts/analyze_experiment_errors.py \
  --cases <evaluate_NAME.cases.json> --output <diagnose_NAME.json>
```

`--predictions` 格式：`[{"image_id":1,"category_id":2,"score":0.8,"bbox_xyxy":[x1,y1,x2,y2]}, ...]`。

## 2. 尝试方向清单与状态

| 编号 | 方向 | 针对错误 | 启动入口 | 状态 |
|---|---|---|---|---|
| Y1 | cross-fit 校准（FRACAL） | 域间分数漂移 | — | ✅ 已完成（C2 准入） |
| Y2 | P2 正式基线 | 车辆无候选 | `scripts/server/run_y2_p2.sh` | ✅ 就绪 |
| Y3 | 层次粗细类辅助损失 | TU-160/F-22/LQS/HM 类间混淆 | 待补（loss 模块） | ⏳ 训练期代码待写 |
| Y4 | AFSS 反遗忘采样 | 困难样本采样 | `afss_diagnose.py`（诊断） | ⏳ 诊断就绪，采样器待写 |
| Y5 | 90° 旋转一致性 | 任意朝向 | 待补（增强/loss 模块） | ⏳ 训练期代码待写 |
| N2-CFG | 舰船/车辆前景门控 | FP_BG | `scripts/server/run_n2_cfg.sh` | ✅ 就绪（等 N0 CSV） |
| M3 | RT-DETR-L 异构检测 | FN_MISS | `scripts/server/run_m3_cv3.sh` | 🔄 训练中 |
| E | 10K 全局对象层 | 跨 tile/20s 门禁 | `benchmark_10k_pipeline.py` | ⏳ 等 GPU 实测 |

## 3. 启动入口约定

所有 `run_*.sh` 遵循统一框架（见 `run_m3_cv3.sh` / `run_y2_p2.sh`）：

1. 前置 SHA 校验（预训练权重 / 数据锁 / p02）；
2. 三折：数据锁 verify → materialize → dry-run → 训练 → 推理 → finalize；
3. aggregate 审计 → **统一评估 + 错误诊断**；
4. 无 checkpoint 回传包。

训练引擎：`scripts/train_cv3_oof.py`（支持 `model.architecture`，如
`yolo26s-p2.yaml`，按结构初始化 + 预训练权重迁移）。

## 4. 剩余工作（训练期模块，需 ultralytics 环境验证）

### Y3 层次粗细类辅助损失（材料19 第三优先）
- 保留 25 类主损失；增加 ship/aircraft/vehicle 粗类辅助分类损失；
- 在正样本分类特征上增加分层原型/监督对比约束（简化版：粗类辅助 + 原型）；
- 仅训练期存在，不改输出接口/推理成本。准入：FP_CLS 净下降、目标细类 macro 改善。

### Y5 90° 离散旋转一致性（材料19 第五优先）
- 0/90/180/270° 离散旋转增强（HBB 可精确变换）；
- 旋转前后分类/质量一致性损失（低风险，不改造旋转等变骨干）。

### Y4 AFSS 采样器（材料19 第四优先）
- `afss_diagnose.py` 已产出充分度诊断（R1-6：151 图充分度=0，困难图集中 ship/small）；
- 下一步：训练期采样器（困难图持续参与、容易图低频回看）。

## 5. 每个实验的统一输出（材料19 第6节）

pooled Recall/FDR、macro(4/20/1)、fold0/1/2、TP_new/TP_broken/FP_BG/FP_CLS/FP_DUP/FP_LOC、
车辆 tiny/small、LQS/HM、TU-160/F-22、cross-fit 工作点 —— 全部由 `evaluate_experiment.py`
自动产出，可直接横向对比。
