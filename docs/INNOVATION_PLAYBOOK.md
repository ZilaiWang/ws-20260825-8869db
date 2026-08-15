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
| Y3 | 层次粗细类辅助损失 | TU-160/F-22/LQS/HM 类间混淆 | `scripts/server/run_innovation.sh y3` | ✅ 就绪 |
| Y4 | AFSS 反遗忘采样 | 困难样本采样 | `scripts/server/run_innovation.sh y4` | ✅ 就绪（诊断+采样器） |
| Y5 | 90° 旋转一致性 | 任意朝向 | `scripts/server/run_innovation.sh y5` | ✅ 就绪 |
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

## 4. 训练期模块（已完成，本机 ultralytics 8.4.103 + torch 2.13 MPS 验证）

训练引擎 `scripts/train_cv3_oof.py` 新增 `--innovation {none|y3|y4|y5}` 开关，加对应超参
（`--coarse-gain` / `--suff-json` + `--easy-floor` / `--rotate90-p`）。代码在
`src/rsdet/innovation/`，ultralytics 依赖延迟 import，仅 coarse 无依赖随包导入。

### Y3 层次粗细类辅助损失（`hierarchical_loss.py` + `trainers.hierarchical_trainer`）
- 保留 25 类主损失；ship/aircraft/vehicle 粗类辅助损失**并入 cls 分量**（loss 保持 3 分量，
  与 ultralytics validator/EMA 契约兼容）；
- 粗类 logit = 细类 logit @ 归属矩阵（25→3），正样本上做 BCE；
- 仅训练期存在，不改输出接口/推理成本。准入：FP_CLS 净下降、目标细类 macro 改善。

### Y4 AFSS 反遗忘采样器（`afss_sampler.py` + `trainers.afss_trainer`）
- 充分度 → 权重 `max(1-suff, easy_floor)`，加权采样（困难图持续、容易图低频回看）；
- `afss_diagnose.py` 新增输出 `per_image_suff`（image_id→充分度），供采样器对齐；
- 采样器经 `get_dataloader` 覆盖注入（train 集按 split_view 顺序对齐 suff_list）。

### Y5 90° 离散旋转增强（`rotate90.py` + `trainers.rotate90_augmentations`）
- 经 ultralytics `hyp.augmentations` 注入 `albumentations.RandomRotate90`（HBB 精确变换，bbox 自动同步）；
- 旋转前后一致性损失（两次前向）留作进阶，首版做低风险增强。

### 已验证
- 单元测试 `tests/test_innovation_modules.py`（6 项）：coarse 映射 / 采样权重 / 旋转 /
  Y3 损失前向（box/dfl 不变、cls 含粗类、梯度回传）；
- 端到端（迷你数据 1 epoch 训练+验证）：Y3/Y4/Y5 三个 trainer 注入均无报错。

## 5. 每个实验的统一输出（材料19 第6节）

pooled Recall/FDR、macro(4/20/1)、fold0/1/2、TP_new/TP_broken/FP_BG/FP_CLS/FP_DUP/FP_LOC、
车辆 tiny/small、LQS/HM、TU-160/F-22、cross-fit 工作点 —— 全部由 `evaluate_experiment.py`
自动产出，可直接横向对比。
