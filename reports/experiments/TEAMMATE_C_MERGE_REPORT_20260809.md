# 队友 C（潘扬东杰）代码包合并报告

日期：2026-08-09  
合并方：负责人（王子莱）  
来源：`xh-202625-master/`（队友打包）+ `BHCDETR-UHR-R50-1024-devv2-seed42-5090/`（训练产物）

## 1. 背景

- 队友的任务是 C 分工：**固定预算下的小目标候选恢复**（对应 `doc/第二阶段分工.md` C 节）。
- 核心问题：车辆 402 个 GT 中 81 个连候选都没形成；候选形成是首要瓶颈（crop 分类约 0.985，不是识别问题，是"没报出来"）。
- 队友实现：**BHC-DETR**（arXiv:2512.24074v1，CVPR 2026：解耦查询 + BHCL 层级对比学习）+ **UHR 小目标扩展**（arXiv:2604.21435v1：Gain Map / LPM / ISSGA / C4 局部细节）。
- 已在 `dev_v2_airport_proxy_k60` 划分上完成 12 epoch 训练（seed42、R50-1024、从 BHCDETR 基线 warm-start）。

## 2. 合并内容

### 2.1 新增源码（src/rsdet/）

| 文件 | 说明 |
|---|---|
| `models/bhcdetr.py` | 论文对齐网络：分类/定位查询解耦、共享自注意力、独立分支 + UHR 可选路径 |
| `models/bhcdetr_adapter.py` | 项目 `Prediction` 契约推理适配器（letterbox、坐标恢复） |
| `models/bhcl.py` | Balanced Hierarchical Contrastive Loss（Eq.7-10：层级权重、类别平衡、EMA 原型） |
| `models/hierarchy.py` | 3 大粗类 + 25 细类层级标签树 |
| `models/detection_loss.py` | Hungarian 匹配 + focal + L1/GIoU + BHCL |
| `models/uhr_small_object.py` | Gain Map head、IoF-sum target、DFL、LPM、ISSGA 稀疏 token 选择 |
| `models/prototype_refiner.py` | 原型细化器（可选） |
| `models/ultralytics_adapter.py` | 旧 YOLO/RT-DETR 历史适配器（不参与当前入口） |
| `data/bhcdetr_dataset.py` | 双视图翻转/平移增强数据集 |
| `data/imbalance.py` | 不均衡清单生成（Ultralytics 消费格式） |
| `data/object_crops.py` | 对象裁剪工具 |
| `engine/trainer.py` | **完整替换主项目 stub**：可执行 BHC-DETR 训练引擎 |
| `engine/inference.py` | 单图/10K 滑窗推理核心 |
| `evaluation/coco_metric.py` | COCO 指标 |
| `postprocess/nms.py` | **替换 stub**：确定性框架无关 NMS |
| `postprocess/tile_fusion.py` | **替换 stub**：跨 tile 坐标恢复 + 分组 NMS 融合 |

### 2.2 修改文件

- `models/registry.py`：增加 bhcdetr 惰性注册
- `scripts/train.py` / `infer.py` / `benchmark.py` / `export_coco.py`：接入 BHC-DETR 主线
- `configs/train.example.yaml` / `infer.example.yaml`：更新为 BHC-DETR 配置格式
- `README.md` / `pyproject.toml`（新增 `[model]` 可选依赖）/ `requirements-model.txt`（新增）/ `docs/CHANGELOG.md`

### 2.3 新增配置

- `configs/models/`：bhcdetr R50 1024（含 UHR 变体、5090 变体）、m1/m2/m3 模型配置
- `configs/bhcdetr.smoke.yaml` / `bhcdetr_uhr.smoke.yaml`：冒烟链路配置

### 2.4 新增测试（16 个文件）

`test_bhcdetr_*`、`test_bhcl`、`test_uhr_small_object`、`test_imbalance`、`test_tile_fusion`、`test_inference_pipeline` 等。

### 2.5 文档

`docs/BHCDETR_IMPLEMENTATION.md`、`docs/UHR_BHCDETR_TRAINING.md`、`docs/MODEL_TECHNICAL_ROUTE.md`、`docs/PROJECT_AUDIT.md`。

### 2.6 训练产物归档

`outputs/BHCDETR-UHR-R50-1024-devv2-seed42-5090/`（含 best.pt / last.pt / metrics.jsonl / val 预测与评估 JSON / threshold_sweep）。权重不进 git（已被 gitignore 排除）。

## 3. 训练结果摘要（队友实测）

- 训练：`dev_v2_airport_proxy_k60` 划分，3548 训练 / 933 验证图，12 epoch，global_step 10644，best_val_loss 6.399。
- 低阈值（score≥0.001）验证：Overall Recall **0.9337**，FDR 0.9883（大量低分候选，供阈值扫描）。
- coarse 调阈后工作点：Overall Recall **0.8835**、FDR **0.1982**（通过硬门槛 Recall≥0.85/FDR≤0.20，但仅 dev 单折）。
- 分大类似：船 Recall 0.752 / FDR 0.289；飞机 Recall 0.917 / FDR 0.176；车辆 Recall 0.539 / FDR 0.453（调阈后）。
- 车辆 Recall 0.539 仍低于 M1（0.617），**该模型目前不满足 C 分工"车辆 Recall 目标 0.85"的要求**，需与 M1 OOF 配对分析后再决策。

## 4. 合并后验证

- `pytest tests/`：**324 passed, 5 skipped**。
- `scripts/train.py --config configs/bhcdetr_uhr.smoke.yaml --dry-run`：通过（3548/933、25 类、UHR 使能）。

### 4.1 合并时清理的两个遗留问题

1. `tests/test_splits.py`：引用了不存在的 `splits.GroupStats` / `derive_source_key` / `stratified_group_split`（两个代码包中均不存在该 API），**删除**。主项目已有 `test_airport_proxy_split.py`、`test_cv3_split.py` 覆盖等价划分逻辑。
2. `tests/test_imbalance.py::test_training_dry_run_materializes_both_stages`：引用旧 YOLO `train()` 接口（`model: {"family": "yolo"}`），与新 trainer 只接受 `bhcdetr` 冲突，在队友包中同样失败。**标记 skip 并注明原因**（非合并引入的回归）。

## 5. 给团队的注意事项

1. **该模型是 C 分工方向的强基线之一**：它属于"换架构补盲区"路线，与 M1（YOLO26-s）形成异构对照，可支撑 D 分工的 M1/M3 配对分析和 C 分工的候选恢复。
2. **不可直接作为最终模型**：仅 dev 单折、无正式 CV3 OOF、车辆 Recall 未达标；正式三折 CV3 训练 + 低阈值 OOF 尚未完成。
3. **下一步建议**：
   - 在正式 `cv3_airport_proxy_k60_v2` 上跑三折，与 M1 OOF 做四类配对（both / M1-only / BHC-DETR-only / neither）；
   - 重点读出车辆 unique GT 增益（candidate 机制核心指标）与新增 FP；
   - 若车辆增益不明显，按 C 分工的 near-miss 审计路径检查候选消失原因，而非继续堆模型容量。
