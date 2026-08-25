# Y1 严格交叉拟合校准结果

日期：2026-08-11

范围：M1 YOLO26-s 正式 CV3 OOF，4481 张图、20933 个 GT，候选下限 0.001。

## 1. 目的与协议

本实验不重新训练模型，而是在冻结 M1 OOF 上回答：指标过门后，能否通过无泄漏的分数校准进一步降低 FDR。

每次把一折作为完全 held-out 评估折，只用另外两折选择阈值、类别先验和空间统计，最后合并三份 held-out 预测。指标全部调用冻结的 `official_eval_v1` 和 `official_ranking_v1_6`。

四组方法：

- C0：单一全局阈值，正式参考组；
- C1：舰船、飞机、车辆三个粗类阈值；
- C2：细类频率先验校准 + 全局阈值；
- C3：在 C2 上增加类别空间分布分形维数。

OOF 只保存 NMS 后的已选类别和单个置信度，没有 NMS 前完整类别 logits。因此 C2/C3 是**后 NMS 标量校准筛选**，C3 只是 FRACAL-inspired proxy，不是 FRACAL 方法的完整复现。

## 2. 合并 held-out 结果

| 方法 | pooled Recall | pooled FDR | macro Recall | macro FDR | 官方硬门槛 |
|---|---:|---:|---:|---:|---|
| C0 global | 0.9176 | 0.1990 | 0.8654 | 0.2383 | 通过 |
| C1 coarse | 0.9166 | 0.2020 | 0.8644 | 0.2399 | **不通过** |
| C2 prior | 0.9135 | **0.1593** | 0.8631 | **0.2085** | 通过 |
| C3 fractal proxy | 0.9134 | **0.1590** | 0.8631 | 0.2090 | 通过 |

C2 相对 C0：

- pooled Recall：-0.0041（-0.41 个百分点）；
- pooled FDR：-0.0397（-3.97 个百分点）；
- macro Recall：-0.0024；
- macro FDR：-0.0297。

C2 三折 FDR 相对 C0 的变化依次为 -0.0526、-0.0152、-0.0486，方向一致；Recall 三折分别下降 0.0059、0.0015、0.0050。这不是某一折偶然带来的 FDR 收益。

## 3. 粗类结果解读

| 粗类 | C0 pooled R/FDR | C2 pooled R/FDR | 判断 |
|---|---:|---:|---|
| 舰船 | 0.8531 / 0.3899 | 0.8386 / 0.2987 | 明显降低虚警，但 Recall 有代价 |
| 飞机 | 0.9342 / 0.1482 | 0.9319 / 0.1260 | 收益稳定、Recall 代价小 |
| 车辆 | 0.6119 / 0.6239 | 0.5970 / 0.5266 | 虚警改善，但不解决车辆候选和定位不足 |

这进一步确认：校准能优化“输出哪些已有候选”，但无法替代 P2 高分辨率候选通路。车辆仍是结构性问题。

## 4. 准入决策

- **C2 准入**：当前可用的推理后校准分支；
- **C1 不准入**：合并 held-out FDR=0.2020，跌破官方 0.20 门槛，三折相对 C0 均恶化；
- **C3 不准入**：相对 C2，macro FDR 反而 +0.00042，Recall -0.00005，没有空间分形项的独立价值；
- **不启动完整 FRACAL 改造**：当前空间项无增量，不值得为完整 pre-NMS logits 链路立即增加工程复杂度。

自动准入规则和结果写入 `outputs/Y1-CROSSFIT-CALIBRATION-V1/decision.json`。

## 5. 对后续的影响

1. Y2 正式 P2 首先使用 C0 作为与 M1 的结构对照，不把校准收益混入 P2 因果判断。
2. P2 通过结构准入后，再单独报告 P2+C2，检查 C2 是否能在候选增多后继续控制 FDR。
3. Y3 只修复 P2 新候选的质量，不继续堆叠校准变体。

## 6. 索引

| 内容 | 路径 |
|---|---|
| 配置 | `configs/experiments/y1_crossfit_calibration.yaml` |
| 入口 | `scripts/y1_crossfit_calibration.py` |
| 核心实现 | `src/rsdet/postprocess/yolo_calibration.py` |
| 单元测试 | `tests/test_yolo_calibration.py` |
| 完整结果 | `outputs/Y1-CROSSFIT-CALIBRATION-V1/calibration_result.json` |
| 摘要 | `outputs/Y1-CROSSFIT-CALIBRATION-V1/summary.json` |
| 机器准入决策 | `outputs/Y1-CROSSFIT-CALIBRATION-V1/decision.json` |
