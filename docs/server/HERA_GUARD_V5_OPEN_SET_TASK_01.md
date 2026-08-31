# HERA-Guard V5 Open-Set TASK-01：proposal-domain 三分类复核

## 1. 目的

本任务验证一个单因素问题：在已经有效的 `base_crop` OMQ 证据上，增加真正的三分类
proposal-domain 像素复核，能否提高正式 Normal-CV3 的官方指标前沿。

三类固定为：

1. `foreground`：当前 proposal 是一对一匹配中的有效前景；
2. `structured_background`：不可匹配，且 crop 教师将其判为舰船/车辆等结构化易混目标；
3. `ordinary_background`：其余不可匹配背景。

可匹配但不是唯一获胜候选的 proposal 标为 `ignore=-1`，不参与三分类头训练，但保留
OOF 推理分数，避免把重复候选错误教成背景。

## 2. 冻结输入

- proposal ledger：`y5_proposal_inference_manifest.csv`；
- 一对一匹配证据：`nodes.csv`；
- 原始低阈值预测：`y5-all-preds-d4.json`；
- 正式三折 crop manifest：`formal_crop_manifest.csv`；
- P03 ConvNeXt-T 三折 checkpoint；
- 已完成并通过的 `base_crop/cache.npz`。

训练和评估始终按 held-out fold 交叉拟合。每个候选的 tight/context embedding 只由该候选
所属 held-out fold 的 P03 checkpoint 提取，三分类头和最终质量头也都不接触该 fold 的标签。

## 3. 冻结实现

- tight crop：正方形 `1.00×`，224；
- context crop：正方形 `1.25×`，224；
- 特征：两路 ConvNeXt-T 768D embedding + coarse one-hot；
- 三分类头：hidden dim 256，10 epoch，AdamW；
- 采样：coarse×label 分层逆频率，最大权重比 20；
- 最终 OMQ 增量：三类概率与 `p_foreground - max(p_bg)`，共 4 维；
- 最终质量头：与 `base_crop` 相同的 192 hidden、20 epoch、uniform ERM；
- 外层阈值：官方 matching，0.005 网格，FDR 上限 0.15。

唯一实验变量是上述 4 维开放集证据。不得同时改变 detector、crop 教师、D4、OTO、融合
权重、阈值口径或质量头容量。

## 4. 执行

```bash
bash scripts/server/run_hera_guard_v5_open_set_v1.sh
```

状态文件：

```text
/workspace/results/HERA-GUARD-V5-OPEN-SET-V1/status.txt
```

正常阶段依次为：`manifest → extract → open_set_train → augment → quality_train → evaluate → complete`。

## 5. 验收

技术门禁：

- 65,301 个候选全覆盖、唯一、顺序对齐；
- 三折均完整，embedding、概率和增量 OMQ 全部有限；
- OOF score 每行恰好来自对应 held-out fold；
- 原始 `base_crop` 行数、fold、image/category/bbox 不得改变；
- 三类概率逐行和为 1；
- `SHA256SUMS` 五项全部匹配。

科学比较基准：`base_crop` Normal-CV3 Recall=0.923613433、FDR=0.152686（FDR15
外层前沿口径）。主要报告总体 Recall/FDR、macro coarse Recall，以及舰船/飞机/车辆各类
Recall；不得只按三分类 balanced accuracy 宣告成功。

准入要求：相对 `base_crop` Recall 至少提高 0.5pp、FDR 不恶化到不可用，并且任一粗类
Recall 降幅不超过 0.5pp。未达到则记录为负向消融，停止该路线，不扫描融合权重。

## 6. 后续分支

- 通过：补 full-data 三分类头和 crop-only 部署等价闭环，再跑困难集与来源互斥 sentinel；
- 未通过：保留 crop-only 为当前最强归因基线，转入 crop-only GroupDRO/分组均衡单因素；
- 车辆表现显示 224 分辨率证据不足：只允许追加预注册的 selective high-resolution 复核，
  不对全量 proposal 做双尺度推理。
