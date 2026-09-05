# 固定测试：一个候选，一次训练，一张结果表

**历史入口：2026-09-03晚已降为诊断旁证，不再是默认必跑。**
当前精简 Hard→Sentinel 流程见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。

版本：`paired_trend_review_v1`；计分：`platform_observed_20260831`。

**2026-09-03 22:20状态：17879固定160+40/A/B/anchor已完成；19864的BN开发负向并停止。**
新A的开发质量方向+3.244、确认−10.194，未通过已知官方方向核验；
不能称为已经可靠预测官方的筛选器。数据/规则保留，不事后改门禁。
完整[实跑结果](../reports/experiments/PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)。
成熟基线缓存已存在，**不要重复执行下面的启动命令**。
进度、预声明趋势核验和历史方向审计见[GPU执行报告](../reports/experiments/PAIRED_TREND_GPU_AND_DIRECTION_AUDIT_20260903.md)。
下面保留标准使用命令，供未初始化的新环境使用；不重新划分或改配方。

等待基线期间准备的下一项独立候选及开发区错误诊断，见
[P40后续方向与BN候选说明](../reports/experiments/P40_NEXT_IMPROVEMENT_AUDIT_20260903.md)。
它不改变本页冻结的A/B规则；已在独立代码快照执行，BN开发−2.962、确认未运行。

## 1. 实际固定了什么

|区域|图像|来源组|GT框|作用|
|---|---:|---:|---:|---|
|train|3,136|190|15,328|唯一训练来源|
|development|673|31|2,852|日常选一个阈值、比较方向|
|confirmation|672|34|2,753|开发增益>0.5质量贡献分才运行，不重新选点|

三部分每部分25类。按已有255个组分配，使用标签支持而非模型分数；每类至少50%的框进入
train。全部4,481图图像/标签逐项SHA校验，训练标签与COCO GT逐框核对，完整组和完全相同
图像无跨区。来源组仍是视觉代理，并不证明不存在所有近似重复图。

TU-160 train/dev/confirm为352/8/1框；HM为12/3/2；FSC为282/60/60。
稀有类的单框变化必须查看细类TP/FP/FN，0.5工程筛选带不是显著性检验。
这些图历史上被分析过，因此是**新模型未训练的固定留出来源**，不是全新盲测数据。
效果测试没有整张纯负图，背景风险由单独的已审核382张/100.139008MP回归明确报告，
不把背景比例调到与平台FDR相等。

索引：[合同](../data/splits/paired_trend_v1/contract.json)、
[样本及SHA](../data/splits/paired_trend_v1/manifest.json)、
[逐类支持](../data/splits/paired_trend_v1/support.json)、
[部署回归合同](../configs/experiments/paired_deployment_v1.json)。
这些文件包含相对路径，可随Git代码同步，不需要复制另一份图像数据。

## 2. 一次性准备和基线

在项目根目录运行。下面的环境变量是机器本地路径，服务器上按实际位置设置：

```bash
export PYTHONPATH="$PWD:$PWD/src"
PY=.venv/bin/python
DATA=../data
WEIGHTS=outputs/yolo26s.pt
BASE=outputs/PAIRED-TREND-BASELINE-V1

"$PY" scripts/run_paired_trend.py prepare \
  --data-root "$DATA" --weights "$WEIGHTS" --output "$BASE"

# GPU服务器：准备通过后执行；Mac只执行上面的CPU预检。
"$PY" scripts/run_paired_trend.py baseline --execute \
  --data-root "$DATA" --weights "$WEIGHTS" --output "$BASE" --device cuda:0
```

服务器使用现有YOLO26训练环境，不能误用已经转为BHC-DETR的 `scripts/train.py`。
依赖为PyTorch CUDA、Ultralytics（须支持YOLO26及自定义augmentations）、Albumentations、
NumPy、SciPy、Pillow、PyYAML。只验证已冻结数据不需要重新运行MILP划分器。

官方初始化SHA必须为`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`。
配方固定：YOLO26s+Rot90，S1024/160e → 无Mosaic、小学习率S1280/40e，固定last。
foundation batch12、adaptation batch8，单GPU；不复刻历史中途迁移三卡的随机性/有效batch。
它与最终P40具有相同训练成熟度，不承诺数值逐位等于旧full。

训练YAML的`train`和`val`都指向训练区。即使上游最后一次验证忽略`val=False`，也不会读到
开发或确认图。完整阶段校验连续epoch、有限loss、25类名称、checkpoint/results SHA。
已完成阶段可按SHA跳过；未完成阶段保留现场并报错，不自动resume或覆盖。
运行加文件锁防止同目录重复训练。路径、代码或参数变化要求新目录，不静默续接旧缓存。

基线训练结束自动缓存同一个checkpoint的开发和确认预测；开发自动选择自己的阈值，
不再从另一折模型迁移0.536。输出：

- `plan.json`、`materialized.json`：配方、训练来源、代码SHA、实际训练列表；
- `foundation/`、`adaptation/`：两阶段权重、日志及完成收据；
- `lineage.json`：从官方初始化到最终模型的来源记录；
- `evaluation/threshold.json`：开发选点及预测SHA；
- `evaluation/review.json`：基线缓存入口。

## 3. 每个新方法

新方法只用`$BASE/dataset.yaml`指定的训练图。后处理直接复用低阈值预测；短适配可复用
该基线的训练区checkpoint。新架构另训一个模型，不自动三折。方法改变需记录在训练配方中。
当前GPU推理入口支持标准25类YOLO；其他架构必须先接对应适配器，不能把DFINE权重当YOLO。

```bash
"$PY" scripts/run_paired_trend.py candidate --execute \
  --data-root "$DATA" --checkpoint /path/to/candidate/last.pt \
  --lineage /path/to/candidate/lineage.json \
  --baseline "$BASE/evaluation" --output outputs/CANDIDATE-ID --device cuda:0
```

`lineage.json`由训练流程导出，格式同基线，包含bundle SHA、最终checkpoint SHA、完整
train_image_ids、training_recipe和逐个祖先训练来源。入口要求所有祖先只用此train，
起点必须是已审核官方初始化；旧full、其他折checkpoint、未审核外部预训练均拒绝。
这验证的是所记录血缘与资产一致性，不是从权重数学推断训练历史，禁止手填假血缘绕过。

自动顺序只有：开发推理 → 官方计分最大化选点 → 开发比较 → 明显正向才确认 → 单份review。
负向或小于等于0.5分增益不会运行确认推理。确认使用同一checkpoint及已冻结开发阈值，
只做一次应用，不生成确认oracle。确认与开发同方向后进入B回归，不自动full或提交。
同样训练配方、同样推理下的分差优先；`recipe_matches_baseline=false`必须解释预算差异。
没有大图实测时延时只报六项质量贡献/7，`delta_total_score=null`，不把0秒填入总分。

## 4. B：交付前工程回归

```bash
"$PY" scripts/run_paired_deployment_regression.py run \
  --data-root "$DATA" --background outputs/MACROSHIFT-BACKGROUND-100MP-FROZEN \
  --checkpoint /path/to/candidate/last.pt --review outputs/CANDIDATE-ID \
  --output outputs/CANDIDATE-ID-DEPLOYMENT --device cuda:0
```

基线先运行一次同样命令并缓存。回归输出只看`regression.json`：

- 已审核Background-100MP的FP总数、FP/100MP和细类归因；
- 两张固定100MP画布的TP/FP/FN与接缝变化（部分taxonomy、训练来源，不报平台精度总分）；
- 当前safe1024/overlap256、batch4、S1280实际入口与离线逐框一致；
- 预热后大图时延（含预处理、切片、融合和结果序列化，不含读图）、GPU/软件身份。

两张画布是六张训练区稀疏/密集原图的原像素摆放，空白为黑色；这只检查切片与速度，
不能代表未知测试场景的背景纹理。真实背景FP仍由已审核背景集单独判断。
GPU入口通过不等于镜像测试；未来实际打包后再跑一次真正的容器/输入输出挂载检查。
不同GPU或软件环境下的时延不可直接相减。不因一次质量增益忽略明显背景恶化。

## 5. 当前边界

已完成真实数据冻结、标签/图像完整性、可运行入口、背景合同、原像素画布几何检查、
CPU契约端到端测试（替换GPU后端，保留真实评分及分支逻辑）。
**GPU基线与B工程验收已完成，缓存可复用；但新A确认方向与已知官方增益相反。**
工程可运行不等于官方趋势可靠。当前不以新A单独决定正式提交，也不能用历史seen高分
或测试桩的完美预测证明可靠；不为追平一次正式结果事后重分数据或调整确认工作点。
