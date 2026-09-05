# P40 船舶/车辆召回：当前漏检诊断、阈值否决与温和采样实验

日期：2026-09-04。状态：**诊断与受控阈值完成；RFS与targeted EQLv2均在Hard为负向并停止；RFS/EQL/EFL方法族关闭**。

保留 [P40 + view-consistency D4 飞机候选](FIXED_PROXY_LITE_AND_P40_VIEWS_20260903.md)，
本轮不覆盖它的权重、配置、阈值或结果。正式最优仍为 v2.0 / 76.6010；没有 full 训练、
Docker 打包或官方提交。下面两项训练实验都已有完整三折和Hard结论，均未获准进入Sentinel。

## 1. 本轮回答的问题

用户希望继续改善 Ship / Vehicle，尤其 Recall。重新对照[方案13](../../../改进方案13.md)的细类阈值/层级收缩、
[方案14](../../../改进方案14.md)的整图采样与长尾检测、[方案15](../../../改进方案15.md)的尺度机制及已有失败后，先区分：

1. 当前检测器根本没有候选，还是正确候选存在但分数低？
2. 只调阈值，能否在保留所有当前检测的前提下获得净质量增益？
3. 如不能，保留已有效的 P40 配方，仅温和增加弱类训练曝光能否改善分数分离？

本轮固定使用 [`fixed_proxy_lite_v1`](../../configs/experiments/fixed_proxy_lite_v1.json)：
Normal 只提供 OOF 校准和解释；Hard 质量贡献增益 **>0.5** 才进入 Sentinel；Sentinel
增益 **>0** 才成为待成本/风险审查的候选。不增建测试集，不要求所有六个率都严格同涨，
但必须完整报告三粗类和细类变化。质量贡献是六个指标点数之和除以7，**不含时延**。

这些历史 Hard/Sentinel 已被多次查看，不能称作盲测，也未证明能稳定预测正式分数趋势。
研究 P40 是每折 **S1024/40e → 1280/40e**；正式已提交权重是 **160e → 40e**。
本地研究召回率与正式页面召回率不可直接混算或相加。

## 2. 当前 P40 的实际 Normal OOF 漏检结构

输入为现有三折4481图、20933 GT、0.001低阈值预测，工作点继承 Normal FDR15 的
fold0/1/2 阈值 `0.546 / 0.516 / 0.501`。每一项都是同一模型、同一预测缓存的两个端点，
没有重新训练或用 Hard 标签选择阈值。

|细类|GT|工作点漏检|在0.001候选中重新匹配|0.001下仍漏检|
|---|---:|---:|---:|---:|
|HM|17|15|15|0|
|LQS|30|26|22|4|
|QHS|641|274|255|19|
|MS|1994|634|542|92|
|Vehicle / FSC|402|269|249|20|

船舶949个漏检中，834个（87.9%）在低分候选里可匹配；车辆269个漏检中有249个
（92.6%）。这支持优先检查**评分/排序、弱类学习不足**，而不是继续追加昂贵的检测视图。
但它不是“把阈值降低就能无成本获得88%/93%的增益”。

例如Vehicle：工作点133TP/40FP，floor为382TP/16104FP。四船类floor FP分别709、1569、
12293、14874。巨量误检与低分TP混在一起，决定了单纯宽松阈值的代价。
floor经过既有NMS和max_det截断，既不是检测器理论上限，也不是可实现的模型指标。

工作点 Ship 的 FN_CLS 只有63个，远少于 FN_MISS 863个；Vehicle FN_CLS=0、FN_MISS=263。
因此立即复制一个“只改细分类名”的船舶精识别头，不能解决大多数漏检。不过，这也**不能
否定分类训练损失**：弱类分类分数过低，同样会表现为阈值后的 FN_MISS，不一定表现为错类。

数据来源与代码：

- [诊断JSON](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/capacity_named.json)
- [当前P40诊断脚本](../../scripts/analyze_p40_recall_capacity.py)
- [此前新A诊断](PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)是另一套160+40血缘/划分，
  不能用它的60个Vehicle GT代替本节402个GT。

## 3. P40-RECALL-CALIBRATION-V1：已完成并停止

这是新P40上的受控校准，不是重跑旧ScaleRoute R2 Ship-only MacroRisk。

冻结方法：只处理0/1/2/3/24，飞机完全旁路；每折仅在另外两折Normal OOF上拟合。每个
细类只允许降低原工作点阈值，保留原有框、分数及类别。用真实**粗类内细类宏平均**的
质量点数做一遍固定顺序坐标选择，粗类拟合FDR上限为 `max(0.15, 原拟合FDR)`。
继承MacroRisk的50 GT先验、0.2/0.4/0.8 logit位移限制，锚定在原P40阈值；不足10 GT或
2个独立组则不动作。收缩后向更保守的格点取整，再检查拟合质量，未改善则回退原点。

格点0.001–0.996、步长0.005，加精确原阈值。Hard/Sentinel不参与拟合、不扫阈值。
既有MacroRisk中间的coarse anchor使用pooled曲线，不等同于最终macro目标；本实验
直接计算macro目标，**没有改写旧MacroRisk算法或历史负向结果**。

实际拟合几乎全部保持原阈值：仅fold0的QHS和Vehicle由0.546微降至0.541。

|Hard指标|原P40|新阈值|差异|
|---|---:|---:|---:|
|Ship Recall|42.5686%|42.6773%|+0.1087pp|
|Ship FDR|7.0603%|7.0420%|−0.0184pp|
|Vehicle Recall|22.8261%|22.8261%|0|
|Vehicle FDR|26.3158%|26.3158%|0|
|质量贡献（不含时延）|44.762524|44.778738|**+0.016214**|

仅QHS多1TP，没有其他TP/FP变化；飞机逐框不变。在外层Normal结果上，QHS反而多1FP，
没有增加TP。Hard不达+0.5，**停止，不运行Sentinel，不扩大阈值搜索**。
这只否定本配方，不证明任何校准都不可能有效；但当前证据不支持继续把主要时间放在阈值上。

实现复用现有部署 `fine > coarse > global` 阈值语义，单测覆盖逐框相等、边界等于阈值、
飞机旁路、原检测保留、非有限值拒绝、稀疏支持回退和FDR风险。
Normal跨拟合复用历史OOF，并非重新训练嵌套CV；保留这一有限独立性说明。

- [配置](../../configs/experiments/p40_recall_calibration_v1.json)
- [拟合/回放入口](../../scripts/run_p40_recall_calibration.py)
- [核心逻辑](../../src/rsdet/postprocess/recall_calibration.py)
- [拟合证据](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-RECALL-CALIBRATION-V1/fit.json)
- [Hard结果](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-RECALL-CALIBRATION-V1/hard.json)
- [可随代码保留的小型Hard汇总](p40_recall_results_20260904/calibration_hard.json)

## 4. P40-WEAK-RFS-V1：已完成并在Hard停止

### 4.1 为什么值得试，但不保证有效

方案14的MacroExpert同时改六类标签头、模型容量、12/8倍重复和大量飞机图保留比例。
其失败不等于“标准25类P40上的温和整图采样”已经失败。

本次不改模型头，不加新loss，不加外部数据，不删飞机图，不追加背景标签。目标是让已能
产生弱类候选的检测器，在适配阶段更经常看到弱类训练图，尝试把正确框分数与误检分开。
不从Normal OOF、Hard或Sentinel错例采训练图，权重只由本折训练标签的图像频率决定。

对类别 `c∈{HM,LQS,QHS,MS,Vehicle}`，训练图出现频率为 `f_c`：

`r_c = min(3, max(1, sqrt(0.10 / f_c)))`；整图权重取其包含弱类的最大r，否则为1。

按此权重有放回采样，每轮抽取次数仍为原训练图数N，保持优化步数预算。
这是明确限定的capped weak-image sampler，不声称原样实现某论文的全部采样/损失配方。

fold0训练集2974图上的预检：

|图像包含类别|原每轮图数|新期望抽样次数|含义|
|---|---:|---:|---|
|HM|11|31.15|约2.83倍|
|LQS|16|45.31|约2.83倍|
|QHS|128|213.41|约1.67倍；含其他弱类的共现图也被加权|
|MS|789|771.31|近似保留，略降约2.2%|
|Vehicle|45|109.19|约2.43倍|

飞机图通常期望次数约降5.6%，仍全部有非零采样概率，不是旧专家只保留约30%。
这些是期望值，不是实际每轮固定次数；重复采样改变了自然无放回训练的随机轨迹，不能
将结果归因为某单个类别的梯度变化。仍可能过拟合稀有组或恶化FP，必须完整评估。

### 4.2 冻结训练/评价合同

- 三折分别继承与原P40完全相同的本折S1024/40e last权重。
- YOLO26-s，原25类，1280，40e，batch8，workers4，seed42。
- AdamW lr0=0.0002、lrf=0.10，mosaic=0，Rot90 p=1，fixed last，不按验证选模型。
- 唯一实验因素是上述训练图采样。每轮N抽样、优化步数与原40e适配相同。
- 单3090依次完成fold0/1/2；不是对全部4481图训练一个部署模型。
- 各折低阈值OOF → 候选自己的其他两折FDR15阈值 → 固定Hard。
- Hard质量增益>0.5才自动运行固定Sentinel；否则停止。
- 所有25类输出一起比较，不在看到结果后只保留获益类别来掩盖退步。
- Sentinel通过也只标记候选待审，不自动full、组合飞机模块、打包或提交。

三个训练fold不是额外三个评测榜单：它们保障固定大图使用其来源未参与训练的模型。
Normal缓存只做必要的新模型校准，日常裁决仍只有Hard→Sentinel两步。

### 4.3 工程核验和运行位置

Linux固定环境28项测试通过；本机31项纯CPU相关回归通过，训练采样的Torch测试由Linux
执行。本机未装Torch的这一venv不把skip冒充通过。Ruff通过。另完成真实3090、1280、
batch8、原25类的一轮32张训练图GPU smoke，loss有限，生成/重载权重成功。
smoke的8张验证图也来自训练区，**仅检验工程，不作为精度证据，其权重不进入正式实验**。

输入审计核对三份初始权重SHA、原P40全部匹配超参数、4481图精确train/val互补、25类
标签范围及本折训练标签SHA。运行后还核对实际dataloader类别频率与预检一致。派生
Ultralytics标签cache可按列表重建；原图/标签/冻结权重未改。

2026-09-04 00:48:30 CST快照：fold0第1/40轮，loss有限，约4.6–4.7 it/s，GPU76%，
显存11280MiB，约300W。历史同长适配fold0耗时3212秒；按当前速度估计三折及校准/
固定复核约**3小时**，不是承诺完成时刻，以日志为准。

00:52:42复核：fold0已完成3/40，box/cls/dfl loss为0.90504/0.50699/0.00636，全部有限；
第2、3轮约79.4秒/轮。GPU76%、显存11308MiB、约307W，screen持锁运行。
两份源代码快照当前一致；[启动进度回执](p40_recall_results_20260904/training_start.json)。

- SSH：现有别名`cv3-seetacloud-2`，本轮显式使用端口19864；未改默认别名端口。
- screen：`p40-weak-rfs`。
- 冻结代码：`/root/autodl-tmp/xh-p40-weak-rfs-v1`。
- 结果：`/root/autodl-tmp/results/P40-WEAK-RFS-V1`。
- `status.txt`、`driver.log`、`train_fold_N.log`记录阶段和失败现场；进程持排他锁。
- 本机[预检回执](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-RFS-V1/preflight.json)。

代码索引：

- [训练配置](../../configs/experiments/p40_weak_rfs_v1.json)
- [采样器/Trainer](../../src/rsdet/innovation/weak_rfs.py)
- [原适配入口的可选采样开关](../../scripts/train_progressive_resolution_adaptation.py)
- [输入审计、GPU smoke、串行训练及固定评价驱动](../../scripts/run_p40_weak_rfs.py)
- [采样测试](../../tests/test_weak_rfs.py)、[校准测试](../../tests/test_recall_calibration.py)

### 4.4 三折、Hard结果与否决原因

三折均完成40轮，loss全部有限；last.pt SHA、训练回执和实际dataloader采样审计
一致。合并OOF为4481个唯一图像、20933 GT，0.001 floor的93258个预测且覆盖全部
4481图。候选Normal外折校准阈值为`0.471/0.451/0.446`，比基线
`0.546/0.516/0.501`分别低0.075/0.065/0.055。

|Hard宏指标|基线|RFS候选|变化|
|---|---:|---:|---:|
|Ship Recall|42.5686%|51.0317%|+8.4631pp|
|Ship FDR|7.0603%|10.6531%|+3.5927pp|
|Aircraft Recall|80.1652%|80.3337%|+0.1685pp|
|Aircraft FDR|15.2765%|18.5434%|+3.2669pp|
|Vehicle Recall|22.8261%|26.6304%|+3.8043pp|
|Vehicle FDR|26.3158%|31.9444%|+5.6287pp|
|gate Recall|48.5200%|52.6653%|+4.1453pp|
|gate FDR|16.2175%|20.3803%|+4.1628pp|
|Hard质量贡献|44.762524|43.453592|**-1.308932**|

池化计数从1443TP/244FP变为1479TP/278FP：Ship `+27TP/+9FP`，Vehicle
`+7TP/+8FP`，Aircraft `+2TP/+17FP`。召回确实上升，但候选自身Normal校准把三折
阈值全部拉低，在Hard上的FP代价超过TP收益；FDR门禁也从通过变为20.38%
不通过。因Hard未达`>+0.5`，按合同不跑Sentinel、不拆出单类路由、不full。

为分离“模型变化”与“新校准阈值”，只做了一个不用于选模的同阈值诊断：
候选在旧阈值下Ship macro Recall `+3.8077pp`、Ship FDR `-0.1046pp`；Vehicle
Recall不变、FDR `-2.6794pp`；Aircraft Recall `-0.3769pp`、FDR `+2.1804pp`。
质量只`+0.039968`，仍远小于0.5。这说明整图采样确实将学习重心移向尾类，
但破坏了整体分数尺度和头类/飞机的分离，不能作为部署配方。

评价首次因旧基线`run_summary.json`没有后来新增的`imgsz`字段而触发
`KeyError`。修复为：只对SHA锁定的两个历史基线回执允许旧schema，仍强制检查
tile/overlap；新候选必须显式携带`imgsz=1280`。恢复阶段未重跑训练或Hard推理，
13项专项测试通过，两个驱动SHA和三份权重SHA记入[恢复回执](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-RFS-V1/recovery_legacy_summary_keyerror.json)。

- [结果摘要与完整性](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-RFS-V1/result_summary.json)
- [Hard完整对比](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-RFS-V1/hard_comparison.json)
- [Normal外折frontier](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-RFS-V1/aggregate/crossfit_frontier.json)

## 5. P40-WEAK-EQLV2-V1：从RFS结果推导的唯一后续单因素

RFS在旧阈值下使LQS从3TP增至6TP、QHS从131增至132TP，且Ship/Vehicle FDR
不升，证明弱类分类梯度仍有可利用信号；但整图有放回采样让非目标类分数标尺也
变化，在候选自校准后引入过多FP。因此下一步不继续放大采样倍数，也不扫阈值，
而是恢复自然采样，只将细类`0/1/2/3/24`的分类BCE替换为动态正负梯度均衡。

冻结的 targeted EQLv2-style 实现使用`gamma=12, mu=0.8, alpha=4`；正样本仍保留
YOLO task-aligned soft target，正负累积量使用加权logit梯度绝对值。非目标19类的每个
loss element恒为原BCE；box/DFL、assigner、模型、数据、轮数和推理都不变。这是
针对本任务的审计适配，不声称逐行复制论文实现。

21项Linux专项测试与Ruff通过；真实3090上用原25类、1280、batch8做1轮32图
GPU smoke，loss有限，last.pt可重载，one2many/one2one两分支均有有限的梯度累积回执。
smoke不是精度证据。三折已在同一单3090串行启动，预期约3小时；Hard仍要`>+0.5`
才跑Sentinel，不自动full。

- [冻结配置](../../configs/experiments/p40_weak_eqlv2_v1.json)
- [EQLv2分类损失与训练器](../../src/rsdet/innovation/eqlv2.py)
- [三折驱动](../../scripts/run_p40_weak_eqlv2.py)
- [回归测试](../../tests/test_eqlv2.py)

### 5.1 三折完整性、工程恢复与Normal结果

三折均完成40轮，loss有限；fold0/1/2 last.pt SHA分别为
`1fa43a32… / 04dc4ed9… / 3ce7be35…`，与训练回执一致。每折训练合同均为
1280、batch8、seed42、AdamW、lr0=2e-4、Rot90 p=1；one2many和one2one两条
criterion分支分别累计14880/14360/15600次更新。每个分支的五组25维量均有限，
非目标19类正负权重逐项严格为1。

fold0训练结束后的首次低阈值推理因子进程没有继承仓库`src`路径，报
`ModuleNotFoundError: rsdet`。这是训练完成后的导入错误，不涉及权重或指标。
修复仅为所有子进程显式注入冻结代码目录的`PYTHONPATH`；恢复入口严格验证原状态、
错误文本、配置SHA、非恢复代码SHA、fold0 40行和权重SHA，且拒绝已有预测的歧义现场。
fold0没有重训，随后fold1/2按原合同执行。[恢复回执](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-EQLV2-V1/recovery_infer_import.json)
记录修复前后驱动SHA和`training_not_restarted=true`。

合并Normal OOF有4481个唯一图像、107138条0.001预测，预测覆盖4481图。
候选自己的其他两折FDR15阈值为`0.546/0.556/0.531`，相对基线
`0.546/0.516/0.501`在fold1/2更保守。Normal门禁宏指标如下；这只是阈值拟合域，
不替代Hard裁决。

|Normal宏指标|基线|EQLv2|变化|
|---|---:|---:|---:|
|Ship Recall|37.6392%|40.9785%|+3.3392pp|
|Ship FDR|15.2186%|13.3709%|−1.8477pp|
|Aircraft Recall|84.4993%|84.4178%|−0.0815pp|
|Aircraft FDR|8.3386%|7.4255%|−0.9131pp|
|Vehicle Recall|33.0846%|34.3284%|+1.2438pp|
|Vehicle FDR|23.1214%|25.0000%|+1.8786pp|
|gate Recall|51.7410%|53.2415%|+1.5005pp|
|gate FDR|15.5595%|15.2655%|−0.2941pp|

Normal显示分类梯度均衡确实改变了学习方向，但Vehicle FDR已经反向；必须看未用于
阈值拟合的Hard是否保持同向。

### 5.2 Hard结果与失败机理

|Hard宏指标|基线|EQLv2|变化|
|---|---:|---:|---:|
|Ship Recall|42.5686%|39.7501%|−2.8186pp|
|Ship FDR|7.0603%|6.8183%|−0.2420pp|
|Aircraft Recall|80.1652%|80.3168%|+0.1515pp|
|Aircraft FDR|15.2765%|14.1067%|−1.1698pp|
|Vehicle Recall|22.8261%|21.1957%|−1.6304pp|
|Vehicle FDR|26.3158%|35.0000%|+8.6842pp|
|gate Recall|48.5200%|47.0875%|−1.4325pp|
|gate FDR|16.2175%|18.6417%|+2.4241pp|
|Hard质量贡献|44.762524|43.802096|**−0.960428**|

池化计数为基线1443TP/244FP/715FN，对候选1439TP/236FP/719FN：总体少8FP，
但也少4TP；宏门禁恶化的关键是FP/TP在类别间重新分布。Ship总TP不变而FP少2，
但MS `+9TP`被QHS `−8TP`、HM `−1TP`抵消；四船细类宏Recall因此下降。
Vehicle从42TP/15FP退化为39TP/21FP，是质量损失最集中且方向最确定的一项。
飞机整体少1TP、少12FP，属于次要正向副作用，不能补偿Ship/Vehicle目标失败。

固定使用旧基线阈值的诊断也没有挽救结论：候选Hard gate Recall为47.4644%、
gate FDR为19.4327%，仍弱于基线48.5200%/16.2175%；池化虽为1454TP/246FP，
Vehicle仍只有39TP并增至23FP。这排除了“仅候选自校准阈值选错”的主要解释。
EQLv2在Normal上得到的正向变化没有跨到Hard，说明当前弱类梯度比率同时编码了
域特定的正负样本难度；它并未提高弱类正确框与Hard背景混淆框的稳定排序。

Hard要求质量增益严格大于0.5，实测为−0.960428，故状态为
`complete_stopped_hard`：不跑Sentinel、不做飞机模块组合、不full、不打包。
按照事前约定，停止RFS/EQL/EFL这一方法族，不以EFL改名或调整gamma/mu/alpha重跑。

- [结果摘要与完整性](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-EQLV2-V1/result_summary.json)
- [Hard完整对比](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-EQLV2-V1/hard_comparison.json)
- [Normal外折frontier](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-EQLV2-V1/aggregate/crossfit_frontier.json)
- [三折训练合同与EQL审计目录](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/P40-WEAK-EQLV2-V1/)

## 6. 文档中还剩什么，为什么不同时堆上去

|方向|现有证据/本轮处理|
|---|---|
|新P40的细类阈值、组级收缩|本轮已实测，+0.016停止；不继续扫更松FDR预算|
|原25类的温和弱类采样|Hard质量-1.309停止；召回增长伴随明显FP和分数尺度偏移|
|EFL/EQLv2式正负梯度均衡|targeted EQLv2完成三折；Normal正向但Hard质量−0.960，Ship/Vehicle Recall下降且Vehicle FDR恶化；停止整个RFS/EQL/EFL方法族|
|Ship coarse-preserving residual/精识别头|只纠错类不足以解释当前大部分召回损失；若新候选错误结构转为FP_CLS主导再考虑，不直接复制飞机头|
|Vehicle缩小tile放大、rot90直接补框/受支持补框|前轮三个具体配方均未达Hard门槛；停止，不改名重跑|
|背景/quality重排、coarse purity、外部EXT-V/HAD、M模型|已有对应负向闭环；不能因急于提分同时叠加|
|兼容权重soup|当前只有独立训练结果通过后才谈兼容成员；不平均不同标签头，也不拿隐藏集选混合比例|

EQLv2已经按**每类TP/FP/FN、Normal阈值与分数尺度、Hard同向性**完成判断，未修复
RFS暴露的弱类稳定排序问题，因此不再跑EFL变体，也不与飞机候选组合。已独立通过的
P40飞机view-consistency候选继续原样保留；本轮没有可辩护的新增长时训练，不为了让
GPU保持运行而追加变量。不能承诺“代理+几分=正式+几分”。

## 7. 资产索引与完整性

本轮本地目录：[P40-SHIP-VEHICLE-RECALL-20260904](../../outputs/P40-SHIP-VEHICLE-RECALL-20260904/)。
`inputs/normal`含从服务器下载的原始OOF；`inputs/hard`和`inputs/sentinel`含原P40缓存。
Sentinel本轮只准备输入，**没有运行失败阈值配方的Sentinel评价**。

|输入|SHA256|
|---|---|
|Normal GT|`c4290b542ffdafe62d5dbcb575f0b3431d46721bbcb366f8ef05291653fcb975`|
|Normal prediction|`e96870c9e10bdd8022846b03ed40ec7700c822be81433d74c4245cad7cedfdbc`|
|Normal frontier|`545e02b2d252909400ff5cf8f9ea7768bb8438dd99c6e26269fc0807132c81be`|
|group map|`7ae203b589bf57a02bc57eac411b69c9e40040abd9201fa86231b82872040e97`|
|Hard prediction|`69e7bf44fe2dad9a8830da07029db6f71386aaac0bcf88f057df029badb5c7ba`|
|Sentinel prediction|`3b37ce01ff3d8a471b2f1b846ffc417f8c53e3dde22baa6d6ea6a61b7fa9f0e9`|

三份S1024初始化权重SHA、全部训练标签SHA、运行快照代码SHA见预检回执和训练配置。
EQLv2下载目录只含配置、CSV、audit、frontier、Hard对比、日志与SHA回执，不含checkpoint、
大型Normal/Hard预测或GT。正式/飞机候选的既有文件未移动、未覆盖。没有Git提交或上传
请求，本轮只更新本地记录。结果下载和文档闭环后，AutoDL控制台核对实例
`432541b66e-c76917a0`为“已关机”，无继续计费的运行实例；对应heartbeat随后删除。
