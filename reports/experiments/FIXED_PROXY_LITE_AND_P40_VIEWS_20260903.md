# 固定代理精简流程与P40新一轮实测

日期：2026-09-03晚至09-04凌晨。当前主线仍为正式v2.0 / P40 / **76.6010分**。
本轮没有新检测器训练、full训练、Docker打包或官方提交。

## 1. 结论与当前工作

已按负责人要求落地 **旧Hard主比较→有收益才Sentinel确认→入围后工程回归**。
Normal只保留校准、血缘及必要的细类诊断，已有预测尽量复用；不把MacroRisk、多个FDR
前沿、背景、容器和三折重训练都串成每个想法的前置门禁。

三个直接补框方案收益不够，停止。**P40＋飞机CE-D4精识别** 两套代理均正向；
随后固定单视图消融也正向。再复用原R1-5视图一致性分类头，**直接相对CE-D4**仍在
Hard提高0.5767、Sentinel提高0.3551，成为当前质量优先的研究候选。

|方法|Hard质量贡献差|Sentinel质量贡献差|对照|
|---|---:|---:|---|
|CE identity + 飞机NMS|+2.6404|+2.6448|P40 + 飞机NMS-only|
|CE-D4 + 飞机NMS|+2.7644|+2.3850|P40 + 飞机NMS-only|
|View-consistency D4 + 飞机NMS|**+3.3411**|**+2.7401**|P40 + 飞机NMS-only|
|View-consistency D4 的独立增量|**+0.5767**|**+0.3551**|上面的CE-D4候选|

船和车辆输出不变。相对完全原始P40，最后一项为Hard **+3.3670**、Sentinel **+2.7401**。
这些是排除时间的内部质量贡献，不是官方分数增量，也还不是可直接打包的full资产。
所有配方使用相同0.90改类门槛，没有扫描概率阈值、融合权重或恢复已失败补框配方。

## 2. 为什么这样调整测试

新paired-A已知S1024→P40方向：开发质量贡献+3.24395，确认−10.19419，
正式同次质量贡献+4.62205。因此新A尚不能担任可靠的提交筛选器。
它的划分/结果保留作旁证；没有重抽数据、重新调确认阈值或删除失败。

旧Hard/Sentinel此前对这一次官方升级同向，但只有极少官方锚点，不能据此宣称
具有普遍趋势预测能力。尤其当前研究权重S1024/40e→P40/40e，正式full是
S1024/160e→P40/40e，成熟度与场景仍有差异。旧代理已被反复分析，不称全新盲集。

新默认规范：[EXPERIMENT_PROTOCOL](../../docs/EXPERIMENT_PROTOCOL.md)；
机器策略：[fixed_proxy_lite_v1.json](../../configs/experiments/fixed_proxy_lite_v1.json)。
历史paired版：[归档](../../docs/EXPERIMENT_PROTOCOL_PAIRED_20260903.md)；
新A失败：[完整原始记录](PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)。

固定筛选带为Hard质量贡献>0.5、Sentinel质量贡献>0；不是统计显著性。
不要求六项率全部上升，也不以代理低于85%Recall为由停止全部探索。
正式门线余量与部署时间仍单列，不能被代理方向通过替代。

## 3. 共同实验合同

- YOLO26-s，P40三折last，固定原有fold0/1/2，不重新训练。
- 网络1280、基线source tile1024/overlap256、batch4、FP16、safe fusion；候选底阈值0.001。
- 工作阈值来自既有P40 Normal OOF FDR15：fold0=**0.546**，fold1=**0.516**，fold2=**0.501**。
  本轮继承历史工作点作控制变量，不在Hard/Sentinel重新优化。
- 计分使用`platform_observed_20260831`：细类匹配后4/20/1宏平均，七子分；
  质量贡献=六个R/FDR子分之和/7，不含时间，不能叫官方总分。
- Hard：6张100MP伪大图、600来源、2158GT；Sentinel：6张、另外600来源、1969GT。
  两者来源basename互斥，按原折使用对应模型。
- 19864 RTX3090执行。17879在本轮后段SSH连接被关闭，未假定它仍可用，未在其上启动任务。

两次当前代码基线推理均与历史基线预测SHA相同：
`69e7bf44fe2dad9a8830da07029db6f71386aaac0bcf88f057df029badb5c7ba`。
车辆放大分析还核验了工作阈值后逐框一致。不是换环境造成了本轮方向差异。

关键输入SHA：

|输入|SHA256|
|---|---|
|P40 fold0|48434e5206058ed767abea1b2f1fbd5252daf131e09b095a68678c6e39ead5c1|
|P40 fold1|1d65e9fde45e5f7d2c7909ef02baa81628f117453ed1d713cb9b96a433f97281|
|P40 fold2|09787a83a6564dbcb084a2d90ca9f7468c19528d7d19203588c8e528f4f8decb|
|Normal frontier|545e02b2d252909400ff5cf8f9ea7768bb8438dd99c6e26269fc0807132c81be|
|Hard GT|e02e139e5b7f440ee1107fa49c1673d9dc5dc40ed2b3e92de74af6460b885195|
|Sentinel GT|eb1f8850624d77252b568b3515b1953d57adbd6234eed1dec6b467ff5f48c211|

## 4. 三个补框试验：停止并保留结果

|候选|Hard质量贡献差|实际变化|结论|
|---|---:|---|---|
|车辆放大补检：tile512→network1280|+0.1397|车辆新增17TP、15FP；R22.826%→32.065%，FDR26.316%→33.708%|收益不足，停止|
|同尺度rot90全类补检|−2.5503|合入202框；三类R提高，但三类FDR也提高|停止|
|Normal-only旋转新增框可信度筛选|−0.0999|202个新增候选只接受7个；Ship+6TP，Vehicle+1FP|停止|

### 4.1 放大与直接旋转

保留基线工作阈值以上所有框，仅添加辅助视图同阈值以上、未与原有同细类框重叠的框。
车辆去重IoU0.35；旋转全类时船/飞机0.50。没有借GT筛框，没有删除原有框。

放大组合耗时估计由同卡顺序组件相加：4.584→17.992秒，计入时间后的代理分差约
**−1.7758**；旋转为4.538→10.411秒、约**−3.3894**。它们不是Docker端到端时延。
即使忽略时间，两个配方也没有通过事前质量筛选带，因此没有花费Sentinel或重训预算。

旋转新增框的净变化：Ship+67TP/+32FP，Aircraft+45TP/+39FP，Vehicle+8TP/+11FP。
说明额外视图包含真实独立漏检，也包含很多不可靠新框；不能把“召回变高”直接解释成总分变高。

### 4.2 Normal-only可信度

4,481张按原折权重做rot90预测，全部唯一覆盖。已有identity OOF复用。
每个外层折仅用另外两折Normal拟合3个粗类逻辑回归，5个特征：辅助分数、identity同细类
最大IoU、该匹配分数、框宽/高log1p。C=1、max_iter=1000、seed42、接受概率0.90固定。
正/负样本各少于10时放弃补框；实际9个头均满足最低支持。

拟合时两折统一使用被评外层折的阈值，避免用“各自阈值”间接引入被评折标签。
Hard/Sentinel未参与拟合。该处理是既有OOF的后处理交叉拟合，不冒充嵌套重训练。
筛选确实减少误检，但最后6个Ship TP收益太小，1个Vehicle FP仍抵消了收益；固定配方停止，
没有临时改成“只开Ship头”或继续调0.90阈值来改写结果。

证据：[zoom](p40_view_results_20260903/zoom_hard.json)、
[rot90](p40_view_results_20260903/rot90_hard.json)、
[supported](p40_view_results_20260903/supported_hard.json)、
[Normal拟合模型与样本数](p40_view_results_20260903/supported_models.json)。

## 5. 正向候选：P40 + Aircraft CE-D4

这不是继续补框，而是在**现有可靠框内纠正飞机细类**。
复用原R1-1 CE五轮proposal-domain适配的ConvNeXt-Tiny三个fold classifier；
逐个检查embedded held_out_fold、method=ce、fixed-last及source P03 checkpoint SHA。
复用原xyxy tight render_crop224、D4八视图；只在20种飞机细类内求平均概率。
最高概率≥0.90才换细类，原检测分数和坐标不动；最后飞机同细类NMS0.50。
Ship和Vehicle结构旁路，输出内容精确不变。

同时算飞机NMS-only控制，避免把重复框清理的收益算到分类器头上。

|条件|Aircraft Recall|Aircraft FDR|Aircraft TP/FP/FN|
|---|---:|---:|---|
|Hard原P40|80.1652%|15.2765%|844 / 125 / 176|
|Hard NMS-only|80.1652%|15.1858%|844 / 124 / 176|
|Hard CE-D4+NMS|**85.0872%**|**7.3331%**|**900 / 59 / 120**|
|Sentinel原P40 / NMS-only|84.8549%|13.1931%|763 / 91 / 111|
|Sentinel CE-D4+NMS|**87.3589%**|**8.0420%**|**784 / 57 / 90**|

Hard改类69个，处理969个飞机候选；Sentinel改类42个。
相对NMS-only质量贡献分别 **+2.7644 / +2.3850**。
相对完全原P40，Hard **+2.7903**，Sentinel **+2.3850**。
两套测试都没有靠损害Ship/Vehicle换取收益。

机制：同细类才能匹配，分类错误会同时增加FP和FN。修正已有框类别可同时恢复TP、减少FP，
比当前直接添加缺乏精度保障的新框更符合官方公式。这个结论只适用于本次飞机分支；
不据此假设Ship/Vehicle能原样照搬。

计时：Hard完整三fold加载、读图、分类合计19.256秒，Sentinel17.940秒。
不是每图19秒，也不是完整检测链19秒；不能直接给出Docker总时间。
候选还需单模型驻留、稀疏/密集场景、背景标签迁移与完整入口回归。
这也是继续排队单视图CE消融的原因：检验是否能保留收益并减少8视图计算。

证据：[Hard](p40_view_results_20260903/aircraft_hard.json)、
[Sentinel](p40_view_results_20260903/aircraft_sentinel.json)。
过去已有效模块的来源：[R1-1](R1_AIRCRAFT_REFINEMENT_RESULT_20260814.md)、
[R1后NMS](R1_POST_RERANK_NMS_RESULT_20260814.md)。

### 5.1 完整Normal OOF诊断：CE-D4

固定原有阈值，在4,481张OOF图上处理16,676个飞机框，531个改类。
相对NMS-only：飞机macro Recall **84.4993%→87.2684%**，FDR **8.0597%→4.0901%**；
TP **15,656→16,073**，FP **968→548**，FN **2,193→1,776**。
质量贡献+2.0488，20种飞机细类均未出现TP下降或FP增加。

主要纠错净TP：KC-135 +89、SU-35 +53、F-22 +47、SU-34 +33、F-15 +30。
Hard最大的纠错是F-22 +18、F-16 +14、KC-135 +12。
但TU-160在Normal仍有278个FN，说明精识别没有补齐所有漏检。
它是既有OOF的诊断，不新增第三套必过筛选，也不把这些数据回灌分类器训练。

### 5.2 单视图与八视图的代价与风险

单视图在Hard的飞机R/FDR为84.1619%/7.3549%，Sentinel为87.8073%/7.7306%。
总体正向不等于每个细类都改善：Hard的E-8、TU-22、FA-18分别少1、2、1个TP；
Sentinel的F-22少2个TP。CE-D4没有上述TP损失，但Hard的KC-10、Sentinel的KC-10/SU-24
各增加1个FP，不能表述成每种细类的六项指标全优。

19864 / RTX3090的驻留分类器工程计时：每折预热3batch，每图重复3次，交替单/八视图顺序。
包括重新读图、解码、crop、归一化、分类、CPU回传、换类和NMS；不含检测器、checkpoint加载、
容器启动。文件缓存可能已经热。共72个image-repeat，**全部与原缓存最终逐框决定完全一致**。

|条件|CE identity均值 / p95|CE-D4均值 / p95|
|---|---:|---:|
|Hard|1.452 / 1.588秒|2.497 / 2.750秒|
|Sentinel|1.358 / 1.499秒|2.264 / 2.611秒|

若仅按此增量时间折算子分，单视图比八视图节省约0.1492/0.1295分；
因此CE-D4并不在质量+时间上严格支配identity。该估计不代替Docker单卡完整时延。
下述view-consistency与CE-D4架构/视图数量相同，但不能把CE-D4实测时延冒充其独立测量。

### 5.3 进一步正向：旧R1-5视图一致性分类头迁移到P40

唯一改动是CE checkpoint替换为原R1-5的view-consistency checkpoint，仍为ConvNeXt-Tiny、
同样P03 fold-specific初始化、5epoch、同样D4和p≥0.90。不是新训练，也不复制旧R1的融合扫描。
预先冻结以本轮CE-D4为对照（结果SHA写入新合同），Hard增量>0.5才运行Sentinel。
实际按这个顺序自动完成，未把对照退回裸检测器以获得更容易的“增量”。

|条件|原始P40 Aircraft R / FDR|View-consistency D4 R / FDR|候选TP / FP / FN|
|---|---:|---:|---:|
|Hard|80.1652% / 15.2765%|**86.0852% / 6.6454%**|907 / 51 / 113|
|Sentinel|84.8549% / 13.1931%|**87.7917% / 7.3763%**|787 / 50 / 87|

相对CE-D4，Hard净+7TP/−8FP、Sentinel净+3TP/−7FP。
相对NMS-only，两套测试20细类没有TP下降；KC-10仍各增加1个FP。
这证明的是既有飞机分类能力在P40上的增量，而不是Ship/Vehicle的改进。
完整Normal逐细类诊断已完成：仍是16,676个飞机候选，569个改类；
飞机macro Recall **87.3907%**、FDR **3.8146%**，TP/FP/FN为**16,090/530/1,759**。
相对NMS-only净**+434TP/−438FP**，20细类均无TP下降或FP增加；
相对CE-D4再+17TP/−18FP、质量贡献+0.1253。这是风险诊断，不新增筛选门禁。

## 6. 当前决策、边界与下一步

1. 保留View-consistency D4为**质量优先研究候选**，CE identity保留为低成本参照；
   CE-D4作为独立增量对照。当前部署回退仍是正式v2.0。
2. 不启动新检测器160e训练，不再为本轮失败补框方案追加Sentinel；精简流程已节省这些工作。
3. full候选必须解决分类器的full训练/资产血缘和研究40+40与正式160+40的成熟度差异。
   不能拿某一折分类器当作已经验收的唯一full分类器；本轮不训练full、不打包、不提交。
4. 只对最终模块做背景细类FP迁移与完整入口时延/逐框回归。换细类不能增加框数，但可能把
   背景FP移到稀有细类而改变宏FDR，不能仅用“框数不增加”宣布背景风险为零。
5. 真正跨80/85仍需要Ship和Vehicle的独立突破：正式v2飞机已经94.5967%R/3.7265%FDR。
   **假设飞机完美、其余指标和时间不变，总分算术上限也仅79.7241**。
   因此不得将代理+3.367机械加到正式76.601，更不能承诺这个飞机模块单独达到80/85。

本轮下一步应先完成这个正向模块的风险/部署资格梳理；Ship/Vehicle仅依据当前低分TP与
剩余漏检结构设计一个新单因素，不把已经失败的拒识、补框或更大模型重新改名启动。
不为保持GPU占用而重跑已完成任务。

## 7. 文件索引与复现

|用途|入口|
|---|---|
|当前默认流程|[docs/EXPERIMENT_PROTOCOL.md](../../docs/EXPERIMENT_PROTOCOL.md)|
|固定策略|[fixed_proxy_lite_v1.json](../../configs/experiments/fixed_proxy_lite_v1.json)|
|双视图GPU推理|[run_p40_vehicle_zoom_inference.sh](../../scripts/server/run_p40_vehicle_zoom_inference.sh)|
|双视图重放/审计|[analyze_p40_vehicle_zoom.py](../../scripts/analyze_p40_vehicle_zoom.py)|
|旋转逆坐标|[rot90_view.py](../../src/rsdet/models/rot90_view.py)|
|不改旧框的补检|[vehicle_rescue.py](../../src/rsdet/postprocess/vehicle_rescue.py)、[complementary_rescue.py](../../src/rsdet/postprocess/complementary_rescue.py)|
|Normal旋转预测|[run_p40_rot90_normal.py](../../scripts/run_p40_rot90_normal.py)|
|Normal-only拟合及固定应用|[calibrate](../../scripts/calibrate_p40_supported_rescue.py)、[evaluate](../../scripts/evaluate_p40_supported_rescue.py)|
|飞机精识别|[run_p40_aircraft_ce_d4.py](../../scripts/run_p40_aircraft_ce_d4.py)|
|飞机D4合同|[p40_aircraft_ce_d4_v1.json](../../configs/experiments/p40_aircraft_ce_d4_v1.json)|
|单视图成本消融合同/队列|[identity合同](../../configs/experiments/p40_aircraft_ce_identity_v1.json)、[串行队列](../../scripts/server/run_p40_aircraft_identity_queue.sh)|
|视图一致性增量合同/队列|[consistency合同](../../configs/experiments/p40_aircraft_view_consistency_v1.json)、[串行队列](../../scripts/server/run_p40_aircraft_consistency_queue.sh)|
|共享无分数融合的换类逻辑|[aircraft_relabel.py](../../src/rsdet/postprocess/aircraft_relabel.py)|
|驻留时延与逐框回放|[benchmark_p40_aircraft_runtime.py](../../scripts/benchmark_p40_aircraft_runtime.py)、[合同](../../configs/experiments/p40_aircraft_runtime_v1.json)|
|自动结果/细类诊断汇总|[summarize_p40_aircraft_results.py](../../scripts/summarize_p40_aircraft_results.py)、[aircraft_analysis.json](p40_view_results_20260903/aircraft_analysis.json)|
|完整小型原始产物|[outputs/P40-VIEW-EXPERIMENTS-20260903](../../outputs/P40-VIEW-EXPERIMENTS-20260903)|

各运行保留独立服务器代码快照及preflight SHA；后续扩展不会修改先前正在运行的脚本。
目录分别为`xh-p40-zoom-v1`、`xh-p40-rot90-v1`、`xh-p40-supported-v1`、
`xh-p40-aircraft-v1`、`xh-p40-aircraft-normal-v1`、`xh-p40-aircraft-identity-v1`、
`xh-p40-aircraft-runtime-v1`、`xh-p40-aircraft-consistency-v1`和`xh-p40-aircraft-consistency-normal-v1`。
运行状态来自19864上的对应`results/P40-…/status.txt`，不是另外启动相同任务。

当前50项新增/推理/评分/NMS相关测试通过；Ruff、shell语法通过；
原29项正式计分入口迁移审计通过。旋转逆坐标包含非方形图像的像素/连续框单元测试。
尚无新full模型或新的可提交镜像，因此不改变正式v2的回退地位。

2026-09-04 00:18后闭环：本轮19864上的所有GPU推理和诊断均结束，screen为空、GPU无计算进程。
共**82个文件/87,242,846字节**已回传并逐个核对服务器SHA与本地SHA，零缺失、零差异；
[下载校验清单](p40_view_results_20260903/download_receipt.json)。没有改写原始运行目录。
supported-rescue的顶层status保留在collection阶段，最终科学终态以其`hard/comparison.json`
和本报告的negative结论为准，不因顶层状态旧文字重复启动。

复算小型汇总：

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_p40_aircraft_results.py \
  --root outputs/P40-VIEW-EXPERIMENTS-20260903 \
  --output reports/experiments/p40_view_results_20260903
```

三个补框负向结果、CE两种视图、视图一致性增量及Normal诊断均已写入实验账本；
账本不填写不可比的p50/p95总时延，不把组件测速伪装成正式端到端时间。
