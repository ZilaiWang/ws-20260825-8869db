# 实验规范：固定代理精简流程

2026-09-03晚按负责人要求调整，默认协议 `fixed_proxy_lite_v1`。
[机器合同](../configs/experiments/fixed_proxy_lite_v1.json)；计分唯一使用
`platform_observed_20260831`，代码见 [platform_protocol.py](../src/rsdet/evaluation/platform_protocol.py)。

## 1. 改了什么、没有改什么

回到既有 Hard / Sentinel 数据，不恢复整套多榜单必过流程。新的 paired-A 在已知
S1024→P40方向核验中失败，因此降为诊断旁证，不再用它单独否决或批准新方法。
失败数据、阈值及原始报告全部保留；不是重新抽一套更容易的数据来把失败改成成功。

旧 Hard/Sentinel 同样经过多次分析，**不是盲集，也没有充分证据保证官方同趋势**。
它们用于比较，不预测官方绝对分：代理58分和正式76分不是同一种数据难度。
旧CV3研究权重为S1024/40e→P40/40e；正式full为S1024/160e→P40/40e，
训练成熟度仍有差异，报告必须注明，不能把这个代理分直接当成full预期。

历史：[原全量规范](EXPERIMENT_PROTOCOL_PRE_TREND_20260903.md)、
[paired-A规范](EXPERIMENT_PROTOCOL_PAIRED_20260903.md)、
[paired-A与BN实跑失败](../reports/experiments/PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)。

## 2. 默认顺序：一项筛选、一项确认

|阶段|做什么|什么时候需要GPU|
|---|---|---|
|准备|明确一个改动、对照权重、训练血缘、数据/代码SHA和工作点来源|通常不需要|
|Hard主比较|同一固定数据比较三类R/FDR及官方质量贡献；可直接重放缓存|模型/输入视图改变才推理|
|Sentinel确认|仅Hard质量贡献提高超过0.5分的候选；工作点不变|不能复用预测时才推理|
|入围后工程回归|背景FP/100MP、单卡时延、输出/容器一致性|仅有变化的项目；其余按SHA复用|

Normal 不消失：它提供 OOF 校准和训练/验证血缘，而不是每次另跑一个完整必过榜单。
已有三折权重就用它们做held-out推理，**不等于重新训练三折**。
新训练方法先做同初始化、同训练预算的一折比较，只有正向才扩展必要的确认；
没有对应held-out权重时，不能将full权重拿来冒充独立验证。

不强制每次运行MacroRisk、所有FDR前沿、三套背景或完整Docker模拟。
后处理改动优先CPU重放；同一权重改变输入视图，基线缓存经一次逐框核验后继续复用。

## 3. 一个计分口径，工作点事前冻结

- 船/飞机按同细类IoU≥0.50匹配，车辆IoU≥0.35；重复框计FP。
- 船4细类、飞机20细类、车辆1细类分别等权平均，三大类均值作为平台R/FDR门线。
- 三个Recall子分、三个FDR子分和一个时延子分之和/7为平台总分。
- 没有可比时延，比较**六项质量子分之和/7**，不是虚构总分；时延代价另外列出。
- Hard/Sentinel不扫描阈值、不选checkpoint、不参与训练。每个实验明确来自Normal或
  独立development的工作点。新模型分数尺度变化时需独立校准，不能默认沿用他人阈值。
- 对当前P40输入视图实验，使用同一P40 OOF FDR15的三个外层折阈值；这是控制变量，
  不是声称FDR15是所有新方法唯一最佳工作点。

## 4. 精简判断与停止条件

1. Hard质量贡献差值 **> +0.5分**：进入Sentinel；否则该固定配方停止。
2. Sentinel差值 **> 0**：方向确认，进入部署成本/风险讨论；否则停止该配方。
3. 不要求六个R/FDR同时变好；必须列出每类损失、平台过线余量以及新增计算的代价。
4. 代理绝对Recall未达85%不直接否决全部研究；正式交付仍关注R≥85%、FDR≤20%、
   单卡时延≤20秒及安全余量。无可靠时延或部署检查时，不宣称正式准入。
5. 0.5是固定工程筛选带，不是统计显著性。明显方向矛盾要解释，不得再换测试集。
6. **不自动训练full、不自动打包、不自动提交。** 当前正式回退资产仍为v2.0 /76.6010。

已知官方变化仍应记录到趋势账本；积累足够多独立候选前，不保证“本地94、官方至少92”。

## 5. 当前可执行入口与产物

- 固定数据模型推理：[run_multifamily_cv3_pseudo_eval.py](../scripts/run_multifamily_cv3_pseudo_eval.py)。
- 固定工作点指标：[evaluate_fixed_score_threshold.py](../scripts/evaluate_fixed_score_threshold.py)。
- 配对指标校验：[compare_candidate_trend.py](../scripts/compare_candidate_trend.py)。
- 简化方向判断：[fixed_proxy.py](../src/rsdet/experiments/fixed_proxy.py)。
- 当前P40双视图端到端：[run_p40_vehicle_zoom_inference.sh](../scripts/server/run_p40_vehicle_zoom_inference.sh)
  和 [analyze_p40_vehicle_zoom.py](../scripts/analyze_p40_vehicle_zoom.py)。后者也支持旋转视图，
  校验数据/权重/代码SHA、折归属、600来源互斥及不改变原有框的约束。
- 当前记录：[精简流程及双视图实验](../reports/experiments/FIXED_PROXY_LITE_AND_P40_VIEWS_20260903.md)。

每个实验只需小型 `comparison.json` + 一份结论记录。大预测与日志留在`outputs`。
只对有希望或方向矛盾的结果展开细类/大小/背景诊断；不为每个失败配方再写一套长报告。
