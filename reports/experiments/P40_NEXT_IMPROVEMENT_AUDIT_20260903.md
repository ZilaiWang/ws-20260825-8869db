# 正式76.601分之后：剩余方向核对与下一步代码

日期：2026-09-03。状态：**下文事前设计已执行；BN开发负向，停止**。
当前正式最优仍为v2.0 / 76.6010分，剩余三次正式机会；本轮没有full训练、Docker打包或提交。

22:13闭环：19864上BN开发质量贡献−2.9617，确认未运行；新A本身的已知方向核验也未通过。
完整实测、错误结构、SHA和产物见[实跑结果](PAIRED_TREND_REFERENCE_AND_BN_RESULT_20260903.md)。
以下保留方法选择与预检历史，不把失败重写为待测或成功。

## 1. 结论

继续以 **YOLO26-s + Rot90 + S1024/160e → S1280/40e P40** 为主体。
现在不是“文档里的方法全部试尽”，而是必须把四种情况分开：已成功、具体配方已失败、
前提未成立因而未启动、真正尚未试过。不能把负向路线换个名字重新跑，也不能把旧模型上
的失败泛化为该方法对任何新模型都无效。

下一步收敛为两个不增加新测试集的工作：

1. 从新A基线的开发预测中分解漏检：已有低分正确候选，还是最低输出阈值下仍没检出。
2. 验证方案15尚未落地的 **train-only BN统计重估**；原始P40权重保留，另存一个候选。

BN是低成本待验证机会，不是已经定位出的bug，更不是保证从76直接到85的训练方法。
更大的提升应由开发集错误结构决定：有候选就修分数分离/校准，没有候选才考虑新的检测训练。

## 2. 76分具体卡在哪里

以下全部来自[正式v2原始响应与复算](FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)，
不是本地代理分，也不是训练集重测分。

|粗类|Recall|FDR|当前含义|
|---|---:|---:|---|
|Ship|80.9894%|6.7489%|虚警显著减少，但召回低于85%分段点；不能只继续提高拒绝阈值|
|Aircraft|94.5967%|3.7265%|不是100%了，恢复召回仍有明显计分价值，不能永久假定“已经解决”|
|Vehicle|83.1579%|21.0000%|召回和虚警都有空间；本次95个GT中79TP、21FP、16FN|

三粗类平均Recall=86.2480%，仅高出硬门 **1.2480pp**；平均FDR=10.4918%；
时延3.551833秒。三门均过，但继续靠大幅丢TP降低FP存在过门风险。

正式v1→v2总分 **+4.4680**：质量贡献+4.6221，新增时延扣0.1541。
三类Recall实际上都下降，三类FDR都下降。这说明P40已带来有价值的改进，但不能把它说成
“各类检测能力全面提升”。模型、分数工作点和包装差异一起参与了这次变化。

### 2.1 必须更新旧文档里的边际收益判断

用当前[统一计分器](../../src/rsdet/evaluation/absolute_score.py)计算，其他量固定：

|当前指标单独改善1个百分点|总分增加|
|---|---:|
|Ship Recall +1pp|+0.10084|
|Aircraft Recall +1pp|+0.38095|
|Vehicle Recall +1pp|+0.10084|
|Ship或Aircraft FDR −1pp|+0.28571|
|Vehicle FDR 21%→20%|+0.10714|
|Vehicle FDR在20%以下再降1pp|+0.28571|
|时延减少1秒|+0.14286|

这是分段函数在**当前点附近**的变化；Recall越过85%后每pp价值上升，不能线性外推到
任意目标。旧方案“1pp召回值多少分”的表格基于当时的工作点，不应机械沿用。
Ship/Aircraft是细类宏平均，不能由合并TP/GT推算页面Recall；只有Vehicle单细类可以。

例如Vehicle在不增加FP的理想条件下多找回2个目标，79TP→81TP、21FP不变，
总分约增加0.3301。即便只把Vehicle FDR降至0、其他指标完全不变，也仅约82.4225分。
因此解决少量FP有用，但要明显跨档，仍须恢复召回和提高TP/FP可分性。

两个**纯算术目标例子，不是预测，也不是从隐藏集反推的调参目标**：

- Aircraft指标和时延不变，Ship Recall=86%、FDR保持6.7489%，Vehicle Recall=89%、
  FDR=16%：约 **80.3460** 分。
- Ship Recall=90%、FDR保持6.7489%，Aircraft Recall=96%、FDR保持3.7265%，
  Vehicle Recall=92%、FDR=10%，时延不变：约 **85.2615** 分。

这说明80/85需要什么级别的联合改善，不证明任何单个方法能实现它。

## 3. 文档与实际实验逐项对照

本次重点对照[方案13](../../../改进方案13.md)、[方案14](../../../改进方案14.md)、
[方案15](../../../改进方案15.md)，向前追溯已闭环实验报告。下载中的方案15补丁也已读取：
`hera_guard_scaleroute_patch_20260902.zip`，SHA
`0b2cadfa219b4f2890f653ebd6b6241bed4646e909aadd8dd9e281cd10f9d9ef`。
它原有BN脚本只声明纯逻辑/语法验证，不能当作已完成实际BN实验。

|方向|实际状态及证据|现在怎么处理|
|---|---|---|
|S/M × 1024/1280|四格已完成；S→M在两个尺度均负；尺度S1280正向。[容量/尺度闭环](IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md#13-下一轮独立的容量尺度-22)|保留s模型尺度路线；不重跑“大模型试试”|
|训练尺度×推理尺度交叉、max_det审计|已做；不是单靠推理放大获得收益；原饱和审计不支持简单提高上限。[方案15执行](IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)|不盲升1536或max_det|
|R2按类尺度路由|已做，形成过候选；P40单模型随后成为正式主体。[同上](IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)|不用旧路由覆盖新单模型结果|
|R3按类tile几何|已做；Hard差于R2、Sentinel以召回换微小FDR，未准入。[6.5节](IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md#65-ship-only-macrorisk-与正式-r3-tile-几何)|停止该固定几何配方|
|Progressive 20/40与full|已做；40e适配已正式得76.6010。[正式结果](FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)|作为当前主体，不是未来待试项|
|BN统计重估|方案15提出、补丁有原型；此前未找到实际候选/结果。本轮完成受控实现|**下一项低成本独立候选**|
|DDP local batch4→10 / full batch30|原文针对早期global12/3卡；当前P40最终DDP为global24/local8，新A为单卡batch8|原问题条件已变；不因旧假设再重训160e，更不改正在运行的batch|
|MacroRisk V2 / 25细类阈值|代码与部署逐框能力均有；旧模型代理分+1.7624但Recall−8.596pp。[七步闭环](MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md)|不是“没做过”；只在新P40存在可恢复低分TP且有足够支持时考虑迁移|
|Ship-only MacroRisk|已做：Gate R+1.196pp/FDR+0.637pp，平台分56.453→56.402。[方案15§6.5](IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md#65-ship-only-macrorisk-与正式-r3-tile-几何)|保留负向历史，不以旧阈值直接改P40|
|Ship fine-tail / EFL / EQLv2|并非所有loss都训练过；原审核显示BG/漏检为主，未满足细类混淆分流条件。[七步§7](MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md#7-step-5ship-分层审核和唯一方向)|标为“前提未成立”，不是已证伪整个方法族。须用当前P40错误结构重新决定|
|Ship objectness-quality metadata头|已做三折；Ship R−1.058pp、FDR+1.150pp。[七步报告](MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md)|不继续堆同一63维特征的MLP；不能把旧FP_BG全部当真实负样本|
|Vehicle Reject+Rescue / selective D-FINE|代码已实现、replay已做；R+1.244pp同时FDR+1.169pp，分数近乎不变。[七步§6](MACROSHIFT_FROZEN_7STEP_IMPLEMENTATION_AND_LOCAL_RESULTS_20260901.md#6-step-4vehicle-reject--rescue-与-selective-d-fine)|不直接叠加；只有当前P40尾部存在独立可靠信号才重提|
|MacroExpert六类专家、DEIM-HCL|已完成；专家未改善宏指标，DEIM正式口径复核为R/FDR交换不合算。[复核报告](IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md)|不把改头/换架构默认当下轮主线|
|coarse purity / score乘法|已有多轮负向，S1280 purity也明确负向。[S1280最终分析](S1280_CV3_FINAL_ANALYSIS_20260903.md)|不重复扫融合权重|
|HAD / DOTA EXT-V / reviewed patch|HAD三折、EXT-V×patch四格均闭环否决。[Final闭环§8](HERA_GUARD_FINAL_PREFLIGHT_EXECUTION_20260831.md#81-ext-v--annotation-patch-完整-22-结果)|不加epoch重跑原配方|
|权重平均 / greedy soup|方案13有建议；尚无当前新A训练区、同架构/初始化/阶段的多个合格成员|真未充分尝试，但**当前资产前提不足**。不能拿旧full或不同标签头直接平均，再在A上宣称独立增益|
|全新Sentinel、背景压力、大图一致性|背景和部署能力已实现；旧比赛来源已反复查看，“全新盲测”不能靠改名字创造|当前固定A/B继续使用，不再增建一批会改变排名的小榜单|

旧负向结论按原合同保留；新A改变评估方式不等于自动撤销旧失败。若复试，必须注明新父模型、
新的单因素问题和新实验ID。也不把所有未触发的条件分支都强行跑一遍。

## 4. 本轮已实现的两个入口

### 4.1 开发区漏检容量诊断：先回答到底该改哪里

代码：[analyze_paired_development_capacity.py](../../scripts/analyze_paired_development_capacity.py)。

只分析A的development缓存。复用现有官方同细类匹配及守恒错误分解，在两个**已有端点**比较：
该模型开发选出的阈值与已输出的0.001 floor。不新增阈值扫描、不重选点、不用confirmation
挑方向、不生成新的测试数据。合同整体SHA会校验，但不会解析确认预测或确认指标作诊断。

每个细类输出GT、当前FN、低阈值可恢复FN、最低阈值剩余FN，以及
FP_BG/FP_CLS/FP_LOC/FP_DUP、FN_MISS/FN_CLS/FN_LOC，附误检代价。

- 若多数漏检在floor已正确匹配：优先改善分数分离、有限度按类风险控制；不急着加检测器。
- 若多数在floor仍缺失：阈值不能创造候选；应依据尺度/场景/训练样本再立训练单因素。
- 若FN_CLS显著占主导：才重新考虑对应细类分类loss或head修复。

floor已受NMS、max_det、输入视野影响，不是模型理论上限；多找到的TP也同时带来FP，
不能称“无成本可恢复召回”。FP_BG仍只是标签匹配角色，不证明那是纯背景。
开发集用于诊断，后续任何训练样本都必须从train区构造，不能把开发错例回灌训练。

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_paired_development_capacity.py \
  --evaluation outputs/PAIRED-TREND-BASELINE-V1/evaluation \
  --output outputs/PAIRED-TREND-BASELINE-V1/development_capacity.json
```

这只是固定A结果的解释附页，**不是第三套评测**。

### 4.2 P40-BN-TRAINONLY-V1：一个固定候选，不扫描

实现与合同：

- [BN核心](../../src/rsdet/experiments/bn_recalibration.py)
- [候选驱动](../../scripts/run_paired_bn_recalibration.py)
- [冻结配置](../../configs/experiments/paired_bn_recalibration_v1.json)
- [A评估入口](../../scripts/run_paired_trend.py)

固定新A的P40 adaptation last为父模型，仅使用 **3,136张train图，全部各一次，batch8，
1280正方形LetterBox，padding114，无随机增强，固定hash顺序**。共392个前向batch。
重置并累计BN的batch统计；不是训练、不反传、不改学习率、不增加epoch。
使用训练图自然频率，不按隐藏集、确认区或背景测试集构造“目标分布”。

原文建议代表性部署tile；本次先做一个更严格、可追踪的**训练图1280视图**版本，
不声称它等同正式10K场景的BN分布。重估采用等batch估计的累计平均，也不冒充精确总体方差。

保护点：

1. 父模型必须完成160+40、CSV连续、SHA和血缘正确且A基线缓存完整。旧4481图full不能混入。
2. 模型总体eval，只有BN更新统计；Dropout/Detect不切训练分支，所有参数禁用梯度。
3. BN running_mean/running_var/num_batches_tracked以外，所有持久状态逐位相同。
4. FP32做统计，按父模型原始dtype写回；重新加载保存文件再次验证非BN参数完全相同。
5. 显式选原checkpoint的EMA（如有），清掉输出中的旧EMA，避免加载器实际读到未修改的影子模型。
6. 原权重不覆盖；无BN的fused模型、非有限统计、不等长尾batch、未遍历BN、来源混入都拒绝。
7. 失败恢复内存状态并保留独立运行目录；已有目录不自动续跑，不自动重试。
8. 新血缘显式记录BN变换、父SHA、plan/receipt SHA；A入口验证实际回执，不伪装成又训练了一轮。

原ZIP脚本允许任意图片列表，缺少当前固定训练区约束；调用`YOLO.save`还可能保留原EMA。
这次复用了重估思路，但补上了来源、模式、非BN权重守恒和保存后检查，未直接运行原脚本。

成本主要是392次前向与常规A推理，不是再跑160e。具体分钟数等GPU实跑记录；
部署不增加模型或后处理分支，但仍需B实测时延及背景误检，不能直接宣布时延完全不变。

## 5. 运行顺序与停止条件

1. **现有服务器链保持不动**：160e → 40e → A → B → 已声明的S1024/P40趋势核验。
2. 从已完成A缓存生成开发区错误诊断。
3. 在独立代码目录执行唯一BN候选；不改当前运行快照、不与主链争GPU。
4. BN候选复用同一A：开发质量贡献增益>0.5才进入确认；确认不重新选阈值，增益须为正。
5. 若A通过再做B，报告背景FP/100MP、真实GPU时延和逐框一致性；小样本类单框变化单独解释。
6. 不满足则停止这一固定BN配方，不扫描momentum、样本量、阈值或新数据混合比例。
7. 再由当前开发错误结构决定是迁移已有风险控制，还是立一个train-only训练改进；
   只有明确获益才讨论唯一full和下一次正式机会。不能保证未来三次分数单调上涨。

新BN程序默认只准备或显示未执行；必须显式`--execute`。示例在独立代码目录运行：

```bash
PY=/root/autodl-tmp/venvs/cv3-model-cu121/bin/python
BASE=/root/autodl-tmp/results/PAIRED-TREND-BASELINE-V1
OUT=/root/autodl-tmp/results/PAIRED-P40-BN-TRAINONLY-V1
DATA=/root/autodl-tmp/data

"$PY" scripts/run_paired_bn_recalibration.py prepare --baseline "$BASE" --output "$OUT"
# 以下为复现命令，当前实验已完成；不要对已完成目录重复执行。
"$PY" scripts/run_paired_bn_recalibration.py calibrate --execute \
  --baseline "$BASE" --output "$OUT" --data-root "$DATA" --device cuda:0
"$PY" scripts/run_paired_bn_recalibration.py evaluate --execute \
  --baseline "$BASE" --output "$OUT" --data-root "$DATA" --device cuda:0
```

不要把新src/scripts覆盖到`/root/autodl-tmp/xh-paired-trend-v1`；它是持锁运行的原始快照。
本次Linux CPU验证使用`/root/autodl-tmp/paired-bn-cpu-preflight-20260903`，设置
`CUDA_VISIBLE_DEVICES=""`、CPU线程1；不启动BN GPU计算或任何新训练。

## 6. 验证记录与边界

测试入口：[BN真实PyTorch测试](../../tests/test_bn_recalibration.py)、
[来源/血缘合同测试](../../tests/test_paired_bn_contract.py)、
[开发区诊断测试](../../tests/test_paired_development_capacity.py)，加原固定A/B流程测试。

本机通过13项真实PyTorch CPU测试（含实际YOLO26网络）及36项来源/评估/诊断回归；
Linux固定环境合并运行 **49/49通过**。详情见
[preflight记录](p40_next_preflight_20260903.json)。新增正式评估入口加入统一协议registry，
29/29绑定审计通过，不新建计分函数。

本机与服务器均已完成CPU `prepare`，两份plan SHA相同：
`8a8bbc8ce39e2361caf0c783b64e9d881806cb64bfa17e21a82868f570a23abc`。
上述预检时，17879的`/root/autodl-tmp/results/PAIRED-P40-BN-TRAINONLY-V1`只有准备合同；
之后用户提供独立19864，本试验在19864同名结果目录执行并完成；17879准备目录未执行。

覆盖内容包括：训练区全覆盖/无尾batch、EMA选择、半精度保存不改卷积权重、NaN回滚、
Dropout保持eval、BN计数、fused拒绝、RGB/padding、旧权重不覆盖、不可绕过回执、
开发显著改善才确认，以及漏检可恢复性与错类/漏检角色分离。

**这些是工程与合同测试，不是模型精度结果。** 后续BN实测已在顶部结果链接闭环；
没有官方分数预测。当前新A仅能减少换模型阈值迁移和成熟度混杂，不能证明70%训练模型的
阈值可以直接迁移至100%训练模型，也不能靠一对历史提交证明长期趋势可靠。

21:50附近只读快照：foundation已160/160，adaptation为29/40，损失有限；
anchor仍等待原主锁。此快照只记录当时状态，实时进度应再次读服务器。
