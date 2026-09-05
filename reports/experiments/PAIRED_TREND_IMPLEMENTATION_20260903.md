# 固定测试流程落地与验收

日期：2026-09-03。本文保留白天落地时点：**`frozen_data_and_cpu_pipeline_verified / gpu_baseline_pending`**。

晚间后续：17879服务器已开通，真实GPU基线及串行回归已启动；最新状态、40项跨平台流程测试、
预声明趋势核验见[GPU执行报告](PAIRED_TREND_GPU_AND_DIRECTION_AUDIT_20260903.md)。
以下“服务器无法连接/未启动”描述的是此前时点，不代表当前状态。

不是另一份待实施规划：本轮已生成真实划分、完整GT快照与SHA，写好并测试训练/评估入口，
物化本机基线的实际训练列表和dataset.yaml。没有新GPU训练结果，不宣称新协议已能预测
官方趋势。三台已知SSH端口33070、17879、47096均返回连接关闭；本机无CUDA/PyTorch。

## 一、数据已落盘

|区域|图像|组|GT|TU-160|HM|FSC|
|---|---:|---:|---:|---:|---:|---:|
|训练|3,136|190|15,328|352|12|282|
|开发|673|31|2,852|8|3|60|
|确认|672|34|2,753|1|2|60|

输入为既有K60 CV3来源组及20,933框官方GT。分配器只接受来源/标签，不接受模型预测。
比例目标70/15/15，完整组不可拆分，三部分均25类，训练保留各类至少50%框。
HiGHS完成可行分配，报告相对MIP gap约0.00992；不把近似分配称唯一数学最优。
全部图像与标签SHA、标签与GT一致性、完全相同图像不跨区已检查。原始数据和旧CV3未移动/覆盖。

限制：TU-160确认仅1框、HM仅2框；正负变化看逐类计数，不能声称稳定统计显著。
来源分组为机场视觉代理；历史数据被看过，不称全新盲集或不存在任何近似重复的证明。
主评分集纯负图为0；382张已审核背景另计FP/100MP，不混入未知比例伪造平台FDR。

## 二、可执行的两条链

**A `run_paired_trend.py`**：数据检查 → 成熟基线160e+40e（一次） → 同模型开发选点 →
缓存基线；以后一个候选只推开发，质量贡献增益>0.5才推确认，同一阈值只应用一次。
不默认三折、不执行确认oracle、不因一个粗类Recall降低就直接否决总分正向方案。
每个比较输出细类TP/FP/FN、粗类R/FDR贡献、质量分差、血缘与配方/环境一致性。
缺失可比100MP时延时总分留空。该入口当前支持标准25类YOLO，新架构需接对应适配器。

**B `run_paired_deployment_regression.py`**：已审核Background-100MP → 两张100MP接缝回归
画布 → 实际提交Python入口和离线逐框对照 → 真实同步GPU计时 → 一份regression.json。
六张原始训练图按三粗类分别选稀疏/密集，不按候选得分选；两次原像素摆放改变接缝位置，
无缩放/截断。空白为黑色，仅为工程夹具；不是新版Hard/Sentinel分数榜，也不用于选阈值。
B返回部分taxonomy的TP/FP/FN，不冒充完整平台精度。入口测试不等于实际GPU容器测试。

## 三、已经执行的检查

- 4,481图和标签全量哈希及逐框标签/GT核对；
- 25类三部分覆盖、组互斥、重复图片SHA互斥；
- 背景382文件哈希与原有审核冻结manifest一致；
- 两张10000×10000画布，每张6处原像素粘贴逐像素相等，GT在图内；
- 原始官方YOLO26s初始化SHA正确，生成训练区唯一清单、dataset.yaml；
- **86项相关测试通过**，ruff和diff whitespace检查通过；
- CPU端到端测试替换了GPU模型，真实运行官方评分、开发选点、确认分支及输出合同；
- 测试真实覆盖：负图FP、无预测图、错误分类/框/图片ID、训练祖先泄漏、SHA变更、
  基线连续epoch/有限loss、完成阶段缓存、负向不运行确认、确认不选点、官方result.json格式。

CPU测试中的模型与预测为夹具，不能当成“新基线取得好分数”或GPU链已经实测通过。
本轮还修正了开发时发现的B输出缺status/image_id/timestamp问题，再用真实输出校验器测试。

## 四、剩余动作明确且只有初始化

1. 一台GPU可连接后使用现有YOLO26环境准备实际路径，运行已写好的`baseline --execute`。
2. 验收两阶段160/40连续行、25类名称、权重/环境/SHA及开发/确认缓存。
3. 同一个基线跑一次B作为以后对照；记录实际时延与背景FP。
4. 才能把状态改成`baseline_cached_gpu_verified`。后续候选直接复用数据和缓存。

不启动全量4481图训练，不打包Docker，不提交官方，不创建定时任务。
与旧P40同训练成熟度，但固定单卡batch12/8，不宣称复现历史中途迁移DDP的每一步。
若代码/路径/参数改变，预检要求新目录；中断阶段保留，不自动resume。

## 五、索引

- [一页使用入口与实际命令](../../docs/PAIRED_TREND_QUICKSTART.md)
- [当前规范](../../docs/EXPERIMENT_PROTOCOL.md)
- [冻结合同](../../data/splits/paired_trend_v1/contract.json)
- [完整来源及文件哈希](../../data/splits/paired_trend_v1/manifest.json)
- [逐类支持](../../data/splits/paired_trend_v1/support.json)
- [A主入口](../../scripts/run_paired_trend.py)
- [分配/完整性逻辑](../../src/rsdet/experiments/paired_trend.py)
- [B回归入口](../../scripts/run_paired_deployment_regression.py)
- [B固定配置](../../configs/experiments/paired_deployment_v1.json)
- [新增测试](../../tests/test_paired_trend_pipeline.py)
- [最初审计](EVALUATION_WORKFLOW_SIMPLIFICATION_20260903.md)

实际机器路径只在`outputs/PAIRED-TREND-BASELINE-V1`的本地plan/train列表中；不提交凭据或
服务器私有路径。没有把当前大量无关改动一起提交，也没有执行GitHub push。
