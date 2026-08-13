# R1-4 飞机物理属性辅助监督实验计划（2026-08-14）

## 1. 进入原因

R1-1 已证明，实际 proposal crop 上的短微调和 D4 聚合有效；同图 KD 没有独立价值。
R1-2 的早期诊断又表明，直接把训练域中的类别特征拉向动态中心会伤害来源偏移最强的
fold0。问题不再是模型容量不足，而是需要一种比“20 个互相独立的类别”更稳定、又不
依赖训练域视觉中心的共享监督。

ExpertDet 报告训练期属性监督在 PSP.Plane 上可独立改善基线，SLIP-RS 则进一步把飞机
分解为互斥的物理属性维度。本实验只抽取二者共同且适合本项目的最小思想：从 ConvNeXt
的 768 维对象特征同时预测五个 class-level 物理属性，推理时删除全部辅助头。

这一路线还经过了现有错误对齐。将 R1-1 CE bundle 与 N2-v2 正提议按
`fold + image_id + bbox` 回连，得到 17,907 个 held-out 正提议：identity 分类错误 807 个，
D4 后为 683 个。identity 最大的几组定向混淆为：

| GT → prediction | 数量 | 当前属性是否区分 | 主要维度 |
|---|---:|---|---|
| TU-160 → TU-22 | 115 | 是 | engine_count |
| SU-35 → SU-34 | 72 | 是 | mission_role |
| KC-135 → E-8 | 66 | 是 | mission_role |
| FA-18 → SU-35 | 59 | 否 | 当前已知盲区 |
| SU-34 → SU-35 | 49 | 是 | mission_role |
| E-8 → KC-135 | 37 | 是 | mission_role |
| F-15 → F-16 | 28 | 是 | engine/wing/tail |

因此属性监督覆盖了主要错误质量，而不是从论文中任意抽取模块。该诊断只用于立项，不
代替正式官方匹配评价。

## 2. 冻结属性字典

文件：`configs/metadata/aircraft_physical_attributes_v1.yaml`。

- propulsion；
- engine_count；
- mission_role；
- wing_configuration；
- tail_configuration。

保留原则是“俯视、224 crop 中仍有物理意义”。绝对尺寸、制造商/国家、含变体歧义的
发动机位置和鸭翼均排除。字典只是公开语义元数据，不引入 PSP 图像、MAR20 bridge
图像、外部权重或伪标签。自动审计必须确认 20 类完整覆盖、状态合法、共享签名与成对
区分率；审计结果写入正式产物。

## 3. 模型与损失

主模型、初始化、proposal manifest、D4 训练增强、五轮固定微调和学习率全部复用 R1-1
CE。只增加训练期线性属性头：

`L = L_fine_CE + ramp(epoch) * 0.10 * mean_d L_attribute_d`

属性头使用每维独立 categorical CE，维度等权。它们不写入 checkpoint；正式 checkpoint
仍只包含原 ConvNeXt-T 25 类模型，因此参数量、推理接口和时延与 CE 完全相同。

服务器同环境 smoke 已通过：五维属性训练准确率为 96.8%--97.8%，平均属性损失
0.283，`0.10 × loss` 相当于主 CE 的约 8.4%；正式五轮采用线性 ramp，首轮实际权重仅
0.02，避免随机初始化辅助头在开始阶段冲击已收敛的 P03 表征。

## 4. 单变量比较

- reference：R1-1 `ce_identity`；
- primary：`structured_attribute_identity`；
- additional：`structured_attribute_d4`；
- ship/vehicle 结构旁路必须逐位等价；
- 三折固定 epoch，不读取 held-out 指标选 checkpoint；
- 仍采用既有 outer cross-fit 阈值选择与冻结 C2。

属性 identity 若未在 Recall/FDR、飞机 macro Recall/FDR 和 paired net TP 上达到门禁，停止
该实现，不扫描属性权重。D4 只用于判断属性监督是否能与当前最强聚合方式叠加；最终仍
需与 R1-1 `ce_d4` 直接比较。

## 5. 解释边界

这是 ExpertDet/SLIP-RS 思想的轻量、闭集适配，不是两篇方法的完整复现。成功只能说明
“物理属性辅助监督对本项目 proposal 域有增益”；失败不能否定大规模属性预训练，但足以
停止当前比赛工程中的 class-derived 多头版本。
