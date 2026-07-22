# MAR20 机场代理分组与正式 CV3 构建协议 v1

## 0. 文档状态与用途

| 项目 | 内容 |
| --- | --- |
| 文档性质 | 问题说明、研究方案、执行协议与验收标准 |
| 当前状态 | 设计冻结前，供项目组与 GPT Pro 深度审查 |
| 直接目标 | 在缺少“图像—机场”官方映射的前提下，为竞赛中的 3,073 张 MAR20 飞机图像建立可审计的机场/来源代理组，并据此生成正式三折交叉验证划分 |
| 最终用途 | 替换当前不可靠的编号递增段分组，解锁 P04 正式教师比较、P06 正式 OOF 和后续模型选择 |
| 不是目标 | 不恢复机场名称或经纬度；不宣称代理组等于真实 60 个机场；不使用外部 MAR20 图像训练比赛模型 |
| 核心原则 | 高置信证据自动合并，中置信证据人工复核，低置信样本保留不确定性；同时发布精确版与防泄漏版分组 |

本文件必须在代码实现前完成一次人工审查。任何会改变输入范围、自动合并条件、人工判定合同或正式 CV3 约束的修改，都要升级协议版本并记录原因。

---

## 1. 问题的完整描述

### 1.1 我们真正要解决的不是普通图像聚类

比赛训练集共有 4,481 张图像、20,933 个标注框，其中 3,073 张飞机图像来自 MAR20，包含 20 个细粒度飞机型号。MAR20 原始公开数据有 3,842 张图像、22,341 个实例，论文称图像采集自全球 60 个军用机场。竞赛中的 3,073 张 MAR20 图像与原始数据对应 JPEG 字节一致，因此可以可靠追溯到原始编号。

目标不是根据飞机型号把图像分成 20 类，也不是根据视觉风格得到若干“看起来相似”的簇，而是尽可能恢复下列隐藏变量：

\[
g_i = \text{图像 } i \text{ 所属的真实机场或强相关采集来源}
\]

正式交叉验证要求同一隐藏来源的所有图像只进入一个 fold，避免模型在训练阶段看到相同机场的停机坪、跑道、建筑、道路、植被和成像风格，再在验证阶段利用这些背景捷径。

这属于**缺少地理标签的视觉地点识别、来源去重和分组交叉验证联合问题**，不是单一聚类算法能够可靠解决的问题。

### 1.2 为什么机场隔离是科学要求，而不是额外洁癖

MAR20 论文明确指出：同一机场、同一型号仍具有相似空间和背景信息；如果同一机场同时进入训练集和测试集，会严重影响泛化性能评估。因此论文声称先筛选机场，再分配训练集和测试集。论文原文见项目内 PDF 第 4 页（期刊页码 2691），官方页面也说明 3,842 张图像来自 60 个机场。[MAR20 论文官方页面](https://www.ygxb.ac.cn/zh/article/doi/10.11834/jrs.20222139/)

遥感研究中，空间自相关会使随机或相邻样本交叉验证产生乐观偏差。Kattenborn 等在遥感 CNN 评估中指出，空间相关的训练/验证样本会系统性抬高性能估计；Roberts 等则系统讨论了具有空间、时间和层级结构的数据应采用与结构相匹配的交叉验证策略。[Kattenborn et al., 2022](https://doi.org/10.1016/j.ophoto.2022.100018)；[Roberts et al., 2017](https://doi.org/10.1111/ecog.02881)

对本项目而言，机场背景与飞机型号还可能存在非均匀共现。例如某些机场主要出现少数型号，模型很容易把跑道、气候、建筑或分辨率当成类别提示。这会同时影响：

- 小样本尾类的准确率与 macro recall；
- DINOv2、ConvNeXt、扩散特征等教师的公平比较；
- 困难样本定义、错误分析和蒸馏目标；
- P06 正式 OOF proposal/crop 的可信度；
- 最终隐藏测试集上的跨场景泛化。

### 1.3 官方公开材料为什么不足以直接分组

完整 MAR20 归档包含 JPEG、HBB/OBB XML、`train.txt` 和 `test.txt`，但没有公开：

- 图像到机场名称的映射；
- 经纬度或 Google Earth placemark；
- 场景 ID、拍摄批次或机场 ID；
- 可据此恢复机场的 EXIF GPS；
- 说明 `test.txt` 内编号段边界的元数据。

原始 XML 中 `folder/path` 为未知值，`train.txt/test.txt` 也只是编号列表。因此，官方 train/test 只能作为**弱先验和审计字段**，不能作为机场真值或 cannot-link 约束。

### 1.4 当前编号分段方法的问题

当前 `build_split.py` 在 `test.txt` 中遇到编号回退时新建一段，产生 173 个递增段，再用 dHash 近重复结果合并整段。该方法有两类确定性失败：

1. **过度切分**：同一机场或同一停机区会落在不同递增段，甚至跨当前 train/val；
2. **错误合并**：单一 dHash 假阳性会把两个图片所属的整段做并查集合并，形成链式扩散。

此外，某个原始递增段包含 19 种飞机型号，而论文列出的单机场通常只有少数主要型号，这说明编号段也会把多个机场混在一起。由此可知，编号顺序既不是机场 ID，也不是可靠的采集批次 ID。

### 1.5 已确认泄漏的规模

对完整 3,842 张原始图像进行“背景候选检索—遮挡飞机后的局部特征匹配—RANSAC 几何验证—人工复核”后，得到以下保守结果：

| 范围 | 极严格直接证据 | 扩大候选后人工确认/高度可信 | 解释 |
| --- | ---: | ---: | --- |
| MAR20 官方 train/test 两侧 | 12 对、24 张图 | 约 25 对、约 48 张图 | 仅统计近乎相同或明显重叠场景，占完整归档约 0.62%～1.25% |
| 当前 `dev_v1` train/val 两侧 | 10 对 | 17 对清晰、另 2 对高度可能 | 涉及验证侧 70～144 个飞机对象，占 4,154 个验证飞机对象约 1.69%～3.47% |

这只是**可由画面直接证明的同场景泄漏下限**，不包括同一机场的不同跑道、不同停机区或没有画面重叠的不同年份影像。当前清晰泄漏还集中在部分类别：例如一版保守统计中，A18/KC-10 有 8/31 个验证对象位于已确认泄漏场景，比例达到 25.8%。因此几个百分点的总体泄漏不能被视为对 macro recall 无关紧要。

### 1.6 对前序 P 系列实验的影响

前序实验不需要推翻，但必须保持“探索性”定位：

- P03 仍证明 ImageNet 预训练模型经遥感域微调后具有很高的 GT crop 分类上限；
- P03 的约 0.97 不能解释为可靠的跨机场泛化水平；
- P04 中 DINOv2-B 的领先是值得保留的候选结论，但正式教师排序必须在新 CV3 上重跑；
- P04 的特征缓存和代码可复用，换 fold 后 probe 成本很低；
- P06 的正式 OOF 依赖可信 CV3，因此机场代理分组处于当前关键路径。

---

## 2. 任务定义、成功标准与边界

### 2.1 主任务

给定图像集合 \(I=\{I_i\}_{i=1}^{3842}\)、飞机框掩码 \(M_i\) 和多种无监督证据，为每张竞赛目标图像产生：

\[
\hat g_i^{core},\quad \hat g_i^{guard},\quad q_i,\quad P_i
\]

其中：

- \(\hat g_i^{core}\)：高精度机场/来源代理组；
- \(\hat g_i^{guard}\)：偏防泄漏、允许适度过合并的代理组；
- \(q_i\)：分组置信度与不确定性；
- \(P_i\)：支持该决定的边、分数、人工判定和来源记录。

随后将完整比赛训练集的来源组分配至三个 fold，保证每个图像恰好作为验证样本一次，同一来源组不跨 fold。

### 2.2 两套分组而不是伪造一个唯一真值

在没有机场真值时，错误代价不对称：

- **漏合并**同一机场会继续产生泄漏；
- **误合并**不同机场不会造成泄漏，但会减少有效独立组、增加类别平衡难度。

因此发布两套分组：

| 版本 | 合并证据 | 主要用途 |
| --- | --- | --- |
| `airport_proxy_core_v1` | 仅自动高置信边、人工“确定相同”边 | 分组精度高，便于解释和敏感性分析 |
| `airport_proxy_guard_v1` | core 加人工“很可能相同”边和满足保护条件的簇级合并 | 正式 CV3 首选，偏向降低残余泄漏 |

如果 core 与 guard 上的主要模型排序一致，说明分组不确定性对结论影响有限；若排序变化，则必须报告结论依赖分组假设，不得选择对模型最有利的一版。

### 2.3 明确不做的事情

- 不根据类别标签直接聚类机场；
- 不把官方 train/test 设为强制 cannot-link；
- 不强制聚成 60 组；
- 不把 DINOv2 余弦距离直接当成机场真值；
- 不用单一 dHash、感知哈希或局部特征内点数自动合并整段；
- 不借助飞机本身的位置、数量和型号作为主要同机场证据；
- 不用完整 MAR20 的额外 769 张图训练最终检测或分类模型。

---

## 3. 文献依据与方案选择

### 3.1 DINOv2：通用图像与局部表征底座

DINOv2 通过大规模自监督学习产生可跨数据分布和任务迁移的图像级、像素/patch 级特征，并在实例检索上进行过验证。其官方实现直接提供 CLS token、patch token 和中间层特征。[DINOv2 论文](https://arxiv.org/abs/2304.07193)；[DINOv2 官方代码](https://github.com/facebookresearch/dinov2)

本任务选择 P04 已冻结并在服务器存在的 **DINOv2 ViT-B/14 无 registers 版本**作为主干，原因是：

- 权重、官方仓库 commit、环境和许可证已经在 P04 审计；
- ViT-B/14 具备 768 维 patch 特征，3,842 张图规模下成本很低；
- P04 已观察到 DINOv2-B 的 CLS+patch 表征优于 DINOv2-S 和 ConvNeXt 对照；
- 不需要为分组额外引入 ViT-G 的大权重和环境风险。

但 P04 的对象 crop 结论不能直接替代场景检索实验，因此仍需在机场配对校准集上选择层和聚合方式。

### 3.2 AnyLoc：为什么不能只用 CLS

AnyLoc 将 DINO/DINOv2 的局部特征与 GeM 或 VLAD 聚合结合，在城市、室内、航空、卫星、地下和水下场景做视觉地点识别。论文报告局部聚合明显优于直接使用 foundation model 的 CLS；航空域中，基于域内图像构建 VLAD vocabulary 也优于全局或不匹配域的 vocabulary。[AnyLoc 论文与项目](https://anyloc.github.io/)；[AnyLoc 官方代码](https://github.com/AnyLoc/AnyLoc)

因此本方案把 DINOv2 用于**候选召回与组件相似度**，并比较：

- CLS；
- 掩码 patch mean / GeM；
- 域内 VLAD（首选 16/32 个视觉词）；
- VLAD 经 PCA-whitening 后的紧凑描述子。

NetVLAD 和 GeM 分别为地点识别与图像检索提供了成熟的局部特征聚合依据。[NetVLAD, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html)；[GeM, TPAMI 2019](https://arxiv.org/abs/1711.02512)

### 3.3 Copy detection：相同影像与变体的高召回检索

SSCD 面向图像拷贝与编辑变体检索，采用“全局描述子召回—候选局部验证”的两阶段思路，适合发现缩放、裁剪、颜色或压缩变化后的同源图像。[SSCD, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.html)

本项目第一版不强依赖 SSCD 权重：SHA/pixel、pHash 和 DINOv2 已足以产生候选；若候选召回门禁不足，再加入 SSCD 作为独立召回器。无论是否使用 SSCD，任何感知哈希都不能直接做簇级并查集合并。

### 3.4 几何验证：必要但不能单独相信内点数

局部描述子匹配与 RANSAC 可以验证两图是否共享真实地面结构，但机场存在大量重复飞机、规则停机位、跑道线和矩形建筑，可能产生“几何爆发”式假匹配。地点识别研究已经指出，在无关地点中也可能出现相似几何配置，因此不能只按内点数判断地点相同。[Sattler et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Sattler_Large-Scale_Location_Recognition_CVPR_2016_paper.html)

本方案必须同时检查：

- 匹配是否来自遮挡飞机后的背景；
- RANSAC 内点数和内点率；
- 内点在两图中的覆盖面积和网格分布；
- 对称传递误差；
- 估计变换是否合理；
- 是否仅集中在重复跑道线或少量纹理；
- DINOv2/结构描述子是否提供独立支持。

### 3.5 HDBSCAN/层次聚类：只做候选组织，不制造真值

机场图片数量不均，存在单例、稀疏机场和密集机场，因此不宜强制 K-means=60。HDBSCAN 能处理不同密度并保留噪声点，适合生成候选簇和层次结构。[HDBSCAN 软件论文](https://joss.theoj.org/papers/10.21105/joss.00205)

但聚类结果只作为人工复核队列：自动高置信边先形成基础组件，HDBSCAN/层次聚类只建议哪些组件可能进一步合并，不能直接覆盖证据图。

---

## 4. 输入、数据边界与外部数据登记

### 4.1 输入集合

| 集合 | 数量 | 角色 |
| --- | ---: | --- |
| 比赛 MAR20 子集 | 3,073 张 | 最终必须输出 group/fold 的目标节点 |
| 原始 MAR20 全集 | 3,842 张 | 分组检索全集 |
| 额外桥接图像 | 769 张 | 只用于连接不同裁剪、年份或局部视角，不进入比赛模型训练 |
| 比赛舰船/车辆 | 1,408 张 | 沿用并重新审计各自来源分组，最终与飞机组共同生成 CV3 |

完整 3,842 张图应优先用于分组，因为额外 769 张可能构成两张竞赛图之间的视觉桥梁。为评估外部数据依赖，同时生成一次仅 3,073 张目标图的分组诊断，记录有多少 guard 合并依赖桥接节点。

### 4.2 允许使用的信息

分组阶段允许：

- 原始 JPEG 像素；
- 原始 HBB/OBB 或比赛 bbox，仅用于遮挡飞机；
- 原始编号与 official side，仅用于追溯和诊断；
- 预训练 DINOv2/可选 SSCD 特征；
- 人工查看遮挡前后图像和背景结构。

飞机细类标签在**分组完成后**仅用于 CV3 类别平衡，不得进入视觉相似度、聚类或人工“是否同机场”判断。

### 4.3 必须冻结的输入指纹

- 3,842 张 JPEG 的逐文件 SHA-256、尺寸、模式；
- 3,073 张竞赛对应图与原图逐文件一致性；
- HBB/OBB 与比赛 YOLO 标签的映射审计；
- DINOv2 权重 SHA-256、官方仓库 commit、torch/torchvision/CUDA；
- full/target ID 列表 SHA-256；
- 本协议版本与配置 SHA-256。

任何输入漂移均使旧特征、旧边表和旧分组失效，禁止静默续跑。

---

## 5. 总体流程

```text
冻结 3,842 图及标注
        │
        ├── 原图规范化 ── SHA/pixel/pHash/可选 SSCD ── 拷贝候选
        │
        ├── 飞机框膨胀掩码 ── DINOv2 patch ── GeM/VLAD ── 地点候选
        │
        └── 掩码灰度/边缘 ── SIFT + RANSAC ── 几何证据
                                                   │
                                                   ▼
                                     多召回器候选边并集
                                                   │
                          ┌────────────────────────┼──────────────────────┐
                          ▼                        ▼                      ▼
                   自动高置信边             人工复核边             明确否定边
                          │                        │                      │
                          └───────────── 证据图与基础组件 ───────────────┘
                                                   │
                              DINO/HDBSCAN/层次聚类建议组件合并
                                                   │
                              ┌────────────────────┴───────────────────┐
                              ▼                                        ▼
                      airport_proxy_core_v1                   airport_proxy_guard_v1
                              │                                        │
                              └──────────── CV3 约束分配与反向审计 ────┘
```

---

## 6. 阶段 A：图像规范化与飞机背景隔离

### 6.1 图像规范化

原始图大多为 800×800，但存在数百种其他方形尺寸。所有派生图必须从原文件按固定流程生成：

1. 读取并转换为 RGB；
2. 不裁中心、不改变长宽比；
3. DINO 输入缩放至 518×518（14×37，满足 ViT-B/14 patch 网格）；
4. 经典特征保留较高分辨率，最长边建议 800；
5. 保存 canonical bytes SHA，验证 Pillow/OpenCV 版本不会改变输入；
6. 每个随机操作使用冻结 seed；首版原则上不需要随机增强。

### 6.2 三种视图

每张图生成三种逻辑视图，不一定全部落盘：

| 视图 | 内容 | 用途 |
| --- | --- | --- |
| `original` | 原始 RGB | SHA/pixel/压缩变体召回；人工终审 |
| `bg_masked_rgb` | 飞机及阴影邻域被掩码 | DINOv2 场景特征、人工机场判断 |
| `bg_gray_edge` | 掩码后的灰度与边缘 | SIFT/结构检索、减少颜色和季节影响 |

### 6.3 掩码合同

每个飞机框向外膨胀，初始建议：

\[
d = \operatorname{clip}(0.15\max(w,h), 8, 40)\text{ pixels}
\]

膨胀用于覆盖飞机阴影、边缘和框误差。DINO patch 与掩码相交面积超过 20% 时，该 patch 不参与 GeM/VLAD；SIFT keypoint 落入掩码时直接删除。可视图采用局部中值/模糊填充，仅方便模型输入；几何验证仍以“排除掩码区域的 keypoint”为准，避免填充纹理产生伪匹配。

必须检查：

- 掩码面积分布；
- 高密度机场是否因大量飞机而几乎没有背景；
- 掩码后有效 patch 少于阈值的图像；
- 掩码膨胀 10%/15%/20% 的候选稳定性。

如果某图有效背景过少，它不能仅凭 DINO 相似度自动合并，只能依赖原图拷贝证据、可靠几何或人工判断。

---

## 7. 阶段 B：多路候选召回

### 7.1 B0：精确与近拷贝召回

按以下顺序建立候选：

1. 文件 SHA-256 完全相同；
2. 解码后像素 SHA 完全相同；
3. 多尺度 pHash/dHash 距离候选；
4. 规范化灰度/边缘全局相关候选；
5. 可选 SSCD top-K 候选。

哈希和 SSCD 只产生候选，不直接合并。唯一可以无人工直接合并的是完全相同像素或经过可逆规范化后完全一致的图像。

### 7.2 B1：DINOv2-B/14 场景地点描述子

#### 模型与输入

- 模型：P04 已冻结的官方 `dinov2_vitb14_pretrain.pth`；
- 代码：P04 资产锁中的官方 DINOv2 commit；
- 模式：eval、fp32 提取，缓存可用 fp16，输出 L2 归一化；
- 输入：`bg_masked_rgb` 518×518；
- 方向：主流程计算 R0/R90/R180/R270 四个旋转；镜像只做诊断，不作为首版主特征。

旋转不变相似度定义为：

\[
s_{rot}(i,j)=\max_{r_a,r_b\in\{0,90,180,270\}}
\cos(f_{i,r_a}, f_{j,r_b})
\]

旋转最大值只用于召回，不能单独自动合并，因为机场跑道、农田和停机位可能产生方向性别名。

#### 层与聚合的小规模校准矩阵

ViT-B/14 共有 12 个 block，首轮比较接近后段的 0-based layer 9、10、11：

| 编号 | 特征 | 角色 |
| --- | --- | --- |
| D0 | final CLS | 低成本全局基线 |
| D1 | masked patch mean | 局部均值基线 |
| D2 | masked GeM | 紧凑地点描述子 |
| D3 | masked VLAD-16 | 局部结构聚合 |
| D4 | masked VLAD-32 | AnyLoc 风格主候选 |
| D5 | VLAD-32 → PCA-whiten-512 | 紧凑检索与稳定性候选 |

VLAD vocabulary 在 3,842 张 MAR20 背景 patch 上无监督拟合；采样 patch 时每图等额，防止飞机密集图或大图支配词典。选择标准是已标注配对校准集上的高精度召回，而不是无标签簇看起来最漂亮。

实现采用两遍流式提取，避免缓存全部 patch token：

1. 第一遍从每张图等额 reservoir-sample 少量有效背景 patch，拟合 16/32 个 VLAD 中心及 PCA；
2. 第二遍在 GPU 上即时聚合为 CLS/GeM/VLAD，只落盘全局描述子、质量统计和必要的可视化 patch；
3. 不保存 `3842×4 rotations×1369 patches×768D` 的完整张量，防止产生数十 GB 无必要缓存；
4. 特征缓存按模型、层、输入视图、掩码版本、旋转和 vocabulary 指纹命名，任何一项变化均不得复用。

RTX 4080 SUPER 32GB 上建议从 batch 8 开始，根据峰值显存逐步提高；全局描述子缓存预计远低于 1 GB，DINOv2-B 权重直接复用 P04 服务器资产。

#### 候选生成

对每个描述子计算全量余弦矩阵或精确 kNN。3,842 张图仅约 1,476 万对，完全可以精确计算。每个图保留：

- 每种主描述子的 top-50；
- mutual top-K；
- 高于校准阈值的非 mutual 邻居；
- target-target、target-bridge、bridge-bridge 均保留。

最终取多描述子候选并集，记录每条边由哪些召回器支持。

### 7.3 B2：经典背景结构召回

并行计算低成本、可解释的结构特征：

- 灰度缩略图相关；
- Canny/梯度方向直方图；
- 颜色直方图，仅作弱特征；
- 可选 GIST/HOG 空间金字塔；
- SIFT/ORB 局部描述子倒排或仅在候选边上匹配。

这些特征有两类价值：

1. 找回 DINOv2 因颜色、季节、模糊或分辨率变化漏掉的近同场景；
2. 暴露 DINOv2 仅因飞机型号或通用机场语义产生的假近邻。

---

## 8. 阶段 C：配对校准与几何验证

### 8.1 建立配对校准集

在冻结自动阈值前，建立一次性 `pair_calibration_v1.csv`，至少包含：

| 标签 | 定义 | 建议数量 |
| --- | --- | ---: |
| `same_frame` | 同一画面、重裁、压缩或色彩变化 | ≥50 对 |
| `same_local_site` | 明确共享停机坪/跑道/建筑布局，但不是完全相同裁剪 | ≥50 对 |
| `likely_same_airport` | 没有直接画面重叠，但组合证据强 | ≥50 对 |
| `hard_negative` | 高 DINO/哈希/几何分但人工确认不同 | ≥150 对 |
| `uncertain` | 无法可靠判断 | 单独保留，不参与自动阈值拟合 |

已确认的官方跨侧和当前 dev 跨侧配对必须纳入。负例要优先抽取最相似的误匹配，而不是随机选择容易负例。

人工首轮尽可能隐藏算法分数，只展示匿名图像对、掩码图、原图和必要的局部放大，减少确认偏差。

### 8.2 几何验证字段

对每条候选边保存：

- `n_keypoints_a/b`；
- mutual ratio-test 匹配数；
- affine 与 homography RANSAC 内点数；
- 内点率；
- 对称传递误差中位数/p95；
- 两图内点凸包覆盖率；
- 4×4 网格占用数；
- 估计旋转、尺度、透视条件数；
- 匹配落入飞机掩码或掩码边缘的比例；
- DINO GeM/VLAD/CLS、结构与哈希分数。

### 8.3 自动边分级

阈值不得凭经验拍定，应由校准集给出。初始结构为：

| 边级别 | 条件 | 动作 |
| --- | --- | --- |
| H0 | pixel/规范化像素完全一致 | 自动 must-link |
| H1 | 几何验证达到校准后的极高精度区间，且背景覆盖充分 | 自动 must-link |
| M1 | 两种以上独立召回器强支持，但几何不足 | 优先人工复核 |
| M2 | 单一 DINO/VLAD 高相似或聚类同簇 | 普通人工复核 |
| N | 人工确认不同或存在明显结构冲突 | cannot-link/禁止自动链式合并 |
| U | 信息不足 | 保留不确定，不自动合并 |

H1 的首要门槛是**高精度**，不是召回最大化。校准报告必须给出不同阈值下的 precision、已知正例 recall 和 Wilson 置信区间。若没有足够证据证明自动边可靠，就将其降为 M1 人工复核。

---

## 9. 阶段 D：证据图、组件与聚类建议

### 9.1 基础证据图

构建无向图：

\[
G=(V,E_H,E_M,E_N)
\]

- 节点为 3,842 张图；
- \(E_H\) 为 H0/H1 高置信 must-link；
- \(E_M\) 为待审或人工接受的中置信边；
- \(E_N\) 为明确否定边。

先只对 \(E_H\) 求连通组件。组件内每条边保留证据来源，禁止像旧逻辑一样只保存最终并查集根节点而丢失合并原因。

### 9.2 防止链式污染

视觉分组最危险的失败是单链连接两个无关机场。所有组件合并必须满足：

1. 不存在跨组件 N 边；
2. 不是仅由一个低质量边连接两个大组件；
3. 合并后组件内部不存在显著双峰或极低相似成员；
4. 大组件合并需要至少两条独立 M1 支持，或人工簇级确认；
5. 每次合并保存触发边、前后组件 ID、操作者和时间。

除单链相似度外，还要检查 average-link/complete-link、组件 medoid 相似度和跨组件边密度。

### 9.3 DINO 聚类只负责排队

以 H 组件为单位聚合 DINO VLAD/GeM 描述子，运行：

- HDBSCAN 参数扫描；
- average-link 层次聚类树；
- mutual-kNN 社区候选；
- PCA/UMAP 仅用于可视化，不把二维距离作为合并依据。

输出“建议一起复核”的组件集合。不得直接采用某次 HDBSCAN 标签作为机场组，不强制簇数为 60。

### 9.4 core 与 guard 的形成

| 人工结论 | core | guard |
| --- | --- | --- |
| 确定相同画面/地点 | 合并 | 合并 |
| 确定同一机场 | 合并 | 合并 |
| 很可能同一机场 | 不合并 | 合并 |
| 不确定 | 不合并 | 原则上不合并；只有多证据簇级审查通过才合并 |
| 确定不同 | 禁止合并 | 禁止合并 |

guard 不能变成“只要相似就全并”。其目标是合理过合并以降低漏泄风险，而不是获得尽可能少的组。

---

## 10. 阶段 E：人工复核协议

### 10.1 配对复核卡

每张卡包括：

- 原图 A/B；
- 飞机遮挡图 A/B；
- 边缘图与几何内点叠加；
- 必要时旋转对齐后的局部区域；
- 匿名 pair ID；
- 第一轮不展示算法预测和 official side。

判定选项固定为：

1. `same_frame`；
2. `same_local_site`；
3. `likely_same_airport`；
4. `different`；
5. `uncertain`。

原因码包括：共享独特建筑、跑道/滑行道拓扑、停机位编号、道路水体、同一纹理重叠、仅飞机相似、仅通用跑道相似、分辨率不足等。

### 10.2 簇级联系表

完成边审后，为每个候选组件生成：

- 按 medoid 距离排序的掩码图网格；
- 原图网格；
- official side、原始编号仅在第二轮显示；
- 类别分布仅在分组决定完成后显示；
- 簇内最弱边和最远成员；
- 与最近其他组件的 hard-negative 对照。

人工必须能执行“拆分组件”“接受 core 合并”“只接受 guard 合并”“保持不确定”四类操作。

### 10.3 复核优先级

优先顺序：

1. 当前/官方跨侧强匹配；
2. 会连接两个大组件的桥边；
3. 稀有类涉及的跨组边；
4. DINO 与几何结论冲突的边；
5. top-K 高相似但未合并边；
6. 普通单例。

一人可以完成 v1，但对“连接大组件”及所有 `likely_same_airport` guard 决定，建议二次独立复核。无法取得第二人时，保留为不确定并在敏感性分组中处理。

---

## 11. 阶段 F：正式 CV3 生成

### 11.1 分组输入

- 飞机：默认使用 `airport_proxy_guard_v1`；
- 舰船：沿用可从文件名恢复的原始大图 scene ID，并重新检查跨 scene 近重复；
- 车辆：根据真实来源和近重复审计生成组，若无来源信息则至少保证确认重复不跨 fold；
- 所有 4,481 张图都要进入恰好一个 validation fold。

不能在 CV3 中继续把 MAR20 官方 train 侧的 1,083 张竞赛图永久固定为训练样本，否则它们没有 OOF 预测，也不构成完整三折交叉验证。

### 11.2 硬约束

1. 每个图像只属于一个 group；
2. 每个 group 只属于一个 fold；
3. 每个图像恰好在一个 fold 中作为 validation；
4. 确认 must-link 组件不可拆分；
5. 若某细类分布在至少 3 个独立组中，则三个 fold 均须有该类；
6. 25 个细类在可行情况下均有验证对象；
7. 分组与 fold 生成确定性可复现。

### 11.3 优化目标

设组 \(g\) 在类别 \(c\) 上有 \(n_{gc}\) 个对象，分配变量 \(x_{gf}\in\{0,1\}\)。优化：

\[
\begin{aligned}
J = &\lambda_1\sum_f \frac{|N_f-N/3|}{N}
+\lambda_2\sum_{f,c}\frac{|N_{fc}-N_c/3|}{N_c+\epsilon}\\
&+\lambda_3\sum_f\frac{|I_f-I/3|}{I}
+\lambda_4\,\text{worst\_class\_imbalance}
\end{aligned}
\]

其中同时平衡图像数、对象数和各细类对象数。类别项使用相对误差，使尾类不会被头类数量淹没。

先复用 B 已有的确定性分组分配框架，增加多起点贪心和局部 swap；若不能通过平衡门禁，再使用 CP-SAT/整数规划。不得为了达到完美比例拆分来源组。

### 11.4 输出视图

- `fold0/1/2` 三折完整映射；
- `dev_formal_v1`：默认以 fold0 为 val，其余为 train；
- `dev_relaxed_v1`：保留当前 `dev_v1` 作为同分布开发参考；
- `cv3_core_v1`：core 分组敏感性版本；
- `cv3_guard_v1`：正式首选版本。

---

## 12. 门禁与验收标准

### 12.1 输入与工程门禁

- 3,842/3,073/4,481 数量一致；
- 所有 SHA、尺寸、标注映射通过；
- 特征提取无 NaN/Inf；
- 固定输入重复提取 cosine p05 ≥ 0.999；
- resume 不重算已完成 shard；
- 所有派生产物记录代码、配置和模型指纹。

### 12.2 候选召回门禁

- 所有已知 `same_frame` 必须进入候选并集；
- 已确认当前 train/val 及官方 train/test 泄漏对召回率为 100%；
- 每种召回器报告独立召回与新增边数；
- 若 DINO-VLAD 相比 CLS/GeM 没有增加有效正例召回，则不为“文献上更强”而强行使用；
- top-50 仍漏掉已知正例时扩大 K 或加入 SSCD，不得降低几何精度阈值补召回。

### 12.3 自动合并门禁

- H0/H1 校准集中不得出现已知 hard-negative 自动合并；
- 所有自动边可追溯到原始证据；
- 自动组件中若存在人工 N 边，立即失败；
- 大组件、长链组件和单桥合并全部进入人工复核；
- dHash 单独支持的边自动合并数必须为 0。

### 12.4 分组科学门禁

- 所有 strict 已确认同场景对在 core/guard 中同组；
- guard 不得拆开 core 组件；
- 所有 `different` 对在 core/guard 中不同组；
- 依赖外部 769 张桥接图的合并单独统计；
- 最大组、组数、单例率、组件直径和类分布有完整报告；
- 组数接近或远离 60 都不直接决定通过，异常只触发审查；
- 随机抽查每个大组及至少 10% 的普通组。

### 12.5 CV3 门禁

- `group_cross_fold_count=0`；
- `must_link_cross_fold_count=0`；
- `high_conf_geometry_cross_fold_count=0`；
- 4,481 张图均恰好作为验证一次；
- 20,933 个对象均恰好产生一份 OOF 归属；
- 25 类 coverage 满足可行性合同；
- 每折图像/对象/类别分布偏差报告齐全；
- fold 之间 top-K DINO 最近邻和几何候选完成反向人工审计；
- core/guard 的关键模型排序敏感性完成后，才把 guard 标为 `formal_split_admission=true`。

### 12.6 不能设置的伪门禁

- “最终正好 60 组”；
- “official train/test 绝不混组”；
- “DINO 相似度高于某经验值就同机场”；
- “交叉组 DINO 高相似边为 0”；
- “类别比例必须完美一致”。

这些条件不是由公开真值支持的，强制满足反而可能引入新的错误。

---

## 13. 分组质量的模型敏感性实验

分组不是只看联系表。完成 core/guard 后，必须用低成本实验量化影响：

| 实验 | 训练/特征 | 比较 |
| --- | --- | --- |
| G0 | P04 缓存 ConvNeXt probe | relaxed vs core vs guard |
| G1 | P04 缓存 DINOv2-B CLS+patch probe | relaxed vs core vs guard |
| G2 | 最近邻场景检索 | 跨 fold strict match 数、top-K 人工命中 |
| G3 | P03 tight-224 单 seed 快速微调 | core vs guard；仅在 probe 结论稳定后运行 |
| G4 | 类别/机场背景诊断 | 掩码前后分类性能差异 |

关键读法：

- relaxed 明显高于 core/guard：以前存在来源捷径，正式结论应使用 guard；
- core 与 guard 接近且模型排序一致：分组不确定性可接受；
- core 与 guard 差异大：人工分组还不足以形成唯一正式结论，应报告区间或继续复核；
- 掩码后分类明显下降：背景确实携带类别信息，需要在训练中加强背景去相关；
- DINOv2 比 ConvNeXt 对 split 更敏感：教师比较可能受到场景检索能力影响，必须在 guard 上形成正式选择。

不得通过查看哪个 split 让创新模型分数最高来选择分组版本。

---

## 14. 产物合同

### 14.1 核心数据文件

建议目录：`data/groups/mar20_airport_proxy_v1/`

1. `image_registry.csv`
   - `image_uid, original_id, target_or_bridge, competition_stem, official_side`
   - `image_sha256, width, height, annotation_sha256`
2. `pair_edges.csv`
   - 两端 UID、所有召回分数、几何字段、边级别、人工决定、原因码；
3. `groups_core.csv`
   - `image_uid, group_core, confidence, evidence_count, review_status`；
4. `groups_guard.csv`
   - `image_uid, group_guard, confidence, guard_reason`；
5. `bridge_dependencies.csv`
   - 依赖额外 769 张图形成的目标图连接；
6. `review_decisions.csv`
   - 不可修改的人工判定原始记录；
7. `config.yaml` 与 `meta.json`
   - 完整阈值、版本、模型和输入指纹。

### 14.2 CV 文件

- `data/splits/cv3_airport_proxy_core_v1.json`；
- `data/splits/cv3_airport_proxy_guard_v1.json`；
- `data/splits/dev_formal_v1.json`；
- 原 `data/splits/dev_v1.json` 保留并标注为 relaxed/legacy，不覆盖。

### 14.3 报告与可视化

- 配对校准报告；
- 自动边阈值报告；
- core/guard 组统计报告；
- 人工联系表与高风险边清单；
- CV3 类别/来源平衡报告；
- 跨 fold 反向泄漏报告；
- relaxed/core/guard 模型敏感性报告；
- 外部数据来源、范围、用途和未用于训练的声明。

---

## 15. 推荐的实际执行顺序与时间

### Day 0：协议确认与输入冻结（约 1 小时）

1. GPT Pro/项目负责人审查本文；
2. 冻结 full/target/bridge 列表和指纹；
3. 冻结人工判定合同；
4. 明确 core/guard 的正式用途。

### Day 1 上午：掩码、特征与候选（约 2～3 小时）

1. 生成掩码审计；
2. 提取 DINOv2-B 多层 patch 特征；
3. 校准 GeM/VLAD/CLS；
4. 计算全量 top-K；
5. 完成经典特征和几何字段。

### Day 1 下午：人工复核与分组（约 3～4 小时）

1. 先复核所有强边和桥边；
2. 生成基础组件；
3. 复核 DINO/HDBSCAN 组件建议；
4. 冻结 core/guard；
5. 生成组统计和风险清单。

### Day 1 晚间：CV3 与门禁（约 1～2 小时）

1. 分配三折；
2. 反向扫描跨 fold 候选；
3. 修复任何分组证据遗漏，而不是手工移动单图；
4. 输出 provisional CV3。

### Day 2：低成本科学验收

1. 用 P04 现有缓存重跑 probe；
2. 比较 relaxed/core/guard；
3. 若排序稳定，正式冻结 guard CV3；
4. 若不稳定，回到人工边/组件审查，不立即启动 P06 大规模 OOF。

一工作日可以得到高质量代理组和 provisional CV3；第二天的缓存 probe 用来决定能否正式放行。这里的“一天完成”不等于宣称一天恢复 60 个真实机场。

---

## 16. 主要风险与回退策略

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| DINO 仍关注飞机 | 同型号高相似、背景不同 | 膨胀 mask、排除 patch、加入掩码前后差分和 hard-negative |
| 通用机场纹理别名 | 不同机场跑道/停机位被召回 | 几何覆盖、建筑/道路拓扑、人工复核，不按相似度自动合并 |
| 季节/颜色/分辨率变化 | 同机场 DINO 分数下降 | 灰度边缘、VLAD、旋转候选、桥接图、经典局部特征 |
| SIFT 被规则线条欺骗 | 内点多但集中在跑道线 | 掩码、网格覆盖、对称误差、几何爆发检查、DINO独立支持 |
| 单链污染大组件 | 一个错误边合并两个机场 | 大组件双边支持、N 边约束、簇级复核、保存合并历史 |
| 同机场完全不重叠 | 只有弱语义相似 | guard 合并/不确定性；core/guard 敏感性报告 |
| 过度合并导致类别失衡 | 巨大组难以分折 | 优先保持来源完整；若无法三折覆盖，报告不可行而非拆组 |
| 外部桥接图引发争议 | 分组依赖完整 MAR20 | 单独记录依赖，并提供 target-only 敏感性分组 |
| 人工判断不一致 | likely/different 摇摆 | 固定原因码、二次复核高风险边、保留原始判定 |
| 时间超出一天 | 中置信边过多 | 当天完成 H0/H1+高风险 M1 和 provisional core；guard 不成熟则不放行正式 CV |

### 最小可接受回退

如果 DINO 聚类无法可靠恢复机场，仍应完成：

1. 确定同场景/重叠图的高置信组件；
2. 用这些组件替换旧 dHash 并查集；
3. 对跨 fold top-K 邻居做人工排查；
4. 将划分命名为 `source_overlap_guarded`，不宣称 airport-disjoint；
5. 用 core/guard 性能区间表达剩余不确定性。

即使只能做到这一层，也显著优于编号递增段方案。

---

## 17. 需要 GPT Pro 重点审查的问题

1. 在没有机场真值时，同时发布 core/guard 两套组是否比冻结单一分组更科学？
2. guard 应如何控制“宁可过合并”的程度，避免形成不合理巨型组件？
3. DINOv2-B/14 patch + GeM/VLAD 是否足够，是否值得引入 SSCD、AnyLoc 原始 hook 或更大的 DINOv2-G？
4. 飞机掩码膨胀 15%、patch 覆盖 20% 的初始合同是否合理，是否应加入阴影检测？
5. full-MAR20 的 769 张桥接图用于“分组但不训练”是否应作为主方案，还是只做审计方案？
6. 自动 H1 边应追求怎样的统计精度/召回门槛？
7. same-local-site 与 likely-same-airport 的人工视觉判据是否足够明确？
8. HDBSCAN、average-link 和 mutual-kNN 中，哪一种最适合只作为组件复核排序器？
9. CV3 平衡目标是否应以图像、对象、细类 macro 权重或来源组数量为主？
10. core/guard 模型排序不一致时，应继续人工分组、报告区间，还是采用更保守的 leave-group-out 方案？
11. 是否需要增加一个完全独立的固定 holdout，避免 CV3 反复用于架构选择后发生验证过拟合？
12. 对最终比赛而言，应如何在“跨机场泛化估计”和“充分利用所有训练数据”之间组织模型选择与最终重训练？

---

## 18. 当前推荐结论

1. **主表征选 DINOv2-B/14，但使用掩码 patch 的 GeM/VLAD，而不是只用 CLS。**
2. **DINOv2 负责高召回候选和组件排序，不直接宣判同机场。**
3. **同场景自动合并必须经过背景几何验证；重复跑道纹理和飞机布局是主要假阳性来源。**
4. **完整 MAR20 的 769 张额外图作为桥接节点有高价值，但必须单独登记且不用于模型训练。**
5. **不强制 60 组，不再使用编号递增段，不把 official side 当机场真值。**
6. **发布 core/guard 两套代理组，正式 CV3 默认采用 guard，core 用于敏感性分析。**
7. **分组完成后先复用 P04 缓存做低成本科学验收，再启动 P06 正式 OOF。**
8. **如果无法恢复全部机场，至少要消除所有已知直接重叠和高置信跨 fold 来源边，并诚实命名为机场代理/来源保护划分。**

---

## 19. 主要参考资料

1. Yu et al. MAR20: A benchmark for military aircraft recognition in remote sensing images. *National Remote Sensing Bulletin*, 2023. <https://doi.org/10.11834/jrs.20222139>
2. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. 2023. <https://arxiv.org/abs/2304.07193>
3. Keetha et al. AnyLoc: Towards Universal Visual Place Recognition. *IEEE RA-L*, 2023/2024. <https://anyloc.github.io/>
4. Arandjelović et al. NetVLAD: CNN Architecture for Weakly Supervised Place Recognition. *CVPR*, 2016. <https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html>
5. Radenović et al. Fine-Tuning CNN Image Retrieval with No Human Annotation. *TPAMI*, 2019. <https://arxiv.org/abs/1711.02512>
6. Pizzi et al. A Self-Supervised Descriptor for Image Copy Detection. *CVPR*, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Pizzi_A_Self-Supervised_Descriptor_for_Image_Copy_Detection_CVPR_2022_paper.html>
7. Sattler et al. Large-Scale Location Recognition and the Geometric Burstiness Problem. *CVPR*, 2016. <https://openaccess.thecvf.com/content_cvpr_2016/html/Sattler_Large-Scale_Location_Recognition_CVPR_2016_paper.html>
8. McInnes et al. hdbscan: Hierarchical density based clustering. *JOSS*, 2017. <https://doi.org/10.21105/joss.00205>
9. Kattenborn et al. Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks. *ISPRS Open Journal of Photogrammetry and Remote Sensing*, 2022. <https://doi.org/10.1016/j.ophoto.2022.100018>
10. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 2017. <https://doi.org/10.1111/ecog.02881>
