# P03-04 多随机种子稳定性验收与 P03 封板报告

## 1. 结论先行

P03-TASK-04 的 6 个新训练 run、6 个 jitter eval-only run 和回传包通过独立验收。结合 seed=42 的历史结果，本地已从原始 logits 统一复算 3 seed × 3 fold 的 9 个 clean run 和 9 个 jitter run。

P03 普通 crop 分类实验可以封板，最终探索性工作点保持不变：

```text
model: ConvNeXt-Tiny
initialization: ImageNet-1K V1
crop: tight
resolution: 224
sampler: natural
canonical seed: 42
split: exploratory leakage-aware 3-fold
```

核心证据为：

- clean 三个 seed 的三折 mean macro recall 为 0.97029、0.96743、0.96797，最大差 0.00286；
- 9 个 clean run 的总均值为 0.96856；同 fold 跨 seed 的 fold 均值跨度为 0.01769，约为 seed 均值跨度的 6.2 倍；
- 描述性双因素分解中，clean macro recall 的 fold 主效应占 93.5%，seed 占 2.3%，剩余 seed×fold 交互占 4.2%；
- clean 条件下 20,933 个对象有 97.27% 在三个 seed 上预测类别完全一致；
- clean→jitter 的 pooled macro recall 在三个 seed 上均下降，分别为 -0.00394、-0.00864、-0.00388；
- fold0 在三个 seed 的 clean 和 jitter 中均为最难折，说明其困难主要来自数据/来源组成，而非单次优化偶然性。

seed=42 继续作为 canonical seed，是因为它是预注册主 seed、已有完整本地 checkpoint，且指标位于稳定范围内；不是根据验证集事后挑选“最好 seed”。不采用三 seed 集成作为正式方案。

## 2. 完整性与运行条件

| 检查 | 结果 |
| --- | --- |
| TASK-04 回传包 | 7,698,879 bytes，SHA-256 `f15658202d388d6539f9037bf322b98afe0babf4683d48e536206d69c37d5923` |
| TASK-04 包内清单 | 140/140 文件 SHA-256 匹配，无清单自包含问题 |
| 历史 TASK-02/TASK-03 清单 | 90/90、110/110 匹配 |
| 代码门禁 | 14/14 与当前本地冻结训练代码匹配 |
| 新训练 | 6/6 完整，5 个早停，1 个跑满 30 epoch |
| 新 jitter 评估 | 6/6 完整，checkpoint 来源条件和加载 SHA-256 均正确 |
| checkpoint 清单 | 6 个 seed/fold 组合完整，run summary 与清单 SHA-256 一致 |
| 独立复算 | 18 个相关 run 的 logits、预测、混淆矩阵和存储指标全部一致 |
| 对象对齐 | 6 个 seed/条件均为相同 20,933 个 OOF `annotation_uid`、相同 fold 和标签 |
| 环境 | Python 3.10.12、torch 2.5.1+cu121、torchvision 0.20.1+cu121、PyTorch CUDA runtime 12.1 |

本次回传包不含 checkpoint 本体，因此本地能验证的是 checkpoint 清单与训练/评估元数据的闭环一致性，不能重新读取 6 个权重文件的实际字节。它们仍保留在服务器。

## 3. 对服务器回报的校正

服务器回报中的训练、clean 指标、checkpoint 来源和资源记录均正确。clean→jitter macro recall 的两处表述需要区分统计口径：

| seed | 三折等权 mean 差 | pooled OOF 差 | 服务器文字 |
| ---: | ---: | ---: | ---: |
| 42 | -0.00371 | -0.00394 | -0.0039，基本对应 pooled |
| 3407 | -0.00880 | -0.00864 | -0.0055，不正确 |
| 202625 | -0.00406 | -0.00388 | -0.0044，近似但不对应已登记口径 |

`-0.0055` 实际接近三个 seed 的 pooled macro recall 降幅均值 `-0.00548`，不是 seed=3407 自身的降幅。后续报告统一同时给出“三折等权 mean”和“20,933 对象 pooled OOF”，不再简称“均值”。

新 checkpoint 比 seed=42 大 832 bytes 不是风险，但“由不同 seed 独立初始化所致”并不准确。权重张量形状完全相同，数值内容变化通常不改变未压缩张量的字节数；该小差异更可能来自 checkpoint 中 seed 等序列化元数据及 ZIP/pickle 布局。是否为独立训练应由配置、日志和权重 SHA-256 证明，不能由文件大小证明。

## 4. clean GT-crop 的三 seed 结果

### 4.1 每 seed 三折与 pooled OOF

| seed | 三折 macro R mean±std | pooled macro R | pooled macro F1 | pooled accuracy | ship4 macro R | aircraft20 macro R | vehicle1 R |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.97029 ± 0.00778 | 0.97080 | 0.97128 | 0.97970 | 0.89948 | 0.98435 | 0.98507 |
| 3407 | 0.96743 ± 0.01126 | 0.96754 | 0.97401 | 0.98137 | 0.86693 | 0.98653 | 0.99005 |
| 202625 | 0.96797 ± 0.01030 | 0.96753 | 0.96874 | 0.97803 | 0.88556 | 0.98292 | 0.98756 |

三个 seed 的主指标处于同一区间，没有一个 seed 在 macro recall、macro F1、accuracy 和三大类上全面占优。seed=3407 的 accuracy/F1 较高，但 ship macro recall 较低；seed=42 的 macro recall 较高，但 accuracy 并非最高。这符合“总体稳定、少数小 support 类影响 macro 排名”的解释。

### 4.2 seed 与 fold 的相对影响

clean macro recall 的三 seed 均值：

```text
seed42      0.97029
seed3407    0.96743
seed202625  0.96797
range       0.00286
```

同 fold 跨 seed 的均值：

```text
fold0  0.95745
fold1  0.97514
fold2  0.97310
range  0.01769
```

9 run 的描述性加性分解为：seed 2.3%、fold 93.5%、seed×fold 交互 4.2%。这不是总体统计显著性检验，但足以说明在当前 3×3 结果中，数据折差异远大于初始化差异。

因此：

- fold0 应继续作为关键压力折，不应通过换 seed 回避；
- 后续正式 split 应检查 fold0 的来源域、舰船构成、QHS/MS 和边界样本；
- 教师方法若只在一个容易 fold 提升，不能宣称稳定有效。

## 5. 对象级和类别级稳定性

### 5.1 同对象三 seed 一致性

| clean 对象状态 | 对象数 | 比例 |
| --- | ---: | ---: |
| 三 seed 预测类别完全一致 | 20,362 | 97.27% |
| 恰有两个 seed 预测相同 | 541 | 2.58% |
| 三个预测均不同 | 30 | 0.14% |
| 三 seed 全部正确 | 20,207 | 96.53% |
| 恰有两个正确 | 368 | 1.76% |
| 恰有一个正确 | 167 | 0.80% |
| 三 seed 全部错误 | 191 | 0.91% |

这将困难对象分成两类：

1. **稳定困难对象**：191 个三个 seed 都分错，更可能是细类证据不足、标签/来源问题或稳定混淆；
2. **模型不确定对象**：571 个预测不完全一致，适合用于后续教师互补、置信度和困难门控诊断。

三 seed 概率平均的离线诊断结果为 macro recall 0.97295、accuracy 0.98424，高于任一单 seed。但它需要三倍对象头推理且仍只在 GT crop 上成立，因此只证明模型误差存在部分互补，不作为当前部署方案，也不据此下载全部 checkpoint。

### 5.2 尾类和关键波动类

| 类别 | support | seed42 R | seed3407 R | seed202625 R | range |
| --- | ---: | ---: | ---: | ---: | ---: |
| HM | 17 | 0.9412 | 0.9412 | 0.9412 | 0.0000 |
| LQS | 30 | 0.8000 | 0.7000 | 0.7333 | 0.1000 |
| QHS | 641 | 0.9064 | 0.8612 | 0.9189 | 0.0577 |
| A1_SU-35 | 1,317 | 0.9651 | 0.9772 | 0.9408 | 0.0364 |

HM 的总体 recall 数值相同，但预测一致率只有 88.24%，说明三个 seed 不一定错在同一对象上；17 个样本不足以由一个比例宣称完全稳定。LQS 仍是最明显的 seed 敏感类。QHS 和 A1_SU-35 表明波动也不只存在于极少样本类，部分近邻混淆边界同样受优化路径影响。

按当前 support 三层划分，clean macro recall 的三 seed 均值为：

| 层级 | 三 seed 均值 | seed 间 std |
| --- | ---: | ---: |
| tail | 0.95049 | 0.00492 |
| middle | 0.97648 | 0.00354 |
| head | 0.98117 | 0.00399 |

尾类仍是后续教师实验的主要增益观察区，但不能只靠 HM/LQS 的 1—2 个对象决定方法入选。

## 6. clean→jitter 的跨 seed 结论

### 6.1 pooled 同对象配对

| seed | macro R 差 | macro F1 差 | accuracy 差 | clean-only / jitter-only | 净正确对象 | macro R 聚类 bootstrap 95% 区间 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | -0.00394 | -0.00467 | -0.00205 | 89 / 46 | -43 | [-0.01190, +0.00385] |
| 3407 | -0.00864 | -0.00647 | -0.00162 | 81 / 47 | -34 | [-0.01554, -0.00271] |
| 202625 | -0.00388 | -0.00175 | -0.00053 | 78 / 67 | -11 | [-0.01022, +0.00031] |

三个 seed 的方向一致：人工轻扰动都没有改善总体结果。降幅大小受 seed 和 fold 影响，尤其 seed=3407/fold0 的 macro recall 下降 0.01986；不能把单次 seed=42 的 -0.00394 当作固定常数。

### 6.2 风险子集的三 seed 平均

| 子集 | 对象数 | macro R 平均差 | 三个 seed 均为负 | accuracy 平均差 |
| --- | ---: | ---: | ---: | ---: |
| all | 20,933 | -0.00548 | 是 | -0.00140 |
| ship | 2,682 | -0.03203 | 是 | -0.00597 |
| aircraft | 17,849 | -0.00037 | 否，2/3 为负 | -0.00071 |
| vehicle | 402 | -0.00166 | 否，2/3 为负 | -0.00166 |
| GT coverage `<0.90` | 2,296 | -0.02516 | 是 | -0.00479 |
| edge-risk | 933 | -0.00813 | 是 | -0.00607 |
| scale log 扰动 `>0.07` | 10,715 | -0.00455 | 是 | -0.00205 |
| 原生短边 `<48 px` | 1,543 | -0.00447 | 是 | -0.00540 |

最稳定的结论是舰船、低 coverage、边界风险、较大尺度扰动和原生小对象更脆弱。舰船 macro 的绝对降幅受到 HM/LQS 等小 support 类放大，但舰船 accuracy 也在三个 seed 中全部下降，因此不能完全归因于宏平均噪声。

jitter tail macro recall 的三 seed 均值为 0.93808，seed 间 std 为 0.01124，明显比 clean tail 的 0.95049 ± 0.00492 更不稳定。后续教师或对象学生必须同时报告 clean 和真实 proposal/jitter 条件，不能只在理想 crop 上选型。

## 7. P03 最终工程决策

1. P03 普通对象 crop 基线封板，不再增加分辨率、context、sampler、seed 或普通骨干搜索。
2. 保留 seed=42 作为可复现 canonical baseline，不按验证集事后挑 seed=3407 或 202625。
3. 服务器上的 6 个新 checkpoint 不必下载；本地已保存完整 logits、预测、配置、日志和 SHA-256，且 seed=42 三折 checkpoint 已在本地保全。待本报告确认后可删除新 checkpoint。
4. 当前探索性结果不能替代 B 的正式 split。正式 manifest 到达后，只重跑被保留的 canonical baseline 和教师对照，不机械复跑全部 P03 网格。
5. 下一项负责人独立实验转入 P0-4：在 `tight-224`、相同 fold 和相同对象上公平比较 ConvNeXt ImageNet、DINOv2 与扩散特征。
6. P0-4 必须重点观察 tail、QHS/A1_SU-35 等波动类、稳定困难对象和 jitter/未来 Pred-OOF，而不是只争夺已经接近 0.97 的 clean 总体指标。
7. 一旦 M1 OOF proposal 到达，优先级立即转到真实 proposal crop、背景 FP、重复框和官方 Recall/FDR；GT crop 结果只保留为条件上限。

## 8. 复现产物

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_p03_seed_stability.py \
  --seed42-clean-root outputs/P03-TASK-02 \
  --seed42-jitter-root outputs/P03-TASK-03 \
  --task-root outputs/P03-TASK-04 \
  --task-archive outputs/P03-TASK-04-results-no-checkpoints.tar.gz \
  --manifest outputs/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --repo-root . \
  --output-dir outputs/P03-TASK-04/local_analysis \
  --bootstrap-resamples 10000
```

本地分析产物包括：完整性报告、6 个 seed/条件汇总、18 run 单折指标、三组 clean→jitter 配对、六组跨 seed 配对、seed/fold 描述性分解、单类与头中尾稳定性、jitter 几何子集、集成诊断和 20,933 对象级稳定性表。

本地验收脚本 `scripts/analyze_p03_seed_stability.py` 的 SHA-256 为
`af98896a726c249bc6eb470ccdd876aa4264836072cb11c4ed815535ea69c0a9`。全仓
Pytest 136 项通过，相关 Python 文件 Ruff 通过，`git diff --check` 通过。
