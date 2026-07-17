# P03-03 类别均衡与 jitter 鲁棒性验收报告

## 1. 结论先行

P03-TASK-03 的 3 个 `sqrt_inverse` 训练 run、6 个 jitter eval-only run 和两个回传包均通过独立验收。

最终 sampler 冻结为：

> **`tight-224 + natural sampler`**

`sqrt_inverse` 不保留。它没有达到预定的“稳定改善真实小样本/尾类”目标：

- clean 三折 mean macro recall 比 natural 低 0.00065，pooled OOF 低 0.00198；
- 单折只有 1/3 正向，折间 standard deviation 更大；
- tail/middle macro recall 分别下降 0.00393/0.00293，只有 head 上升 0.00115；
- HM 完全不变，LQS 少分对 2 个，QHS 少分对 25 个；
- 虽然 overall accuracy 净多 12 个对象，但 McNemar `p=0.548`，且舰船子集净少 12 个。

`jitter_light` 对 natural 基线造成小但可测的损失：pooled accuracy 下降 0.00205，净少分对 43 个对象；pooled macro recall 下降 0.00394。损失主要集中在 GT coverage `<0.90`、舰船、边界风险、较大尺度扰动和原生小对象。这支持后续做 proposal-aware 训练和原图完整重裁，但仍不能代替 M1 真实 OOF proposal 实验。

## 2. 产物完整性

| 检查 | 结果 |
| --- | --- |
| 小型结果包 | 5,725,986 bytes，SHA-256 `92abff1e5170ad7a952d3486d81556e021ae35f00a0ad00f9336072e7185fe93` |
| 包内清单 | 110/110 文件 SHA-256 匹配，无清单自包含问题 |
| natural checkpoint 包 | 334,284,800 bytes，SHA-256 `864a59d3a3adfb5fbe5be5af8937c31aeea66d1060d17ff55d17e461293fbb4b` |
| checkpoint 内容 | 3 个模型逐个流式解包复算，大小与 SHA-256 均与 TASK-02 一致 |
| 代码门禁 | `CODE_SHA256.txt` 14/14 与当前本地代码匹配 |
| 新训练 | 3/3 完整，2 个早停、1 个跑满 30 epoch |
| jitter 评估 | 6/6 eval-only 完整，checkpoint 来源条件和加载 SHA-256 均匹配 |
| 独立复算 | 12 个相关 run 的 logits、CSV 预测、混淆矩阵和存储指标全部一致 |
| 对象配对 | 4 个条件均为同一批 20,933 个 OOF `annotation_uid` |
| 环境 | 与 TASK-02 一致：Python 3.10.12、torch 2.5.1+cu121、torchvision 0.20.1+cu121、PyTorch CUDA runtime 12.1 |

这里的 12 个相关 run 包括 TASK-02 的 3 个 natural clean、TASK-03 的 3 个 sqrt-inverse clean 和 6 个 jitter eval-only；TASK-03 本身新执行的是 9 个 run。

## 3. 对服务器回报的校正

回报中“HM/LQS 获得最大权重 10”不准确：

- HM 三折训练 support 为 12/11/11，未截断平方根反频率大于 10，因而确实被 cap 到 10；
- LQS 训练 support 为 19/21/20，实际权重是 8.784/8.086/8.529，没有达到 cap；
- “高频飞机类权重 1”也只对每折最高频的 A16_FA-18 成立，其他高频类仍大于 1。

实际理论抽样质量为：

| 类别 | natural 每 epoch 概率范围 | sqrt-inverse 期望概率范围 |
| --- | ---: | ---: |
| HM | 0.078%–0.086% | 0.519%–0.566% |
| LQS | 0.136%–0.152% | 0.787%–0.830% |
| A16_FA-18 | 9.93%–10.51% | 6.71%–6.92% |

因此该 sampler 是温和的平方根均衡，不是 25 类均匀采样。上述文字校正不影响已完成训练，因为 resolved config 中的逐类权重和理论概率都已正确记录。

服务器说“fold0 困难不在类别不均衡”的方向基本合理，但更准确的表述应是：**sqrt-inverse 未改善 fold0，说明类别频率不是其当前主要可修复因素；真正原因仍需结合来源域、QHS/MS 组成和真实 proposal 分析。**

## 4. natural 与 sqrt-inverse clean 对比

### 4.1 三折等权汇总

| sampler | macro recall | macro F1 | accuracy | aircraft20 macro R |
| --- | ---: | ---: | ---: | ---: |
| natural | 0.9703 ± 0.0078 | 0.9709 ± 0.0054 | 0.9797 ± 0.0010 | 0.9843 ± 0.0037 |
| sqrt-inverse | 0.9696 ± 0.0098 | 0.9735 ± 0.0089 | 0.9803 ± 0.0045 | 0.9871 ± 0.0055 |

sqrt-inverse 减 natural 的单折 macro recall 差为：

| fold | macro R 差 | accuracy 差 | 方向 |
| ---: | ---: | ---: | --- |
| 0 | -0.00318 | +0.00057 | macro 退化 |
| 1 | -0.00173 | -0.00295 | 两者退化 |
| 2 | +0.00297 | +0.00424 | 两者改善 |

只有 1/3 折在主指标上正向，不满足预先规定的至少 2/3 折同向。

### 4.2 pooled 同对象配对

| 指标 | natural → sqrt-inverse |
| --- | ---: |
| 两者都对 | 20,346 |
| 仅 natural 对 | 162 |
| 仅 sqrt-inverse 对 | 174 |
| 两者都错 | 251 |
| 净增正确对象 | +12 |
| accuracy 差 | +0.00057 |
| macro recall 差 | -0.00198 |
| macro F1 差 | +0.00209 |
| McNemar p | 0.548 |

按 4,481 张源图聚类的 10,000 次 bootstrap：

- macro recall 差 95% 区间 `[-0.01044, +0.00609]`；
- accuracy 差 95% 区间 `[-0.00154, +0.00270]`；
- 两者都包含 0。

### 4.3 头中尾和大类

| 频率层 | natural clean macro R | sqrt clean macro R | 差 |
| --- | ---: | ---: | ---: |
| tail，9 类 / 3,044 对象 | 0.9562 | 0.9522 | -0.0039 |
| middle，8 类 / 6,314 对象 | 0.9761 | 0.9732 | -0.0029 |
| head，8 类 / 11,575 对象 | 0.9819 | 0.9831 | +0.0011 |

| 子集 | natural accuracy | sqrt accuracy | 净对象差 | natural macro R | sqrt macro R |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship4 | 0.9381 | 0.9336 | -12 | 0.8995 | 0.8749 |
| aircraft20 | 0.9858 | 0.9869 | +19 | 0.9843 | 0.9862 |
| vehicle1 | 0.9851 | 0.9975 | +5 | 0.9851 | 0.9975 |

sqrt-inverse 的 overall accuracy 小幅提升来自飞机和车辆，却伤害了尾类和舰船。这与本次 sampler 的立项目标相反。

典型类别变化：

- HM：16/17 → 16/17，无改变；
- LQS：24/30 → 22/30，少 2 个；
- QHS：recall 0.9064 → 0.8674，少 25 个；
- A18_KC-10：+4 个；
- A19_SU-34：+8 个；
- FSC：+5 个。

不均衡采样不但没有稳定帮助 HM/LQS，还改变了 QHS/MS 和若干飞机近邻类的决策边界。

## 5. clean → `jitter_light` 鲁棒性

### 5.1 natural 基线的配对变化

| 指标 | clean → jitter |
| --- | ---: |
| accuracy | 0.97970 → 0.97764，差 -0.00205 |
| macro recall | 0.97080 → 0.96686，差 -0.00394 |
| macro F1 | 0.97128 → 0.96661，差 -0.00467 |
| clean 对 / jitter 错 | 89 |
| clean 错 / jitter 对 | 46 |
| 净变化 | -43 个正确对象 |
| 预测一致率 | 0.99269 |
| McNemar p | 0.00030 |

accuracy 差的源图聚类 bootstrap 95% 区间为 `[-0.00315, -0.00095]`，不包含 0；macro recall 差区间为 `[-0.01190, +0.00385]`，仍受小 support 类影响而包含 0。

三折 mean macro recall 从 0.97029 降到 0.96658，折间差为：

- fold0：-0.00884；
- fold1：-0.00229；
- fold2：+0.00001，基本不变。

这表明相同的人工轻扰动在不同来源折上效果不同，不能只报一个总体降幅。

### 5.2 几何与数据子集

natural 基线的主要变化为：

| 子集 | 对象数 | accuracy 差 | macro R 差 | clean-only / jitter-only |
| --- | ---: | ---: | ---: | ---: |
| GT coverage `<0.90` | 2,296 | -0.0070 | -0.0275 | 21 / 5 |
| coverage `0.90–0.95` | 7,156 | -0.0027 | -0.0051 | 32 / 13 |
| coverage `0.95–0.99` | 6,924 | +0.0001 | +0.0004 | 20 / 21 |
| scale log 扰动 `>0.07` | 10,715 | -0.0032 | -0.0094 | 56 / 22 |
| ship | 2,682 | -0.0067 | -0.0189 | 33 / 15 |
| aircraft | 17,849 | -0.0013 | -0.0010 | 53 / 29 |
| edge-risk | 933 | -0.0096 | -0.0072 | 11 / 2 |
| 原生短边 `<48 px` | 1,543 | -0.0065 | -0.0051 | 13 / 3 |

中心偏移幅度与错误不呈简单单调关系；更直接的危险信号是实际 GT coverage、尺度扰动、边界和原生像素证据。因此后续困难对象门控应优先使用完整度/边界/质量信号，而不是只根据 proposal 中心移动量。

### 5.3 sqrt-inverse 的 jitter 表现不足以改变决策

sqrt-inverse 在 jitter 下比 natural：

- pooled accuracy `+0.00277`，净多 58 个正确对象，McNemar `p=0.0032`；
- pooled macro recall 仅 `+0.00063`，聚类 bootstrap 95% 区间 `[-0.00450, +0.00584]`；
- tail macro recall 仍比 natural 低 0.00036，middle 低 0.00251，head 高 0.00489。

这说明 sqrt-inverse 对人工轻扰动的自然频率 accuracy 更稳，但增益仍主要由头部类驱动，没有转化为稳定的小样本 macro 收益。考虑其 clean 主指标更低、舰船更差、训练时间更长，不为一份人工 jitter 分布而改变最终 sampler。

## 6. 工程决策

P03-3/4 后的唯一基线为：

```text
model: ConvNeXt-Tiny
initialization: ImageNet-1K V1
crop: tight
resolution: 224
sampler: natural
seed for primary comparison: 42
split: exploratory leakage-aware 3-fold
```

sqrt-inverse 仅作为已完成的负向消融保留，不再进入后续教师特征、蒸馏或真实 proposal crop 主表。服务器上的 3 个 sqrt-inverse checkpoint 只需在本报告和哈希保全后保留短期备查，不必下载为正式工作点。

已下载的 natural-224 三折 checkpoint 归档继续保留，但它们是探索性折模型，不是未来最终提交模型。

## 7. 下一步

P03-5 补充最终工作点的 seed 稳定性：

- 只训练 `tight-224 + natural`；
- 补 seed `3407` 和 `202625`，每个 seed 跑 3 fold，共 6 个训练 run；
- 用每个 best checkpoint 做同 fold `jitter_light` eval-only，共 6 个快速评估；
- 不重新打开 336、context、sampler 或其他超参数；
- 判断主结论、fold0 困难性和 jitter 损失是否依赖 seed=42。

P03-5 完成后，P03 普通 crop 上限基线即封板。之后不再扩大普通分类网格，而是进入相同数据合同下的 DINOv2/扩散特征教师对照，或等 M1 OOF proposal 后进行真实重裁和背景拒识。

## 8. 复现产物

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_p03_balance_jitter.py \
  --natural-root outputs/P03-TASK-02 \
  --task-root outputs/P03-TASK-03 \
  --manifest outputs/P0-2-exploratory-crop-manifest/crop_manifest.csv \
  --checkpoint-archive outputs/P03-TASK-02-tight-224-natural-checkpoints.tar \
  --repo-root . \
  --output-dir outputs/P03-TASK-03/local_analysis \
  --bootstrap-resamples 10000
```

本地分析产物包括：完整性报告、单折/条件汇总、4 组配对 bootstrap、单类、头中尾、sampler audit、jitter 几何子集和 20,933 对象的四条件对齐表。

本地验收脚本 `scripts/analyze_p03_balance_jitter.py` 的 SHA-256 为
`2f10c852055fc9a9fdd545ea2916402399ad8abf21aa74e9f96ce01348850c9e`。全仓
Pytest 134 项通过，相关 Python 文件 Ruff 通过，`git diff --check` 通过。下一批执行说明见
[`P03_TASK_04_SEED_STABILITY.md`](../../docs/server/P03_TASK_04_SEED_STABILITY.md)。
