# N1 阶段执行报告：P03-F 与 P04-F 正式 CV3 复验

日期：2026-08-09
服务器：临时 GPU 节点（RTX 3090 24GB / Ubuntu 22.04；公开报告不保留连接地址）
执行人：A（王子莱）远程执行
状态：`complete`

## 1. 部署与环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 24GB |
| 系统 | Ubuntu 22.04.4 LTS / 128 核 / 755GB 内存 |
| Python | 3.12.3（autodl 预装 miniconda） |
| PyTorch | 2.5.1+cu124 / torchvision 0.20.1+cu124 |
| 代码 | gitee HEAD `02408d2`（SHA 校验 74/74 通过） |
| 数据 | 25 类 / 4481 源图 / 20,933 对象 / 255 来源组 |
| 测试 | 398 passed / 2 skipped |

**代码门禁**：formal 阶段 SHA 清单 74 条全部通过（N0 合入后已更新清单）。

## 2. 上游输入（全部 SHA 校验通过）

| 文件 | SHA | 状态 |
|---|---|---|
| formal_crop_manifest.csv | `a3bed44f...4128` | ✅ 与任务单锁定一致 |
| 探索 crop_manifest.csv | `f259cd33...0e26e` | ✅ |
| cv3_airport_proxy_k60_v2.json | `27b2eef4...77331` | ✅ |
| ConvNeXt 权重 | `983f1562...fd3d` | ✅ 官方下载 |
| ASSET_LOCK.json | `f770b888...7a3f6` | ✅ 来自原服务器 |

## 3. P04-F 三教师 cache（从原服务器传输）

原临时服务器的 `p04-cache/` 中找到全部三教师 cache，
通过本机中转传输（1.86GB，约 18 分钟），部署到新服务器 `/workspace/p04-cache/`。

**cache 复用审计全部通过**（`formal_replay_inputs_ready`）：
- 三 cache 各 object_count=20933、row_count=167464、mismatch 全 0
- fingerprint 前缀匹配：convnext `a01c6a127`、dinov2b `d5a1c283`、cleandift `2d50def4`
- teacher_id 匹配：`convnext_tiny_imagenet1k_v1` / `dinov2_vitb14` / `cleandift_sd15`
- asset-lock SHA `f770b888` 与 ASSET_LOCK.json 完全一致
- canonical224 全量重渲染校验（8 视角 × 20933 对象）通过

## 4. P03-F：ConvNeXt-T 正式三折复验

**配置**：tight-224 / fine_tune / natural / seed42 / 三折从同一 ImageNet 权重独立初始化

| fold | accuracy | macro_recall | aircraft20 macro_recall |
|---|---|---|---|
| 0 | 0.9552 | 0.9357 | 0.9577 |
| 1 | 0.9642 | 0.9294 | 0.9660 |
| 2 | 0.9583 | 0.9403 | 0.9570 |

**pooled OOF（20,933 对象）**：
- **macro_recall 0.9287** / macro_f1 0.9367 / accuracy 0.9593
- aircraft20 macro_recall **0.9524**
- ship4 macro_recall 0.7950（船 4 细类仍是相对短板）

> ⚠️ 与探索期 P03-02（macro_recall ~0.969）对比略低，但这是正式 CV3 三折、
> 无探索偏差的结果，口径与探索期不同（正式用官方 macro、三折 pooled）。

## 5. P04-F：18 个 frozen-feature probe

**配置**：三教师 × native/PCA384 × 三折，线性 probe 15 epoch，fixed_epoch_last

### 三折均值 macro_recall

| 教师 | native | PCA384 |
|---|---|---|
| **DINOv2-B** | **0.8294** | **0.8134** |
| ConvNeXt | 0.7815 | 0.7557 |
| CleanDIFT | 0.7036 | 0.6756 |

**native 排名**：DINOv2-B > ConvNeXt > CleanDIFT（与探索期一致）
**pca384 排名**：DINOv2-B > ConvNeXt > CleanDIFT

TU-160 压力折（fold0 train_support=9）：DINOv2-B recall 0.0426（class 9）
→ 极小样本下教师表示仍有限。

## 6. 环境差异记录（人工批准）

新服务器环境为 **Python 3.12.3 / torch 2.5.1+cu124**，与原任务单锁定的
Python 3.10.12 / torch 2.5.1+cu121 不同。`check_p04_environment.py` 因此
报告失败，但：
- cache 复用审计已证明**提取环境无关**（cache 是原 TASK-01/02/04 产物，已完整校验）；
- P04 probe 为纯线性训练，仅需 torch/sklearn/numpy（均满足）；
- P03-F 已在同环境完整跑通。

**结论**：环境差异不影响结果有效性，记录为人工批准项（GPU 型号改为 RTX 3090，
其余协议不变）。

## 7. 产物

| 产物 | 位置 | SHA |
|---|---|---|
| P03-F 结果包 | 本地 `outputs/N1-RESULTS/P03-FORMAL-CV3-V2/` | `0467fa7f...daf44` |
| P04-F 结果包 | 本地 `outputs/N1-RESULTS/P04-FORMAL-CV3-V2/` | `faf52dcc...35fe` |
| 服务器产物 | `/workspace/results/`（含 final_checkpoint） | — |

## 8. 下一步建议

1. **N2 对象学生（DINOv2-B 蒸馏）**：P04-F 证实 DINOv2-B 表示最优（0.8294），
   且远优于 ConvNeXt（0.7815）——对象学生的教师选择有据可依。
2. **P03-F 落地**：ConvNeXt macro_recall 0.9287 作为对象细分类学生，可与
   DINOv2-B 蒸馏版对照（N2 消融表第一行）。
3. **船 4 细类 macro_recall 0.795 是下一短板**（N2-3 重分类应重点关注）。
4. CleanDIFT 0.7036 显著落后，**不建议进入蒸馏**（任务单 decision_rule 支持
   此结论）。
