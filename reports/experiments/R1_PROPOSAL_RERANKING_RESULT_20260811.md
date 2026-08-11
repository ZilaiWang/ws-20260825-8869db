# R1-0 P03-F 教师对 M1 OOF 候选框重排结果

日期：2026-08-11
技术状态：`complete`
科学状态：`mechanism_admitted_global_policy_not_admitted`
下一步：`R1-1_aircraft_only_proposal_domain_refinement`

## 1. 本次实际完成了什么

R1-0 使用 P03-F 三份 fold-specific ConvNeXt-T checkpoint，对 M1 正式 OOF 的全部
55,548 个低阈值候选框提取 25 类 logits。方法和阈值只在另外两折选择，再应用到
held-out 折；类别只允许在原检测大类内部变化，bbox 始终不变。

实验同时比较：

1. 原始 detector cross-fit 工作点；
2. outer cross-fit 选择的 R1 变体；
3. 原始预测经过已冻结 Y1-C2；
4. R1 原始输出经过同一套已冻结 Y1-C2 参数。

完整预注册见
[`R1_PROPOSAL_RERANKING_PLAN_20260811.md`](R1_PROPOSAL_RERANKING_PLAN_20260811.md)。

## 2. 技术验收

| 项目 | 结果 |
|---|---|
| GPU | NVIDIA RTX 3090 24GB |
| 环境 | torch 2.5.1+cu121 / torchvision 0.20.1+cu121 / numpy 1.26.4 / Pillow 10.4.0 |
| Python | 3.10.13；相对任务单 3.10.12 仅补丁版本不同，已记录 |
| 图像 | 4,481，全部一次覆盖 |
| proposal | 55,548；fold 0/1/2 = 20,115 / 18,437 / 16,996 |
| manifest SHA | `48747c3bba75ec5226e52fff5b488bb92eaa17d9722e377ec393a9cadeafdab0` |
| checkpoint | 三折 SHA 全部与 P03-F 冻结合同一致 |
| logits | 三折均为 25 维；UID 恰好一次；无 NaN/Inf |
| C2 重建 | Recall/FDR/macro Recall/macro FDR 四项误差均为 0 |
| 专项测试 | 12 passed；ruff 全绿 |
| 最终门禁 | `R1_0_TASK_PASS` |

每折推理：

| fold | proposal | 耗时 | 吞吐 | 峰值显存 |
|---:|---:|---:|---:|---:|
| 0 | 20,115 | 14.88 s | 1,352/s | 1.05 GiB |
| 1 | 18,437 | 14.20 s | 1,298/s | 1.05 GiB |
| 2 | 16,996 | 13.37 s | 1,271/s | 1.05 GiB |

首次启动在 prepare 前因服务器 `PYTHONPATH` 未包含 `src/` 而停止；没有运行模型或
改写科学产物。修复提交为 `c09d8e5`，失败目录已独立归档，正式任务从空目录重新运行。

## 3. 预注册主结果

### 3.1 相对原始 detector

| 指标 | D0 | R1 selected | delta |
|---|---:|---:|---:|
| pooled Recall | 0.91759 | 0.92524 | **+0.00764** |
| pooled FDR | 0.19897 | 0.19841 | **−0.00056** |
| 25 类 macro Recall | 0.86545 | 0.87232 | **+0.00687** |
| 25 类 macro FDR | 0.23825 | 0.21514 | **−0.02311** |

配对转移：500 个 new TP、340 个 broken TP、净增 160 TP；FP 增加 23，FN 减少
160。说明 GT-crop 教师直接迁移到真实 proposal crop 已存在明确细类纠错能力，但也会
破坏一部分原本正确的预测。

### 3.2 相对当前采用的冻结 C2

| 指标 | C2 | C2 + R1 | delta |
|---|---:|---:|---:|
| pooled Recall | 0.91349 | 0.92151 | **+0.00803** |
| pooled FDR | 0.15929 | 0.15672 | **−0.00257** |
| 25 类 macro Recall | 0.86310 | 0.86916 | **+0.00606** |
| 25 类 macro FDR | 0.20854 | 0.18610 | **−0.02244** |

配对转移：506 个 new TP、338 个 broken TP、净增 168 TP；FP 减少 38，FN 减少
168。预注册自动门禁因此给出
`admit_r1_inference_and_run_short_proposal_domain_finetune`。

## 4. 折间稳定性

| held-out fold | 选择方法 | threshold | Recall delta | FDR delta | net TP |
|---:|---|---:|---:|---:|---:|
| 0 | R4 gate+fusion, p=0.90, margin=0.15, α=0.30 | 0.061 | +0.01728 | −0.00439 | +127 |
| 1 | R2 gate, p=0.75, margin=0.15 | 0.061 | −0.00042 | +0.00778 | −3 |
| 2 | R2 gate, p=0.75, margin=0.15 | 0.041 | +0.00562 | −0.00473 | +36 |

两个问题不能被总体均值掩盖：

- fold1 没有净 TP 收益且 FDR 变差；
- fold0/2 的 held-out FDR 仍高于 0.20，虽然三折合并满足总体硬门槛。

因此方法存在信号，但门控参数和折间行为尚未达到最终模块的稳定程度。

## 5. 按官方三大类检查后的科学修正

冻结 C2 与 C2+R1 的大类 macro 变化为：

| 大类 | Recall delta | FDR delta | 判断 |
|---|---:|---:|---|
| ship | **−0.05268** | −0.07900 | 虚警下降但召回损失不可接受 |
| aircraft | **+0.01787** | **−0.01602** | 明确双向改善，是主要有效信号 |
| vehicle | +0.00498 | **+0.07535** | 少量召回换取严重 FDR 退化 |

官方 V1.6 最终比较的是三大类各自 Recall/FDR，而不是只看 25 类 Overall macro。
所以自动门禁证明的是“R1 机理值得继续”，不能据此直接把全类别 R1 接入最终系统。

科学结论修正为：

> 接受 P03-F 在真实 aircraft proposal 上的细类纠错机理；拒绝当前全类别统一策略作为
> 正式推理模块。舰船和车辆必须保持原路径，除非后续有各自独立证据。

## 6. 事后 aircraft-only 安全诊断

为判断退化是否可以通过结构性隔离解决，使用同一份 R1 结果进行了事后诊断：仅对原
检测大类为 aircraft 的 proposal 使用 R1 输出，ship/vehicle 的类别和 score 逐条恢复为
原始 M1，然后应用完全冻结的 C2 参数。

这不是预注册结果，只用于生成下一实验假设。

| 指标 | C2 | C2 + aircraft-only R1 | delta |
|---|---:|---:|---:|
| pooled Recall | 0.91349 | 0.92161 | **+0.00812** |
| pooled FDR | 0.15929 | 0.14762 | **−0.01167** |
| 25 类 macro Recall | 0.86310 | 0.87739 | **+0.01429** |
| 25 类 macro FDR | 0.20854 | 0.19572 | **−0.01282** |

配对转移为 new TP 458、broken TP 288、净增 170 TP、FP 减少 282。ship 和
vehicle 的四项指标逐项完全不变；aircraft macro Recall 从 0.90548 提升至 0.92335，
macro FDR 从 0.13881 降至 0.12278。

该诊断说明按大类路由不是事后“挑好看数字”的部署结论，而是下一轮必须预注册并重新
验证的安全结构。

## 7. 飞机细类收益与风险

aircraft-only 诊断中较明确的改善：

| 类别 | Recall delta | FDR delta |
|---|---:|---:|
| F-22 | +0.0872 | −0.0708 |
| E-8 | +0.0486 | −0.0578 |
| TU-160 | +0.0388 | −0.0479 |
| TU-22 | +0.0223 | −0.0410 |
| F-16 | +0.0177 | −0.0621 |
| E-3 | +0.0146 | −0.0865 |
| C-17 | +0.0110 | −0.0715 |

仍需重点限制的退化：

| 类别 | Recall delta | FDR delta |
|---|---:|---:|
| SU-24 | −0.0293 | +0.1199 |
| FA-18 | −0.0303 | −0.0201 |
| C-5 | +0.0720 | +0.0804 |
| SU-35 | +0.0121 | +0.0535 |

这说明下一步不能简单“把 P03 模型全量微调几轮”。需要保留 detector 原标签的保守
门控，并针对 SU-24/FA-18、C-5/SU-35 等具体流入流出关系记录混淆转移。

## 8. 最终决定与下一步

R1-0 达到了预期的信息目标：

1. 证明已有 proposal 中存在可由对象 crop 模型恢复的细类信息；
2. 证明收益在冻结 C2 之后仍存在，不是只击败旧基线；
3. 定位出收益几乎全部来自 aircraft，避免继续把一个统一对象头错误地铺到三大类；
4. 暴露 proposal-domain 偏移和少数细类的破坏风险。

下一阶段只启动一个主实验：`R1-1 aircraft-only proposal-domain refinement`。

- 复用本次 55,548 行 manifest 和 logits，不重新生成 M1 OOF；
- 每个 held-out fold 只用另外两折 proposal 训练；
- 从对应 P03-F checkpoint 初始化；
- 先做 aircraft 20 类、无背景类的短微调，对照“零训练 P03-F inference”；
- ship/vehicle 在结构上旁路，保证指标逐条不变；
- 再把显式 background rejector 作为独立消融，不能与细分类微调一次性合并；
- 快筛预算应控制在单折短周期，只有 aircraft macro Recall/FDR 双安全且 broken TP
  明显下降才进入三折确认。

当前不启动 DINOv2、CleanDIFT、扩散或新检测层。R1-0 已经给出了比这些方向更直接、
更便宜的真实 OOF 信号。

## 9. 产物索引

- 本地结果目录：`outputs/R1-0-P03-TEACHER-M1-OOF/`；
- 回传包：`outputs/R1-0-P03-TEACHER-M1-OOF-return.tar.gz`；
- 回传 SHA256：`64f1d1a65e91163f9475f57b50fb531f92f2dff454cb537b8a9599e628bbeeeb`；
- Gitee Release：`v0.2-r1-evidence`；
- 主结果：`evaluation/reranking_result.json`；
- 自动决策：`evaluation/decision.json`；
- 三折 logits：`logits/fold_{0,1,2}_logits.npz`；
- 服务器任务单：
  [`R1_PROPOSAL_RERANKING_TASK_00.md`](../../docs/server/R1_PROPOSAL_RERANKING_TASK_00.md)。
