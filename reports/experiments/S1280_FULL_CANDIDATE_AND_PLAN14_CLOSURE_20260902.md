# S1280 全量候选、风险边界与方案14闭环

日期：2026-09-02  
状态：`full_and_cv3_complete_purity_rejected_no_submission_authorized`

## 1. 当前判断

S1280 值得训练成一个正式候选，但现有证据还不足以称为“确定优于正式主线”。它在
同长度 fold0/40ep 中相对 S1024 呈现本轮迄今最强的单因素正向信号，且在相同阈值
0.481 下仍取得 Gate Recall `+5.67pp`、Gate FDR `-0.51pp`。这证明尺度保持确实改善
了一部分目标排序，而不只是通过降低阈值增加候选。

风险同样必须保留：该结果只有一个 held-out fold，两个 FDR15 工作点都使用了本折标签；
Ship 宏平均的绝大部分提升来自仅 6 个 GT 的 category 0；S1280 的低阈值 score-floor
候选上限没有全面提高。因此本次 full 是高收益、可解释、但有跨域方差风险的比赛候选，
不是对原预注册失败结论的改写。

## 2. 40 epoch 与 160 epoch 的关系

- `S/M × 1024/1280` 四格全部是 fold0、40 epoch、固定 last，仅用于同长度筛选；
- 当前既有正式 Y5-S 主线使用全部 4,481 图、160 epoch、固定 last；
- S1280 的 40e 训练到末轮时损失仍在下降，因此 40e 不应直接作为提交权重；
- 正在运行的候选沿用正式主线的 160e 全量训练长度，不用验证集选 checkpoint。

## 3. 为什么 S1280 有效，以及上限在哪里

各自 FDR15 oracle 工作点的配对 GT 结果：

| 尺度 | GT | S1024 R | S1280 R | 净找回 GT |
|---|---:|---:|---:|---:|
| `<48 px` | 238 | 43.28% | 54.62% | +27 |
| `48–80 px` | 2,286 | 76.03% | 77.65% | +37 |
| `80–128 px` | 3,242 | 85.75% | 85.60% | -5 |
| `>=128 px` | 1,584 | 82.77% | 81.82% | -15 |

尺度增益集中在 `<80 px`，尤其 Vehicle：TP `48→69`、FP `19→17`，说明 1280 同时
改善 Vehicle 的可见性和排序。中大型目标没有收益，意味着继续无条件提高到 1536/2048
很可能主要增加时延、重复框和背景响应，不应在没有新门禁的情况下扫描。

Ship 不是同样干净的提升。category 0 只有 6 个 GT，TP `1→3` 对 Ship 四类宏平均的
贡献极大；category 2 TP 不变但 FP `+14`，category 3 虽 TP `+17` 也新增 21 FP。
Aircraft 总宏 Recall只增 `0.034pp`，却新增较多重复与背景 FP。S1280 的理论短板已经
从“看不见小目标”转为“Ship/Aircraft 的细类混淆、重复框与背景置信仍未解决”。

## 4. 正在运行的唯一 full 合同

| 字段 | 冻结值 |
|---|---|
| 模型 | YOLO26-s，25 类 |
| 初始化 | `yolo26s.pt`, SHA256 `646f8bc3…384a1b` |
| 数据 | 全部 4,481 张官方训练图 |
| 输入 | 1280 |
| 增强 | Y5 RandomRotate90 `p=1.0` |
| 训练 | 160 epoch，AdamW，cosine，seed42 |
| batch | 全局 12，三卡各 4 |
| checkpoint | 固定第 160 epoch `last.pt` |
| 设备 | 3×RTX 3090 DDP |
| 服务器输出 | `/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1` |

驱动为 `scripts/server/run_y5_s1280_full_3gpu.sh`；`scripts/train_full_y5.py` 新增显式
`--device 0,1,2` 支持。驱动在训练前验证三张 GPU、权重/manifest/8,962 标签的 SHA、
4,481 图计数和 dry-run 合同；训练后要求 `results.csv` 恰好 160 行并锁定 last.pt SHA。

2026-09-02 22:29 CST 只读检查时，训练已完成 23/160；三卡利用率 47%–94% 波动，功耗
275–292W，显存约 6.1GiB/卡，24 个 DataLoader worker 正常，CPU/内存无压力，有限损失。
每轮约 1 分钟，主体训练 ETA 约 2.5 小时。没有为了提高显存占用而改变全局 batch，
因为这会改变优化轨迹并破坏与正式 Y5-S 的可比性。

为避免 full 完成后 GPU 空闲，服务器还启动了只等待、不占 GPU 的
`s1280-postfull` 控制器。它只在 full `status=complete` 后执行：GPU0/1 并行补
S1280 fold1/2，GPU2 串行补 S1024 fold1/2；四个确认折仍是 40e/seed42/batch8。
随后生成两模型各自的 4,481 图 outer-fold frontier 和逐细类配对报告，并单独复核
`coarse_purity_sqrt`。该队列不修改 full 权重，不进行 Docker 打包或正式提交；任何
中间失败均 fail closed，不会覆盖或 resume。

2026-09-03 复核：full 已完成 160/160，`training_result.json` 和固定 last.pt 均存在，
last.pt SHA256 为 `539ab9c9…ed0460`，结果 CSV SHA256 为 `7a8ad52a…4ed0460`。
S1024/S1280 的 fold1/2 四个确认训练、推理和单折 frontier 也全部完成。原控制器随后
在不影响上述产物的 `coarse_purity_sqrt` 推理中因 FP16/FP32 `torch.allclose` dtype
不一致 fail closed；代码已统一转成 FP32 比较并通过专项测试，当前只恢复该推理和三折
aggregate，没有重训任何模型。

确认折各自 FDR15 单折 oracle 结果如下；这些阈值仍使用本折标签，只说明方向：

| fold | S1024 R/FDR | S1280 R/FDR | Recall 差值 |
|---|---|---|---:|
| 0 | 51.143% / 14.251% | 59.406% / 14.718% | +8.263pp |
| 1 | 54.261% / 14.558% | 60.431% / 14.601% | +6.170pp |
| 2 | 39.139% / 7.528% | 41.787% / 14.370% | +2.648pp |

三个折的 Recall 方向一致，但 fold2 付出明显 FDR 代价。是否准入必须读取即将生成的
outer-fold cross-fit 结果，不能把三行 oracle 直接平均。

### 4.1 最终 outer-fold cross-fit 结论

恢复链已经完成，三折阈值均只由另外两折选择，`selection_uses_held_out_labels=false`。
这里的 `FDR15` 是训练折选择目标，不表示 held-out 汇总 FDR 必然不超过 15%。最终结果：

| 条件 | cross-fit 阈值 fold0/1/2 | Gate Recall | Gate FDR |
|---|---|---:|---:|
| S1024 | 0.646 / 0.646 / 0.481 | 43.440% | 17.119% |
| S1280 | 0.706 / 0.706 / 0.436 | 47.544% | 17.869% |
| S1280 - S1024 | — | **+4.104pp** | **+0.750pp** |

按官方平台三个粗类宏平均口径拆分：

| 粗类 | Recall 差值 | FDR 差值 | 判断 |
|---|---:|---:|---|
| Ship | -0.485pp | +1.529pp | 轻微退化 |
| Aircraft | -3.870pp | -0.743pp | 精度改善但召回明显退化 |
| Vehicle | +16.667pp | +1.463pp | 尺度增益强且跨折存在 |

因此 S1280 不是“所有类别共同提升”的正式替代品，也不通过原始逐粗类退化不超过
0.5pp 的严格准入门。它证明了 Vehicle 的尺度瓶颈真实存在，但同时把问题转移为
Aircraft 召回损失和 Ship 稳定性风险。

在两者假定相同时延时，按 2026-08-31 平台确认的七项等权绝对计分公式，S1280 的
纯质量分只比 S1024 高约 `0.926` 分；其质量收益可容忍的额外时延上限约为 `6.481s`。
这个计算只用于同一代理集内比较，不能当作隐藏集绝对分预测。正式候选仍需完成
RTX 3090 Docker 时延和逐框一致性后才能决定是否值得消耗提交机会。

### 4.2 `coarse_purity_sqrt` 最终结论

独立恢复后的 purity 原型已完整运行。在 fold0 各自 oracle FDR15 工作点，相对 S1280
identity 的 Gate Recall 为 `-1.611pp`，Gate FDR 为 `+0.078pp`；其中 Aircraft Recall
`-6.075pp`，Vehicle FDR `+2.455pp`。该变换没有获得准入资格，永久停止，不进入 full
权重、Docker 或任何组合。原始 R1 的失败仅是 FP16/FP32 一致性断言 dtype 不同；修复
后 R2 完成，且确认失败结论来自算法指标而不是工程异常。

## 5. 完成后提交前的硬验收

只有以下项目全部通过，才形成一个可由用户决定是否消耗正式机会的镜像候选：

1. `status.txt=complete`，`results.csv=160` 行，last.pt 与结果清单 SHA 一致；
2. Docker 模型输入同步改为 1280，tile 几何、坐标恢复和输出格式不变；
3. 离线入口与 Docker 对同一小批图逐框一致；
4. RTX 3090 按官方 10K 口径测时，满足 20 秒硬门并记录真实平均时延；
5. 不直接复制 fold0 的 0.436 oracle 阈值到部署；阈值必须来自已有无泄漏校准证据或
   新的 source-grouped CV3；
6. 不把 MacroExpert、DEIM-HCL、外部数据或未准入后处理混进该镜像。

若上述工程验收通过，这个候选值得占用一次正式提交：它是目前最清楚的高收益单因素，
且单模型比双模型专家路线更快、更稳定。但它仍属于“较高上行、有限统计保证”的提交，
不能承诺把单折增益等比例复制到隐藏集。

## 6. 方案14逐项闭环

| 方案14项目 | 状态 | 结论/后续 |
|---|---|---|
| 配置、尺度、时延审计 | 部分完成 | 训练合同已锁；S1280 full 后补 3090 Docker 10K 时延 |
| MacroExpert 6 类视图与 review100 | 完成 | 100/100 审核通过 |
| MacroExpert-M official-only fold0/40 | 完成、否决 | Normal/Hard/Sentinel 不同向，Sentinel Vehicle 明显退化 |
| MacroExpert-M background | 按冻结顺序停止 | official-only 未正向，未获得启动资格 |
| `coarse_purity_sqrt` | 已实现未评估 | 零推理成本原型；不可混入当前 full，可在独立冻结预测上低成本复核 |
| M25-1024 / M25-1280 | 完成同等问题的独立 2×2 | 两个 M 条件均显著差于 S；停止增大容量 |
| S25-1024 / S25-1280 | 完成 | 尺度正向、容量负向；S1280 full 正在运行 |
| 前两名 folds1/2 | 原门禁未授权 | 若用于提交校准，必须新立确认性 CV3 合同，不得改写原门禁 |
| 唯一 full | 正在运行 | 仅 S1280 单模型，不含其他变量 |
| EFL/EQLv2 | 条件未满足 | 原方案只允许 MacroExpert 明显正向后单因素加入；当前不启动 |
| 更大 L 模型 | 条件未满足 | M 已负向，不启动 L |
| Formal-Anchor P10 | 未形成独立冻结集 | 不能虚构；继续以现有 Normal/Hard/Sentinel-B 明确标注局限 |

## 7. S1280 后续真正值得做的提升

按风险和信息价值排序：

1. **S1280 source-grouped CV3 确认与阈值校准**：这是最重要的缺口。fold1/2 必须与
   现有 S1280 fold0 使用相同 40e/seed/batch，再做 outer-fold threshold，判断 Vehicle
   和稀有 Ship 增益是否跨折稳定。
2. **1280 对齐的部署推理**：训练和 Docker 都保持 1280，不能训练 1280 后又在入口
   下采样到 1024，否则会直接丢掉已观察到的小目标收益。
3. **只针对已定位错误做独立模块**：若后续还有计算预算，优先评估 Ship/Aircraft 的
   FP_DUP/FP_CLS 抑制；Vehicle 不需要 reject，应保护其 S1280 新增 TP。每个模块必须
   单独过门后才能组合。
4. **`coarse_purity_sqrt` 离线复核**：只读现有预测即可，不耗训练 GPU；若不能在
   Normal/Hard/Sentinel-B 同向，立即永久关闭。

不建议的方向：继续增加模型容量、提高到更大输入、重开 MacroExpert repeat、重开
DEIM-HCL、扫描融合权重或把多个未通过模块一次性叠加。它们要么已有负向证据，要么会
使正式提交的归因和稳定性不可控。

## 8. 代码与结果索引

- 全量训练：`scripts/train_full_y5.py`；
- 三卡驱动：`scripts/server/run_y5_s1280_full_3gpu.sh`；
- 跨折确认单格：`scripts/server/run_yolo_scale_cv3_confirm_condition.sh`；
- full 后队列：`scripts/server/run_s1280_postfull_confirmation.sh`；
- 未评估分数变换复核：`scripts/server/run_s1280_purity_fold0_eval.sh`；
- CV3 COCO 合并：`scripts/merge_cv3_coco_ledgers.py`；
- 分数变换配置物化：`scripts/materialize_yolo_score_transform_config.py`；
- 单折配对分析：`scripts/analyze_single_split_paired_scale.py`；
- 2×2 总账：`reports/experiments/YOLO_CAPACITY_SCALE_2X2_PLAN_20260902.md`；
- 配对 JSON/CSV：`outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024_vs_s1280_*`；
- 固定阈值对照：`outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/{s1024,s1280}_fixed_*.json`；
- 方案14原实施报告：
  `reports/experiments/IMPROVEMENT_PLAN14_MACROEXPERT_AND_GATE_AUDIT_20260901.md`。
