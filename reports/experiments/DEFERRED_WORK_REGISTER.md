# 延期、待解锁与停止实验台账

更新日期：2026-08-10
状态：`current_pre_innovation_closure`

当前状态和数字先读
[`PRE_INNOVATION_CLOSURE_20260810.md`](PRE_INNOVATION_CLOSURE_20260810.md)。

正式 CV3 已完成，过去“等待 B 划分”的项目已重新分类。当前所有正式
实验固定引用：

```text
split_version: cv3_airport_proxy_k60_v2
manifest_sha256: 27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331
held_out rule: fold == held_out_fold is post-training evaluation/OOF only;
               other folds are train; no checkpoint selection on held-out
```

划分、代码和完整边界见
[`DATA_SPLITS_MASTER_INDEX_v1.md`](../data/DATA_SPLITS_MASTER_INDEX_v1.md)。

M1 结果之后的实际执行顺序统一见
[`NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md`](NEXT_STAGE_AFTER_M1_FORMAL_OOF_MASTER_v1.md)；
本文件只负责记录哪些事项已解锁、暂缓或停止。

## 1. 状态总表

| ID | 项目 | 当前状态 | 还缺什么 | 下一步 |
|---|---|---|---|---|
| CV3-ADAPTER | 正式三折消费适配 | `implemented_local_pass` | 服务器复现 F00 | 六个 CSV 通用视图与逐折 JSON loader view 均有严格门禁 |
| P0-2-FORMAL | crop manifest 重挂 fold | `implemented_local_pass` | 服务器复现 F00 | 已确定性得到 formal crop SHA `a3bed44f…4128` |
| CV3-DETECTION-DATA-LOCK | 正式图像/标签/GT 字节锁 | `implemented_local_pass` | F00 后服务器复现 D00 | 锁定 4,481 图像、4,481 标签、20,933 GT；锁 SHA `03a8d8b5…e77a` |
| CV3-MODEL-ASSET-ENV | M1/M3/E 共用环境与官方权重锁 | `implemented_ready_for_server` | 在同一服务器创建一次不可变锁 | 精确冻结 Python/CUDA/Ultralytics、完整 distribution inventory、两个权重 URL/size/SHA |
| P03-FORMAL | tight-224 ConvNeXt 正式上限 | `complete` | 无 | CV3 macro Recall 0.9287，作为理想 GT-crop 上限 |
| P04-FORMAL | 正式教师比较 | `complete` | 无 | DINOv2-B 0.8294 > ConvNeXt 0.7815 > CleanDIFT 0.7036 |
| C-M1-CV3-OOF | YOLO26-s 正式主检测器 | `complete_formal_with_power_interruption_resume_amendment` | 无 | 4,481 图唯一 OOF、55,548 候选、aggregate 四件套与官方描述性分析均完成 |
| D-M3-CV3-OOF | RT-DETR-L 异构检测器 | `implemented_ready_for_server` | 三折训练和低阈值预测 | 按冻结 foundation 配置跑三折 |
| E-10K-BASE | 10K 工程闭环 | `implemented_ready_for_server` | 可用 checkpoint 与 10K 图像 | 先跑工程与分段计时，最终模型后复测 |
| OOF-AUDIT | 正式预测完整性/官方评估 | `M1_complete_waiting_optional_M3` | M3 paired 分析可选 | M1 官方指标、逐折稳定性和计数守恒错误分解已完成 |
| THRESHOLD-FORMAL | 正式阈值 | `crossfit_pooled_and_macro_complete` | 未来变体的 cross-fit 校准 | 当前 cross-fit pooled Recall 0.9176/FDR 0.1990；V1.6 macro 已补算；fold 0/2 仍超线 |
| M1-M3-PAIRED | 异构互补分析 | `implemented_waiting_both_OOF` | M1 与 M3 同协议 OOF | 已实现 TASK-01：paired TP/FN、IoU、FP 交并和 oracle-union |
| P05-HARDNEG / N2 | 真实背景拒识 | `v1_invalid_v2_optional` | N0-4 v2 人工确认 clear background | 不得将未标 hard negative 自动当背景；详见 N2 v2 修复报告 |
| P2-MAINLINE1 | vehicle 近阈值特征增强 | `stopped_after_paired_fast_screen` | 仅有新的独立机制证据时才重新立项 | s 级完整 P2 在 fold0 配对快筛中未改善候选下限且显著降低可用工作点 Recall；Y2 正式三折、Y3 与 P2-Lite 不启动 |
| P06-REAL | 真实框修正 | `deferred_low_localization_evidence` | 新的边界/尺寸分解反证 | 工作点仅 `FN_LOC=66/1734`，近期不占 GPU |
| P06-DIFF | bbox residual diffusion | `stopped_no_real_admission` | P06-REAL 出现稳定且仍未解决的定位收益空间 | 不能绕过确定性强基线与真实错误门禁 |
| COMBINED-ABLATION | 二阶段组合消融 | `waiting_modules` | 已入选的 P05/P06 模块 | base / +P05 / +P06 / 两者 |
| FINAL-MODEL | 最终模型/架构冻结 | `waiting` | OOF、错误分解、模块消融、10K 时延 | 以官方门槛、稳健性和 20 秒约束决策 |

## 2. 已停止项目

| ID | 停止证据 | 恢复条件 |
|---|---|---|
| P07-SD15 | 仅 1/24 优于传统融合，48 个扩散输出中 43 个有 halo | 新任务定义或不同生成机制给出独立反证 |
| C-HPR | 严格门控只少 1 个 FP；位置也早于全局对象聚合 | 真实 Pred-OOF crop 上新对象模型有稳定净收益 |
| CLEANDIFT-SOLE | 探索 probe 弱于 DINOv2-B 和 ConvNeXt | 正式 CV3 显示互补增益，只能作为对照/辅助教师 |
| P06-DIFF-EARLY | 真实错误类型未知 | 通过 P06-REAL 准入后才可恢复 |
| MULTI-FACTOR-M2 | 同时改模型规模和分辨率无法归因 | M1 正式三折稳定后逐因素立项 |
| R1-9-SHIP-VEHICLE-NMS | 固定官方 IoU 虽减少 219 FP，却误删 58 TP；舰船 pooled Recall 下降 0.02013 | 只有新的实例唯一性证据，不允许继续搜索 NMS 阈值 |

## 3. 执行计划

### Phase A：共同输入层，CPU，已实现并本地通过

目标：让 P 系列、C、D 都读取完全相同的 CV3。实现与服务器任务单见
[`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)；
当前只剩同服务器 F00 的确定性复现。

1. 实现 `held_out_fold` 适配：
   - 输入只允许 `fold=0/1/2`；
   - held-out fold 为 val，另两折为 train；
   - 同一 `group_id` 不得跨集合；
   - 每折输出图像数必须为验收报告中的冻结值。
2. 生成 `formal_crop_manifest_v2`：
   - 复用 P0-2 的 20,933 个对象、62,799 条三 crop 记录；
   - 只按 `source_image_id` 或 `annotation_uid` 重挂 fold；
   - 旧 crop 路径、bbox、像素及 canonical SHA 不变；
   - 每个对象恰好有一次 held-out 归属。
3. 为两种适配增加覆盖、无泄漏、确定性及错误输入测试。

完成门禁：三折图像计数、来源组计数、对象计数和 manifest SHA 全部记录。

### Phase A2：正式检测数据字节锁，CPU，已实现并本地通过

F00 通过后执行 D00，生成唯一
`FORMAL_DETECTION_DATA_LOCK.json`（SHA
`03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`）。
M1/M3 每折开训前必须全量复验 8,962 个图像/标签文件，并把验证报告绑定到
fold metadata。A00 与 F00→D00 可并行，但模型任务必须同时通过 A00 和 D00。

### Phase B：四条可并行支线

#### B1. P03/P04 对象 crop 支线

- P03 只重跑 `tight-224 + natural sampler + seed 42` 三折；
- 不再重跑 336、context、sqrt-inverse 或多 seed 网格；
- P04 只比较：
  1. ConvNeXt-T；
  2. DINOv2-B CLS+patch；
  3. CleanDIFT map0；
- feature cache 可复用的前提是 `annotation_uid` 与 canonical 输入 SHA
  全匹配；
- PCA、train-RMS、分类头和任何 normalizer 都必须逐折只在训练侧拟合。

输出：逐折、均值、标准差、头中尾类和 TU-160 压力折；P04 只在同协议
下判断教师排序及 DINOv2 的独立收益。

#### B2. C 的 M1 正式 OOF

- 固定 YOLO26-s / 1024 foundation；
- 三个 fold 从相同原始预训练权重独立训练；
- 禁止用旧 `dev_v1` best checkpoint 续训或直接生成 OOF；
- 第一轮不引入未完成 rare-rebalance、HPR 或 M2；
- 每折固定跑满预注册 epoch，交 `last.pt` SHA、完整配置、环境、训练日志和
  低阈值预测；
- 合并后 4,481 张图必须各自由“未见过本组”的一个模型预测一次。

#### B3. D 的 M3 正式 OOF

- 固定 RT-DETR-L / 1024 foundation，不重开模型网格；
- 数据、低阈值输出和记录合同与 M1 相同；
- 科学上可与 M1 并行；若只有一张 GPU，可在 M1 后排队，但不依赖
  P05/P06。

#### B4. E 的 10K 工程

- 立即使用现有 M1 adapter/checkpoint 打通切片、坐标恢复、跨 tile
  聚合和 COCO JSON；
- 分别记录读盘、切片、model-only、融合/序列化和完整 pipeline；
- 当前 4080 SUPER 结果只能是工程基线；最终模型和阈值冻结、官方 10K
  manifest 进入代码注册表后，再在独占 RTX 3090 上复测 p50/p95 及 20 秒
  硬门槛。

### Phase C：OOF 汇合与决策门禁

M1 OOF 到达后先做：

1. 完整性审计：4,481 图恰好一次 OOF，输入 fold/组未参与训练；
2. 官方 Recall/FDR、阈值无关候选召回曲线；
3. cross-fit 阈值，禁止在同一 OOF 全集拟合后又对原集合宣告无偏成绩；
4. 互斥且计数守恒的错误分解：
   `FP_BG/FP_DUP/FP_FINE_CLS/FP_LOC` 与
   `FN_FINE_CLS/FN_LOC/FN_MISSING`；
5. `R_loc@oracle-class`、`Acc_fine@localized`、边界/尺寸/头中尾类。

M3 OOF 到达后补做 M1/M3 paired/oracle-union，判断 M3 是主模型、定位
教师、门控互补候选，还是应停止。

### Phase D：有条件放行 P05/P06

- `FP_BG` 是主要 FDR 来源且样本量足够，才放行 P05；
- 近官方 IoU 阈值的定位损失或边界框明显，才放行 P06-REAL；
- P05/P06 自身也必须 cross-fit：某个 held-out fold 的二阶段模型只能
  用另外两折 OOF 派生样本训练；
- P06 先比较 identity、几何/MLP residual 和视觉确定性 refiner；
- 只有确定性强基线后仍有稳定空间，才允许 P06-DIFF。

### Phase E：最终组合

对入选模块做 base、单模块和组合消融，重新 cross-fit 阈值；最后把准确
率、FDR、折间稳定性、模型复杂度和 10K p50/p95 放在同一张决策表中。

## 4. P06-TASK-02 正式输入与当前决策

M1 已交付并通过审计以下四件套：

```text
formal_crop_manifest.csv
oof_metadata.json
oof_images.csv
oof_proposals.csv
```

M1 四件套来自同一个冻结检测器的三折模型：每个模型只对未参与其训练的
held-out fold 预测，合并后 4,481 图恰好一次覆盖。`oof_metadata.json`
证明三个 fold 均绑定 D00 锁 SHA
`03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a`；
完整结论见
[`M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`](M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md)。

当前不再因输入缺失等待，而是因真实定位错误证据过弱主动暂缓：
探索工作点只有 `FN_LOC=66/1734`。此前 P06 服务器代码/任务模板在当前
Git 中仍没有完整归档；若将来出现新的定位准入证据，恢复实验前必须先从
服务器原路径或回传包恢复并核对原始 SHA，不能凭聊天回报重写近似实现。
M3 若完成，也必须形成并单独审计自己的四件套，不能与 M1 混写。

## 5. 台账规则

- `ready` 表示科学输入已经具备，不表示可以跳过工程适配；
- `waiting_*` 项只能准备代码、合同和 smoke，不能报告正式结论；
- `stopped` 项默认不恢复；
- 每次状态变化记录日期、manifest SHA、代码 commit、任务单、checkpoint
  SHA 和验收报告；
- 跨成员大文件只认 Gitee Release/附件与 SHA256，见
  [`ARTIFACT_RELEASE_REGISTER.csv`](ARTIFACT_RELEASE_REGISTER.csv)；
  [`SERVER_ARTIFACT_REGISTER.csv`](SERVER_ARTIFACT_REGISTER.csv) 仅是历史路径快照。
