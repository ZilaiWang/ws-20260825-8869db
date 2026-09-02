# YOLO 25 类容量/尺度 2×2 快筛合同

日期：2026-09-02  
状态：`screen_complete_strict_gate_failed_s1280_full_candidate_running`

## 1. 要回答的问题

当前正式主线仍是 YOLO26-s、1024、25 类、Y5 RandomRotate90。MacroExpert-M 已证明
“六类专家 + 图像 repeat + 互斥替换”会增加 Ship 重复/背景 FP 并损失 Vehicle TP；
DEIM-HCL 则表现为 Recall 上升但 FDR 代价过大。下一轮不再调整这两条失败配方，而是
回答尚未被公平实验回答的基础问题：

1. 仅把 YOLO26-s 换成 YOLO26-m，官方三粗类宏平均是否提高；
2. 仅把输入从 1024 提高到 1280，收益能否超过 FDR 与时延代价；
3. 容量和尺度是否有交互，还是其中一个因素单独有效。

这不是 MacroExpert 的补做控制，而是新的独立、预注册的标准 25 类实验。

### 1.1 正式分数的边际价值

Attempt 1 的正式锚点为 72.1331：Ship `R/FDR=0.874969/0.320177`，Aircraft
`0.967641/0.064691`，Vehicle `0.852632/0.325000`，时延 2.473167s。按官方绝对公式：

- 任一粗类 Recall 增加 1pp，当前区间约增加总分 0.381；
- Ship/Vehicle FDR 降低 1pp（当前均高于 0.2），约增加总分 0.107；
- Aircraft FDR 降低 1pp（当前低于 0.2），约增加总分 0.286；
- 若只把 Ship/Vehicle 同时推到 Recall=0.95、FDR=0.20，其他项不变，理论总分约
  81.328；推到 Recall=0.95、FDR=0.15 时约 84.185。

所以 2×2 的成功标准不是单纯多出若干低分框，而是更强模型是否把 Ship/Vehicle 的
TP 排到结构化背景之前。若只增 Recall、同时像 DEIM-HCL 一样支付更大的 FDR 代价，按
官方公式仍可能净负。

## 2. 冻结矩阵

| key | 模型 | 输入 | 初始化 SHA256 |
|---|---|---:|---|
| s1024 | YOLO26-s | 1024 | `646f8bc3…384a1b` |
| s1280 | YOLO26-s | 1280 | `646f8bc3…384a1b` |
| m1024 | YOLO26-m | 1024 | `401cea9a…5d0b7` |
| m1280 | YOLO26-m | 1280 | `401cea9a…5d0b7` |

共同条件：

- CV3 fold0，split SHA256 `a647ce03…f128943`；
- seed42、40 epoch、总 batch8、AdamW、同一学习率与增强；
- Y5 RandomRotate90 `p=1.0`；
- `close_mosaic=20`、固定第 40 epoch `last.pt`；
- 低阈值 0.001、NMS IoU 0.70、非切片整图推理；
- 同一 1,507 图 GT，SHA256 `2641d3bb…e585977`；
- 所有选择和准入只读 `platform_observed_20260831`。

训练 batch 在四格中保持 8。1280 的推理 batch 固定为 4；推理 batch 不作为方法因素。

## 3. 因子分解

- `s1280 − s1024`：s 模型上的纯尺度效应；
- `m1024 − s1024`：1024 下的纯容量效应；
- `m1280 − s1280`：1280 下的纯容量效应；
- `m1280 − m1024`：m 模型上的纯尺度效应。

主参考始终是同长度 `s1024`。历史 160 epoch 主线只用于绝对能力现实检查，不能拿来
否决尚未充分训练的 40 epoch 候选。

## 4. 单折探索门禁

在各模型自己的 platform Gate-FDR≤0.15 frontier 上：

1. Gate Recall 至少增加 0.5pp；
2. Gate FDR 不增加；
3. Ship/Aircraft/Vehicle 任一粗类 macro Recall 不下降超过 0.5pp；
4. conf=0.001 的 Ship 或 Vehicle macro Recall 至少增加 1pp。

四条全部通过才可扩 source-grouped CV3。单折阈值使用了 held-out 标签，只能用于排序，
不能成为 Docker 阈值。若四格均失败，停止容量/尺度路线，不做 epoch、batch、阈值或融合
权重扫描。

## 5. 服务器执行

每个条件使用同一驱动，通过 `CUDA_VISIBLE_DEVICES` 做物理 GPU 隔离：

```bash
cd /root/autodl-tmp/xh-202625
CUDA_VISIBLE_DEVICES=0 CONDITION=s1024 bash scripts/server/run_yolo_capacity_scale_fold0_condition.sh
CUDA_VISIBLE_DEVICES=1 CONDITION=s1280 bash scripts/server/run_yolo_capacity_scale_fold0_condition.sh
CUDA_VISIBLE_DEVICES=2 CONDITION=m1024 bash scripts/server/run_yolo_capacity_scale_fold0_condition.sh
CUDA_VISIBLE_DEVICES=3 CONDITION=m1280 bash scripts/server/run_yolo_capacity_scale_fold0_condition.sh
```

四格终态均为 `complete` 后执行：

```bash
bash scripts/server/finalize_yolo_capacity_scale_fold0.sh
```

单卡时可按 `s1024 → m1024 → s1280 → m1280` 串行运行；不得重复启动已存在的结果目录，
不得 resume。

单卡服务器可由持排他锁的串行控制器按上述顺序自动执行：

```bash
screen -dmS yolo-capscale bash -lc \
  'cd /root/autodl-tmp/xh-202625 && bash scripts/server/run_yolo_capacity_scale_fold0_single_gpu.sh'
```

控制器只会跳过 `status.txt=complete` 的不可变单格；若发现半成品目录会 fail closed，
不会覆盖或自动续训。

三卡服务器使用 `scripts/server/run_yolo_capacity_scale_fold0_three_gpu.sh`：GPU0 先跑
`s1024` 再接 `m1280`，GPU1 跑 `m1024`，GPU2 跑 `s1280`。物理卡隔离之外的实验
合同与单卡控制器完全相同。

## 6. 无 GPU 验收状态

- 四份配置均通过 YAML/合同测试；
- 训练配置物化器锁定 seed、40 epoch、batch8、尺度和 close_mosaic；
- 推理配置物化器会删除任何专家 `label_map/drop_labels/score_transform`，防止污染标准
  25 类模型；
- 本机用真实 fold0 split 与正式 YOLO26-s 权重完成无 GPU dry-run：train/val=
  2,974/1,507，输出训练合同一致；
- 服务器脚本通过 `bash -n`，禁止 `--resume`，并锁定两份模型、split 与 GT SHA；
- 2×2 汇总器和 platform-only 准入测试通过。
- 正式评测入口注册表审计为 `pass`：24/24 active entrypoints 已绑定
  `platform_observed_20260831`；三个决策器还会校验 payload 内的精确协议名并对旧格式
  fail closed。

## 7. 代码索引

- 配置：`configs/experiments/s25_yolo26s_1024_fold0_40ep.yaml`、
  `s25_yolo26s_1280_fold0_40ep.yaml`、`m25_yolo26m_1024_fold0_40ep.yaml`、
  `m25_yolo26m_1280_fold0_40ep.yaml`；
- 训练物化：`scripts/materialize_yolo_capacity_scale_config.py`；
- 标准推理物化：`scripts/materialize_standard_yolo_infer_config.py`；
- 单格服务器驱动：`scripts/server/run_yolo_capacity_scale_fold0_condition.sh`；
- 单卡串行控制器：`scripts/server/run_yolo_capacity_scale_fold0_single_gpu.sh`；
- 三卡并行控制器：`scripts/server/run_yolo_capacity_scale_fold0_three_gpu.sh`；
- 仅限训练已满 40 epoch 后的推理恢复：
  `scripts/server/recover_yolo_capacity_scale_posttrain.sh`；
- 四格汇总：`scripts/summarize_yolo_capacity_scale_screen.py`；
- 汇总驱动：`scripts/server/finalize_yolo_capacity_scale_fold0.sh`；
- 中断恢复后的只读等待与汇总：
  `scripts/server/wait_finalize_yolo_capacity_scale_fold0.sh`；
- 测试：`tests/test_yolo_capacity_scale_screen.py`。

## 8. 2026-09-02 三卡执行记录

- 正式执行主机：AutoDL 端口 33070，3×RTX 3090；结果根目录
  `/root/autodl-tmp/results/YOLO-CAPACITY-SCALE-FOLD0-V1`；
- 代码使用隔离目录 `/root/autodl-tmp/xh-202625-capscale`，不覆盖服务器旧仓库；
- 模型、split、GT 已逐文件复核冻结 SHA，4,481 张 split 图像零缺失；
- 首次执行中 `s1024` 训练 40/40 正常，但推理配置仍引用旧主机
  `/workspace/.../split_view.json`，推理 fail closed；修复后仅对已满 40 epoch
  的 `s1024` 执行推理与固定 frontier，最终 1,507 图、46,621 框，
  状态 `complete`；
- 第二次三路并发启动时，Ultralytics 三进程同时写入
  `/root/autodl-tmp/data/labels/train.cache`，`s1280` 因读到部分写入的
  NumPy cache 报 `EOFError: No data left in file`。该故障发生在模型初始化阶段，
  不构成容量/尺度科学结果；
- 恢复时为 `s1280` 建立独立 label-cache 数据视图，8,962 个标签文件
  的相对路径+内容 SHA256 聚合值与共享数据完全一致
  (`929d7321…64a2128`)，不改动数据、模型或训练参数；
- 当前 GPU0=`m1280`、GPU1=`m1024`、GPU2=`s1280`，三路均已稳定进入训练；
  `s1024` 已完成。四格全部 `complete` 后，恢复收尾器才会执行唯一
  `screening_result.json` 汇总；
- 端口 19864 的单卡重复执行在 s1024 第 9 epoch 后主动停止并保留现场，不纳入比较。

## 9. 最终结果（2026-09-02 21:42 CST）

四格均完成 40/40 epoch、`last.pt`、1,507 图低阈值推理和固定 frontier；
frontier 与汇总的 `RESULT_SHA256.txt` 全部校验通过。表内是每个候选在自身
platform Gate-FDR≤0.15 frontier 上的单折诊断值：

| 条件 | 阈值 | Gate Recall | Gate FDR | 相对 s1024 Gate Recall |
|---|---:|---:|---:|---:|
| s1024 | 0.481 | 0.51143 | 0.14251 | 基线 |
| s1280 | 0.436 | **0.59406** | 0.14718 | **+8.26pp** |
| m1024 | 0.661 | 0.36321 | 0.14272 | -14.82pp |
| m1280 | 0.491 | 0.47475 | 0.14814 | -3.67pp |

因子结论：

- S 模型的 1024→1280 尺度效应很强：Gate Recall +8.26pp，其中 Ship
  +8.96pp、Vehicle +15.79pp、Aircraft +0.03pp；但 Gate FDR 也增加 0.47pp，
  因而未通过预注册的“FDR 不增加”严格门禁；
- M 模型的 1024→1280 同样有 +11.15pp Gate Recall 的尺度效应，但
  `m1280` 仍比 `s1280` 低 11.93pp；
- 相同 40 epoch/总 batch8 条件下，S→M 在两个尺度都显著降低 Gate Recall，
  本轮没有证据支持增大模型；
- `s1280` 的低阈值 score-floor Ship/Aircraft Recall 分别下降 6.94pp/1.00pp，
  说明它的改善主要发生在 FDR 约束下的排序/校准区间，并非原始候选上限
  全面提高。

预注册结论为 `selected_for_cv3=null`、`next_action=stop_capacity_scale_route`。
这个结论只表示本轮严格准入失败，不应把 `s1280` 的强尺度信号改写为“没有
收益”。若未来验证该信号，必须新建独立预注册的 source-grouped CV3，不得直接
使用本折 oracle 阈值或事后放宽本轮门禁。

小型可审计产物已回传至
`outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/`；服务器保留四份 checkpoint 与低阈值大预测文件。

## 10. S1024→S1280 配对错误分析与后续候选

为区分“尺度本身提高排序能力”和“候选使用更低 oracle 阈值”两种解释，补做同一阈值
`0.481` 的固定工作点比较。该阈值来自本折 S1024 frontier，因此仍是单折诊断，不是
Docker 阈值；但两模型在同一工作点比较可以排除阈值差异：

| 模型 | 固定阈值 | Gate Recall | Gate FDR |
|---|---:|---:|---:|
| S1024 | 0.481 | 0.51143 | 0.14251 |
| S1280 | 0.481 | **0.56811** | **0.13743** |

即 S1280 在相同阈值下 Recall `+5.67pp`，FDR `-0.51pp`。尺度收益不是单纯放低阈值
堆框，而是部分目标置信排序和背景分离确有改善。

在各自 FDR15 oracle 工作点（S1024=`0.481`、S1280=`0.436`）做逐 GT 配对后：

| 原生框尺度（sqrt area） | GT | S1024 Recall | S1280 Recall | 差值 |
|---|---:|---:|---:|---:|
| `<48 px` | 238 | 43.28% | **54.62%** | **+11.34pp** |
| `48–80 px` | 2,286 | 76.03% | 77.65% | +1.62pp |
| `80–128 px` | 3,242 | 85.75% | 85.60% | -0.15pp |
| `>=128 px` | 1,584 | 82.77% | 81.82% | -0.95pp |

主要收益确实来自小目标。Vehicle 增加 21 TP、减少 2 FP，Recall `+15.79pp`、FDR
`-8.59pp`，是最可信的尺度收益。Ship 的宏平均增益则主要受 category 0 影响：该类
本折只有 6 个 GT，TP 从 1 增至 3；其 `+33.33pp` Recall 经四船类等权后对 Ship
宏平均贡献约 `+8.33pp`，存在很高抽样方差。category 3 另增 17 TP，但增 21 FP；
category 2 TP 不变而增 14 FP。Aircraft 总体近乎持平，且不同细类正负互抵。

错误守恒分解进一步显示：

- Ship：`FN_MISS -27`，但 `FP_BG +26`、`FP_LOC +7`；
- Aircraft：`FN_MISS -3`，但 `FP_DUP +30`、`FP_BG +28`；
- Vehicle：`FN_MISS -22`、`FP_BG -3`，是唯一 Recall/FDR 同向改善的大类。

因此 S1280 是“Vehicle 与小目标明确正向、Ship 有收益但伴随方差和 FP、Aircraft
总体中性”的候选，不应描述成所有 25 类全面提升。结构化原件：

- `scripts/analyze_single_split_paired_scale.py`；
- `outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024_vs_s1280_paired_diagnosis.json`；
- `outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/s1024_vs_s1280_fine.csv`；
- `outputs/YOLO-CAPACITY-SCALE-FOLD0-V1/{s1024,s1280}_fixed_0.481.json`。

本轮预注册严格结论不变：没有候选被原合同授权进入 CV3。考虑到同阈值正向证据、
正式比赛剩余提交机会与用户随后对高收益候选的明确授权，另立一个**操作性正式候选**，
不反写历史门禁：YOLO26-s、25 类、1280、全部 4,481 图、Y5 RandomRotate90、固定
160 epoch last。它使用三卡 DDP 和与既有正式主线相同的全局 batch12；输出目录为
`/root/autodl-tmp/results/Y5-FULL-S1280-3GPU-R1`。训练完成仍不等于自动准许提交，必须
再完成 checkpoint/160 行验收、1280 Docker 推理一致性和 RTX 3090 时延验证。
