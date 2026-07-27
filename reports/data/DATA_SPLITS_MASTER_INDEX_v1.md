# 数据分组与划分总索引 v1

更新日期：2026-07-23  
状态：`current`

## 1. 结论

当前两份可用划分均已完成、冻结并通过来源组隔离检查：

| 划分 | 用途 | 归属字段 | 规模 | SHA256 |
|---|---|---|---|---|
| `dev_v2_airport_proxy_k60` | 单次快速开发、接口联调和便宜探索 | `split=train/val` | 3,548 / 933 图 | `99d9e8885fd1adddc507cce77c4f5b8ecdda9117dd24b387ca5c2d7a94992322` |
| `cv3_airport_proxy_k60_v2` | 正式模型选择、三折 OOF、P 系列正式复验 | `fold=0/1/2` | 1,507 / 1,613 / 1,361 图 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |

两份划分覆盖相同的 4,481 张图和 255 个不可拆来源组，并逐图使用相同
`group_id`。其中 MAR20 的 3,073 张图完整使用同一套 K=60
机场代理视觉域。它们不是两份竞争方案：`dev_v2` 用于一次性开发，
`cv3_v2` 用于正式三折结论。

机场代理视觉域不等于真实机场标签；正式材料不得写成已经获得真实
airport-disjoint ground truth。

## 2. 共用上游分组

### 2.1 权威输入

| 内容 | 路径 | 作用 |
|---|---|---|
| MAR20 K60 两列映射 | [`data/groups/mar20_airport_proxy_k60_for_b.csv`](../../data/groups/mar20_airport_proxy_k60_for_b.csv) | 3,073 张竞赛 MAR20 图到 60 个不可拆代理组 |
| K60 完整形成过程 | [`MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md`](MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md) | 问题定位、描述子、检索、几何、人工复核及 K60 收尾的全链索引 |
| K60 最终验收 | [`MAR20_AIRPORT_PROXY_K60_ACCEPTANCE_v1.md`](MAR20_AIRPORT_PROXY_K60_ACCEPTANCE_v1.md) | 60 组统计、使用边界和给 B 的合同 |
| K60 随机图像复核 | [`RANDOM_VISUAL_AUDIT_RESULT.md`](../../outputs/MAR20-AIRPORT-PROXY-K60-v1/visual-audit/RANDOM_VISUAL_AUDIT_RESULT.md) | 最终映射的抽样视觉边界 |

K60 两列映射 SHA256：
`3cd2fdb1db0b95ec3069db569fe56019942b9760a3a261f44800164135256f00`。

### 2.2 非 MAR20 分组

- 舰船按文件名中的场景 ID 分组；
- 车辆按文件名中的经纬度站点分组；
- 两份当前划分对这些来源组使用同一推导规则；
- MAR20 官方 `train.txt`/`test.txt` 只作为外部元数据和历史复现输入，
  不能替代 K60，也不能证明机场互斥。

## 3. 单次开发划分 `dev_v2`

### 3.1 正式入口

| 类型 | 路径 |
|---|---|
| manifest | [`data/splits/dev_v2_airport_proxy_k60.json`](../../data/splits/dev_v2_airport_proxy_k60.json) |
| 验收报告 | [`MAR20_DEV_V2_AIRPORT_PROXY_K60_SPLIT_ACCEPTANCE_v1.md`](MAR20_DEV_V2_AIRPORT_PROXY_K60_SPLIT_ACCEPTANCE_v1.md) |
| 求解与门禁 | [`src/rsdet/data/airport_proxy_split.py`](../../src/rsdet/data/airport_proxy_split.py) |
| 生成入口 | [`scripts/build_dev_airport_proxy_split.py`](../../scripts/build_dev_airport_proxy_split.py) |
| 测试 | [`tests/test_airport_proxy_split.py`](../../tests/test_airport_proxy_split.py) |

### 3.2 核心统计

| 域 | train | val |
|---|---:|---:|
| 全部 | 3,548 | 933 |
| 舰船 | 1,036 | 305 |
| 飞机 | 2,458 | 615 |
| 车辆 | 54 | 13 |

MAR20 为 43 个训练代理组、17 个验证代理组；跨 train/val 来源组为 0；
25 类训练侧和验证侧均有样本。该划分保留为便宜开发入口，不用于替代
正式三折汇总。

`dev_v2` manifest 已冻结并可直接使用。它的原始精确生成元数据引用
`outputs/MAR20-AIRPORT-PROXY-K60-v1/final/` 中的详细 K60 CSV
（SHA `afde2a3d...`），该目录受 Git 忽略但已登记在 K60 证据链中。
新机器可用已跟踪的两列 K60 CSV 科学等价地重建相同逐图分组；由于输入
文件名和输入 SHA 元数据不同，重建 JSON 不会与上述冻结 manifest
逐字节同哈希。需要精确字节复现时必须先按服务器/回传登记恢复原详细
CSV，不应静默改写已冻结的 `dev_v2`。

## 4. 正式三折 `cv3_v2`

### 4.1 正式入口

| 类型 | 路径 | SHA256 |
|---|---|---|
| 三折 manifest | [`data/splits/cv3_airport_proxy_k60_v2.json`](../../data/splits/cv3_airport_proxy_k60_v2.json) | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| 冻结组分配 | [`data/splits/cv3_airport_proxy_k60_v2_groups.json`](../../data/splits/cv3_airport_proxy_k60_v2_groups.json) | `5b87536cd49eb1ebcb79a0d1bc539a623d07c19557e07a082954cdb793ce2033` |
| 机器审计 | [`cv3_airport_proxy_k60_v2_audit.json`](cv3_airport_proxy_k60_v2_audit.json) | `b0bd834c8760617d0b8258295bc14d4748df34760c060cc52bcf6c35e5039e4d` |
| 人类验收 | [`CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md`](CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md) | — |
| 生成入口 | [`scripts/build_cv3.py`](../../scripts/build_cv3.py) | — |
| 核心门禁 | [`src/rsdet/data/cv3_split.py`](../../src/rsdet/data/cv3_split.py) | — |
| 测试 | [`tests/test_cv3_split.py`](../../tests/test_cv3_split.py) | — |

### 4.2 核心统计

一次运行以 `fold == held_out_fold` 为验证集，其余两折为训练集。

| 指标 | fold 0 | fold 1 | fold 2 |
|---|---:|---:|---:|
| 验证图像 | 1,507 | 1,613 | 1,361 |
| 验证框 | 7,350 | 7,179 | 6,404 |
| 验证来源组 | 82 | 95 | 78 |

三折覆盖 4,481 张图、20,933 个框、255 个来源组；同组不跨折，每张图
恰好验证一次，25 类在每个验证折均有来源证据。完整逐类表见验收报告。

TU-160 的 361 个框分布为 352 / 8 / 1，严格分组后存在一个仅剩 9 个
训练框的压力折。这是来源高度集中造成的结构事实，不得拆组掩盖；正式
结果必须保留逐折值、均值、标准差和 TU-160 单类说明。

### 4.3 复现

```bash
PYTHONPATH=src python scripts/build_cv3.py \
  --data-root ../data \
  --airport-groups data/groups/mar20_airport_proxy_k60_for_b.csv \
  --assignment data/splits/cv3_airport_proxy_k60_v2_groups.json \
  --near-duplicates reports/data/near_duplicates_mar20.json \
  --output data/splits/cv3_airport_proxy_k60_v2.json \
  --audit-output reports/data/cv3_airport_proxy_k60_v2_audit.json
```

`near_duplicates_mar20.json` 在这里仅审计，不合并或改写 K60。重建后的
manifest 与审计文件必须分别得到上表 SHA。

## 5. 两份划分的关系与禁止混用

- 4,481 条记录的 `image_id`、`relative_path`、`group_id` 和
  `group_rule` 逐项完全一致；
- 两者只在同一套 255 个来源原子上分别记录单次 `split` 和三折 `fold`；
- `dev_v2` 的 933 张验证图在 CV3 中分别落入 fold 0/1/2 的
  253 / 355 / 325 张（框为 807 / 1,264 / 1,414，来源组为
  44 / 58 / 40），因此它不等于任何一个 CV3 fold；
- 旧 `dev_v1` checkpoint 训练时见过的新 CV3 验证图不能用于正式 OOF；
- CV3 的三个训练集分别包含 680 / 578 / 608 张 `dev_v2` 验证图，因此
  不能把完整 `dev_v2 val` 再当作 CV3 模型的独立测试集；
- 任何正式三折模型必须从相同预训练初值独立训练三次；
- 每个预测、GT、checkpoint 和实验登记都要记录 manifest 版本及完整
  SHA，不能只写“CV3”。

机器读取字段和 held-out 语义见
[`docs/INTEGRATION_CONTRACT.md`](../../docs/INTEGRATION_CONTRACT.md) 第 4 节。

## 6. 历史与废弃边界

| 项目 | 状态 | 处理 |
|---|---|---|
| `dev_v1.json` | `historical` | 保留，用于复现 C 已交付的旧结果和构造 dev_v2 |
| `scripts/build_split.py` | `historical` | 保留，只可复现 dev_v1，已加警告 |
| `scripts/analyze_groups.py` | `historical` | 保留初期问题诊断，不可生成当前划分 |
| `cv3_airport_proxy_k60_v1.json` | `invalid/deleted` | 曾错误地在 K60 上再做 dHash 合并并导致覆盖失败，已从工作树删除 |
| MAR20 `train.txt`/`test.txt` | `metadata_only` | 保留许可与历史复现，不作机场真值或当前分组 |

早期报告、盲评合同和失败状态属于完整证据链，不能因为结果被替代而删除；
它们不再作为新实验入口。

## 7. CV3 完成后放行的实验

当前执行状态和优先顺序统一见
[`CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md`](../experiments/CV3_FORMAL_EXPERIMENT_EXECUTION_MASTER_v1.md)
与 [`DEFERRED_WORK_REGISTER.md`](../experiments/DEFERRED_WORK_REGISTER.md)。最短
逻辑不是一条完全串行链，而是：

```text
CV3 v2 适配与 SHA 冻结（F00）
├─ P0-2 重挂 fold → P03 / P04 正式复验
└─ 正式图像/标签/GT 字节锁（D00）
      ├─ M1 三折重训与低阈值 OOF
      └─ M3 三折重训与低阈值 OOF

独立模型环境/官方权重锁（A00）
├─ M1 / M3 的第二前置门禁
└─ E 的 10K 工程基线
          ↓
OOF 完整性审计、官方评估与 cross-fit 阈值
          ↓
M1 错误分解及 M1/M3 配对
          ↓
按背景 FP / 定位错误证据有条件放行 P05 / P06
          ↓
组合消融、最终阈值、10K 时延与模型冻结
```

F00→D00 是数据链，A00 可与其并行；P03/P04、M1、M3 和 10K 工程可按依赖
与 GPU 资源排队。P05/P06 必须由真实 OOF 错误类型触发，不能因为 CV3 已
完成就自动宣告需要全部运行。
