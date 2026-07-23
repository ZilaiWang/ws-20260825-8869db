# `dev_v2_airport_proxy_k60` 划分生成与验收

## 1. 当前结论

已在 B 的 [`dev_v1.json`](../../data/splits/dev_v1.json) 上生成新的来源隔离开发划分：

- [`dev_v2_airport_proxy_k60.json`](../../data/splits/dev_v2_airport_proxy_k60.json)；
- 给 B 直接读取的两列表：[`mar20_airport_proxy_k60_for_b.csv`](../../data/groups/mar20_airport_proxy_k60_for_b.csv)；
- 机器可读审计：[`summary.json`](../../outputs/MAR20-DEV-V2-AIRPORT-PROXY-K60-v1/summary.json)。

没有覆盖旧 `dev_v1`。按照项目接口合同，成员归属发生变化时必须更换版本号，因此新版本命名为 `dev_v2_airport_proxy_k60`。

## 2. 复用了 B 的哪些内容

以下内容完全复用：

1. 4,481 张图的稳定 `image_id`；
2. `relative_path` 与 JSON manifest 字段格式；
3. `data_version=official_raw_v1`；
4. `val_fraction=0.20` 与原 seed 记录；
5. 舰船和车辆共 1,408 张图的 `split`、`group_id`、`group_rule` 全部逐条不变；
6. 使用标注细类数量和分组原子共同求解划分的基本思想。

MAR20 的 3,073 张图只替换两项：

- `group_id` 改为最终 K=60 机场代理视觉域；
- `group_rule` 改为 `mar20_airport_proxy_k60`，并以完整组为原子重新分配 train/val。

## 3. 为什么没有机械调用原 `assign_splits`

把 K=60 组直接交给旧贪心器时，得到19个验证组、707张验证图，但 TU-160 的3个来源代理组会全部进入验证集：

```text
TU-160 train boxes = 0
TU-160 val boxes   = 361
```

这不满足训练基本条件。因此新求解器保留 B 的20%目标和细类感知思想，补充了两项必要约束：

1. 每个飞机细类至少保留1个训练来源组；
2. 若一个细类覆盖不少于3个来源组，则验证集至少包含2个来源组。

目标函数优先使各飞机细类的验证框数量接近20%，再以很小权重尽量保留旧 `dev_v1` 的图像归属。实现位于：

- [`airport_proxy_split.py`](../../src/rsdet/data/airport_proxy_split.py)；
- [`build_dev_airport_proxy_split.py`](../../scripts/build_dev_airport_proxy_split.py)；
- [`test_airport_proxy_split.py`](../../tests/test_airport_proxy_split.py)。

## 4. 最终统计

| 项目 | train | val | 合计 |
|---|---:|---:|---:|
| 全部图像 | 3,548 | 933 | 4,481 |
| 舰船 | 1,036 | 305 | 1,341 |
| 飞机 | 2,458 | 615 | 3,073 |
| 车辆 | 54 | 13 | 67 |
| MAR20 代理组 | 43 | 17 | 60 |

验证集占全部图像 `20.82%`；MAR20 验证图严格为 `round(3073×0.20)=615`。

完整性门禁：

| 门禁 | 结果 |
|---|---:|
| 样本覆盖 | 4,481 / 4,481 |
| MAR20 映射覆盖 | 3,073 / 3,073 |
| MAR20 `group_id` 数 | 60 |
| 跨 train/val 分组 | 0 |
| 非 MAR20 归属变化 | 0 |
| 训练侧缺失细类 | 0 |
| 验证侧缺失细类 | 0 |
| 相对旧 `dev_v1` 改变的 MAR20 图 | 1,018 |

飞机细类中最需要说明的是 TU-160。其361个框高度集中于3个机场代理组：一个组含352框，另两个组合计9框。不存在接近20%的机场隔离分法。当前选择为：

```text
train: 1 group / 352 boxes
val:   2 groups / 9 boxes
```

这样优先保证模型有足够训练样本，同时在验证侧保留两个独立代理来源。后续解读 TU-160 单类验证指标时必须注明其验证证据量低，不能把该类的单次波动当成稳定结论。

## 5. 给 B 的文件合同

两列 CSV 格式为：

```csv
image_name,group_id
MAR20_1.jpg,mar20-airport-proxy-001
MAR20_10.jpg,mar20-airport-proxy-001
```

文件共有3,073条数据行、3,073个唯一图像名和60个唯一 `group_id`。B 可以用图像原文件名直接 join；若现有代码使用 stem，则对 `image_name` 调用 `Path(image_name).stem` 即可。

正式 train/val 可直接使用 `dev_v2_airport_proxy_k60.json`，不需要 B 再求解一次。若 B 后续生成 CV3，则仍须把同一 `group_id` 当作不可拆分原子。

## 6. 哈希与复现

| 文件 | SHA256 |
|---|---|
| 旧 `dev_v1.json` | `bcb6fdb909df3421db800ea248022a39dd7e596c815192b97c388f836cd32aed` |
| K60 原始映射 | `afde2a3d9b9941ad5fc603d979adcdf68a0c9819541eeb96a06993654529cf87` |
| 新 `dev_v2_airport_proxy_k60.json` | `99d9e8885fd1adddc507cce77c4f5b8ecdda9117dd24b387ca5c2d7a94992322` |
| 给 B 的两列 CSV | `3cd2fdb1db0b95ec3069db569fe56019942b9760a3a261f44800164135256f00` |

复现命令：

```bash
PYTHONPATH=src python scripts/build_dev_airport_proxy_split.py \
  --base-manifest data/splits/dev_v1.json \
  --group-map outputs/MAR20-AIRPORT-PROXY-K60-v1/final/mar20_airport_proxy_assignments_target.csv \
  --label-dir /path/to/official/data/labels/train \
  --output data/splits/dev_v2_airport_proxy_k60.json \
  --b-mapping-output data/groups/mar20_airport_proxy_k60_for_b.csv \
  --summary-output outputs/MAR20-DEV-V2-AIRPORT-PROXY-K60-v1/summary.json
```

同一输入独立执行两次时，manifest 和两列 CSV 均逐字节一致。

## 7. 使用边界

本划分用于比旧 `dev_v1` 更严格地评估 MAR20 场景/来源泛化，但 K=60 仍是机场背景视觉代理组，不是真实机场标签。正式文字应写作：

> MAR20 airport-proxy grouped development split

不得写成已经获得真实 airport-disjoint ground truth。
