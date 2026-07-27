# 数据、分组与划分

更新日期：2026-07-23  
状态：`current`

## 当前可用版本

| 版本 | 用途 | 边界 |
|---|---|---|
| `dev_v1` | 历史复现；C 当前 YOLO 结果所在划分 | MAR20 来源隔离不够可靠 |
| `dev_v2_airport_proxy_k60` | 较严格的单次 train/val 探索划分 | 机场代理视觉组，不是真实机场标签；不能替代三折 |
| `cv3_airport_proxy_k60_v2` | 正式模型选择、OOF 和 P 系列复跑 | 当前正式三折；TU-160 含 9-shot 压力折 |

`dev_v2` 统计：

- 全部：3,548 train / 933 val；
- 舰船：1,036 / 305；
- 飞机：2,458 / 615；
- 车辆：54 / 13；
- MAR20：43 个训练代理组 / 17 个验证代理组；
- 跨 train/val 组数：0。

正式 CV3 统计：

- 4,481 张图、20,933 个框、255 个不可拆来源组；
- 三折验证图：1,507 / 1,613 / 1,361；
- 三折验证框：7,350 / 7,179 / 6,404；
- 每张图恰好验证一次，同组不跨折，25 类每折均有验证来源。

## 当前入口

- 划分说明：[`data/splits/README.md`](../../../data/splits/README.md)
- 两份划分总索引：[`DATA_SPLITS_MASTER_INDEX_v1.md`](../../../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)
- `dev_v2`：[`data/splits/dev_v2_airport_proxy_k60.json`](../../../data/splits/dev_v2_airport_proxy_k60.json)
- 正式 `cv3_v2`：[`data/splits/cv3_airport_proxy_k60_v2.json`](../../../data/splits/cv3_airport_proxy_k60_v2.json)
- CV3 冻结组分配：[`data/splits/cv3_airport_proxy_k60_v2_groups.json`](../../../data/splits/cv3_airport_proxy_k60_v2_groups.json)
- 给 B 的映射：[`data/groups/mar20_airport_proxy_k60_for_b.csv`](../../../data/groups/mar20_airport_proxy_k60_for_b.csv)
- `dev_v2` 验收：[`MAR20_DEV_V2_AIRPORT_PROXY_K60_SPLIT_ACCEPTANCE_v1.md`](../../../reports/data/MAR20_DEV_V2_AIRPORT_PROXY_K60_SPLIT_ACCEPTANCE_v1.md)
- `cv3_v2` 验收：[`CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md`](../../../reports/data/CV3_AIRPORT_PROXY_K60_V2_ACCEPTANCE.md)
- `cv3_v2` 机器审计：[`cv3_airport_proxy_k60_v2_audit.json`](../../../reports/data/cv3_airport_proxy_k60_v2_audit.json)
- MAR20 全链索引：[`MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md`](../../../reports/data/MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md)

## 使用规则

1. 便宜单次探索可使用 `dev_v2`；正式模型选择、OOF、P03/P04 复验统一使用 `cv3_v2`。
2. C 的现有 checkpoint 在 `dev_v1` 训练，不能直接在 `dev_v2` 上与新模型作公平结论：两个划分间有 1,018 张 MAR20 图改变归属，`dev_v2` val 可能包含旧训练图。
3. 若正式比较 C 与 D，必须各自从原始预训练权重按相同 CV3 独立训练三折；旧 checkpoint 不能生成正式 OOF。
4. CV3 必须保持 K=60 `group_id` 不跨 fold；不能为了平衡类别手工拆组，也不能恢复已删除的 v1。
5. TU-160 等来源高度集中的类别需要报告实际验证证据量，不能只给宏平均。

## CV3 v2 已通过的冻结门禁

- 覆盖 4,481 张图且无重复/遗漏；
- 25 类每折验证均有来源证据；
- 同一 `group_id` 不跨 fold；
- 舰船场景组、车辆地理组、飞机 K=60 代理组均遵守相同不可拆分原则；
- 输出 fold 统计、每类框数、每类组数、最差 fold 和跨 fold 反向泄漏审计；
- manifest SHA256 已冻结为
  `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331`；
  后续所有 OOF 和 P 系列正式复跑必须引用该完整哈希。
