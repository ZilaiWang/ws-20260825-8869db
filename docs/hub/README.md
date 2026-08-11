# 项目资料导航

更新日期：2026-08-10

> 开始新创新实验前，先读
> [`PRE_INNOVATION_CLOSURE_20260810.md`](../../reports/experiments/PRE_INNOVATION_CLOSURE_20260810.md)；
> 它统一了官方 V1.6 口径、可信基线、作废结论和后续准入。

本目录是项目文档的非破坏式导航层。原始报告、数据清单、任务单和服务器产物仍保留在原路径，避免破坏相对链接、复现命令和 SHA 合同。

## 阅读顺序

1. [`00_project_status`](00_project_status/README.md)：当前结论、总体架构和模型角色；
2. [`10_team_tasks`](10_team_tasks/README.md)：当前 A—E 分工及 D 的立即任务；
3. [`20_data_and_splits`](20_data_and_splits/README.md)：数据、机场代理分组、`dev_v2` 与 CV3；
4. [`30_p_series`](30_p_series/README.md)：P0-1 至 P07 的结果、证据等级和保留/停止决定；
5. [`40_member_deliveries`](40_member_deliveries/README.md)：成员交付及验收入口；
6. [`50_server_tasks`](50_server_tasks/README.md)：服务器资产和回传登记；
7. [`90_deferred`](90_deferred/README.md)：等待条件、解锁顺序和停止项。

## 状态词

| 状态 | 含义 |
|---|---|
| `current` | 当前执行依据 |
| `exploratory` | 可指导方向，但不能作为正式成绩 |
| `historical` | 历史讨论或旧版方案 |
| `superseded` | 已被新版本替代 |
| `waiting` | 已有明确输入合同，等待外部产物 |
| `stopped` | 已达到停止条件，不继续投入 |
| `missing` | 曾完成或回传，但当前仓库缺少正式归档 |

## 当前唯一事实源

- 类别、IoU 和项目配置：[`configs/project.yaml`](../../configs/project.yaml)
- 接口合同：[`docs/INTEGRATION_CONTRACT.md`](../INTEGRATION_CONTRACT.md)
- 实验协议：[`docs/EXPERIMENT_PROTOCOL.md`](../EXPERIMENT_PROTOCOL.md)
- 当前单次开发划分：[`data/splits/dev_v2_airport_proxy_k60.json`](../../data/splits/dev_v2_airport_proxy_k60.json)
- 当前正式三折：[`data/splits/cv3_airport_proxy_k60_v2.json`](../../data/splits/cv3_airport_proxy_k60_v2.json)
- 两份划分总索引：[`reports/data/DATA_SPLITS_MASTER_INDEX_v1.md`](../../reports/data/DATA_SPLITS_MASTER_INDEX_v1.md)
- 机场代理分组主链：[`reports/data/MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md`](../../reports/data/MAR20_AIRPORT_PROXY_GROUPING_MASTER_CHAIN_INDEX_v1.md)
- 延期实验台账：[`reports/experiments/DEFERRED_WORK_REGISTER.md`](../../reports/experiments/DEFERRED_WORK_REGISTER.md)
- 大文件交付登记：[`reports/experiments/ARTIFACT_RELEASE_REGISTER.csv`](../../reports/experiments/ARTIFACT_RELEASE_REGISTER.csv)
- 历史服务器路径快照：[`reports/experiments/SERVER_ARTIFACT_REGISTER.csv`](../../reports/experiments/SERVER_ARTIFACT_REGISTER.csv)
