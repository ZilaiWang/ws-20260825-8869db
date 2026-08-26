# 数据划分清单

这里只提交小型 JSON manifest，不放图像、标注副本或个人路径。文件名应包含版本，例如 `dev_v1.json`。

- `dev_v1.json`：B 最初生成的 relaxed 开发划分，保留用于历史复现；
- `dev_v2_airport_proxy_k60.json`：舰船和车辆完全复用 `dev_v1`，MAR20 飞机改用 K=60 机场代理视觉域作为不可拆分 `group_id`。这是当前来源隔离开发划分。
- `cv3_airport_proxy_k60_v2_groups.json`：经硬约束审查后冻结的 255 个来源组到三折的映射；
- `cv3_airport_proxy_k60_v2.json`：当前正式三折划分。K60 的 3,073 张图、60 个代理组逐项保持不变，25 类每折均有验证来源。

错误的 `cv3_airport_proxy_k60_v1.json` 已从工作树删除；它曾在 K60 之上再次用旧 dHash 合并来源组并导致 TU-160 覆盖失败，只可从 Git 历史和审计记录了解，禁止恢复到实验入口。

唯一格式说明见 [`docs/INTEGRATION_CONTRACT.md`](../../docs/INTEGRATION_CONTRACT.md)，本目录不再维护第二份定义。
