# 数据划分清单

这里只提交小型 JSON manifest，不放图像、标注副本或个人路径。文件名应包含版本，例如 `dev_v1.json`。

- `dev_v1.json`：B 最初生成的 relaxed 开发划分，保留用于历史复现；
- `dev_v2_airport_proxy_k60.json`：舰船和车辆完全复用 `dev_v1`，MAR20 飞机改用 K=60 机场代理视觉域作为不可拆分 `group_id`。这是当前来源隔离开发划分。

唯一格式说明见 [`docs/INTEGRATION_CONTRACT.md`](../../docs/INTEGRATION_CONTRACT.md)，本目录不再维护第二份定义。
