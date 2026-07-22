# 来源分组映射

本目录只保存可提交仓库的小型图像级分组映射，不保存图像或标注副本。

- `mar20_airport_proxy_k60_for_b.csv`：给 B 直接读取的两列文件，格式严格为 `image_name,group_id`。`image_name` 是竞赛训练集中的实际文件名，`group_id` 是 K=60 机场代理视觉域编号。

这些 `group_id` 是用于来源隔离划分的视觉代理组，不是真实机场身份标注。
