# MAR20 官方划分列表

来源：MAR20 数据集 `ImageSets/Main/`，下载自
<https://gcheng-nwpu.github.io/>。

许可：CC BY-NC 4.0，仅限研究用途。

本目录只包含图像编号列表（`train.txt` 1,331 行、`test.txt` 2,511
行），不含图像、标注或任何派生像素内容。

## 当前允许用途

这两个列表只可作为 MAR20 发布侧元数据和历史复现输入。它们没有提供
逐图机场标识；列表顺序和编号递增段也不能证明同段等于同一机场，或
train/test 两侧机场严格互斥。因此：

- 禁止把列表侧别或编号段当作当前 `group_id`；
- 禁止据此声称正式验证集实现真实机场互斥；
- `scripts/build_split.py` 仅保留为 `dev_v1` 历史复现入口。

当前 MAR20 来源隔离采用经人工校准、检索和几何核验得到的 K=60
机场代理视觉域。权威逐图映射是
`data/groups/mar20_airport_proxy_k60_for_b.csv`；正式三折入口是
`data/splits/cv3_airport_proxy_k60_v2.json`。这些组是机场代理视觉域，
不是机场真值。
