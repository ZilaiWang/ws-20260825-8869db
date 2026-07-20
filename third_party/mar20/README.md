\# MAR20 官方划分列表



来源：MAR20 数据集 `ImageSets/Main/`，下载自 https://gcheng-nwpu.github.io/

许可：CC BY-NC 4.0，仅限研究用途



本目录只包含图像编号列表（train.txt 1331 行 / test.txt 2511 行），

不含图像、标注或任何派生像素内容。



用途：`scripts/build\_split.py` 依据这两个列表推导飞机图像的防泄漏分组。

MAR20 论文说明其 train/test 按机场划分且两侧机场互斥；test.txt 内部

呈递增段排列，每段机型组合与论文表 2 的机场清单吻合，据此作为场景分组依据。

