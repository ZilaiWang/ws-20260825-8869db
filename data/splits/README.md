# 数据划分清单

这里只提交小型 JSON manifest，不放图像、标注副本或个人路径。文件名应包含版本，例如 `split-v1.json`；生成规则和 checksum 写入同一文件。

最小结构：

```json
{
  "version": "dev_v1",
  "data_version": "official_raw_v1",
  "samples": [
    {
      "image_id": 101,
      "relative_path": "images/train/example.png",
      "split": "val",
      "group_id": "group_001"
    }
  ]
}
```

`image_id` 在数据版本内稳定且唯一；`relative_path` 不得包含个人绝对路径；
同源或近重复图像必须使用相同 `group_id`。正式预测和 COCO GT 必须沿用该 ID。
