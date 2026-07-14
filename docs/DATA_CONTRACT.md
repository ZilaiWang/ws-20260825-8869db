# 数据契约

## 规则

1. **原始数据只读**。处理后的数据写入 outputs/。
2. **原始数据不进 Git**。
3. 统一通过 `configs/local.yaml` 指定数据根目录。
4. 数据版本使用 manifest 和 checksum。
5. 内部 bbox 统一 `xyxy`（像素坐标）。COCO 导出时转为 `xywh`。
6. `category_id` 不得随意重排。
7. `image_id` 必须稳定，不随数据划分变化。
8. 切片必须保留 `parent_image_id` 和 offset。
9. 训练/验证划分必须保存 manifest JSON。
10. 可共享的数据报告不得包含原图。
11. 数据集结构审计完成前，标注为 TBD。

## 统一数据结构

| 结构 | 定义位置 |
|------|----------|
| `ImageRecord` | `src/rsdet/contracts.py` |
| `AnnotationRecord` | `src/rsdet/contracts.py` |
| `Prediction` | `src/rsdet/contracts.py` |
| `TileRecord` | `src/rsdet/contracts.py` |

## 数据集类别映射

TBD — 待数据审计完成后填充 `configs/project.yaml` 中 `dataset_category_mapping`。
