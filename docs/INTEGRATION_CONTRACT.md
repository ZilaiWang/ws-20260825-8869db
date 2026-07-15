# 最小协作与交接契约（V1）

本契约只统一跨成员交接内容，不统一各模型的训练框架和内部实现。目标是让数据、模型和大图工作可以并行开始。

## 1. 现在即可开始的工作

- B 可以独立生成 `dev_v1`、`cv3` 和稳定图像 ID；
- C、D 可以使用各自熟悉的原生框架训练 M1/M2/M3，不需要等待公共训练器；
- E 可以先用 Dummy 或模拟预测完成切片、坐标恢复和融合；
- A 负责公共预测校验、官方评测和最终集成。

公共的 `scripts/train.py`、`scripts/infer.py` 目前仍是入口骨架。C、D 不必等待它们，第一阶段只需按本文件交付预测结果。

## 2. 最低交付：标准 COCO detection JSON

模型成员最早可以只交一个标准 JSON 顶层列表：

```json
[
  {
    "image_id": 101,
    "category_id": 4,
    "bbox": [120.5, 80.0, 32.0, 20.0],
    "score": 0.87
  }
]
```

字段约定：

- `image_id`：必须来自 B 冻结的 split manifest，禁止用推理循环序号临时代替；
- `category_id`：内部暂用官方训练细类 ID 0-24，不得先归并为三大类；
- `bbox`：原图像素坐标 COCO `xywh`，不是归一化坐标；
- `score`：有限数，范围 0-1；
- 每个预测对象只含一张图中的一个框；无预测图像不需要写空记录。

交付前运行：

```bash
PYTHONPATH=src python scripts/validate_predictions.py \
  --pred outputs/实验ID/predictions.json \
  --gt outputs/dev_v1_gt.json

PYTHONPATH=src python scripts/evaluate.py \
  --gt outputs/dev_v1_gt.json \
  --pred outputs/实验ID/predictions.json \
  --output outputs/实验ID/metrics.json
```

预测校验通过不代表指标达标，只代表字段、类别、图像 ID 和坐标满足公共交接要求。

## 3. 推荐交付：轻量模型 adapter

当首轮模型已经能输出 JSON 后，再接入 Python adapter。adapter 只负责加载权重和推理，不要求统一训练。

```python
from collections.abc import Sequence

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.models.registry import register_model


@register_model("my_detector")
class MyDetector(BaseDetector):
    def load(self, checkpoint_path: str) -> None:
        ...

    def to(self, device: str) -> None:
        ...

    def eval(self) -> None:
        ...

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        # 适配器负责预处理，并把 resize/padding 后的框恢复到每个输入的像素坐标。
        ...
```

公共输入：

```text
InferenceSample(image_id, image, width, height, metadata)
```

公共输出：

```text
Prediction(image_id, boxes_xyxy, scores, labels)
```

要求：

- 一张输入恰好返回一个 `Prediction`；
- 输出顺序与输入顺序一致，`image_id` 原样返回；
- `boxes_xyxy` 是当前输入图像或 tile 上的绝对像素坐标；
- `boxes_xyxy`、`scores`、`labels` 数量一致；
- label 保留 0-24 细类；
- adapter 内部可以自由使用 NumPy、PIL、PyTorch 或模型框架原生预处理。

不要求 adapter 实现训练步骤，也不要求 C、D 改写现有训练代码。

## 4. B 的最小 split manifest

B 冻结的每个 split 使用小型 JSON manifest，至少提供：

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

- `image_id` 在该数据版本中稳定且唯一；
- `relative_path` 相对数据根目录，不含个人绝对路径；
- `split` 至少区分 train/val；
- `group_id` 用于说明同源或近重复图像分组；
- split 更新时必须更换版本号，不静默改变已有 ID 和成员归属。

正式预测、COCO GT 和实验台账必须使用同一份 manifest 的 `image_id`。

## 5. E 的大图交接

- tile 进入模型时，每个 tile 使用唯一 tile ID；
- 模型输出 tile 内绝对像素 `xyxy`；
- E 负责恢复为 parent image 坐标、裁剪边界和跨 tile 融合；
- 融合后输出新的 parent image `Prediction`；
- A 负责将最终 `Prediction` 校验并导出 COCO JSON；
- 模型 adapter 不感知 NMS、WBF 或大图坐标策略。

## 6. 实验记录与暂缓项

实验目录、必填字段和 M1/M2/M3 编号只在
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) 中维护，本文不重复。

当前不统一训练循环、优化器、增强实现、25 类独立阈值和部署后端；大图融合策略由 E 先完成可替换实现。上述事项不阻塞各模块开工，在 M1 首轮结果产生后再决定是否补充。
