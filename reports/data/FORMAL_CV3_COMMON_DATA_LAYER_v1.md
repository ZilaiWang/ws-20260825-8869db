# 正式 CV3 通用消费层与 crop manifest v2 验收报告

## 1. 结论

正式三折的数据入口已经可用：

- `cv3_airport_proxy_k60_v2.json` 是 fold 与来源组的唯一权威；
- 任意 `held_out_fold` 都严格解释为“该 fold 验证、其余两 fold 训练”；
- P0-2 只复用对象、裁剪框和渲染元数据，不再提供正式 fold/group；
- 62,799 条记录已按 `source_relative_path` 无损重挂到正式 CV3；
- 全过程不打开源图、不重裁 crop，不改变 `crop_id`；
- 新消费层拒绝覆盖已有正式产物，复跑必须使用新目录。

这解决了“正式 CV3 已更新，但 P03/P04 仍可能误用探索性 P0-2 分组”的公共
数据层问题。P03/P04 训练实现及总索引未在本工作中修改。

## 2. 文件链

| 角色 | 文件 |
|---|---|
| 正式 CV3 输入 | `data/splits/cv3_airport_proxy_k60_v2.json` |
| P0-2 几何输入 | `outputs/P0-2-exploratory-crop-manifest/crop_manifest.csv`（非 Git 大文件） |
| CV3 冻结配置 | `configs/analysis/formal_cv3.yaml` |
| crop v2 冻结配置 | `configs/analysis/formal_crop_manifest.yaml` |
| 通用 CV3 API | `src/rsdet/data/formal_cv3.py` |
| 元数据重挂 API | `src/rsdet/analysis/formal_crop.py` |
| CV3 CLI | `scripts/build_formal_cv3_views.py` |
| crop v2 CLI | `scripts/build_formal_crop_manifest.py` |
| 单元测试 | `tests/test_formal_cv3.py`、`tests/test_formal_crop.py` |
| 服务器任务单 | `docs/server/FORMAL_CV3_CROP_TASK_01_CPU.md` |
| 代码指纹 | `docs/server/FORMAL_CV3_CROP_TASK_01_CODE_SHA256.txt` |

本地验收输出位于：

- `outputs/CV3-FORMAL-CONSUMER-LOCALCHECK/`
- `outputs/P0-2-FORMAL-CROP-LOCALCHECK/`

输出目录不是 Git 接口。服务器验收后，P03/P04 及 D00 数据字节锁只允许复用
以下唯一正式路径：

```text
/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
SHA256 a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
```

正式服务器合同统一冻结 `/workspace`；若宿主机目录不同，应先建立挂载或
符号链接，不得在不同子任务中替换出不同根前缀。框架无关图像视图的唯一路径
为同一验收副本的
`/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/cv3/`；其 6 个完整 SHA
见第 4 节。`run-b` 只做确定性证明，禁止配置给训练任务。

## 3. 输入合同

| 输入 | 冻结值 |
|---|---|
| 正式 CV3 SHA256 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| CV3 版本 | `cv3_airport_proxy_k60_v2` |
| 图像 / 来源组 / fold | 4,481 / 255 / 3 |
| 各 fold 图像数 | 1,507 / 1,613 / 1,361 |
| 各 fold 来源组数 | 82 / 95 / 78 |
| P0-2 SHA256 | `f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e` |
| P0-2 行 / annotation / 图像 | 62,799 / 20,933 / 4,481 |
| crop policy | `tight`、`context_1p25`、`jitter_light` |

任何 SHA、版本、样本数、来源组数、fold 计数、路径覆盖或组隔离不一致均在
写正式产物前失败。

## 4. CV3 消费语义

调用方只需指定 `held_out_fold ∈ {0,1,2}`：

```text
val   := sample.fold == held_out_fold
train := sample.fold != held_out_fold
```

每次构造都会重新检查图像 ID、相对路径和 `group_id` 三种 train/val 交集均为
空。输出 CSV 不依赖 PyTorch、MMDetection 或 Ultralytics，可作为各训练框架的
共同索引。

| held-out fold | train 图像 | val 图像 | train 组 | val 组 |
|---:|---:|---:|---:|---:|
| 0 | 2,974 | 1,507 | 173 | 82 |
| 1 | 2,868 | 1,613 | 160 | 95 |
| 2 | 3,120 | 1,361 | 177 | 78 |

六个视图的冻结 SHA：

| 文件 | SHA256 |
|---|---|
| `formal_cv3_fold0_train.csv` | `93b0cf3782d4c8da3004c7b7a98093b83a57edbe03527d0d950861c413642db5` |
| `formal_cv3_fold0_val.csv` | `f03683689bc17c3bdbecba874f0fa686663527f24e47e347b9f8a2b71107494d` |
| `formal_cv3_fold1_train.csv` | `d3c74835ebbfc4e4b79c8303a83c437560a20817a7552150afe410068a906429` |
| `formal_cv3_fold1_val.csv` | `257598ed0e3ffae4b9a539f395f7a3d9845531ad6f79b6ddb0453ae01d301070` |
| `formal_cv3_fold2_train.csv` | `236aff8d96380eb6e0e7f4cec4cee6b9be328a774a804c8caf29210cfc598286` |
| `formal_cv3_fold2_val.csv` | `1c8c37a2dad0f16ee0f568156559c18fbb374359fe91d527882d08656f71ac78` |

## 5. crop manifest v2 的数据语义

连接键固定为 P0-2 的精确 `source_relative_path`。该键在 P0-2 与 CV3 中均
唯一覆盖全部 4,481 张图；缺图、多图或路径归一化后不一致均失败。

输出字段分三类：

1. 正式活动字段：`fold`、`group_id`、`leakage_group_id`、`group_rule`，全部
   只来自正式 CV3；
2. 历史字段：P0-2 的旧分割/分组字段全部改名为 `historical_p02_*`，逐值保留，
   不再具有训练语义；
3. 几何与身份字段：`crop_id`、`annotation_uid`、源图身份、GT/proposal、
   crop 坐标、padding、render/resize 等原样保留。

没有保留易误用的活动态 `main_split`。正式 train/val 始终由
`fold == held_out_fold` 动态得到。

## 6. 本地实跑结果

| 项目 | 结果 |
|---|---|
| 正式 crop 行数 | 62,799 |
| annotation 数 | 20,933 |
| 源图数 | 4,481 |
| 每个 annotation 的 policy 数 | 恰好 3 |
| 每个 policy 的 fold 对象数 | 7,350 / 7,179 / 6,404 |
| 每个 policy/fold 的细类覆盖 | 25/25 |
| annotation/source/group train-val overlap | 0 / 0 / 0 |
| `crop_id` | 62,799 个全部唯一且顺序不变 |
| 几何重算 | false |
| 读取像素 | 0 |
| 正式 CSV 大小 | 71,995,981 bytes |
| 正式 CSV SHA256 | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |

历史 P0-2 分配字段不仅在写入时改名，写完重读后还会逐行核对其值。所有非
分配字段另有整表规范化指纹：
`8c5f1bcf9be7a7ab2b1dd14ccdeb87fa41d0db0f386b244015faa9869f4529da`。

## 7. 使用方法

生成框架无关三折视图：

```bash
PYTHONPATH=src python scripts/build_formal_cv3_views.py \
  --output-dir /new/path/formal-cv3
```

重挂 crop 元数据：

```bash
PYTHONPATH=src python scripts/build_formal_crop_manifest.py \
  --exploratory-manifest /verified/path/crop_manifest.csv \
  --output-dir /new/path/formal-crop-v2
```

Python 调用可使用 `load_formal_cv3_manifest(...).view(held_out_fold)`。训练代码
读取 crop v2 时，只过滤活动字段 `fold`；不得读取 `historical_p02_fold` 做
任何正式训练或模型选择。

## 8. 复跑、覆盖与跨机器说明

- 正式输出目录不可变；底层代码检测到既有 CSV 会抛出 `FileExistsError`。
- 公共 CPU 任务是幂等的：产物不存在时生成；完整存在时先核对精确 SHA 和审计，
  然后标记 `verified_existing_skip`，不调用写入程序。
- 只有部分文件、SHA 不一致或审计不通过时直接失败，绝不“修补式覆盖”。
- 确定性复验使用独立 `run-b`，然后逐字节比较正式 CSV。
- `formal_crop_manifest.csv` 与 6 个 CV3 视图的 SHA 是验收依据。
- 审计 JSON 记录绝对输入/输出路径，因此不同机器或不同输出目录的审计 JSON
  本身不要求同 SHA；其中的计数、状态和正式 CSV SHA 必须一致。
- 服务器上 P0-2 的绝对路径尚未冻结。任务单按上述 P0-2 SHA 唯一定位；找不到
  时必须停在 `waiting_for_p02_manifest_input`，不得用近似文件替代。

## 9. 验证记录

- 新增专项测试：`11 passed`；
- scoped Ruff：pass；
- 真实 P0-2/CV3 本地运行：pass；
- 正式 crop 复建 SHA：与冻结值一致；
- 无图像 I/O、无 GPU 依赖；
- 未修改 P03/P04 训练代码或项目总索引。
