# 外部遥感粗类预训练合同

## 1. 使用边界

外部数据只用于 `aircraft / ship / vehicle / other_remote_object` 四粗类与
objectness 表示学习。外部类别不得映射为比赛的 25 个军机/舰船型号，外部数据不得参与
Normal-CV3、Hard10K、source-disjoint sentinel 的阈值、checkpoint 或融合权重选择。

每个资产必须记录官方来源、许可文本、下载时间、原始压缩包 SHA256、转换脚本 SHA256、
输出 COCO SHA256 和切片审计。许可不明确或禁止当前用途的资产不得进入正式权重。

## 2. 第一资产：DOTA-v1.0

DOTA-v1.0 首轮采用官方 train+val。`plane / ship / small-vehicle /
large-vehicle` 映射为前三个粗类；直升机、储罐和集装箱吊机映射为紧凑的
`other_remote_object`。港口、桥梁、环岛、球场和泳池的大范围标注不作为巨型前景框，
其像素仍保留在图中作为结构化背景，避免与舰船/车辆框形成大面积重叠监督。
`difficult > 0` 首轮丢弃。

OBB 转 HBB 使用四顶点外接轴对齐框并裁剪到图像。大图以 1024、重叠 256 做保尺度切片，
不缩放原图；每个标注只归属最大可见率的一个中心覆盖 tile，可见率低于 0.7 时丢弃；
每张源图最多保留两个确定性空 tile。转换、切片与 YOLO 导出分别由：

- `scripts/import_dota_to_coarse_coco.py`
- `scripts/slice_external_coarse_coco.py`
- `scripts/export_coarse_coco_to_yolo.py`
- `scripts/merge_external_coarse_coco.py`（train/val ID 重排并加路径前缀）

官方 train/val 文件 ID 已锁在 `configs/external/dota_v1_coarse.json`；
`scripts/download_external_dota_v1.py` 在下载前要求至少 20GiB 可用空间，并在完成后写
`ASSET_LOCK.json`。当前本地磁盘只保留官方 train part1 与全量 train labelTxt 用于真实
流水线 smoke；完整 train+val 必须在正式训练服务器下载，禁止在空间不足时留下无法审计的
半套资产。官方目录中多出的嵌套 `images/1/part1.zip` 不进入完整资产清单。

## 3. 第二资产与来源顺序

第二个已实现转换管线的数据源为 DIOR。官方来源固定为
`https://gcheng-nwpu.github.io/` 所列 Google Drive 文件夹，使用范围按
CC BY-NC 4.0 / 学术非商业用途记录。`airplane / ship / vehicle` 分别映射为前三粗类；
`chimney / storagetank / windmill` 仅作为紧凑的 `other_remote_object`；机场、港口、桥梁、
球场等大场景区域不生成巨型前景框，其像素保留为结构化背景。VOC 一基闭区间坐标严格转为
零基半开区间。

对应资产与实现：

- `configs/external/dior_coarse.json`；
- `scripts/download_external_gdrive_folder.py`；
- `scripts/import_dior_to_coarse_coco.py`；
- `scripts/discover_dior_layout.py`；
- `scripts/server/run_hera_guard_final_prepare_dior.sh`；
- `src/rsdet/external/dior.py`。

首轮因果筛选顺序冻结为 DOTA 单源 EXT-G/EXT-V，再做 DIOR 单源或 DOTA→DIOR 顺序续训。
不得在单源收益尚未确认前把全部来源混合。SODA-A 对 tiny aircraft/vehicle/ship 有价值，但
官方 OneDrive/Baidu 下载要求会话，列为第二轮储备。xView 与依赖它的 AI-TOD 需要注册；
FAIR1M 体量与 share-alike 边界需要单独审计；这些来源均不得以非官方镜像替代。

完整来源状态锁在 `configs/external/source_admission_matrix.json`。任何新增来源都必须先更新
该矩阵，再产生下载锁和转换审计。

## 4. 训练、迁移与并发合同

第一轮外部训练固定 Y5-S、单视图、四类 head。外部 checkpoint 转回官方 25 类时，只迁移
shape 一致的 backbone/neck；检测 head 由 Ultralytics 原生 fresh `DetectionModel` 重建后
移植，不能对已有 Detect 模块递归调用通用 `reset_parameters`。后者会破坏 Detect 的原生
`bias_init`，已经在真实 smoke 中表现为数量级异常的分类损失。

正式迁移固定两阶段：前 8 epoch 冻结前 10 层稳定 fresh 25 类 head，随后重建 optimizer、
全模型训练 32 epoch。两个阶段必须使用同一个冻结 Python/NumPy/PyTorch/Ultralytics 环境；
不允许在 NumPy 1.x 与 2.x 环境之间载入训练 checkpoint。迁移/跳过参数、head 初始化方式、
两阶段 checkpoint 和 SHA256 都必须写入审计。

4 GPU 并行时 EXT-G 与 EXT-V 共享只读图像字节，但必须拥有独立的 labels 目录和
`labels/train.cache`。`scripts/materialize_external_role_view.py` 以图像目录软链接和标签硬链接
产生缓存隔离视图；任何直接让两个训练进程写同一个 Ultralytics cache 的执行均无效。

仅当 fold0 快筛同时满足候选地板下降不超过 0.3pp、Normal macro 下降不超过 0.3pp、
Hard ship 或 vehicle 固定风险 Recall 至少提高 0.5pp且 Sentinel 同方向，才扩展 CV3。

## 5. 固定评测与准入

所有候选只允许使用两套主评测：

1. Normal-CV3：候选地板、25 类 macro 与粗类不可退化；
2. Hard10K + source-disjoint Sentinel：在 FDR=0.15 的冻结外层风险工作点检验方向。

首轮只训练 fold0，HAD 另加历史反向风险 fold2。候选只替换对应 fold 的预测，另外两折必须
逐字节复用基线。仅当 Normal Recall/macro 下降均不超过 0.3pp、任一粗类 Recall 下降不超过
0.5pp、Hard ship 或 vehicle 提高至少 0.5pp且 Sentinel 同方向、FDR 恶化不超过 1pp，才准入
CV3。不得在 Hard/Sentinel 上选择 checkpoint、阈值或融合权重。

## 6. 后续资产

xView、AI-TOD、FAIR1M、RarePlanes 等只有在来源、许可、磁盘和统一转换审计完成后才能加入。
多数据源训练必须做 dataset-aware sampling，并保留单数据源消融；不能把多个来源一次性混成
无法归因的训练集。
