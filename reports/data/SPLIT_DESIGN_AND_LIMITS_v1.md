# 数据划分设计与使用边界（dev_v1 / cv3）

本文说明 B 交付的两份划分的分组依据、证据强度和使用限制。
数字来自 `scripts/build_split.py` 与 `scripts/build_cv3.py` 的运行输出。

## 1. 交付物

| 文件 | 说明 |
|---|---|
| `data/splits/dev_v1.json` | 固定开发划分，train 3456 / val 1025 |
| `data/splits/cv3_airport_proxy_k60_v1.json` | 三折划分，fold0 1609 / fold1 1574 / fold2 1298 |
| `reports/data/near_duplicates_mar20.json` | MAR20 近重复检测结果 |
| `third_party/mar20/` | MAR20 官方划分列表（编号，CC BY-NC 4.0） |

`dev_v2_airport_proxy_k60.json` 由 A 在 dev_v1 基础上生成，见
`MAR20_DEV_V2_AIRPORT_PROXY_K60_SPLIT_ACCEPTANCE_v1.md`。

## 2. 分组依据与证据强度

防泄漏的基本单位是 `group_id`：同一组的图像必须整组进入同一划分，不得拆分。

| 大类 | 图像数 | 分组依据 | 组数 | 证据强度 |
|---|---:|---|---:|---|
| 舰船 | 1,341 | 文件名景 ID（`L\d+A\d+` / `L\d{11}`） | 174 | 强：拍摄元数据 |
| 发射车 | 67 | 文件名经纬度（`N..-E..`） | 21 | 强：拍摄元数据 |
| 飞机 | 3,073 | K=60 机场代理组（A 提供） | 60 | 中高：视觉聚类代理 |

飞机分组直接采用 `data/groups/mar20_airport_proxy_k60_for_b.csv`，
按其验收报告要求「同一 `group_id` 当作不可拆分原子」。

**K=60 是机场背景视觉代理组，不是真实机场标签。** 正式文字写作
"MAR20 airport-proxy grouped"，不得声称已获得 airport-disjoint ground truth。

### 2.1 舰船与发射车的分组细节

舰船文件名含卫星影像元数据，例如
`01-PAN-20240418-318-232-L00000010061-CCD14_3_crop1`：
`PAN` 为全色、`20240418` 为拍摄日期、`L00000010061` 为景 ID、`CCD14` 为片号、
`crop1` 为裁片序号。同一景的裁片背景与光照一致，按**景**（而非景+CCD）分组，
避免同一次过境的相邻条带跨划分。

发射车文件名含经纬度与影像源，例如
`fsc_AGZ-N23.00-E120.33-lv20-Bing_crop0001`。同一坐标存在 Bing 与 Google
两个来源，按**经纬度**分组，两个来源整组同进同出。

### 2.2 已排除的分组方案

以下方案经验证不可用，记录以免重复尝试：

- **按 MAR20 编号连续性分组**：切分后得到 2,508 个块、中位数 1 张，
  无防泄漏效果（`scripts/check_mar20_blocks.py`）。
- **按 MAR20 官方 test.txt 递增段分组**：可覆盖 1,990 张，但经 A 核对
  存在十几对跨段泄漏，已由 K60 机场代理组取代（`scripts/check_mar20_imagesets.py`）。
- **对舰船使用 dHash 近重复合并**：全色海面图纹理不足（4,481 张中 501 张
  缩略图灰度标准差 < 8），dHash 将不同海域、不同年份的空旷海面判为近重复，
  产生跨景假阳性。舰船分组仅采用景 ID，不采用近重复结果。

MAR20 部分保留 dHash 近重复合并（阈值 6，7 组）。其中 `MAR20_440` 与
`MAR20_441` 编号相邻且视觉近重复，属双重证据。检出总量与 P0-2 报告的
11 条候选边数量级一致，但未逐条比对，不能声称两者识别的是同一批图像。

## 3. dev_v1 验收

| 指标 | 结果 |
|---|---|
| 跨 train/val 的分组数 | 0 |
| 25 个细类在验证集有样本 | 25 / 25 |
| val 占比 | 22.9% |

val 占比高于 80/20 目标，原因是分组整块分配且优先保证稀有类验证覆盖。
已与队长确认该偏差可接受。

## 4. cv3 验收

| 指标 | 结果 |
|---|---|
| 跨折的分组数 | 0 |
| 各折图像数 | 1609 / 1574 / 1298 |

分配方法：按组从大到小依次试放，选使各细类在三折间分布标准差最小的那一折。
舰船与发射车三折近似等分（QHS 214/213/214，MS 665/664/665，FSC 134/132/136）；
飞机受机场组粒度限制存在波动。

## 5. 使用限制（重要）

### 5.1 TU-160 的三折指标不可用

`A6_TU-160` 共 361 框，其中 360 框集中于单一机场代理组：

| | fold0 | fold1 | fold2 |
|---|---:|---:|---:|
| 作验证时的框数 | 360 | 0 | 1 |

fold1 作验证时无样本，fold0 作验证时训练侧仅剩 1 框。
**该类的三折结果不得用于任何模型比较或结论。** 数据本身高度集中，
非划分方法可解决；已与队长确认照做并单列记录。

### 5.2 折间波动需注明的类

| 细类 | 各折框数 |
|---|---|
| A7_E-3 | 82 / 372 / 93 |
| A8_B-52 | 490 / 141 / 119 |
| A16_FA-18 | 300 / 1710 / 137 |

这些类可用，但单折结果受分布影响大，须报告三折均值与折间波动，
不得用单折结果立论。

`A10_B-1B` 与 `A17_TU-95` 各有一折的样本仅来自 1 个代理组，代表性有限。

### 5.3 验证证据量偏低的细类（dev_v1）

| 细类 | val 框 | val 图 | val 组 |
|---|---:|---:|---:|
| HM 航母 | 5 | 5 | 4 |
| LQS 两栖舰 | 11 | 9 | 5 |
| A15_F-22 | 34 | 8 | 8 |
| A18_KC-10 | 31 | 16 | 12 |
| FSC 发射车 | 76 | 13 | 6 |

这些类漏检一两个目标即可造成 Recall 大幅波动，**dev_v1 上的单类指标仅供参考，
结论以 cv3 三折为准**。FSC 是车辆大类唯一细类，直接影响该大类的官方指标。

### 5.4 不移动原始文件

验证集只在 manifest 中以 `split` / `fold` 字段逻辑标注，**未移动任何原始图像与标签**。

原因：`XHDataset` 的 `image_id` 由目录内文件名排序生成
（`enumerate(sorted(image_by_stem), start=1)`）。将任何图像移入 `images/val/`
都会使剩余图像的 `image_id` 整体错位，导致全队的预测、COCO GT 与实验台账失效。

## 6. 版本管理

按 `INTEGRATION_CONTRACT.md` 第 4 节，split 更新必须更换版本号，
不得静默改变已有 ID 和成员归属。已发布版本不修改；分组证据更新时新增版本。

## 7. 复现

    PYTHONPATH=src python scripts/build_split.py \
      --data-root data/raw \
      --imagesets third_party/mar20 \
      --near-duplicates reports/data/near_duplicates_mar20.json \
      --output data/splits/dev_v1.json

    PYTHONPATH=src python scripts/build_cv3.py \
      --data-root data/raw \
      --imagesets third_party/mar20 \
      --airport-groups data/groups/mar20_airport_proxy_k60_for_b.csv \
      --near-duplicates reports/data/near_duplicates_mar20.json \
      --output data/splits/cv3_airport_proxy_k60_v1.json

随机种子 42，数据版本 `official_raw_v1`。