# MAR20-GROUPING-TASK-00A 验收与人工复核交接

## 1. 验收结论

TASK-00A 的计算产物完整，可以进入人工复核；当前状态必须保持为
`waiting_for_method_specific_view_and_pair_reviews`，不得提前形成分组或描述子入选结论。

回传包 `outputs/MAR20-00A-return.tar.gz` 的 SHA256 为：

```text
90e2f04a6fd1c3d0f403ea1b7ecb09b38a264a921fd00c008257cfbcf1854232
```

原回传包错误包含了应在服务器封存的 `blind_card_mapping.csv`。本地验收未读取其内容，
只通过归档文件名发现该问题，因此匿名卡片仍可继续使用。人工评审者不得打开原回传包，
只使用重新生成的安全盲评包。

## 2. 已确认的技术结果

- 方法分列视图复核模板：120 个节点，其中 114 个有背景 tile、6 个无可用背景 tile；
- calibration：360 个唯一 pair、29 张盲重复卡，共 389 张卡和 98 张 contact sheet；
- 盲重复卡最小间距为 30，涉及 672 个唯一节点；
- DINOv2-B smoke：20 行、15 路特征，每路 768 维，`nonfinite_count=0`；
- blur、local_mean、Telea 三份缓存均覆盖同一批 672 个节点，每份 6,397 行、50 个 shard；
- 三份缓存均无 NaN/Inf，23 个节点没有合格背景 tile，三种方法的缺失集合一致；
- resume 复检均为 `computed_rows=0`、`skipped_rows=6397`，证明缓存可安全续用；
- 三种填充的缓存指纹不同，未发生相互覆盖。

## 3. 人工复核一：背景视图质量

逐行查看 15 张 `view-review/contact_sheets/sheet-*.jpg`，填写
`manual_view_review_v2.csv`。所有方法必须分别判断，不能用一项结论代替三种填充。

字段规则：

- `valid`：该行图像完整、清晰，足以作出判断时填 1，否则填 0；
- `*_aircraft_remnant`：填充后仍可辨认飞机主体、机翼、机尾或稳定轮廓时填 1；
- `*_inpaint_artifact`：明显方块、拉伸、重复纹理或填充边界足以主导地点特征时填 1；
- `background_tile_aircraft`：背景 tile 中仍有明显飞机主体时填 1；机器预填的
  `background_tile_available` 不得修改，无 tile 的行可将该字段留空。

机场跑道、机坪、建筑和道路仍可见不是飞机残留；这些正是地点描述子需要保留的信息。

冻结门槛为：有效率不低于 90%；候选方法的飞机残留率不高于 5%，填充伪影率不高于
10%；背景 tile 中明显飞机数必须为 0。至少一种填充方法通过，才允许继续。

## 4. 人工复核二：calibration pair

按 `CAL-0001` 至 `CAL-0389` 的自然顺序查看 98 张 contact sheet 并填写
`manual_calibration_decisions.csv`。不要跳着寻找相似卡片，也不要尝试识别盲重复。

标签定义：

- `same_frame`：两图实质上是同一帧或同一源图；
- `geometric_overlap`：不同帧，但存在可对齐的重叠地面区域；
- `same_local_site`：无须直接重叠，但跑道、滑行道、建筑、道路等稳定布局能证明属于同一局部地点；
- `likely_same_airport`：有同机场迹象，但证据不足以证明同一局部地点；
- `not_same_local_site`：看不出同一局部地点，但也不足以断言机场不同；
- `different_airport`：存在足够强的互斥地点证据，谨慎使用；
- `uncertain`：信息不足、遮挡严重或正反证据冲突。

`confidence` 表示对所选标签的把握，必须为 0 到 1；它不是图像相似度。飞机型号相同、
色调接近或编号接近都不能单独作为同地点证据。`supporting_evidence` 和
`counter_evidence` 应简短记录可复核的地面结构证据。

编译时只有 `same_frame`、`geometric_overlap`、`same_local_site` 进入正样本；
`not_same_local_site`、`different_airport` 进入负样本；其余标签仅保留用于诊断。
需要至少 30 个正 pair，盲重复标签一致率至少 90%，且当前严格合同不允许重复卡标签冲突。

## 5. 评审组织与回传

建议由同一名主评审者连续完成 389 张卡，以免盲重复一致率混入评审者间差异。另一名成员
可以在编译后只复核低置信、冲突或会连接大组件的高影响 pair，不应在首轮评审前接触映射。

完成后只回传两份已填写 CSV：

```text
manual_view_review_v2.csv
manual_calibration_decisions.csv
```

服务器使用封存映射完成解盲、重复一致性门禁、pair 编译和 Round-A 描述子 bake-off。

