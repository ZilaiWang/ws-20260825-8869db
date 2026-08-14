# N0-4 v3 FP_BG 盲审标注指南

安全审阅包下载地址：

`https://gitee.com/zilai-wang/xh-202625/releases/download/v0.2-r1-evidence/N0-FP-BG-AUDIT-R1-6-V3-review-package.tar.gz`

SHA256：
`57f7056e63e956046d851d026a8c49a93cfbd261d05b6a6b3bf2a993a3b13d4f`

解压后审阅 `review-blind/` 中的 322 张卡。
红框是待审预测，绿框是数据中已知 GT。每张卡从左到右为全图、4 倍
上下文和 1.35 倍紧裁剪。

## 标签

- `clear_background`：红框内没有可合理解释为目标的结构，也不是附近已知
  GT 的定位误差、重复框或碎片。只有这一类可进入背景白名单。
- `plausible_unlabeled_or_ambiguous_target`：红框内存在未标注的飞机、舰船、
  车辆/装备，或图像分辨率不足以安全断言为背景。不确定时优先用此标签，
  不要冒险标为 `clear_background`。
- `poor_localization_of_known_target`：红框对应某个已知绿框目标，但中心、
  宽高、截断或边界偏差较大，官方 IoU 未达门槛。
- `duplicate_or_fragment_not_captured`：红框是已知目标的重复输出、部件/
  碎片，或与附近绿框实际上属于同一对象，但自动错误分解没有捕捉。
- `invalid_crop_or_render`：源图缺失、卡片渲染错位、红框完全不可见或其他
  使该卡无法审阅的技术问题。

## 程序约束

1. 只填写 `manual_review_decisions.csv` 的 `label`/`labeler`/`notes`，不改 `card_id`。
2. 审阅期间不得打开 `sealed_card_mapping.csv`、`audit_samples.csv` 或任何候选
   身份表。
3. 卡中含 54 张盲重复卡；不要主动查找重复，按每张卡独立判断。
4. 322 行全部完成后才允许解封编译；一致率门槛为 `>=0.85`。
5. 标签的目的是识别安全负样本，不是强行证明 `FP_BG` 大部分为背景。
