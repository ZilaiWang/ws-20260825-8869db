# N0-4 FP_BG 盲化可视审阅包补齐记录

日期：2026-08-14  
状态：`waiting_for_manual_review`  
科学权限：`automatic_background_admission=false`

## 1. 为什么补这一项

`outputs/N0-FP-BG-AUDIT-v2/audit_samples.csv` 已冻结 324 条分层审计记录，
但原产物只有候选坐标，无法直接判断：

- `clear_background`；
- `plausible_unlabeled_or_ambiguous_target`；
- `poor_localization_of_known_target`；
- `duplicate_or_fragment_not_captured`；
- `invalid_crop_or_render`。

因此，N2-v2 当前只能训练纯重分类，不能把 3,242 个未确认 hard negative 当作
背景。该限制保持不变。

## 2. 新增实现

- `src/rsdet/analysis/fp_bg_review.py`：审计记录与 formal manifest 对齐、GT 去重、
  review card 和 contact sheet 渲染；
- `scripts/render_fp_bg_review.py`：命令行入口；
- `tests/test_fp_bg_review.py`：源图关联、重复 annotation 去重、盲化卡片与产物合同测试。

卡片固定显示三个面板：

1. 全图；
2. 预测框附近 4 倍上下文；
3. 预测框 1.35 倍紧裁剪。

红框为待审预测，绿框为当前数据中的已知 GT。卡面只显示新的 `card_id`，不显示
`audit_uid`、`proposal_uid`、是否盲重复等字段。人工填写的
`manual_review_decisions.csv` 也只含 `card_id` 和标签；真实映射单独存放在
`sealed_card_mapping.csv`，审阅结束前不得打开。

## 3. 当前产物

正式盲化目录：`outputs/N0-FP-BG-AUDIT-v2-review-blind/`。较早的
`outputs/N0-FP-BG-AUDIT-v2-review/` 未分离映射表，只作本地渲染调试，禁止用于
正式盲审。

| 项目 | 数量/状态 |
|---|---:|
| 单卡 | 324 |
| contact sheet | 81（每张 4 卡） |
| 有序单卡 SHA256 | `5671255ac22059930c5a86ff439315e7c1ac614b8ce017eaa8a53c5112ab0da8` |
| 自动背景准入 | false |

首次可视抽查已经确认“`FP_BG` 不等于纯背景”的风险是真实的：同一页中可同时看到
明显的无标注疑似飞机、车辆/装备和舰船，也有纹理或设施型纯背景候选。这个观察只
证明人工审阅必要，不代替 324 卡的正式标签汇总。

## 4. 后续门禁

人工结果必须满足：

1. 324 行全部填写合法标签；
2. 盲重复卡一致率不低于 0.85；
3. 只有去重后的 `clear_background` proposal UID 可进入背景训练白名单；
4. 白名单、决策 CSV 和汇总 JSON 分别冻结 SHA256；
5. 背景拒识与飞机细分类保持独立消融，不能一次性合并后只报总体结果。

在上述条件完成前，P05/N2 背景拒识继续标记为等待人工证据，而不是代码阻塞。
