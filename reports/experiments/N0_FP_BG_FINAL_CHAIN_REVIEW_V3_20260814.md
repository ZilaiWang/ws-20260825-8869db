# N0-4 R1-6 最终候选链 FP_BG 盲审包 v3（2026-08-14）

## 1. 当前状态

`waiting_for_manual_review`。自动背景准入仍为 `false`。

正式审阅任务：Gitee Issue `IK8V54`
`https://gitee.com/<UPSTREAM_OWNER>/xh-202625/issues/IK8V54`

旧包基于 N0 早期候选链，270 个唯一审阅对象中仅 196 个（72.6%）在
R1-6 输出中仍有坐标级对应。因此旧卡片不用于最终背景白名单。

v3 从冻结的 `CE+D4+C2+aircraft post-NMS@0.50` 预测重新执行守恒
错误分解，得到当前真实剩余 `FP_BG=1539`。

## 2. 抽样与盲化

| 项目 | 数值 |
|---|---:|
| FP_BG 母池 | 1539 |
| 唯一正卡 | 268 |
| 盲重复卡 | 54 |
| 总卡片 | 322 |
| contact sheets | 81 |
| 分层 | 3 粗类 × 3 fold × 3 分数档，27 层全覆盖 |

`vehicle|fold1|high` 母池仅足够抽 8 张，其余 26 层各 10 张，因此
唯一卡为 268 而非 270。

有序卡片 SHA256：
`15852d41fe45188a38533e308a086346b8e2931a40772255d9259fa9aff5ae69`。

## 3. 一致性修正

旧 `compute_audit_summary` 只将 `repeat_of` 非空的重复卡放入一致性分组，
没有将对应原卡放入；当每对只有一张重复卡时，会伪造天然 100%
一致。本轮已改为按 `proposal_uid` 将原卡与重复卡共同分组，且将
主卡未标注记为 incomplete。

新增 `scripts/compile_fp_bg_review.py`，仅在：

1. 322 张全部使用合法标签；
2. card/mapping/audit 集合完全一致；
3. 盲重复完整且一致率 `>=0.85`；
4. 同 proposal 没有冲突标签；

时输出去重的 `clear_background_whitelist.csv`。

## 4. 产物索引

- Gitee Release 安全审阅包：
  `https://gitee.com/<UPSTREAM_OWNER>/xh-202625/releases/download/v0.2-r1-evidence/N0-FP-BG-AUDIT-R1-6-V3-review-package.tar.gz`
- Gitee Release SHA256：
  `https://gitee.com/<UPSTREAM_OWNER>/xh-202625/releases/download/v0.2-r1-evidence/N0-FP-BG-AUDIT-R1-6-V3-review-package.tar.gz.sha256`
- 审阅包 SHA256：
  `57f7056e63e956046d851d026a8c49a93cfbd261d05b6a6b3bf2a993a3b13d4f`
- 冻结配置：`configs/experiments/n0_fp_bg_review_r1_6_v3.yaml`
- 最终链证据构建：`scripts/build_final_chain_fp_bg_manifest.py`
- 分层抽样：`scripts/n0_4_fp_bg_audit.py`
- 盲化渲染：`scripts/render_fp_bg_review.py`
- 解封编译：`scripts/compile_fp_bg_review.py`
- 盲审目录：`outputs/N0-FP-BG-AUDIT-R1-6-V3/review-blind/`
- 待填表：`outputs/N0-FP-BG-AUDIT-R1-6-V3/review-blind/manual_review_decisions.csv`

Release 附件只包含盲化卡片、contact sheets、待填决策表和公开审阅说明，
不包含 `sealed_card_mapping.csv`、`audit_samples.csv` 或候选身份表。
