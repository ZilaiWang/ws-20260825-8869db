# N2-CFG 链路预检记录(白名单签发前)

> 日期：2026-08-18 | 目的：确认白名单一签发即可一键跑,不踩隐藏坑
> 预检人：主线 | 状态：**管线可跑,量不足已量化确认**

## 1. 代码锁

- 发现 `N2_CFG_CODE_SHA256.txt` 过期：dc60b68(sealed runner)改了
  `evaluate_bg_gate.py` + `run_n2_cfg.sh` 未同步锁 → sha256sum --check 必失败。
- **已修复**（commit 5365591）：重算两文件 SHA，11 项全对齐。

## 2. 白名单空 → 报错路径验证 ✓

- `clear_background_whitelist.csv` 当前为空（门禁失败未签发），
  `_load_clear_background_uids` 正确 raise：
  `ValueError: clear_background 白名单为空`。
- 行为符合预期：不会静默继续,明确等待白名单。

## 3. 模拟 30 白名单 → manifest 构建全链路 ✓

用编译逻辑提取的 30 个真实 clear_background proposal 模拟白名单,
跑 `build_foreground_gate_manifest`：

```
rows=19500
view_counts: deployable_positive=19470, clear_background=30
skipped: ignored_fp_loc=62, unconfirmed_fp_bg=1509,
         ignored_fp_cls=777, ignored_fp_dup=104
```

- 19470 官方 TP 全部进正样本;30 白名单全进负样本;
- **1509 个未审 FP_BG 正确跳过**（不污染训练）——链路行为符合《改进方案 1》第 2.4 节。

## 4. 30 负样本下三折均衡采样 ✓(能跑,但量不足)

`build_balanced_batch_indices`（batch 64, 200 batch/epoch, seed=202625+fold）：

| fold(held-out) | 训练行 | 正样本 | 负样本 | 结果 |
|---|---|---|---|---|
| 0 | 12,711 | 12,693 | 18 | ✅ 采样正常 |
| 1 | 12,737 | 12,714 | 23 | ✅ 采样正常 |
| 2 | 13,552 | 13,533 | 19 | ✅ 采样正常 |

- 负样本每折仅 **18-23 个**（去重后更少）,32,000 次负采样/折从极少量样本重复抽
  → 门控分类器对"背景"的泛化极其有限;
- **佐证**：N2_CFG_NEGATIVE_SAMPLE_SUFFICIENCY_20260818.md 结论成立——
  三折能跑,但 paired bootstrap 门禁（2000 次,95%CI 下界>0）大概率不过,
  **建议扩审第 2 批补负样本**（目标 600-700 卡,白名单 ~105-115）。

## 5. 结论

1. 管线代码本身无隐藏阻塞（锁已修、报错路径正确、manifest/采样全链路通）；
2. **白名单签发后即可一键跑** `run_n2_cfg.sh`（服务器需 convnext 权重 + GPU）;
3. 但用当前 30 白名单跑,大概率门禁不过 → 建议等扩审后跑正式三折,
   或先跑 smoke（验证 S0/S1/S2 机制）再决定。
