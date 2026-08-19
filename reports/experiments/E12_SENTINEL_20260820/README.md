# Prospective Sentinel 冻结(2026-08-20)

- 从 255 个 source groups 中随机冻结 23 组(555 图, 12.4%), 整组冻结防组内泄漏;
- fold 分布: 0/1/2 = 208/149/198(均匀);
- 用途: 不参与模型选择/阈值/错误挖掘/人工审核, 只评最终 Safe/Balanced/Attack 三个冻结版本;
- 意义: 现有 OOF 已被多轮开发使用, sentinel 快速判断收益是否只是对旧错误清单的记忆;
- 文件: PROSPECTIVE_SENTINEL_20260820.json(frozen_groups + frozen_image_ids)。
