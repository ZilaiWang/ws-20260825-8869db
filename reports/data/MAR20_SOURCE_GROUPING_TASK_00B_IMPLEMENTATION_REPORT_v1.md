# MAR20-GROUPING-TASK-00B 实现与本地验收记录

## 1. 实现范围

本批次已实现：

1. v1.2 同距 clip 外扩飞机掩码；
2. `518×518 / patch14 / 37×37` 掩码映射及 20% patch 前景占比阈值；
3. 原图 DINOv2-B/14 输入下的 masked mean/signed-GeM；
4. 分片缓存、断点跳过和每图等额 patch sample；
5. local PCA128 + MiniBatchKMeans VLAD16/32；
6. mask-aware VLAD 第二遍提取及 global PCA-whiten-512；
7. 多路 DINO/VLAD/pHash/像素 SHA 候选召回；
8. 飞机掩码外 SIFT + RANSAC 候选富集；
9. 240 pair + 10% 盲重复的新标定包；
10. Round-A 旧证据合并、strict component 隔离划分；
11. calibration-only 选路由、held-out-only 验收的 Round-B 分析；
12. 统一终止决策器和服务器两阶段任务单。

VLAD 保持 MiniBatchKMeans 学到的欧氏中心和欧氏分配，不把中心强制单位化后改成余弦分配；Round-B 同时检查全证据集中已知 `same_frame` 的召回，避免其恰好落在 calibration 时被 held-out 指标遗漏。

## 2. 重要不变量

- Round-A v1.1 产物只读，不覆盖；
- `likely_same_airport` 不是 strict positive；
- DINO/VLAD/SIFT 在 00B 只产生候选，不自动 union；
- 769 张 bridge 不进入比赛模型训练；
- 正式分组和 CV3 仍未生成；
- 人工输入不存在时，服务器必须正常停在 waiting 状态。

## 3. 本地验收

### 3.1 自动测试

```text
MAR20 v1.1 + v1.2 scoped pytest: 18 passed, 2 skipped
full-repository pytest: 169 passed, 2 skipped
ruff scoped: pass
compileall: pass
11 scripts --help: pass
```

两个 skip 仅因当前本地 Python 没有 scikit-learn；服务器锁定环境已包含 `scikit-learn==1.5.2`，任务单要求服务器必须运行两个本地 skip 的测试并得到全通过。

### 3.2 真实数据 CPU smoke

使用 Round-A 回传 registry 和本地完整 MAR20：

- 4 张真实图的 10%/15%/20% patch overlay 生成成功；
- `automatic_geometry_gate=pass`；
- 人工可见网格与飞机位置对齐；
- 2 张真实图的 mock masked-feature/cache/sample 全链路通过；
- 缓存行数 2、唯一行数 2、NaN/Inf 0。

本地 smoke 只验证几何、缓存和接口，不进入科学比较。服务器仍必须使用锁定 DINOv2-B 权重执行真实 GPU smoke。

## 4. 服务器预期状态

Phase A 完成无标签提取、VLAD、候选富集和盲评包后，应正常返回：

```text
waiting_for_patch_mask_and_enriched_pair_reviews
```

该状态不是代码失败。Phase B 仅在两份人工 CSV 存在后继续。
