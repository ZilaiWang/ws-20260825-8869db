# 正式提交 1：无外部数据 Safe 基线冻结与新绝对评分审计

日期：2026-08-31  
结论：**正式第一次提交使用预测评 `trial-v2` 的完全相同镜像身份，不加入外部数据、
双视图或新阈值。**

## 1. 冻结候选

| 项目 | 冻结值 |
|---|---|
| 模型 | YOLO26-s |
| 训练数据 | 4,481 张官方训练图；无外部训练图 |
| 训练合同 | 1024 / RandomRotate90 / fixed 160 epochs / `last.pt` |
| 权重 | `outputs/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt` |
| 权重 SHA256 | `f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229` |
| 部署 | identity 单视图 / safe tile fusion / 全类统一阈值 0.15 |
| 配置 SHA256 | `2b47f14e69944c862eab7062b3121a980d24ded21febe1161e15c6de8170f281` |
| 本地镜像 | `xh-detector:y5s-single-thr015-v1` |
| Image ID | `sha256:8574d833876f4fee0556063c10788cf6a6ed335a62184dee0e818c39e57a1a52` |
| 平台 | linux/amd64 |

容器内再次核验得到相同的权重和配置 SHA；入口为
`["python", "/app/main.py"]`。RTX 3090 真实 10K smoke 已通过，冷启动墙钟约
12 秒；预测评平台记录的平均推理时间为 2.704833 秒。

本轮评分/冻结门禁/历史评分回归共 28 项测试通过，ruff 与两份服务器 shell 语法检查
通过，交付 manifest 149/149 文件的大小和 SHA 全部匹配。全仓测试在本机轻量 `.venv`
收集阶段因未安装 PyTorch 而未执行完；这项环境限制不记作全仓测试通过，也不影响已经
在含 CUDA/PyTorch 的 3090 环境完成的正式镜像 smoke。

不要使用 `dist/y5s-rot90tta-trial-v2-calibrated-v1`：其目录名虽含 `trial-v2`，实际是
identity+90° 双视图和分粗类阈值，对应预测评 `trial-v3` 路线。

## 2. 预测评的可复核依据

预测评 `trial-v2`：

| 粗类 | Recall | FDR |
|---|---:|---:|
| ship | 0.942287 | 0.126937 |
| aircraft | 0.999246 | 0.024300 |
| vehicle | 0.946309 | 0.237838 |

预测评旧平台综合分为 86.2274。`trial-v3` 双视图路线为 85.0018，且车辆 Recall 从
0.946309 降至 0.906040；因此双视图不能替代首次 Safe 基线。

## 3. 2026-08-31 新公式复算

新公式已经实现为：

- `src/rsdet/evaluation/absolute_score.py`
- `scripts/score_absolute_preliminary.py`
- `tests/test_absolute_score.py`

公告没有说明平台展示的三个粗类 Recall/FDR 如何归并为公式中的单个 `r/f`。代码不会
自行猜测，而是同时报告“先对三个粗类取均值再评分”“每粗类先评分再平均”；若提供
TP/FP/FN 还报告 pooled-count。仅用平台可见的六个粗类数值，得到：

| 路线 | 粗类原始均值后评分 | 每粗类评分后平均 |
|---|---:|---:|
| `trial-v2` Safe | **84.2245** | **84.9002** |
| `trial-v3` 双视图 | 84.2087 | 84.2087 |

两种可计算解释下 `trial-v2` 均不差于 `trial-v3`。上述数值不能当作正式隐藏集分数，
因为平台最终采用 pooled count、粗类 macro 或其他归并仍未在公式截图中说明。

在 `Recall > 0.85`、`FDR < 0.20`、`t < 20s` 区间：

- Recall +1pp 约增加总分 1.143；
- FDR -1pp 约增加总分 0.857；
- 时延 -1s 只增加总分 0.143。

因此后续优化优先级仍是 Recall，其次 FDR，时延只需保持安全裕量。

## 4. 是否在第一次提交前改阈值

内部全量部署回看：

| 工作点 | pooled Recall / FDR | 新公式分数（同一时延） |
|---|---:|---:|
| 统一 0.15 | 0.961075 / 0.144036 | 82.8191 |
| 统一 0.22 | 0.959222 / 0.137500 | 83.1675 |

0.22 的纸面增益仅约 **+0.35 分**，同时车辆 Recall 从 0.945652 降至 0.923913。
它没有达到占用一次正式机会所要求的约 +0.7 分证据，并存在隐藏域 Recall 放大损失。
所以第一次提交不改阈值。0.22 只作为观测到正式车辆 FDR 明显越线后的风险备选，不能
预先替换 Safe 基线。

## 5. 本轮同时修复的后续候选评估问题

1. 增加新绝对评分器及边界测试，旧 `official_ranking.py` 仅保留为历史 V1.6 排名协议。
2. 候选选择输出同架构、同时延下的新绝对分差。
3. Sentinel 不再在自身标签上拟合阈值：由 Hard 集确定三折阈值后原样应用到 Sentinel；
   决策 JSON 明确记录 `sentinel_thresholds_frozen_from_hard=true`。
4. 后续候选除 Recall/FDR 安全门禁外，还必须在 Hard 和冻结 Sentinel 上绝对分不下降。

这些修复只约束后续候选准入，不改变第一次提交的镜像。

## 6. 五次机会的当前使用原则

1. **Attempt 1：本报告冻结的 Safe `trial-v2` 等价镜像。** 用正式隐藏集建立新公式锚点。
2. Attempt 2：仅给完成 Normal/Hard/冻结 Sentinel 门禁且预期提升至少约 0.7 分的候选。
3. Attempt 3：若首发暴露单一粗类风险，提交一次有独立证据的单因素工作点修订。
4. Attempt 4：留给真正独立的模型/数据路线，不提交等价镜像。
5. Attempt 5：截止日前最终回退或冠军候选，不能提前消耗。

正在训练的 EXT-V 使用外部数据，尚未完成准入；它不得混入 Attempt 1，也不得仅凭训练
完成就占用 Attempt 2。

## 7. 正式推送前的人工操作合同

网站生成本次正式 tag 和临时登录命令后，在本机执行；`TARGET` 必须逐字复制网站显示的
完整镜像地址，不能复用预测评 tag：

```bash
SOURCE="xh-detector:y5s-single-thr015-v1"
TARGET="<网站本次生成的完整正式镜像地址>"

docker image inspect "$SOURCE" \
  --format '{{.Id}} {{.Architecture}} {{.Os}} {{.Size}}'
docker tag "$SOURCE" "$TARGET"
docker image inspect "$SOURCE" "$TARGET" --format '{{.Id}} {{join .RepoTags ","}}'
docker push "$TARGET"
```

两次 `inspect` 必须都显示 Image ID
`8574d833876f4fee0556063c10788cf6a6ed335a62184dee0e818c39e57a1a52`。push 完整成功后
再在网站点击“提交评测”。同一 tag 不允许覆盖；若 tag 已存在，不得用另一个本地镜像
冒充，应在网站生成下一次 tag。
