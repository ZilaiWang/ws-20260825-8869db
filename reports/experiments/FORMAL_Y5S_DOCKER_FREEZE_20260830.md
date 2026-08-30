# Y5-S 正式 Docker 冻结与提交审计

日期：2026-08-30  
代码提交：`d14eef68ff19ff3bf9c58847c48bbc919e57931e`  
状态：两份正式候选镜像已构建并通过 RTX 3090 真实 10K GPU smoke

## 1. 冻结结论

正式权重只选用 Y5-S 全量训练权重：

```text
outputs/Y5-FULL-S-20260829-R1/y5_full_s_sanitized.pt
SHA256 f7e30fac3391d7048314d1c41df0e878a4d5f0423e5c55b68ee7df5917418229
训练数据 4,481 张官方训练图
训练合同 YOLO26-s / imgsz=1024 / RandomRotate90 / 160 fixed epochs / last.pt
```

Y5-L 在同合同的 FDR≈0.10 与 FDR≈0.15 工作点均低于 Y5-S，且速度更慢；E3 双 crop
后验替换和 E4 VOI 门控在正式 OOF 与开发难集上均为负向。因此不再启动新全量训练，
避免在临近提交时用未经独立准入的结构替换已经完成全量拟合的 Y5-S。

## 2. 两个候选

| 候选 | 配置 | 作用 |
|---|---|---|
| A：稳健主镜像 | `y5_full_s_safe_1024_thr015.json` | 单视图、统一阈值 0.15；证据链最简单，作为正式首次提交和回退点 |
| B：高收益候选 | `y5_full_s_safe_1024_rot90cwtta_coarse_v1.json` | identity+90° 双视图；ship/aircraft/vehicle 阈值 0.371/0.301/0.366 |

两者共用同一权重。B 的双视图在独立困难 OOF 上把固定风险附近 Recall 由约 0.7099
提高到 0.7442，说明视图方向成立；但 B 的低粗类阈值来源于全量权重同源部署回看，
不是已验证的隐藏域阈值。因此 A/B 的官方差异只能解释整套部署策略，不能把收益单独归因
于视图或阈值。

## 3. 镜像身份

| 镜像 | Image ID / digest | 架构 | 大小 |
|---|---|---|---:|
| `xh-detector:y5s-single-thr015-v1` | `sha256:8574d833876f4fee0556063c10788cf6a6ed335a62184dee0e818c39e57a1a52` | linux/amd64 | 4,351,055,994 B |
| `xh-detector:y5s-rot90tta-coarse-v1` | `sha256:3a68bfd75dc42b6d333f8a1d60fe94ab60b4b978545c329ebbc1ae34725a84b4` | linux/amd64 | 4,351,056,046 B |

配置 SHA256：

```text
2b47f14e69944c862eab7062b3121a980d24ded21febe1161e15c6de8170f281  y5_full_s_safe_1024_thr015.json
dc3b02ff076b6f832c60f87d80d09be3991288a7817bfc3b858c727aa2b56264  y5_full_s_safe_1024_rot90cwtta_coarse_v1.json
```

两份交付目录的 `BUILD_MANIFEST.json` 均记录干净源树、同一 commit 和同一权重 SHA。
容器内断网静态检查已经验证：入口为 `python /app/main.py`，配置可加载，权重文件 SHA
与 `model.expected_sha256` 完全一致。

## 4. RTX 3090 真实 10K smoke

输入为 `fold0_pseudo_10k_00.jpg`，在 RTX 3090、torch 2.5.1+cu121 上运行官方入口，
模型加载、切片、推理、融合和 `result.json` 原子写出均实际执行：

| 候选 | 冷启动端到端墙钟 | 输出对象 | 结果校验 |
|---|---:|---:|---|
| A 单视图 | 约 12 s | 397 | pass |
| B 双视图 | 约 14 s | 384 | pass |

两者均低于项目冻结的 20 秒上限。上述墙钟包含 Python 启动、权重加载和整图解码，比平台
逐图 `run_end_timestamp` 的纯推理口径更保守。GPU smoke 原始日志与结果位于：

```text
outputs/FORMAL-Y5S-GPU-SMOKE-20260830/
```

## 5. 正式提交顺序

1. 首次正式提交 A，建立全量 Y5-S 的官方域基线；首次提交会被平台优先调度。
2. 第二次提交 B，测试旋转补召回与分粗类风险控制在隐藏域上的净收益。
3. 比较时逐项记录 ship/aircraft/vehicle 的 Recall、FDR 与平均时延，而不是只看平台综合名次。
4. 若 B 的任一粗类 Recall 明显下降或 FDR 没有改善，立即回退 A；不现场扫描更多阈值。
5. 剩余三次只留给有新独立证据的单因素修订，不重复提交等价镜像。

当前内部证据不能保证隐藏测试 Recall 达到 94%；它能保证的是：权重来自全量训练、被否决
路线没有混入镜像、两份候选均为可复现的正式代码路径，并已在与官方算力一致的 RTX 3090
上完成真实 10K 运行。隐藏域分数尺度只能由最少次数的正式 A/B 提交最终确定。
