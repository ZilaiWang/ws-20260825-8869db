# 提交模型与权重冻结审计

日期：2026-08-29
结论状态：`trial_ready_but_formal_model_not_frozen`

## 1. 结论

当前 Docker 中的模型和权重**不是全部实验确定下来的正式最优模型**。它是专门为预测评
工程闭环准备的安全权重：M1 YOLO26-s、formal CV3 fold0、fixed epoch 160 `last.pt`，
SHA256 为：

```text
d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d
```

该权重只见过 formal CV3 的两个训练折；`score_threshold=0.051` 也属于 M1 OOF 的探索性
工作点。它已经通过 Linux/RTX 3090 端到端测试，适合分数无参考意义的预测评，但不能
直接改名为正式 v1.0。

目前实验结论分为三层：

| 层级 | 当前结论 | 能否立刻进入现有 Docker |
|---|---|---|
| 最强检测器训练方法 | Y5-ROT90（YOLO26-s + 90° 旋转增强） | 不能直接换权重，见第 3 节 |
| 最强可信科学链 | Y5 候选 + 14 特征 corrected-OER | 尚未形成部署模型包 |
| 已完整封装并在 3090 跑通 | M1 fold0 + 大图切片/全局融合 | 可以用于预测评 |

## 2. 为什么正式科学基线仍是 Y5 + corrected-OER

修正官方 prediction-first 匹配、tie block 和 formal grouped OOF 后，当前可信基线为：

| 指标 | 数值 |
|---|---:|
| Recall@FDR=.12 | **0.943104** |
| TP / FP | 19,742 / 2,687 |
| Recall@FDR=.10 | 0.936655 |

它以 65,301 条 Y5-ROT90 D4 OOF 候选为底座，使用 Y5 分数、crop 概率与熵、几何、
局部密度、D4 支持和 OTO 支持等 14 个特征进行 formal outer-fold OOF 重排序。绝对数字
以 2026-08-26 修正后的 0.943104 为准；旧的 0.9620 已作废。

后续路线均未获得替换资格：

- PAV-V1 `guard-strong` 仅 `+0.001385`，低于预注册 `+0.002` 准入线；
- PAV-V2 fold0 最佳仅 `+0.000408`，没有扩三折；
- MAR 乐观代理合并仅 `+0.000430`，且六项最差值下降 `-0.011037`；
- D3/D4 训练使用了含 held-out 统计的 curriculum，且 D4 混合了采样与 loss，不具正式
  比较资格；
- hard relabel 和其他后验分类方案没有稳定通过三折门禁。

因此，近期负向实验没有推翻 Y5 + corrected-OER 的科学地位；但“科学 OOF 最优”不等于
“已经有可装入 Docker 的最终模型”。

## 3. Y5 权重的部署阻塞

服务器保存了三个 fixed-epoch-160 Y5 checkpoint：

| fold | SHA256 |
|---:|---|
| 0 | `47d98fab29cbc4b6836a907a77cda33294affbd891e90f9e3aab0b05578e7c96` |
| 1 | `3b175e1471ae139dd4415cac094487e5a4be369cb0f1af6094bd3b2f1f25a9d4` |
| 2 | `8cec3f91cd0421c328f1fa430de2b9b4def4f28fdab41b1d90d56e76bd413304` |

但这些 checkpoint 把训练时的 `albumentations.RandomRotate90` 对象及 NumPy RNG 状态写入
了 pickle：

1. 当前 M1 镜像没有 `albumentations`，直接换入 Y5 时 Ultralytics 会尝试 AutoInstall；
   官方离线运行环境下不可接受。
2. 安装 `albumentations` 后，在冻结的 NumPy `1.26.4` 环境加载原始 Y5 checkpoint 会
   报 `PCG64 is not a known BitGenerator module`；原始 pickle 带有 NumPy 2.x 状态。
3. 所以 Y5 不能靠替换 `models/model.pt` 上线。必须在兼容环境中生成只含推理模型的
   sanitized checkpoint，或重训时禁止把训练 augmentation 对象写进最终部署权重，随后
   在离线 NumPy 1.26 环境重新加载和逐图比对。

此外，目前没有使用全部 4,481 张训练图按 Y5 冻结协议训练的最终权重。三折 checkpoint
分别用于 held-out OOF 估计，不应任意挑一折冒充全量最终模型。

## 4. corrected-OER 尚不能直接部署的原因

`scripts/build_corrected_oer_oof.py` 只生成严格 OOF 概率和评估产物，没有保存一个在全部
OOF 样本上拟合的最终 `HistGradientBoostingClassifier`。仓库和服务器均没有可部署的
OER `joblib/pickle`。正式部署还需要同时冻结：

1. 全数据 Y5 detector 或经过明确论证的 fold ensemble；
2. D4 支持的实时计算与去重对应；
3. proposal-domain crop classifier 权重；
4. OTO 支持来源及其权重；
5. 全 OOF 拟合的 OER 模型、特征顺序、缺失值合同和工作点阈值；
6. 上述完整链在 RTX 3090 上的 10K p95 与最终 `result.json` 复测。

在这些资产没有闭合前，不能把 `corrected_oer_oof_predictions.json` 当作能对未知测试图
运行的模型。

## 5. 提交决策

### 预测评

提交当前 `xh-detector:trial-preflight`。预测评数据和分数没有参考价值，本轮目标是验证
ACR push、平台 GPU、入口参数和 `result.json`。当前镜像已在相同级别 RTX 3090 上完成
10K 与三类真实格式图验收，工程风险最低。

### 正式测评

正式 v1.0 之前至少完成以下门禁：

1. 产出并验证可离线加载的 Y5 推理权重；
2. 用全部允许训练数据训练最终 Y5，或预注册并验证三折 ensemble；
3. 将 corrected-OER 变为完整可运行资产，并实测它是否仍在 20 秒内；若未闭合，则把
   “Y5 单模型安全链”作为正式首发，而不是回退到未经说明的 M1 fold0；
4. 冻结正式阈值、权重 SHA、配置 SHA、镜像 digest 和回退镜像。

## 6. 证据索引

- `reports/experiments/IMPROVEMENT_PLAN7_EXECUTION_CLOSURE_20260826.md`
- `reports/HERA_GUARD_PRECHECK_AND_FAST_SCREEN_20260826.md`
- `reports/experiments/M1_CV3_OOF_FORMAL_RESULT_AND_RECOVERY_AUDIT_v2.md`
- `outputs/HERA-GUARD-PRECHECK/corrected-oer-oof-v1/summary.json`
- `reports/submission/DOCKER_PREFLIGHT_20260829.md`
