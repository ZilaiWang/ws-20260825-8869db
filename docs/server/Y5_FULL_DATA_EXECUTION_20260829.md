# Y5 全量官方数据训练合同（2026-08-29）

目标：在 CV3 已经完成方法选择后，用全部 4481 张官方训练图拟合最终 Y5-Rot90。
该任务不是新的验证实验，不生成“全量验证分数”，也不使用预测评选择 checkpoint。

冻结条件：

- 基座：YOLO26-s 官方初始化，SHA256
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`；
- 数据来源：`cv3_airport_proxy_k60_v2.json` 中恰好 4481 个唯一样本；
- 输入：1024，batch 12，AdamW，lr0 0.002，cosine，160 epoch；
- 单因素创新：`RandomRotate90(p=1.0)`；
- 选择：固定 `last.pt`，禁止 early stop、best 选择、resume 和预测评选模；
- 训练后必须执行 `sanitize_yolo_checkpoint.py`，再做净化前后逐框等价检查。

服务器入口：

```bash
PYTHONPATH=src python scripts/train_full_y5.py \
  --manifest data/splits/cv3_airport_proxy_k60_v2.json \
  --data-root /root/autodl-tmp/data \
  --weights /workspace/cv3-model-assets/yolo26s.pt \
  --expected-weight-sha256 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b \
  --output-dir /workspace/results/Y5-FULL-S-20260829 \
  --dry-run
```

去掉 `--dry-run` 后才进入正式训练。完成门禁为：4481/4481 数据存在、160 行
`results.csv`、`last.pt`、`training_result.json`、净化权重、净化等价检查全部通过。
