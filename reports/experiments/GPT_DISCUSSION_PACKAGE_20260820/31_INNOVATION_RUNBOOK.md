# 创新队列评估对比:Y3 / Y5 / Y4 vs M1(待队列完成执行)

> 状态:**等待队列完成**(预计 2026-08-19 09:30 前后)
> 队列:`run_all_innovations.sh`(Y3→Y5→Y4 串行,三折 ×160 epoch)
> 本文件是"队列跑完后的评估启动入口",脚本已全部就绪,跑完即用。

## 一、队列完成判定

```bash
ssh -p 47096 -o ControlPath=/tmp/n2cfg_ssh root@connect.nmb2.seetacloud.com \
  "tail -3 /workspace/results/innovation-run-all.log; ls /workspace/results/*-CV3-OOF-return-no-checkpoints.tar.gz* 2>/dev/null"
```

- 三个 `*-CV3-OOF-return-no-checkpoints.tar.gz` + `.sha256` 齐全 → 队列完成;
- `run_all_innovations.sh` 每完成一个打 `Yx_CV3_OOF_COMPLETE`。

## 二、拉回三个 evaluate + diagnose 结果

```bash
ssh -p 47096 -o ControlPath=/tmp/n2cfg_ssh root@connect.nmb2.seetacloud.com \
  "cd /workspace/results && for k in Y3-HIER Y5-ROT90 Y4-AFSS; do \
     tar czf /tmp/\$k-eval.tar.gz \$k-CV3-OOF/evaluate_\$k.json \$k-CV3-OOF/evaluate_\$k.cases.json \$k-CV3-OOF/diagnose_\$k.json; \
   done"
# 本地拉回
for k in Y3-HIER Y5-ROT90 Y4-AFSS; do
  scp -P 47096 -o ControlPath=/tmp/n2cfg_ssh \
    "root@connect.nmb2.seetacloud.com:/tmp/$k-eval.tar.gz" /tmp/
  tar xzf /tmp/$k-eval.tar.gz -C /tmp/
done
```

> 服务器 evaluate 输出:`/workspace/results/<KEY>-CV3-OOF/evaluate_<KEY>.json`
> (run_innovation.sh 第 165-170 行,与 diagnose 一起自动产出)

## 三、生成对比汇总

```bash
cd /Users/suzuku/Documents/揭榜挂帅-小样本遥感卫星图像/xh-202625
# 先确保 M1 基线 evaluate(本地已生成于 /tmp/M1-baseline-eval.json)
# 若需重建:
#   python scripts/evaluate_experiment.py \
#     --predictions /tmp/M1-oof-predictions-list.json \
#     --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
#     --project-config configs/project.yaml \
#     --model-key M1 --threshold 0.01 --output /tmp/M1-baseline-eval.json

python scripts/compare_innovations.py \
  --baseline /tmp/M1-baseline-eval.json \
  --innovation Y3=/tmp/Y3-HIER-CV3-OOF/evaluate_Y3-HIER.json \
  --innovation Y5=/tmp/Y5-ROT90-CV3-OOF/evaluate_Y5-ROT90.json \
  --innovation Y4=/tmp/Y4-AFSS-CV3-OOF/evaluate_Y4-AFSS.json \
  --output outputs/innovation-comparison/comparison_20260819.json
```

产出:
- `comparison_20260819.json`:41 行结构化对比(pooled/macro/per_fold/错误分解/专项/GT 尺寸)
- `comparison_20260819.md`:人类可读对比表(直接进报告)

## 四、判读重点(对应各创新假设)

| 创新 | 假设 | 重点看 |
|---|---|---|
| Y3-HIER | 层次损失降 FP_CLS | FP_CLS 是否下降;Recall 是否被 coarse 权重伤到 |
| Y5-ROT90 | 旋转增强提车辆 Recall | vehicle Recall / 车辆专项;是否伤 ship/aircraft |
| Y4-AFSS | 充分度采样改善低质量图 | FP_BG / FN_MISS;配 `diagnose_*.json` 看 source-group 集中度 |

每项再看 `diagnose_<KEY>.json`(错误按类/尺寸/fold/source-group 聚合),定位"哪里好了、哪里坏了",决定是否纳入最终链。

## 五、最终链升级路径(通过对比后)

1. 哪个创新 pooled/macro 显著优于 M1(Recall↑ 且 FDR↓ 或至少不劣化)→ 进最终链;
2. 多创新可组合 → 起新 plan 排队(约 20h/个);
3. 组合前先用 `analyze_experiment_errors.py` 看错误互补性(如 Y3 降 FP_CLS 而 Y5 提车辆 Recall,互补才值得组合)。
