# E 组合审计预演(2026-08-19)

> 数据: M1/M3 各自正式测速的 runtime_samples.jsonl(10 measured)
> 脚本: scripts/audit_combined_runtime.py(预算加总口径)

## 结果

| 指标 | 值 | 门禁 20s | 判定 |
|---|---|---|---|
| combined after-read p50 | **16.80s** | ≤20s | ✅ 余量 1.19× |
| combined after-read p95 | 17.03s | ≤20s | ✅ |
| combined after-read max | 17.03s | ≤20s | ✅ |
| **combined wall p50** | **19.37s** | - | ⚠️ 逼近预算 |
| M1 after-read p50 | 4.42s | - | - |
| M3 after-read p50 | 12.40s | - | - |

## 重要发现

1. **after-read 口径通过**(16.80 ≤ 20), 与预算加总 4.42+12.40=16.82 一致;
2. **wall 时间 19.37s 已非常逼近 20s 预算**(含 image_read 约 2.5s 开销)——若官方
   按 wall 计(含读图), 组合余量仅 0.63s, 风险高;
3. 官方口径是 after-read(任务单: 图像读入内存后的完整 pipeline), 组合按
   after-read 判定 OK; 但需在报告中注明 wall 风险。

## 建议

- 正式组合实测(GPU 空闲后)时, 输出同时给 after-read 与 wall 两套数字;
- 若官方后续改 wall 口径, M1+M3 组合可能不达标 → 需考虑 M3 门控推理
  (T4-GATED-INFER)只对部分区域跑 M3, 以压缩 wall。
