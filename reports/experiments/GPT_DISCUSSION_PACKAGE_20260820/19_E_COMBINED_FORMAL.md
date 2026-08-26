# E 组合系统正式实测报告(2026-08-19 23:40)

> 数据: 本机(同环境)M1 + M3 正式采集(3 warmup + 10 measured, 独占 GPU)
> 口径: 预算加总(逐图 M1 after-read + M3 after-read), 门禁 20s
> 脚本: audit_combined_runtime.py(输入 SHA 已冻结)

## 结果

| 指标 | 值 | 门禁 20s | 判定 |
|---|---|---|---|
| **combined after-read p50** | **14.28s** | ≤20s | ✅ 余量 1.40× |
| combined after-read p95 | 14.46s | ≤20s | ✅ |
| combined after-read max | 14.46s | ≤20s | ✅ |
| combined wall p50 | 16.88s | ≤20s | ✅(wall 也过) |
| M1 after-read p50 | 1.89s | - | 本机独占 GPU 实测 |
| M3 after-read p50 | 12.40s | - | 同环境 |

## 关键说明

1. **gate5(10K p95 ≤ 18s)数据到手**: combined after-read p95 = 14.46s < 18s ✅;
2. **M1 本机实测 1.89s**(历史归档 4.42s 为当时环境/负载下测得, 配置一致
   batch8/tile1280/overlap256)——组合按同环境口径, 余量充足;
3. **wall 16.88s 也 ≤ 20s**, 即使官方改 wall 口径(含读图)组合仍达标(与
   预演时 wall 19.37s 逼近不同, 因 M1 本机更快);
4. 输入 SHA 已冻结(M1 722c6c78 / M3 1bcee237), 可追溯。

## 结论

- **组合系统(或最终链 Y5+M1+M3 类似结构)时延预算充足**:
  after-read 余量 1.40×, wall 余量 1.19×;
- T4-GATED-INFER 的 gate5 条件(10K p95 ≤ 18s)满足;
- 注: 组合口径为"预算加总"(M1+M3 串行), 若官方要求"单模型内部融合后
  并行"口径需另写实现(当前非官方需求)。
