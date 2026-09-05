# Progressive-40 全量候选：本次提交冻结与操作

日期：2026-09-03。当前状态：`submitted_evaluated_all_platform_gates_passed`。

用户已推送并提交正式 `v2.0`（ID3953），2026-09-03 16:20:12成绩为 **76.6010**，
读取时第16名，Recall/FDR/时延三门全部通过。推送digest与下述冻结镜像一致。
完整接口、与v1差分和本地诊断差距见
[正式提交2结果分析](../experiments/FORMAL_ATTEMPT2_P40_PLATFORM_RESULT_ANALYSIS_20260903.md)。
下文保留原准备过程与操作步骤作为历史记录，不构成再次提交指令。

本次用户已明确要求准备提交，覆盖此前“不得打包”的操作限制。这里只准备用户选择的
单个候选，不代为推送、不点击官方评测，也不新增训练或调整阈值。

## 1. 只提交这一份

| 项目 | 冻结值 |
|---|---|
| 本机镜像 | `xh-detector:p40-full-s1280-frozen0536-final` |
| Image ID | `sha256:db2a0eaacc0608eecd80193f2cefb83995214288da0250d405b8f016e8ae1303` |
| 架构 | `linux/amd64` |
| 模型 | YOLO26-s，25类，单模型、单视图、无TTA/二次模型 |
| 训练来源 | S1024全量160e → 1280适配40e；4,481张官方训练图，无新增外部训练数据 |
| 原始last SHA | `904c4935a85484a83d98930b0862bd1b5a1b0e9e7c6ed4eea7525391d383123f` |
| 部署权重SHA | `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012` |
| 配置SHA | `50156c2d3143f930cb6f07f0a72e76b69ad64897f363e2fdea05718c80a52e22` |
| 网络输入 | 1280 |
| 大图切片 | 1024，重叠256，batch4，safe融合 |
| 候选底阈值 | 0.001；每tile上限500、融合后上限4000 |
| 最终输出阈值 | 融合完成后统一过滤 `score >= 0.536`；没有重新选阈值 |

原始last包含训练增强对象，直接放进离线镜像会因缺少albumentations而加载失败。
沿用已有净化导出工具，删除训练元数据、保留1280元信息后，**708个模型张量的键、dtype、
数值全部逐项一致**。两个文件的SHA不同是序列化内容变化，不是换了模型或重新训练。

内部构建尝试 `xh-detector:p40-full-s1280-frozen0536-v1` 因原始权重依赖检查失败而作废，
不可提交。旧trial镜像、旧正式v1镜像也都不属于本次候选。推送脚本只接受上述final的
完整Image ID。

## 2. 封装对齐与验收

1. 旧部署适配器即使只有恒等视图也会追加一次TTA NMS，不等价于本次离线候选生成器。
   新配置显式启用 `model.inference_adapter=shared_offline`，直接复用已测的
   `UltralyticsDetector`（含当前Ultralytics的quantize接口）；旧配置行为不改。
2. 保持低阈值候选及safe融合，再应用 `post_fusion_score_threshold=0.536`，
   不把最终工作点误用为融合前过滤阈值。
3. Linux RTX3090运行**交付目录中的真实官方入口代码**，完整跑Hard与Sentinel各6张
   10000×10000图片。与原始last离线预测过滤后的结果比较：**12/12张、3913个框完全一致**，
   最大坐标差0像素、最大分数差0、类别差0；`result.json`字段/边界校验通过。
4. 入口平均4.439秒/图；首图5.190秒；12图整次冷启动、解码与运行共66.014秒。
   后者不是单图耗时。不是官方机器时延保证。
5. Mac上的Docker在 `--network none` 下通过配置、177个app/权重文件SHA、25类权重加载，
   并完成1280网络输入的CPU空图前向（输出0框）。服务器与镜像342个Ultralytics源码/配置
   文件的合并SHA一致：`5e514b3064da670c74df179c04a1e94929d45b47a0295bbc8909f1f9ddd44cf4`。
6. 必须区分验收边界：本次服务器没有Docker；**GPU验收是交付代码在Linux3090直接运行，
   不是本次镜像在GPU容器内运行**。镜像复用了此前已在官方运行成功的CUDA12.1/torch2.5.1
   运行环境，并单独通过上述断网验收。未声称完成新的GPU容器端到端实测。
7. 47项相关单测通过；新配置验证、实际共享适配器选择、融合后阈值边界都有测试。

## 3. 分数解释与保留风险

| 测试 | 总体Recall | 总体FDR | 本地公式分数 |
|---|---:|---:|---:|
| Normal，4481图 | 91.229% | 3.966% | 不填：没有同口径大图时延 |
| Hard，6图/2158 GT | 88.838% | 3.217% | 84.965 |
| Sentinel，6图/1969 GT | 91.333% | 4.840% | 85.278 |

以上为用户明确允许的**训练同源诊断**：Normal全部4481图已见，Hard、Sentinel各600个
源图也已见。它们不替代原无泄漏CV3，不证明隐藏域可得85分，更不保证官方门槛或排名。
新的包装与这组预测逐框一致，分数没有通过封装或阈值调整被改写。

之前Background-100MP仍是10个误检，相比旧S1024部署点2个有所退化；尚未通过图像
复核将这些误检排除。这里继续保留风险，不因同源诊断较高就把背景退化改为通过。

## 4. 用户在Mac上执行的流程

1. 打开比赛网站“提交评测”，查看本次正式tag与剩余次数。点击“生成”，在**Mac终端**
   执行本次临时 `docker login`，看到 `Login Succeeded`。不要把登录凭证提交进Git或聊天。
2. 同一Mac终端执行：

```bash
cd '/Users/suzuku/Documents/揭榜挂帅-小样本遥感卫星图像/xh-202625'
bash scripts/push_p40_submission.sh
```

3. 脚本先核对Image ID、linux/amd64、权重标签。然后按提示输入**网页当前tag**，例如
   网页显示 `v2.0` 才输入 `v2.0`；不要输入 `trial-v2.0`，不要沿用旧终端的`$TARGET`。
4. 核对脚本显示的完整地址与网页一致：
   `competition-registry.cn-beijing.cr.aliyuncs.com/competition/team612528:<网页tag>`。
   输入 `PUSH` 开始推送。脚本不会自动点官网提交。
5. 等待push正常结束、显示digest及脚本成功提示。若报告tag已存在且不可覆盖，停止，
   核对网页；不自动尝试下一个tag，也不手改为staging等自定义名称。
6. 回网站，确认仍是同一tag/镜像地址后点击“提交评测”。可填备注：
   `P40 full, YOLO26-s, imgsz1280, safe1024/256, post-thr0.536, weight b0df7981f6ad`。
7. 到“我的提交记录”确认任务已受理，等待结果。按平台显示扣减正式机会；不要连续重复点。
   把三类Recall/FDR、平均时延、综合分数和提交tag一起记录。

只想核对镜像而不上传：`bash scripts/push_p40_submission.sh --check`。

## 5. 索引

- 配置：[progressive40_full_s1280_frozen0536_v1.json](../../submission/docker/configs/progressive40_full_s1280_frozen0536_v1.json)
- 推送工具：[push_p40_submission.sh](../../scripts/push_p40_submission.sh)
- 入口：[competition.py](../../src/rsdet/submission/competition.py)
- 导出：[sanitize_yolo_checkpoint.py](../../scripts/sanitize_yolo_checkpoint.py)
- 逐框验收：[validate_progressive_submission_parity.py](../../scripts/validate_progressive_submission_parity.py)
- 构建目录：`dist/p40-full-s1280-frozen0536-final/`，manifest SHA
  `1627addf575858ba39c879fc7f565d10d2cde98f548fdeecb7edb1cd6793312d`。
- 依赖复用构建：`submission/docker/Dockerfile.overlay`，基础Image ID
  `sha256:8574d833876f4fee0556063c10788cf6a6ed335a62184dee0e818c39e57a1a52`。
- 本地证据：`outputs/P40-DEPLOYMENT-PREFLIGHT-20260903/`：净化权重、sanitization.json、
  image_static.log、container_cpu_forward.log、parity/parity_summary.json及result.json。
- 同源诊断：`outputs/SCALEROUTE-PLAN15-PROGRESSIVE40-FULL-SEEN-DIAGNOSTIC-V1/`。
- 主报告：[方案15执行记录](../experiments/IMPROVEMENT_PLAN15_SCALEROUTE_EXECUTION_20260903.md)。
