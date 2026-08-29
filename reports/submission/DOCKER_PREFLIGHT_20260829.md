# Docker 预测评本地预检记录

日期：2026-08-29
结论：代码合同和 `linux/amd64` 镜像预检通过；同一部署目录已在 Linux x86_64、RTX
3090 上完成真实 CUDA 端到端运行。该服务器没有 Docker runtime，因此“容器进程 +
NVIDIA Container Runtime”这一层仍由官方预测评完成。

## 1. 冻结对象

- 本地镜像：`xh-detector:trial-preflight`
- 镜像 ID：交付目录每次重新物化会改变 Docker `COPY` 层元数据；push 前用
  `docker image inspect xh-detector:trial-preflight` 登记本次实际 ID/digest
- 平台：`amd64|linux`
- 入口：`["python", "/app/main.py"]`
- 镜像大小：`4,351,050,544` bytes
- 部署角色：仅用于预测评工程闭环，不是正式最终模型
- 工作点：`trial_m1_fold0_score_0p051`
- 权重 SHA256：`d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d`

交付目录的 `BUILD_MANIFEST.json` 记录来源 commit、dirty 状态、权重和逐文件 SHA。
RTX 3090 验收完成后已提交运行代码并重新物化交付目录；当前 manifest 为
`source_tree_dirty=false`，权重 SHA 与配置一致。

## 2. 已通过门禁

1. 全仓测试：`663 passed, 5 skipped`。
2. 本次新增模块定向 Ruff：通过；`git diff --check`：通过。
3. Docker 构建：`docker build --platform linux/amd64` 成功。
4. 容器在 `--network none` 下完成 Python/模型合同检查，不依赖运行时下载。
5. 容器环境：PyTorch `2.5.1`、CUDA runtime `12.1`、Ultralytics `8.4.103`、NumPy
   `1.26.4`、OpenCV `4.10.0`。
6. 容器内权重 SHA 与配置一致；模型类别恰为 25 类，顺序从 `HM` 到 `FSC`，与
   `FINE_NAMES` 完全一致。
7. 入口 `--help` 明确要求 `--input` 和 `--output`。
8. 单元测试覆盖第一层图片发现、后缀/排序、重复主名拒绝、官方 JSON 写出、类别名映射、
   bbox 越界拒绝和时间戳。
9. 源码预检 ZIP 可完整解压；常见权重、图片、密钥后缀和凭证模式扫描通过。

## 3. RTX 3090 等价运行验收

测试环境：Ubuntu 22.04、Linux x86_64、NVIDIA GeForce RTX 3090 24GB、driver
`570.124.04`、PyTorch `2.5.1+cu121`、Ultralytics `8.4.103`、NumPy `1.26.4`、
Pillow `10.4.0`、OpenCV `4.10.0`。运行时使用交付目录中的同一 `app/`、同一
`config.json` 和 SHA 为 `d403ca...e19501d` 的同一 M1 权重。

1. 一张 `10000×10000` 合成大图：完成 CUDA 检查、模型加载、切片、批推理、全局融合和
   原子写出，总 wall 约 7 秒，输出 1 个对象；`result.json` 严格校验通过。
2. 三张真实训练格式图片（舰船、飞机、车辆）：三张均完成推理，输出对象数分别为
   2、3、4；严格校验得到 `images=3, objects=9`。
3. 三张图的预测与标注语义吻合：舰船图输出 QHS，飞机图输出 C-130/B-1B，车辆图输出
   FSC；该检查只用于发现通道、类别表或坐标映射错误，不作为泛化精度估计。
4. `10000×10000` 结果 SHA256：
   `0b044ebc3d52de7b79b5cc4c0988b5e557b904cc1f82b691a2c1a12107ec7aa3`。
5. 审计文件位于 `reports/submission/rtx3090_preflight_20260829/`。

服务端运行前后均未修改训练 checkpoint；服务器没有 Docker、Podman、Apptainer 或
Singularity，也没有启动 Docker daemon 所需的 `CAP_SYS_ADMIN`，因此不能在该容器内
嵌套执行 `docker run --gpus`。

## 4. 尚未通过的提交硬门禁

真实算法运行、输出合同和 RTX 3090 时延已通过。当前唯一未在自有环境覆盖的层是：

1. 官方平台能否成功拉取 ACR 镜像；
2. 平台 NVIDIA Container Runtime 能否把 GPU 注入容器；
3. 平台追加 `--input /input --output /output` 后能否完整运行；
4. 平台能否读取 `/output/result.json` 并完成一次预测评记录。

这四项正是预测评的目的。当前镜像已经达到“可以推送预测评”的状态，但预测评成功不
等于正式最终模型冻结；模型身份结论见
`reports/submission/MODEL_AND_WEIGHT_FREEZE_AUDIT_20260829.md`。
