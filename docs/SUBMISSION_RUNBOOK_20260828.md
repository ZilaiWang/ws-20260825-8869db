# 初赛线上测评交付与 Docker 提交操作手册

更新日期：2026-08-28
状态：预测评交付链已实现；正式最终模型尚需单独冻结

本手册依据下载目录中的三份官方材料整理，解析时的文件 SHA256 如下；后续若官方重新
发布同名文件，应先比较 SHA，再判断是否需要更新合同实现：

| 官方材料 | SHA256 |
|---|---|
| `目标检测算法Docker封装说明.docx` | `be4badd36a0163416c26c77cda1f412bff6131d1ee5e595025afdd781669cc3c` |
| `赛事评测管理系统参赛队伍使用手册.docx` | `aa5bb6226d20c747942006b995a1130cad1efb405f74bc8a7beb707d34204306` |
| `Dockerfile` | `8447640afc56ff0f7c3077657e535f827ca4d7a9508acd800172a6ee32f4b583` |

## 1. 时间与提交物

| 阶段 | 时间 | 次数 | 本阶段目的 |
|---|---|---:|---|
| 预测评 | 2026-08-27 12:00—08-29 17:00 | 不限 | 只验证 Docker 封装、推送和平台运行；分数没有参考价值 |
| 正式测评 | 2026-08-30 12:00—09-05 17:00 | 最多 5 次 | 取最高分；首次提交会被优先测评 |
| 源码与报告 | 截止 2026-09-05 17:00 | 最新成功上传覆盖旧版 | 上传 ZIP，不放权重和训练图，页面上限 500MB |

当前首要目标不是继续调精度，而是在 **08-29 17:00 前完成一次预测评**，把镜像
架构、GPU、入口参数、输入发现、`result.json` 和 ACR 提交流程全部打通。

## 2. 本仓库交付结构

| 内容 | 位置 | 作用 |
|---|---|---|
| 官方合同实现 | `src/rsdet/submission/competition.py` | GPU 检查、一次加载模型、第一层图片发现、逐图推理、时间戳、结果校验与原子写出 |
| 容器入口 | `submission/docker/app/main.py` | 接收平台追加的 `--input` / `--output` |
| 冻结配置 | `submission/docker/config.json` | 权重 SHA、输入尺寸、切片、融合和阈值 |
| Docker 模板 | `submission/docker/Dockerfile` | Linux/amd64 + CUDA 12.1 + micromamba |
| 环境导出 | `scripts/export_submission_environment.sh` | 在 Linux x86_64 已验证环境中重新导出 `environment.yml` |
| 交付目录生成 | `scripts/build_submission_delivery.py` | 复制运行所需源码与权重，核验 SHA，生成构建清单 |
| 结果校验 | `scripts/validate_submission_result.py` | push 前检查 `result.json` 全字段、类别、坐标和范围 |
| 源码报告打包 | `scripts/package_source_report.py` | 打包 Git 已跟踪和未忽略的新源码/文档及指定报告；排除权重、图像、缓存和 outputs |

## 3. 预测评模型与正式模型必须分开

当前模板默认使用正式 CV3 血缘的 M1 fold0 fixed-epoch-160 `last.pt`：

```text
artifacts/M1-CV3-OOF/last.pt/fold0_last.pt
SHA256 d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d
```

它足以验证预测评工程闭环，并已有 RTX 3090 合成 10K 的 p95 4.6166 秒证据。但它只在
formal CV3 的两个训练折上训练，不是“使用全部训练数据得到的最终模型”。预测评成功
不能把它自动提升为正式 v1.0。正式镜像冻结前至少重新确认：

1. 最终使用 M1、Y5 或其他已准入模型；
2. 是否已经在允许的全部训练数据上按冻结协议训练；
3. 权重 SHA 与 `config.json` 一致；
4. 正式工作点来自官方匹配口径，不能照搬探索性阈值；
5. 在正式候选环境完成一次离线全流程复测。

## 4. 在 Linux x86_64 准备环境

官方要求 `environment.yml` 从实际跑通的 Linux x86_64 Conda 环境自动导出。仓库中的
版本是可构建的冻结 bootstrap；正式提交前必须在目标 Linux 环境执行：

```bash
cd /path/to/xh-202625
bash scripts/export_submission_environment.sh detector submission/docker/environment.yml
```

随后核对文件中没有 `prefix:`、`win-64`、`pywin32` 或本机绝对路径。每次修改依赖后
都重新导出，不在导出结果上手改包版本。

## 5. 生成独立 Docker 交付目录

在仓库根目录执行：

```bash
python scripts/build_submission_delivery.py \
  --weights artifacts/M1-CV3-OOF/last.pt/fold0_last.pt \
  --expected-sha256 d403ca0d5f760f8d7271af33c467426a3bd8ce8095b6d13e5dc63a6d8e19501d \
  --output dist/detector-docker-delivery \
  --force
```

产物中的 `BUILD_MANIFEST.json` 记录 commit、权重来源、权重 SHA 和每个构建文件。
进入目录后不再依赖仓库其他位置：

```bash
cd dist/detector-docker-delivery
find . -maxdepth 3 -type f | sort
```

必须看到 `Dockerfile`、`environment.yml`、`app/main.py`、`app/config.json`、
`app/rsdet/`、`models/model.pt` 和 `BUILD_MANIFEST.json`。

## 6. 构建和本地 GPU 验收

必须在 Linux x86_64、WSL2 或能构建 amd64 且支持 NVIDIA Container Toolkit 的环境中：

```bash
docker build --platform linux/amd64 -t xh-detector:trial-20260828 .
docker image inspect xh-detector:trial-20260828 \
  --format '{{.Architecture}}|{{.Os}}'
```

第二条必须输出 `amd64|linux`。然后准备至少一张真实赛事格式图片：

```bash
mkdir -p test-input test-output
cp /path/to/one-real-image.jpg test-input/

docker run --rm \
  --gpus '"device=0"' \
  --network none \
  -v "$PWD/test-input:/input:ro" \
  -v "$PWD/test-output:/output" \
  xh-detector:trial-20260828 \
  --input /input \
  --output /output

PYTHONPATH=app python /path/to/xh-202625/scripts/validate_submission_result.py \
  test-output/result.json
```

还要人工抽查：输入每张图恰有一个 `images` 条目；`image_id` 为文件主名；
`category_id` 为 0—24；`category_name` 与 25 类表一致；bbox 为原图像素 xyxy；
无目标时 `objects=[]`；时间戳位于每张图片记录内。

## 7. 平台预测评提交顺序

1. 打开 `http://39.96.3.74/`，使用队伍账号登录；首次登录先修改初始密码。
2. 进入“提交评测”，只以页面当次显示的 tag 和完整镜像地址为准。
3. 在页面生成临时 ACR `docker login` 命令；凭证不可写入仓库、文档或聊天记录。
4. 本机执行页面生成的 `docker login`。
5. 执行 `docker images`，确定源镜像是 `xh-detector:trial-20260828`。
6. 复制页面的 `docker tag` 命令，只替换源镜像名，不改目标仓库或 tag。
7. 执行页面给出的 `docker push`，等待完整成功。
8. 回到页面核对镜像地址/tag 后点击“提交评测”；同一个 tag 不能重复提交。
9. 到“我的提交记录”观察拉取、运行、评分或失败原因。
10. 预测评成功后，在官方腾讯表登记本队已完成密码修改和一次预测评。

截图显示的 `trial-v4.0` 只是截图当时页面自动生成的 tag，不可硬编码。每次都复制页面
当前命令。平台说明中的“禁止 docker pull”意味着提交时以已经本地构建/加载的镜像为源，
不要把目标仓库地址误当成供本地拉取的公开镜像。

## 8. 源码与报告 ZIP

最终报告准备好后执行：

```bash
python scripts/package_source_report.py \
  --report /absolute/path/to/final-report.pdf \
  --output dist/XH-202625-source-and-report.zip
```

脚本只收入 Git 已跟踪或未忽略的新运行源码、配置、测试、正式划分/分组、Docker 模板
和交付说明；内部实验报告不会批量进入 ZIP，最终报告只由显式 `--report` 加入。脚本
拒绝常见权重、训练图、缓存和密钥后缀；ZIP 内 `BUNDLE_MANIFEST.json` 记录完整文件清单
和 SHA。上传前再执行：

```bash
unzip -t dist/XH-202625-source-and-report.zip
du -h dist/XH-202625-source-and-report.zip
```

要求：ZIP 可完整解压、低于 500MB，不包含 `.pt/.pth/.onnx/.engine`、训练图片、
服务器密码、ACR 临时凭证、`.env` 或 SSH 私钥。平台只保留最后一次成功上传的 ZIP，
且上传后不能下载，因此本地保留相同文件及 SHA256。

## 9. 五次正式提交的使用原则

正式 v1.0 先提交已经完整离线验收的安全链，不把首次正式机会用于现场排错。其余次数只
用于已有离线证据的单变量升级，并为每次保存：Git commit、权重 SHA、配置 SHA、镜像
digest、提交 tag、平台结果和回退点。由于取最高分，不需要覆盖旧镜像；每次用页面自动
生成的新 tag。

## 10. 提交前最终门禁

- [ ] 预测评在 08-29 17:00 前至少成功一次；
- [ ] 镜像 `amd64|linux`，GPU 检查通过且未回退 CPU；
- [ ] `--network none` 本地运行成功，证明推理不依赖联网下载；
- [ ] 模型只加载一次，输入只读第一层四种允许图片后缀；
- [ ] `/output/result.json` 通过严格校验并有人工作图抽查；
- [ ] `config.json` 的权重 SHA、模型身份和阈值与本轮提交登记一致；
- [ ] 正式提交前已用 Linux x86_64 实际环境重新导出 `environment.yml`；
- [ ] 源码报告 ZIP 无权重、训练图和敏感信息，低于 500MB；
- [ ] ACR 登录命令、账号密码从未进入 Git、ZIP 或日志包；
- [ ] 每次使用网页当前 tag，push 成功后才点击提交。
