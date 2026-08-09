# N1 阶段服务器部署指南（P04-F → P03-F）

日期：2026-08-09
面向：autodl GPU 服务器（Linux）

> 本指南把 `docs/server/P03_FORMAL_CV3_V2_REPLAY.md` 与
> `P04_FORMAL_CV3_V2_REPLAY.md` 两个任务单固化为可直接执行的脚本。
> 服务器上的操作请严格按顺序执行，任何一步失败都不要继续。

## 0. 部署前置（一次性）

服务器路径约定（与任务单一致）：

```text
/workspace/xh-202625          代码（git clone gitee）
/workspace/data               数据集
/workspace/pretrained/        预训练权重
/workspace/venvs/p03-cu121    P03 venv
/workspace/venvs/p04-cu121    P04 venv
/workspace/results            结果根目录
/workspace/p04-cache/         P04 三教师 cache
/workspace/p04-assets/        ASSET_LOCK.json
```

### 0.1 拉取最新代码（必须，含 N0 与 SHA 清单更新）

```bash
git clone https://gitee.com/zilai-wang/xh-202625.git /workspace/xh-202625
cd /workspace/xh-202625 && git pull origin master
# 校验 formal 阶段代码 SHA（必须 74 条全部 OK）
sha256sum -c docs/server/CV3_FORMAL_EXPERIMENT_CODE_SHA256.txt | grep -v OK || true
```

### 0.2 依赖安装

```bash
# P03 venv（P04 venv 类似，用 requirements-p04.txt）
python -m venv /workspace/venvs/p03-cu121
source /workspace/venvs/p03-cu121/bin/activate
pip install -r requirements.txt -r requirements-p03.txt
```

## 1. P04-F 先行（18 个 frozen-feature probe，先验证 cache）

**顺序原因**：P04-F 成本低（18 个线性 probe，15 epoch），先用它验证三个
D4 cache 是否完整、可复用。cache 通过审计后，再跑 P03-F 全微调。

```bash
cd /workspace/xh-202625
source /workspace/venvs/p04-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash docs/server/run_p04_formal.sh
```

`run_p04_formal.sh` 会依次执行：
1. 环境 preflight + pytest（formal 阶段测试）
2. 消费 F00 run-a formal manifest（SHA 校验）
3. 三教师 cache 复用审计（18-run 前硬门禁，任一 mismatch 全停）
4. 18 个 probe（ConvNeXt/DINOv2-B/CleanDIFT × native/PCA384 × 3 折）
5. formal 汇总 + 回传包生成

## 2. P03-F 后行（tight-224 ConvNeXt 三折全微调）

```bash
cd /workspace/xh-202625
source /workspace/venvs/p03-cu121/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash docs/server/run_p03_formal.sh
```

`run_p03_formal.sh` 会依次执行：
1. 环境 preflight + pytest
2. 消费 F00 run-a formal manifest（SHA 校验 + audit 检查）
3. formal 输入独立审计 + 配置冻结 + 环境检查
4. fold 0 smoke（10 项产物检查，分数不入汇总）
5. 三折正式 run（从同一 ImageNet 权重独立初始化）
6. formal 汇总（恰 3 run、TU-160 压力折、pooled OOF 20,933 对象）
7. 回传包生成

## 3. 关键约束（任务单强制）

1. **禁止覆盖**：任何 run 目录已存在且无 `run_summary.json` → 停止，绝不覆盖。
2. **禁止换参数**：不得传入覆盖 epoch/LR/batch/sampler/checkpoint 的额外参数。
3. **P04 cache 任一 mismatch → 整个 18-run 停止**，不得跑剩余教师。
4. **smoke 分数不得进入汇总**。
5. OOM 后保留产物并停止，不允许同任务减小 batch 重试。
6. 服务器不自行宣布 CleanDIFT 入选，只回报数据，由本地 A 决策。

## 4. 回报模板

跑完后把以下内容发给 A：

```
环境: nvidia-smi 快照、python/torch 版本
Git: HEAD SHA、git status 干净
formal SHA: 3 个 SHA（CV3/formal/P0-2）
P04-F: 18/18 run、六组 mean±std、pooled OOF、TU-160、native/PCA384 排名
P03-F: 3/3 run、逐折与 pooled 指标、TU-160、耗时显存
回传包: *.tar.gz SHA256
```

## 5. 回传

```bash
# 服务器执行（脚本已自动生成）
# P04: /workspace/results/P04-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
# P03: /workspace/results/P03-FORMAL-CV3-V2-results-no-checkpoints.tar.gz
# 下载回本地解压到 outputs/ 下，A 验收。
```

## 6. 若遇故障

| 症状 | 处理 |
|---|---|
| `waiting_for_FORMAL_CV3_CROP_TASK_01` | run-a formal 缺失，需先跑 F00 crop 任务，**不能自己生成** |
| pytest FAILED | 停止，回报日志，本地修好再继续 |
| SHA 校验 FAILED | 停止，回报；不要 `--force` |
| GPU OOM | 保留产物，停止，回报日志 |
