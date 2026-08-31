# HERA-Guard Final：4×3090 正式筛选执行合同

状态：`code_ready / one_gpu_smoke_pass / waiting_for_four_gpu_host`

## 1. 目的与唯一实验矩阵

本任务只回答四个配对问题，不做超参数搜索：

|GPU|阶段 1|阶段 2/配对|
|---:|---|---|
|0|DOTA EXT-G 80 epoch|EXT-G→reviewed patch fold0，8+32 epoch|
|1|DOTA EXT-V 80 epoch|EXT-V→reviewed patch fold0，8+32 epoch|
|2|HAD fold0 branch-only→terminal-FPN|官方初始化→reviewed patch fold0 对照|
|3|HAD fold2 branch-only→terminal-FPN|官方初始化→omit patch fold0 对照|

所有训练固定 Y5-S、1024、seed 20260831、最后 epoch checkpoint。禁止 resume、早停、验证集
选 checkpoint、替换为 Y5-L 或临时改变 batch/增强。资源不足应停止并记录，不能静默改合同。

## 2. 环境与磁盘

- 4 张可见 RTX 3090，每张至少 24GiB；
- 推荐可用磁盘不少于 100GiB，DOTA 全量准备门禁为 60GiB；
- 训练、保存、继续微调必须使用同一 venv；
- 运行前记录 `pip freeze`、驱动、CUDA、GPU UUID、Git commit 与 dirty 清单；
- DOTA 与官方训练资产只从已锁官方地址下载，禁止临时镜像。

先执行：

```bash
REPO=/workspace/xh-202625 \
PYTHON_BIN=/workspace/venvs/hera-final-cu121/bin/python \
ASSET_ROOT=/workspace/external-assets/DOTA-v1.0 \
OUT=/workspace/results/HERA-GUARD-FINAL-DOTA-PREP-V1 \
bash scripts/server/run_hera_guard_final_prepare_dota.sh
```

任务产生 96 张确定性审计卡并进入 `waiting_for_agent_visual_review`。代理逐图检查类别、HBB、
边界截断和巨型场景框后写 `visual_decision.json`；再次运行同一命令会逐阶段 resume，最终状态
必须为 `ready_for_external_pretraining`。

## 3. 4 GPU 启动

以下变量必须使用真实绝对路径和对应 SHA，不得从目录中猜测：

```bash
export REPO=/workspace/xh-202625
export PYTHON_BIN=/workspace/venvs/hera-final-cu121/bin/python
export OUT=/workspace/results/HERA-GUARD-FINAL-4GPU-SCREEN-V1
export DOTA_DATASET=/workspace/external-assets/DOTA-v1.0/yolo-dota-v1
export DOTA_PREP_STATUS=/workspace/results/HERA-GUARD-FINAL-DOTA-PREP-V1/status.txt
export Y5_INITIAL=/workspace/assets/yolo26s.pt
export Y5_INITIAL_SHA256=<64-hex>
export CV3_MANIFEST=/workspace/assets/cv3_formal_manifest.json
export DATA_ROOT=/workspace/data/official
export CONFIRMED_MISSING=/workspace/assets/confirmed_missing_v1.json
export IGNORED_AMBIGUOUS=/workspace/assets/ignored_ambiguous_v1.json
export TEACHER_CACHE=/workspace/assets/teacher_cache.npz
export BASE_WEIGHT_0=/workspace/assets/y5-fold0.pt
export BASE_WEIGHT_2=/workspace/assets/y5-fold2.pt
export BASE_SHA_0=<64-hex>
export BASE_SHA_2=<64-hex>

bash scripts/server/run_hera_guard_final_4gpu_screen.sh \
  >"${OUT}.driver.log" 2>&1
```

驱动会先生成两份 cache-isolated DOTA role view，避免 EXT-G/EXT-V 并发写同一个
`labels/train.cache`。出现已有 checkpoint 但没有完成审计时固定退出，不自动 resume。

## 4. 训练后固定评测

候选不得直接全三折重训。先对 fold0（HAD 同时看 fold2）运行
`scripts/server/run_hera_guard_final_candidate_eval.sh`：

- Normal-CV3；
- Hard10K；
- source-disjoint Sentinel；
- 0.001 阈值网格只用于形成冻结 frontier，不用于反复调候选；
- 两个未训练 fold 完全复用基线预测；
- `decision.json` 是唯一准入字段。

准入后才补其余折；外部初始化最多保留 EXT-G/EXT-V 中一个，HAD 最多保留
branch-only/terminal-FPN 中一个。最终只训练一个 full 配方，并保持单 Y5-S 部署。

## 5. 必须回传的小型证据

- DOTA `ASSET_LOCK.json`、转换/切片/role/audit、视觉决定；
- 每个训练的 `training_contract.json`、`training_result.json`、参数迁移审计；
- partial-label 数据集审计；
- HAD adapter 的 base/teacher/cache SHA 与零残差等价门禁；
- Normal/Hard/Sentinel frontiers、candidate fold replacement audit、`decision.json`；
- 全部代码与结果 SHA256；
- checkpoint 保留服务器，只有最终准入权重另行下载。

任何一个来源、环境、SHA、fold replacement 或风险门禁不成立时，结果只能作为诊断，不能
进入正式镜像。
