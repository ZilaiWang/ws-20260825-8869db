# HERA-Guard Sprint20 最终读出执行手册

本手册对应 `reports/experiments/HERA_GUARD_SPRINT20_FINAL_READOUT_EXECUTION_20260905.md`。它只复现实验，不授权打包 Docker 或提交官网。

## 1. 固定要求

- 代码基准必须包含 `ab51106949ad8369a5cccd862fcc19e1739cdeb2`；
- Ultralytics 必须是 `8.4.103`；
- P40 SHA 必须是 `b0df7981f6ad58fe8eca65fb0deef54feed55caf300c2c219ac1ccb3500c8012`；
- Aircraft-D4 SHA 必须是 `5f1b175f8b2c310a3c0583e652f5cf7cfd444d0b0d74d794a9ac59e4537832d5`；
- 不升级环境、不改切片/融合、不覆盖正式权重；
- 所有 OTM/shared 配置默认 `formal_admission=false`。

以下命令假设仓库根目录为当前目录，并设置：

```bash
export PYTHONPATH=src
export PYTHON=/workspace/venvs/p06-cu121/bin/python
```

## 2. 资产审计

```bash
$PYTHON -m sprint20.cli audit \
  --repo . \
  --config configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json \
  --out /path/to/asset_audit.json
```

审计必须确认 OTO/OTM 两套 head 均存在、25 类、源码祖先和关键 blob 未改变、权重 SHA 正确。

## 3. 原生 OTO/OTM 探针

两次探针必须使用相同 COCO、图像根目录和配置：

```bash
$PYTHON -m sprint20.cli probe \
  --config /path/to/frozen_config.json \
  --coco /path/to/ground_truth.json \
  --image-root /path/to/images \
  --head oto --role outer_oof_short \
  --out /path/to/native_oto.json

$PYTHON -m sprint20.cli probe \
  --config /path/to/frozen_config.json \
  --coco /path/to/ground_truth.json \
  --image-root /path/to/images \
  --head otm --role outer_oof_short \
  --out /path/to/native_otm.json
```

每个 head 使用全新模型实例；不得先跑 OTO 再切同一实例。

## 4. 三折聚合与 cross-fit

```bash
$PYTHON scripts/aggregate_sprint20_head_caches.py \
  --fold-roots /path/fold_0 /path/fold_1 /path/fold_2 \
  --oto-output /path/native_oto.json \
  --otm-output /path/native_otm.json

$PYTHON scripts/analyze_sprint20_oof_routing.py \
  --coco /path/to/aggregate_ground_truth.json \
  --oto-cache /path/native_oto.json \
  --otm-cache /path/native_otm.json \
  --groups /path/to/source_split_view.json \
  --bootstrap 3000 \
  --output /path/crossfit_routing.json
```

阈值只由另外两折选择。输出必须保留 `outer_oof_short` 血缘说明、每折分差、255 来源组 bootstrap 的有效次数和缺失类别次数。

这一步只隔离当前折的直接预测模型，不构成严格嵌套独立性：用于选阈值的另外两折模型
训练时接触过当前评估折的来源。若类别范围也是看完三折或 full-seen 后确定，最终结果必须
标为后验选择的开发证据。

## 5. Shared parity

只有原生对照存在方向性时才运行 shared：

```bash
$PYTHON -m sprint20.cli probe \
  --config /path/to/frozen_config.json \
  --coco /path/to/ground_truth.json \
  --image-root /path/to/images \
  --head shared --role full_seen \
  --out /path/shared.json

$PYTHON -m sprint20.cli parity \
  --native /path/native_oto.json --shared /path/shared.json \
  --head oto --out /path/parity_oto.json

$PYTHON -m sprint20.cli parity \
  --native /path/native_otm.json --shared /path/shared.json \
  --head otm --out /path/parity_otm.json
```

两个 parity 必须同时为精确多重集一致。当前全量结果是 OTO 通过、OTM 失败，因此 shared 不准部署；不要添加容差绕过。

## 6. D4 有界早退 AB/BA

```bash
$PYTHON -m sprint20.cli benchmark-d4 \
  --config configs/experiments/p40_aircraft_d4_full_runtime_candidate_v1.json \
  --cache /path/to/frozen_hard_case_cache.json \
  --image-root /path/to/images \
  --out /path/d4_bounded_abba.json \
  --repeats 1
```

必须同时检查 `all_outputs_exact=true` 和总耗时。当前与正式 consistency D4 匹配的 100 张硬例虽然逐框一致，但慢 3.93%，因此保持 `bounded_d4=false`。

## 7. 测试

```bash
$PYTHON -m pytest -q
```

2026-09-05 最终代码的完整服务器结果为 `1231 passed, 3 skipped`。实验代码还需单独通过 scoped Ruff；全仓历史脚本的旧 lint 债务不应混写为 Sprint20 错误。运行全仓测试必须保持仓库规定的 `PYTHONPATH=src`；缺少该环境变量会让3项 CLI 子进程测试因无法导入 `rsdet` 失败，不是模型逻辑失败。

## 8. 当前停止点

```text
保留：frozen P40 path + original Aircraft-D4 only（历史平台未返回 digest/head 字段）
研究保留：OTM 接管 QHS/MS 的方向证据
禁止部署：shared OTM、OTM FSC、bounded D4、双模型 OTO+OTM
禁止：从 full-seen 选阈值、放宽 parity、追加 rescue/融合扫描
```

## 9. 方案21只读证据审计

该步骤不运行模型、不训练、不改政策，也不作部署决定：

```bash
python scripts/build_sprint20_evidence_bundle.py

cd tools/evaluation_evidence_audit
python -m unittest discover -s tests -v
python -m evidence_audit \
  --manifest ../../reports/audits/HERA_SPRINT20_EVIDENCE_20260905/training_and_selection_lineage.json \
  --root ../.. \
  --out /tmp/sprint20_evidence_audit_report.json
```

审计命令预期返回码为 `2`：三个所谓严格独立 fold 声明均存在可追溯的间接数据接触，
并且短 OOF 与成熟 full 训练签名不同。返回码 `2` 是正确发现证据冲突，不是运行失败。
五项最终审计产物固定在 `reports/audits/HERA_SPRINT20_EVIDENCE_20260905/`；工具生成的临时
诊断报告不纳入这五项交付，避免把通用工具输出误写成精度认证。
