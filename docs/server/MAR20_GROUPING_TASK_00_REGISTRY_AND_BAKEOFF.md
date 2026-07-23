# MAR20-GROUPING-TASK-00：registry、背景视图与 DINOv2 Round-A bake-off

## 0. 任务目标与边界

本任务是 MAR20 来源/机场代理分组的第一批服务器实验，只完成 MG00 和 MG01：

1. 核验竞赛中的 3,073 张 `MAR20_*` 与原始 MAR20 3,842 张图像的一一映射；
2. 建立 target 3,073 + bridge 769 的可追溯 registry；
3. 审计飞机背景隔离、inpaint 和纯背景 tile；
4. 构建 360 对盲化人工校准包；
5. 在校准节点上提取冻结 DINOv2-B/14 的 layer 9/10/11、CLS/mean/signed-GeM 描述子；
6. 人工复核回传后，编译 Round-A 结果，为后续 VLAD/旋转和全量召回提供依据。

本任务**不恢复机场真值、不生成 CV3、不训练任何模型、不修改 P03～P07 产物**。必须遵守：

- MAR20 原始 `train.txt/test.txt` 只记为 `official_side`，不得解释为机场 ID；
- DINO 相似度只能用于候选召回，不能生成 strict 同源边或自动 union；
- `likely_same_airport` 不得 union；
- bridge 769 只用于连通性诊断，不得进入竞赛模型训练；
- 人工盲评不得由服务器 AI 代填，不得在盲评前打开 `blind_card_mapping.csv` 给评审者；
- Round-A 即使通过，也只是 `provisional_until_vlad`，不能直接进入正式分组。

## 1. 冻结路径

```text
repo                 /workspace/xh-202625
competition data     /workspace/data
MAR20 original       /workspace/inputs/MAR20
P04 asset lock       /workspace/p04-assets/ASSET_LOCK.json
venv                 /workspace/venvs/mar20-group-cu121
result root          /workspace/results/MAR20-GROUPING-TASK-00
cache root           /workspace/mar20-group-cache
registry             $ROOT/registry
view audit           $ROOT/view-audit
blind review         $ROOT/calibration-review
DINO cache           $CACHE/dinov2b-calibration-round-a-v1
manual return input  /workspace/inputs/MAR20-GROUPING-TASK-00
```

原始 MAR20 必需树：

```text
/workspace/inputs/MAR20/
├── JPEGImages/                         # 3842 张 1.jpg ...
├── Annotations/Horizontal Bounding Boxes/  # 3842 个 XML
└── ImageSets/Main/
    ├── train.txt                       # 1331 ID
    └── test.txt                        # 2511 ID
```

`Oriented Bounding Boxes/` 不参与本任务，可以存在但不得替代 HBB。若完整数据已在服务器其他位置，可把 `MAR20` 变量改成其真实绝对路径，并在 `system_preflight.txt` 记录；禁止重新下载不同版本后混用。

## 2. 状态语义

本任务允许三种正常状态和若干失败状态：

| 状态 | 含义 | 是否故障 |
|---|---|---:|
| `waiting_for_mar20_input` | 原始 MAR20 必需树不存在 | 否 |
| `waiting_for_manual_reviews` | registry、视图包、盲评包、DINO cache 已完成，等待两份人工 CSV | 否 |
| `complete_round_a` | 人工门禁和 descriptor Round-A 已编译 | 否 |
| `complete_round_a_no_admission` | 技术成功，但真实正例不足或 held-out recall 未达门槛 | 否，科学非准入 |
| `failed_*` | SHA、输入映射、缓存完整性、重复一致性或环境失败 | 是 |

不得把正常等待分支报告为代码失败，也不得为了“跑完”伪造人工输入。

## 3. 代码与环境预检

```bash
cd /workspace/xh-202625
set -o pipefail
ROOT=/workspace/results/MAR20-GROUPING-TASK-00
CACHE=/workspace/mar20-group-cache
MAR20=/workspace/inputs/MAR20
DATA=/workspace/data
ASSETS=/workspace/p04-assets/ASSET_LOCK.json
MANUAL=/workspace/inputs/MAR20-GROUPING-TASK-00
VENV=/workspace/venvs/mar20-group-cu121
mkdir -p "$ROOT/logs" "$CACHE" "$MANUAL"

sha256sum -c docs/server/MAR20_GROUPING_TASK_00_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"
```

先检查输入，不存在则正常停止：

```bash
test -d "$MAR20/JPEGImages" \
  && test -d "$MAR20/Annotations/Horizontal Bounding Boxes" \
  && test -f "$MAR20/ImageSets/Main/train.txt" \
  && test -f "$MAR20/ImageSets/Main/test.txt"
```

若任一项不存在：

1. 写 `task_decision.json`，状态为 `waiting_for_mar20_input`、`formal_grouping_admission=false`；
2. 记录缺失路径和上述必需树；
3. 打包 `logs/`、`task_decision.json`、代码 SHA 日志后停止；
4. 不创建空目录冒充输入，不运行后续步骤。

输入存在后，新建独立环境；不得直接使用已经发生版本漂移的 P04 venv：

```bash
if [ ! -x "$VENV/bin/python" ]; then
  python3.10 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip==24.2
  "$VENV/bin/python" -m pip install \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  "$VENV/bin/python" -m pip install -r requirements-mar20-grouping.txt \
    -i https://mirrors.ustc.edu.cn/pypi/web/simple
  "$VENV/bin/python" -m pip install pytest==8.3.5 ruff==0.11.2 \
    -i https://mirrors.ustc.edu.cn/pypi/web/simple
fi
source "$VENV/bin/activate"
export PYTHONPATH=src
export XFORMERS_DISABLED=1

pytest -q tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/pytest-grouping.log"
ruff check \
  src/rsdet/grouping \
  scripts/build_mar20_source_registry.py \
  scripts/audit_mar20_background_views.py \
  scripts/compile_mar20_view_review.py \
  scripts/build_mar20_calibration_review.py \
  scripts/compile_mar20_calibration_review.py \
  scripts/extract_mar20_place_features.py \
  scripts/analyze_mar20_descriptor_bakeoff.py \
  scripts/check_mar20_grouping_environment.py \
  tests/test_mar20_grouping_batch_a.py \
  2>&1 | tee "$ROOT/logs/ruff-grouping.log"

python scripts/check_mar20_grouping_environment.py \
  --asset-lock "$ASSETS" \
  --expected-gpu "NVIDIA GeForce RTX 4080 SUPER" \
  --output "$ROOT/environment_check.json" \
  2>&1 | tee "$ROOT/logs/environment-check.log"

python -m pip freeze > "$ROOT/pip-freeze.txt"
nvidia-smi > "$ROOT/system_preflight.txt"
git rev-parse HEAD >> "$ROOT/system_preflight.txt"
git status --short >> "$ROOT/system_preflight.txt"
df -h /workspace >> "$ROOT/system_preflight.txt"
```

若实际 GPU 是此前经负责人批准的同级 32GB GPU，可只修改任务单调用中的 `--expected-gpu` 为实际完整名称，并记录这一项人工批准；禁止修改代码或放宽其他版本。

## 4. MG00：全量 registry 与映射门禁

```bash
python scripts/build_mar20_source_registry.py \
  --competition-root "$DATA" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/registry" \
  2>&1 | tee "$ROOT/logs/build-registry.log"
```

必须同时满足：

- all=3,842，target=3,073，bridge=769；
- official train=1,331，test=2,511；
- target pixel mismatch=0；
- target annotation class-hist mismatch=0；
- out-of-scope ID=0；
- `registry_summary.json.status=pass`；
- 本项目当前原始数据的确定性锚点：

```text
image_registry.csv       bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d
image_annotations.jsonl  0a350c1d68e63dd134ca5f4b6a1f89c16f9a4d457f34598b72f2c1eface650f4
```

已知原始 MAR20 有 21 个 HBB XML 将 width/height 写为 0；代码应使用真实解码图像尺寸并在 registry 报告 `xml_size_missing_count=21`。这不是允许忽略任意尺寸冲突：非零 XML 尺寸与图像不一致仍须失败。

文件字节 SHA 不同但解码 RGB 像素相同可以记录为重编码；像素不一致绝不能通过。当前本地核验 H0 全像素重复组为 0，这不代表不存在近重复，只代表不存在可自动 union 的全像素 H0。

## 5. MG01-A：背景视图机器审计与人工包

```bash
python scripts/audit_mar20_background_views.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/view-audit" \
  --sample-count 120 \
  --dilation-ratios 0.10,0.15,0.20 \
  --fill-methods blur,local_mean,telea \
  --primary-dilation 0.15 \
  --tile-size 224 \
  --tile-stride 112 \
  --tile-valid-fraction 0.95 \
  --max-tiles 8 \
  --rows-per-sheet 8 \
  2>&1 | tee "$ROOT/logs/view-audit.log"
```

机器门禁要求：输出有限、mask 不覆盖整图、无遮挡像素保持不变、背景 tile 坐标合法、无路径逃逸。此时状态应是 `waiting_for_manual_view_review`，不是最终通过。

人工评审者查看 `view-audit/contact_sheets/`，逐行填写：

```text
view-audit/manual_view_review.csv
```

四个二值列只允许 `0/1`：

- `valid`：该行整体能否用于判断背景隔离质量；
- `aircraft_remnant`：15% mask 后仍有足以识别飞机的主体/明显结构；
- `inpaint_artifact`：填充形状或伪影明显主导图像；
- `background_tile_aircraft`：展示的纯背景 tile 仍含明显飞机主体。

不要把“机场设施仍可见”标为飞机残留；机场设施正是地点描述子需要保留的信息。

## 6. MG01-B：盲化 pair 校准包

复用仓库已有的 `reports/data/near_duplicates_mar20.json` 作为 dHash 候选输入；它只改变候选路由，不提供正标签，其 SHA 已列入代码/输入清单。

```bash
python scripts/build_mar20_calibration_review.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --output-dir "$ROOT/calibration-review" \
  --near-duplicate-json reports/data/near_duplicates_mar20.json \
  --pair-count 360 \
  --duplicate-fraction 0.08 \
  --dilation-ratio 0.15 \
  --fill-method telea \
  --cards-per-sheet 4 \
  2>&1 | tee "$ROOT/logs/build-calibration-review.log"
```

要求：

- 360 个唯一 pair；
- 盲重复约 8%，重复卡间距至少 30；
- 25 类相关候选、跨/同 official side、编号邻近及普通负例均有可追溯 route；
- `calibration_review_summary.json.status=waiting_for_blind_manual_review`；
- 服务器保留 `blind_card_mapping.csv`，盲评阶段不发给评审者打开；
- 回传 `contact_sheets/` 和空白 `manual_calibration_decisions.csv`；
- `blind_calibration_node_uids.txt` 只用于无标签特征提取，不向人工透露 pair route。

人工标签只能使用：

```text
same_frame
geometric_overlap
same_local_site
likely_same_airport
not_same_local_site
different_airport
uncertain
```

`same_local_site` 要求可对齐的跑道、滑行道、建筑、道路或稳定背景布局；“飞机型号相同”“色调相同”“编号接近”均不构成正证据。

## 7. DINOv2 真实 smoke

先用校准 node list 的前 2 个节点进行真实 GPU smoke，禁止使用 `--encoder mock`：

```bash
python scripts/extract_mar20_place_features.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$ROOT/dino-smoke" \
  --encoder dinov2_vitb14 \
  --scope target_only \
  --node-list "$ROOT/calibration-review/blind_calibration_node_uids.txt" \
  --max-nodes 2 \
  --view-types original,masked_inpaint,background_tiles \
  --rotations 0 \
  --input-size 518 \
  --dilation-ratio 0.15 \
  --fill-method telea \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --batch-size 4 \
  --shard-size 64 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --device cuda \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/dino-smoke.log"
```

必须得到 15 个特征：每层 `CLS`、`mean`、signed-GeM p=2/3/4，均无 NaN/Inf。DINO repo commit、权重 SHA 与 P04 asset lock 必须匹配。记录实际峰值 VRAM；smoke 结果不参与 descriptor 选择。

## 8. 无标签 Round-A cache

人工复核尚未完成时允许先提取特征，因为本节不读取人工标签。Round-A 固定 0°，不提前跑四旋转或 VLAD：

```bash
python scripts/extract_mar20_place_features.py \
  --registry "$ROOT/registry/image_registry.csv" \
  --annotations "$ROOT/registry/image_annotations.jsonl" \
  --mar20-root "$MAR20" \
  --asset-lock "$ASSETS" \
  --output-dir "$CACHE/dinov2b-calibration-round-a-v1" \
  --encoder dinov2_vitb14 \
  --scope target_only \
  --node-list "$ROOT/calibration-review/blind_calibration_node_uids.txt" \
  --view-types original,masked_inpaint,background_tiles \
  --rotations 0 \
  --input-size 518 \
  --dilation-ratio 0.15 \
  --fill-method telea \
  --tile-size 224 \
  --tile-stride 112 \
  --tile-valid-fraction 0.95 \
  --max-tiles 8 \
  --layers 9,10,11 \
  --gem-powers 2,3,4 \
  --batch-size 16 \
  --shard-size 128 \
  --compute-dtype float16 \
  --storage-dtype float16 \
  --device cuda \
  --seed 202625 \
  2>&1 | tee "$ROOT/logs/extract-round-a.log"
```

若首次真实 OOM，唯一允许的变更是 batch 16→8→4；保留失败日志，其他参数不变。batch 是缓存指纹的一部分，因此重试必须写入带 `-b8` 或 `-b4` 后缀的新 cache，禁止和原 cache 混写；后续命令统一改用实际成功的唯一 cache 路径。正式成功后完全相同的命令再运行一次，必须 `computed_rows=0`、全部 shard 跳过、index SHA 不变。缓存留在服务器，不打包 `.npz`。

完成本节后写 `task_decision.json`：

```json
{
  "status": "waiting_for_manual_reviews",
  "formal_grouping_admission": false,
  "descriptor_selection_admission": false,
  "next_inputs": [
    "manual_view_review.csv",
    "manual_calibration_decisions.csv"
  ]
}
```

然后打包 Phase-A 回传包，内容见第 11 节，正常停止。

## 9. 人工回传合同

人工完成后，把两份文件上传至：

```text
/workspace/inputs/MAR20-GROUPING-TASK-00/manual_view_review.csv
/workspace/inputs/MAR20-GROUPING-TASK-00/manual_calibration_decisions.csv
```

服务器 AI 必须确认文件存在且非空，不能根据图片自行填写。原空白模板 SHA、回传文件 SHA 和评审日期都写入日志。若未齐，保持 `waiting_for_manual_reviews`，不得运行第 10 节。

## 10. Phase-B：编译人工门禁与 descriptor bake-off

### 10.1 视图人工门禁

```bash
python scripts/compile_mar20_view_review.py \
  --view-audit "$ROOT/view-audit/view_audit.csv" \
  --review "$MANUAL/manual_view_review.csv" \
  --output "$ROOT/view-review-decision.json" \
  --minimum-valid-rate 0.90 \
  --maximum-aircraft-remnant-rate 0.05 \
  --maximum-inpaint-artifact-rate 0.10 \
  --maximum-background-tile-aircraft 0 \
  2>&1 | tee "$ROOT/logs/compile-view-review.log"
```

失败是科学门禁失败：保留产物并停止 descriptor 准入，不改阈值重跑。可另立 protocol amendment 调整 mask 后重做视图审计。

### 10.2 pair 标签与盲重复门禁

```bash
python scripts/compile_mar20_calibration_review.py \
  --mapping "$ROOT/calibration-review/blind_card_mapping.csv" \
  --decisions "$MANUAL/manual_calibration_decisions.csv" \
  --output-dir "$ROOT/calibration-compiled" \
  --minimum-positive-pairs 30 \
  --minimum-repeat-agreement 0.90 \
  2>&1 | tee "$ROOT/logs/compile-calibration-review.log"
```

重复卡 label 冲突或一致率低于 0.90 是人工质量失败，必须回查冲突卡，不能自动采用多数票。真实 strict positive 少于 30 时允许继续生成探索报告，但 `formal_threshold_admission=false`。

### 10.3 Round-A 分析

```bash
set +e
python scripts/analyze_mar20_descriptor_bakeoff.py \
  --cache-dir "$CACHE/dinov2b-calibration-round-a-v1/cache" \
  --calibration-pairs "$ROOT/calibration-compiled/calibration_pairs.csv" \
  --output-dir "$ROOT/descriptor-bakeoff-round-a" \
  --k-values 20,50,100 \
  --minimum-heldout-positive-directions 10 \
  --heldout-recall-target 0.95 \
  2>&1 | tee "$ROOT/logs/analyze-round-a.log"
BAKEOFF_EXIT=${PIPESTATUS[0]}
set -e
test "$BAKEOFF_EXIT" -eq 0 -o "$BAKEOFF_EXIT" -eq 2
```

退出码 2 表示真实证据/recall 不足，不是代码崩溃。报告必须同时解释：

- held-out strict positive recall@20/50/100；
- hard negative top-k rate与 AUC；
- original/masked/background 三种视图；
- original→masked 邻居 Jaccard；
- `FI=sim_original-sim_masked` 的正/负例分布；
- 计算维度和缓存成本。

只有“视图人工门禁 pass + 盲重复门禁 pass + strict positive 足够 + held-out recall 达标”时状态为 `complete_round_a`；否则为 `complete_round_a_no_admission`。两者都必须保持：

```text
formal_grouping_admission=false
selection_is_provisional_until_vlad=true
```

Round-A 通过后下一步是实现/运行 Round-B VLAD+四旋转，不是直接构建机场组。

## 11. 回传与保留

### Phase-A 回传

应包含：

- `registry/` 全部小型产物；
- `view-audit/` 的 CSV、JSON、contact sheets 和空白 review CSV；
- `calibration-review/` 的 contact sheets、空白 decisions、summary、node list；
- **不包含** `blind_card_mapping.csv`，它留在服务器封存至盲评完成；
- DINO smoke 全部小型产物；
- Round-A cache 的 meta/index/sidecar、extraction summary，不含 NPZ；
- `logs/`、`system_preflight.txt`、`pip-freeze.txt`、`task_decision.json`。

### Phase-B 回传

在 Phase-A 基础上增加：

- 两份原始人工决定及 SHA；
- 解盲后的 `blind_card_mapping.csv`；
- `view-review-decision.json`；
- `calibration-compiled/`；
- `descriptor-bakeoff-round-a/`；
- 最终 `task_decision.json` 和全部新增日志。

服务器保留：

```text
/workspace/mar20-group-cache/dinov2b-calibration-round-a-v1
/workspace/results/MAR20-GROUPING-TASK-00
/workspace/inputs/MAR20
```

不得删除或覆盖 P03～P07 的 cache/checkpoint/结果。

## 12. 最终执行回报格式

请逐项报告：

1. 状态：waiting/complete/科学非准入/技术失败；
2. Git commit、dirty 项数、代码 SHA 门禁；
3. GPU、driver、Python、torch/CUDA、六个固定 Python 包版本；
4. MAR20 输入树、3,842/3,073/769/1,331/2,511 计数和 registry 两个 SHA；
5. target 像素/标注不一致数、21 个 0×0 XML 的处理、H0 重复组数；
6. 120 图视图审计的 mask fraction、背景 tile 数和人工门禁状态；
7. 360 pair、盲重复数/间距、唯一节点数和人工状态；
8. DINO smoke 15 个特征维度、有限性、VRAM；
9. Round-A cache 行数、shard、fingerprint、速度、耗时、resume；
10. 若已人工回传：repeat agreement、正/负/uncertain 数量与 held-out 数；
11. 15 个 descriptor×3 views 的关键指标、FI、入选项及为何仅为 provisional；
12. 回传包路径、大小、SHA；
13. 任何 OOM/重试/唯一参数变更；
14. 明确声明：未把 official side 当机场、未自动 union、bridge 未训练、未修改 P03～P07。
