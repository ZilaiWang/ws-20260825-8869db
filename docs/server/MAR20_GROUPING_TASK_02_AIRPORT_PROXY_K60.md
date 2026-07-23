# MAR20-GROUPING-TASK-02：机场级代理 K=60 最终收尾

## 0. 任务边界

前一版 `mar20_final_group_assignments.csv` 有 2,882 个组，实际是高置信局部同源连通分量，不是机场级分组，禁止直接交给 B。

本任务只做一次确定性收尾：复用服务器现有的两路入选 masked-VLAD PCA512 缓存，在完整 MAR20 3,842 张图上形成 60 个机场代理视觉域；strict local-scene 组件先折叠成不可拆原子。K=60 来自 MAR20 公开的 60 个来源机场，只作为结构先验，不宣称恢复了机场真值或机场名称。

不新增模型，不重新提特征，不重新人工审图，不修改已有缓存。

## 1. 输入准备

将本地文件：

```text
outputs/MAR20-FINAL-GROUPING-v1/mar20_group_assignments_all.csv
```

上传为：

```text
/workspace/inputs/MAR20-AIRPORT-PROXY-K60-v1/mar20_group_assignments_all.csv
```

若服务器根目录实际为 `/root/autodl-tmp/workspace`，全文的 `/workspace` 前缀统一替换一次，禁止混用。

```bash
cd /workspace/xh-202625
set -euo pipefail

ROOT=/workspace/results/MAR20-AIRPORT-PROXY-K60-v1
REPEAT=/workspace/results/MAR20-AIRPORT-PROXY-K60-v1-repeat
PREV=/workspace/results/MAR20-GROUPING-TASK-00
B2=/workspace/results/MAR20-GROUPING-TASK-00B1
INPUT=/workspace/inputs/MAR20-AIRPORT-PROXY-K60-v1
VENV=/workspace/venvs/mar20-group-cu121

mkdir -p "$ROOT/logs" "$REPEAT" "$INPUT"
source "$VENV/bin/activate"
export PYTHONPATH=src
```

## 2. 代码、环境和输入门禁

```bash
sha256sum -c docs/server/MAR20_GROUPING_TASK_02_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

pytest -q \
  tests/test_mar20_airport_proxy.py \
  tests/test_mar20_grouping_task01.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

ruff check \
  src/rsdet/grouping/airport_proxy.py \
  scripts/compile_mar20_airport_proxy.py \
  tests/test_mar20_airport_proxy.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import numpy, sklearn
assert numpy.__version__ == '1.26.4'
assert sklearn.__version__ == '1.5.2'
print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)
PY

printf '%s  %s\n' \
  bcdc4b697532be0899b84abb7f218a0fc2fec2c08e5aaeb3460336eea7b7201d \
  "$PREV/registry/image_registry.csv" \
  d46205a27cf55d74c2cdcf56b9d0c4c98cb584b6261c3289731fc24687c1444f \
  "$B2/round-b/round_b_decision.json" \
  e095e52130e3849c2ee4b43be8a90b2d61a73cc2482da5e88c175021a32305e9 \
  "$INPUT/mar20_group_assignments_all.csv" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/input-sha256.log"

test -s /workspace/mar20-group-cache/dinov2b-vlad-pca512-full-v1p2/cache/index.json
test -s /workspace/mar20-group-cache/dinov2b-vlad-pca512-full-v1p2/cache/cache_meta.json
```

上述 scoped pytest 在服务器上不得 skip；本地无 scikit-learn 时出现的单项 skip 不适用于服务器门禁。

## 3. 正式 K=60 编译

```bash
python scripts/compile_mar20_airport_proxy.py \
  --registry "$PREV/registry/image_registry.csv" \
  --local-scene-groups "$INPUT/mar20_group_assignments_all.csv" \
  --routes-json configs/grouping/mar20_00b_round_b_routes_server.json \
  --round-b-decision "$B2/round-b/round_b_decision.json" \
  --output-dir "$ROOT/final" \
  --n-clusters 60 \
  2>&1 | tee "$ROOT/logs/compile.log"
```

## 4. 科学与完整性门禁

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/final-gate.json"
import csv, json
from collections import Counter
from pathlib import Path

p=Path('/workspace/results/MAR20-AIRPORT-PROXY-K60-v1/final')
s=json.loads((p/'airport_proxy_summary.json').read_text())
d=json.loads((p/'task_decision.json').read_text())
with (p/'mar20_airport_proxy_assignments_all.csv').open(newline='') as f:
    all_rows=list(csv.DictReader(f))
with (p/'mar20_airport_proxy_assignments_target.csv').open(newline='') as f:
    target=list(csv.DictReader(f))

assert s['status']=='airport_proxy_k60_ready_for_cv3'
assert s['formal_grouping_admission'] is True
assert s['registry_nodes']==3842 and s['target_nodes']==3073 and s['bridge_nodes']==769
assert s['airport_proxy_groups_all']==60
assert s['strict_component_split_count']==0
assert d['airport_proxy_group_count']==60
assert len(all_rows)==3842 and len(target)==3073
assert len({r['node_uid'] for r in all_rows})==3842
assert len({r['competition_image_id'] for r in target})==3073
assert len({r['group_id'] for r in target})==s['airport_proxy_groups_with_target']
assert not any(not r['group_id'] for r in target)

sizes=Counter(r['group_id'] for r in target)
print(json.dumps({
  'status': s['status'],
  'groups_all': s['airport_proxy_groups_all'],
  'groups_with_target': len(sizes),
  'target_group_min': min(sizes.values()),
  'target_group_median': s['target_group_size_median'],
  'target_group_max': max(sizes.values()),
  'strict_component_split_count': s['strict_component_split_count'],
  'route_partition_agreement': s['route_partition_agreement'],
  'confidence': s['confidence'],
}, indent=2))
PY
```

组规模、两路描述子 ARI 和 margin 只做透明报告，不因“不够漂亮”临时修改 K、路线或算法。

## 5. 确定性复跑

```bash
python scripts/compile_mar20_airport_proxy.py \
  --registry "$PREV/registry/image_registry.csv" \
  --local-scene-groups "$INPUT/mar20_group_assignments_all.csv" \
  --routes-json configs/grouping/mar20_00b_round_b_routes_server.json \
  --round-b-decision "$B2/round-b/round_b_decision.json" \
  --output-dir "$REPEAT/final" \
  --n-clusters 60 \
  > "$ROOT/logs/compile-repeat.log" 2>&1

sha256sum "$ROOT/final/mar20_airport_proxy_assignments_all.csv" \
          "$REPEAT/final/mar20_airport_proxy_assignments_all.csv" \
          "$ROOT/final/mar20_airport_proxy_assignments_target.csv" \
          "$REPEAT/final/mar20_airport_proxy_assignments_target.csv" \
  | tee "$ROOT/logs/determinism-sha256.log"

cmp "$ROOT/final/mar20_airport_proxy_assignments_all.csv" \
    "$REPEAT/final/mar20_airport_proxy_assignments_all.csv"
cmp "$ROOT/final/mar20_airport_proxy_assignments_target.csv" \
    "$REPEAT/final/mar20_airport_proxy_assignments_target.csv"
```

## 6. 回传与给 B 的唯一文件

```bash
cd /workspace/results
tar -czf MAR20-AIRPORT-PROXY-K60-v1-return.tar.gz \
  MAR20-AIRPORT-PROXY-K60-v1/final \
  MAR20-AIRPORT-PROXY-K60-v1/logs
sha256sum MAR20-AIRPORT-PROXY-K60-v1-return.tar.gz
```

回传后，给 B 的唯一主文件是：

```text
final/mar20_airport_proxy_assignments_target.csv
```

B 只使用其中的 `competition_image_id → group_id`。`membership_cosine`、`centroid_margin` 仅用于审计；不得把低 margin 图拆成 singleton。旧 `mar20_final_group_assignments.csv` 不再使用。

最终回报必须包含：状态、60/目标侧组数、目标组大小 min/median/max、strict split 数、三项 ARI、两项 margin 摘要、两次 CSV SHA 是否相同、回传包 SHA。
