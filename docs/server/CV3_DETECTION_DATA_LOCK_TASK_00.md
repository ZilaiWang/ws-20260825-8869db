# CV3-DETECTION-DATA-LOCK-TASK-00：正式检测数据字节锁

## 0. 结论、顺序与边界

本任务在 `FORMAL-CV3-CROP-TASK-01` 之后、M1/M3 任一正式三折训练之前执行。
它不训练模型、不重新划分数据，而是生成唯一、只读、可跨机器复现的检测数据锁：

```text
/workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json
```

锁逐图绑定 4,481 个图像文件及其 4,481 个 YOLO 标签文件的相对路径、字节数与
SHA-256，并同时证明：

1. 正式 CV3 精确覆盖这 4,481 张图，分组与三折不变；
2. P0-2 与 formal crop 的 62,799 行 GT 元数据逐 `crop_id` 等价；
3. 每张图的实际字节 SHA/尺寸等于 P0-2/formal crop 中的来源记录；
4. 20,933 个 YOLO 标签框与 formal crop 的 tight-policy GT 完成一一匹配；
5. 25 个类完整覆盖，坐标最大绝对误差不超过 `5e-6`。

本任务不允许修改图像、标签、manifest、fold、类别、框坐标或既有 F00 产物。
若任一字节或 GT 不一致，停止并回传，不得自动修复。任务只需要 CPU。

## 1. 冻结输入、输出与预期结果

```bash
set -euo pipefail

REPO=/workspace/xh-202625
DATA_ROOT=/workspace/data
F00=/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a
CV3="$REPO/data/splits/cv3_airport_proxy_k60_v2.json"
FORMAL_CROP="$F00/crop/formal_crop_manifest.csv"
SPEC="$REPO/configs/experiments/formal_detection_data_lock.json"
LOCK_ROOT=/workspace/formal-detection-data
LOCK="$LOCK_ROOT/FORMAL_DETECTION_DATA_LOCK.json"
ROOT=/workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00

mkdir -p "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTHONNOUSERSITE=1
```

冻结输入：

| 输入 | SHA256 |
|---|---|
| 正式 CV3 | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| P0-2 crop manifest | `f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e` |
| formal crop manifest | `a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128` |

同一官方数据在本地完成过完整独立构建，预期锁：

| 项目 | 冻结值 |
|---|---|
| 锁文件 SHA256 | `03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a` |
| `lock_fingerprint` | `ef8d3ff216e310e6c4dee6bdabdf20ebb5da983bd473c988b790cfc1bb3b7a18` |
| `inventory_fingerprint` | `e094fdd9181c3aecd59e4385b8cd8616047c7c9e98c3717812139944c9e9a280` |
| 图像 / 标签文件 | `4481 / 4481` |
| 对象数 / 类别数 | `20933 / 25` |
| 图像总字节 | `1225990013` |
| 标签总字节 | `820186` |

任何预期值不一致都属于输入不一致，不得放宽。

## 2. 定位唯一 P0-2 manifest

P0-2 是非 Git 大文件，必须按 SHA 查找，禁止只凭文件名选取：

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/locate-p02.log"
import hashlib
from pathlib import Path

expected = "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"
roots = [
    Path("/workspace/xh-202625/outputs"),
    Path("/workspace/artifacts"),
    Path("/workspace/results"),
    Path("/workspace/inputs"),
]
seen = set()
matches = []
for root in roots:
    if not root.is_dir():
        continue
    for path in root.rglob("crop_manifest.csv"):
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(digest, path)
        if digest == expected:
            matches.append(path)
if not matches:
    raise SystemExit("waiting_for_exact_p02_manifest")
selected = sorted(matches, key=str)[0]
Path("/workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00/p02_path.txt").write_text(
    str(selected) + "\n",
    encoding="utf-8",
)
print("P02_READY", selected, "IDENTICAL_COPIES", len(matches))
PY

P02="$(cat "$ROOT/p02_path.txt")"
export P02
```

找不到精确 SHA 时，状态为 `waiting_for_exact_p02_manifest`，回传日志后停止。

## 3. 前置完整性与代码门禁

优先复用 F00 的 Python 3.10 CPU 环境；若不存在，复用 P05 CPU 环境。两个环境都
必须能够导入 Pillow 和当前工作树的 `rsdet`：

```bash
if test -x /workspace/venvs/formal-cv3-cpu/bin/python; then
  source /workspace/venvs/formal-cv3-cpu/bin/activate
elif test -x /workspace/venvs/p05-cpu/bin/python; then
  source /workspace/venvs/p05-cpu/bin/activate
else
  echo "WAITING_FOR_F00_CPU_ENV" >&2
  exit 2
fi

cd "$REPO"
export PYTHONPATH="$REPO/src"
export PYTHONNOUSERSITE=1

python - <<'PY' 2>&1 | tee "$ROOT/logs/import-environment.log"
from pathlib import Path
import PIL
import rsdet

actual = Path(rsdet.__file__).resolve()
expected = (Path.cwd() / "src/rsdet/__init__.py").resolve()
assert actual == expected, (actual, expected)
print("Pillow", PIL.__version__)
print("RSDET_IMPORT_OK", actual)
PY

test -d "$DATA_ROOT/images"
test -d "$DATA_ROOT/labels"
test -f "$CV3"
test -f "$P02"
test -f "$FORMAL_CROP"
test -f "$SPEC"

printf '%s  %s\n' \
  27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 "$CV3" \
  f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e "$P02" \
  a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128 "$FORMAL_CROP" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/input-sha256.log"

sha256sum -c docs/server/CV3_DETECTION_DATA_LOCK_TASK_00_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

python -m pytest -q tests/test_detection_data_lock.py \
  2>&1 | tee "$ROOT/logs/pytest.log"

python -m ruff check \
  src/rsdet/experiments/detection_data_lock.py \
  scripts/lock_formal_detection_data.py \
  tests/test_detection_data_lock.py \
  2>&1 | tee "$ROOT/logs/ruff.log"
```

期望专项测试 `5 passed` 且 Ruff 全绿。

## 4. 不覆盖门禁与创建

只允许两种起始状态：

- `$LOCK_ROOT` 不存在：新建并创建锁；
- `$LOCK` 已存在且为只读：跳过创建，进入第 5 节重新验证。

目录部分存在、锁缺失，或锁有任意写权限时均停止：

```bash
if test -e "$LOCK"; then
  test -f "$LOCK"
  LOCK_PATH="$LOCK" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["LOCK_PATH"])
if path.stat().st_mode & 0o222:
    raise SystemExit("BLOCKED_WRITABLE_EXISTING_DATA_LOCK")
print("VERIFIED_EXISTING_SKIP_CREATE")
PY
elif test -e "$LOCK_ROOT"; then
  echo "BLOCKED_PARTIAL_DATA_LOCK_ROOT" >&2
  exit 2
else
  mkdir "$LOCK_ROOT"
  python scripts/lock_formal_detection_data.py create \
    --config "$SPEC" \
    --data-root "$DATA_ROOT" \
    --cv3-manifest "$CV3" \
    --p02-manifest "$P02" \
    --formal-crop-manifest "$FORMAL_CROP" \
    --output "$LOCK" \
    2>&1 | tee "$ROOT/logs/create.log"
fi
```

创建使用同目录临时文件、`fsync`、原子改名，拒绝覆盖，并将最终锁设为只读。
锁内容不记录绝对数据路径和时间，因此相同数据在不同机器上必须得到同一字节。

## 5. 全量独立 verify 与确定性门禁

`verify` 会重新读取三个 manifest、重新计算 8,962 个文件的大小和 SHA、重新解析
所有 YOLO 框，并从头构造锁与既有锁逐字段比较：

```bash
python scripts/lock_formal_detection_data.py verify \
  --config "$SPEC" \
  --data-root "$DATA_ROOT" \
  --cv3-manifest "$CV3" \
  --p02-manifest "$P02" \
  --formal-crop-manifest "$FORMAL_CROP" \
  --lock "$LOCK" \
  --expected-lock-sha256 \
    03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a \
  --report "$ROOT/verification.json" \
  2>&1 | tee "$ROOT/logs/verify.log"

LOCK="$LOCK" ROOT="$ROOT" python - <<'PY' \
  2>&1 | tee "$ROOT/logs/final-gate.log"
import hashlib
import json
import os
from pathlib import Path

lock = Path(os.environ["LOCK"])
root = Path(os.environ["ROOT"])
report = json.loads((root / "verification.json").read_text(encoding="utf-8"))
assert report["status"] == "pass", report
assert report["image_count"] == 4481, report
assert report["label_file_count"] == 4481, report
assert report["object_count"] == 20933, report
assert report["p02_formal_gt_equivalence"] is True, report
assert report["yolo_formal_gt_equivalence"] is True, report
assert report["lock_fingerprint"] == (
    "ef8d3ff216e310e6c4dee6bdabdf20ebb5da983bd473c988b790cfc1bb3b7a18"
), report
assert report["inventory_fingerprint"] == (
    "e094fdd9181c3aecd59e4385b8cd8616047c7c9e98c3717812139944c9e9a280"
), report
assert hashlib.sha256(lock.read_bytes()).hexdigest() == (
    "03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a"
)
assert lock.stat().st_mode & 0o222 == 0
print("CV3_DETECTION_DATA_LOCK_GATE_PASS")
PY
```

只有出现 `CV3_DETECTION_DATA_LOCK_GATE_PASS` 才算完成。

## 6. M1/M3 每折训练前的强制调用

本任务只创建一次锁，但 M1/M3 的每一个 `held_out_fold` 都必须在启动训练进程之前
重跑完整 verify，并把报告放入该折输出目录。以下命令是训练前合同，不是可选审计：

```bash
FOLD=0  # 由任务循环设置为 0/1/2
FOLD_RUN="/workspace/results/<MODEL-TASK>/fold${FOLD}"
P02="$(
  cat /workspace/results/CV3-DETECTION-DATA-LOCK-TASK-00/p02_path.txt
)"
mkdir -p "$FOLD_RUN/input-gates"
echo \
  "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e  $P02" \
  | sha256sum -c -

python /workspace/xh-202625/scripts/lock_formal_detection_data.py verify \
  --config /workspace/xh-202625/configs/experiments/formal_detection_data_lock.json \
  --data-root /workspace/data \
  --cv3-manifest \
    /workspace/xh-202625/data/splits/cv3_airport_proxy_k60_v2.json \
  --p02-manifest "$P02" \
  --formal-crop-manifest \
    /workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv \
  --lock \
    /workspace/formal-detection-data/FORMAL_DETECTION_DATA_LOCK.json \
  --expected-lock-sha256 \
    03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a \
  --report "$FOLD_RUN/input-gates/detection_data_lock_verification.json"

test "$(
  python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$FOLD_RUN/input-gates/detection_data_lock_verification.json"
)" = pass
```

该命令失败时禁止启动或续跑本折。训练汇总和最终 OOF 元数据必须记录锁文件
SHA、`lock_fingerprint` 与 `inventory_fingerprint`。

## 7. 回传包与最终回报

锁文件约 2 MB，应回传；图像和标签不得重复打包：

```bash
RETURN_LOCK="$ROOT/FORMAL_DETECTION_DATA_LOCK.json"
if test -e "$RETURN_LOCK"; then
  test "$(sha256sum "$RETURN_LOCK" | cut -d' ' -f1)" = \
    03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a
else
  cp "$LOCK" "$RETURN_LOCK"
  chmod 0444 "$RETURN_LOCK"
fi

cd /workspace/results
tar -czf CV3-DETECTION-DATA-LOCK-TASK-00-return.tar.gz \
  CV3-DETECTION-DATA-LOCK-TASK-00
sha256sum CV3-DETECTION-DATA-LOCK-TASK-00-return.tar.gz \
  | tee "$ROOT/return-package-sha256.txt"
```

最终回报必须包含：

1. `status`；
2. 三个输入 SHA；
3. 图像/标签/对象/类别计数；
4. 两种 GT 等价布尔值及最大坐标误差；
5. 锁文件 SHA、两个 fingerprint、只读权限；
6. 专项 pytest/Ruff 状态；
7. 回传包大小与 SHA；
8. 失败、重试或输入漂移（没有也明确写“无”）。

成功状态固定为：

```text
complete_formal_detection_data_lock
```
