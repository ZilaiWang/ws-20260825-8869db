# FORMAL-CV3-CROP-TASK-01：正式三折通用消费层与 crop v2（CPU）

## 0. 任务结论与边界

本任务把已冻结的 `cv3_airport_proxy_k60_v2` 变成所有框架都能直接读取的
三折视图，并把 P0-2 的 62,799 条 crop 几何记录重挂到正式三折。

只允许执行以下工作：

1. 严格校验正式 CV3 的 SHA、版本、4,481 张图、255 个来源组及逐折计数；
2. 按 `fold == held_out_fold` 为验证、其余 fold 为训练，生成 6 个 CSV；
3. 以 `source_relative_path` 精确连接 P0-2 与正式 CV3；
4. 保留原 `crop_id` 和所有裁剪几何，不打开源图、不重新裁图；
5. 在两个全新目录中独立复跑并比较正式 CSV 的字节。

禁止重新求解分组、修改 fold、读取图像像素、修改 P03/P04 代码、覆盖既有
验收目录，或用旧 P0-2 的 `fold/group_id` 作为正式分组。旧字段只允许以
`historical_p02_*` 出现在输出。

本任务不需要 GPU。

## 1. 固定输入与期望指纹

仓库与结果目录：

```bash
set -euo pipefail
export WS=/workspace
export REPO=/workspace/xh-202625
export ROOT=/workspace/results/FORMAL-CV3-CROP-TASK-01
export RUN_A="$ROOT/run-a"
export RUN_B="$ROOT/run-b"
mkdir -p "$ROOT/logs"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
```

本阶段服务器合同统一冻结 `/workspace`。若宿主机没有该路径，应先由运维建立
挂载或符号链接，再使用原任务 ID；不得只改本任务的根前缀，导致后续任务读取
另一套绝对路径。

冻结输入：

| 输入 | SHA256 |
|---|---|
| `data/splits/cv3_airport_proxy_k60_v2.json` | `27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331` |
| P0-2 `crop_manifest.csv` | `f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e` |

P0-2 是非 Git 大文件，服务器上的绝对路径尚未冻结。先检查历史任务的常见
位置，再按 SHA 搜索；不得凭文件名选取，也不得在本任务中重新生成：

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/locate-p02.log"
import hashlib
import os
from pathlib import Path

root = Path(os.environ["WS"])
expected = "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"
candidates = [
    root / "artifacts/P0-2-exploratory-crop-manifest/crop_manifest.csv",
    root / "xh-202625/outputs/P0-2-exploratory-crop-manifest/crop_manifest.csv",
]
for base in (root / "artifacts", root / "results", root / "inputs"):
    if base.is_dir():
        candidates.extend(base.rglob("crop_manifest.csv"))

seen = set()
matches = []
for path in candidates:
    path = path.resolve()
    if path in seen or not path.is_file():
        continue
    seen.add(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(digest, path)
    if digest == expected:
        matches.append(path)

if not matches:
    raise SystemExit("waiting_for_p02_manifest_input: no exact SHA match")
selected = matches[0]
(Path(os.environ["ROOT"]) / "p02_manifest_path.txt").write_text(
    str(selected) + "\n",
    encoding="utf-8",
)
print("P02_INPUT_READY", selected)
print("IDENTICAL_SHA_COPIES", len(matches))
PY

export P02_MANIFEST
P02_MANIFEST="$(cat "$ROOT/p02_manifest_path.txt")"
```

若搜索结果为 0，状态记为 `waiting_for_p02_manifest_input`，回传定位日志后停止；
不要用其他 SHA 的 manifest 继续。同 SHA 出现多份时它们在科学上等价，记录
全部候选并按上述优先顺序使用第一份。

## 2. 环境与代码门禁

优先复用同一服务器上已有的 P05 CPU 环境：

```bash
if [[ -x "$WS/venvs/p05-cpu/bin/python" ]]; then
  source "$WS/venvs/p05-cpu/bin/activate"
else
  python3.10 -m venv "$WS/venvs/formal-cv3-cpu"
  source "$WS/venvs/formal-cv3-cpu/bin/activate"
  python -m pip install -r requirements-dev.txt
  python -m pip install -e . --no-deps
fi

# 即使复用旧 venv，也强制从当前已通过 SHA 门禁的工作树导入 rsdet。
python - <<'PY'
import rsdet
from pathlib import Path

actual = Path(rsdet.__file__).resolve()
expected = (Path.cwd() / "src/rsdet/__init__.py").resolve()
if actual != expected:
    raise SystemExit(f"wrong rsdet import: actual={actual}, expected={expected}")
print("RSDET_IMPORT_OK", actual)
PY

python --version 2>&1 | tee "$ROOT/logs/python-version.log"
python - <<'PY' 2>&1 | tee "$ROOT/logs/environment.log"
import yaml
print("PyYAML", yaml.__version__)
PY

sha256sum -c docs/server/FORMAL_CV3_CROP_TASK_01_CODE_SHA256.txt \
  2>&1 | tee "$ROOT/logs/code-sha256.log"

python -m ruff check \
  src/rsdet/data/formal_cv3.py \
  src/rsdet/analysis/formal_crop.py \
  scripts/build_formal_cv3_views.py \
  scripts/build_formal_crop_manifest.py \
  tests/test_formal_cv3.py \
  tests/test_formal_crop.py \
  2>&1 | tee "$ROOT/logs/ruff.log"

python -m pytest -q \
  tests/test_formal_cv3.py \
  tests/test_formal_crop.py \
  2>&1 | tee "$ROOT/logs/pytest.log"
```

期望 `11 passed` 且 Ruff 全绿。任一门禁失败立即停止，服务器不得改代码。

再校验两项输入：

```bash
printf '%s  %s\n' \
  27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331 \
  data/splits/cv3_airport_proxy_k60_v2.json \
  f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e \
  "$P02_MANIFEST" \
  | sha256sum -c - 2>&1 | tee "$ROOT/logs/input-sha256.log"
```

## 3. 幂等复用与不覆盖门禁

程序本身拒绝覆盖已有正式 CSV；任务驱动按以下规则保持幂等：

- 目标目录中正式产物完全不存在：标记 `generate`，正常生成；
- 正式产物完整存在且 CSV SHA、审计状态全部精确通过：标记
  `verified_existing_skip`，只校验并跳过生成；
- 只有部分产物、SHA 不同或审计不通过：立即失败，禁止覆盖或删除。

执行：

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/existing-output-gate.log"
import hashlib
import json
import os
from pathlib import Path

expected = {
    "cv3/formal_cv3_fold0_train.csv": "93b0cf3782d4c8da3004c7b7a98093b83a57edbe03527d0d950861c413642db5",
    "cv3/formal_cv3_fold0_val.csv": "f03683689bc17c3bdbecba874f0fa686663527f24e47e347b9f8a2b71107494d",
    "cv3/formal_cv3_fold1_train.csv": "d3c74835ebbfc4e4b79c8303a83c437560a20817a7552150afe410068a906429",
    "cv3/formal_cv3_fold1_val.csv": "257598ed0e3ffae4b9a539f395f7a3d9845531ad6f79b6ddb0453ae01d301070",
    "cv3/formal_cv3_fold2_train.csv": "236aff8d96380eb6e0e7f4cec4cee6b9be328a774a804c8caf29210cfc598286",
    "cv3/formal_cv3_fold2_val.csv": "1c8c37a2dad0f16ee0f568156559c18fbb374359fe91d527882d08656f71ac78",
    "crop/formal_crop_manifest.csv": "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128",
}
required_audits = (
    "cv3/formal_cv3_consumer_audit.json",
    "crop/formal_crop_audit.json",
)

for env_name, mode_name in (("RUN_A", "run-a-mode.txt"), ("RUN_B", "run-b-mode.txt")):
    target = Path(os.environ[env_name])
    present = [path for path in target.rglob("*") if path.is_file()] if target.exists() else []
    if not present:
        mode = "generate"
    else:
        missing = [
            relative for relative in (*expected, *required_audits)
            if not (target / relative).is_file()
        ]
        if missing:
            raise SystemExit(f"{env_name} is partial; immutable outputs missing {missing}")
        for relative, wanted in expected.items():
            actual = hashlib.sha256((target / relative).read_bytes()).hexdigest()
            if actual != wanted:
                raise SystemExit(
                    f"{env_name} immutable SHA mismatch for {relative}: {actual}"
                )
        cv_audit = json.loads(
            (target / "cv3/formal_cv3_consumer_audit.json").read_text()
        )
        crop_audit = json.loads(
            (target / "crop/formal_crop_audit.json").read_text()
        )
        if (
            cv_audit.get("formal_cv3_admission") is not True
            or cv_audit.get("group_cross_fold_count") != 0
            or crop_audit.get("formal_crop_admission") is not True
            or crop_audit.get("historical_assignment_fields_preserved") is not True
            or crop_audit.get("pixels_read") != 0
            or crop_audit.get("geometry_recomputed") is not False
        ):
            raise SystemExit(f"{env_name} existing audit is not accepted")
        mode = "verified_existing_skip"
    (Path(os.environ["ROOT"]) / mode_name).write_text(mode + "\n", encoding="utf-8")
    print(env_name, mode)
PY
```

发生失败后不得清空目录重跑；保留失败目录和日志，交由负责人判断新任务后缀。
正式验收后也不得以 `--overwrite` 或脚本修改原文件。

## 4. 生成或复用唯一正式副本

`RUN_A` 是 P03/P04 后续唯一允许消费的副本：

```text
/workspace/results/FORMAL-CV3-CROP-TASK-01/run-a/crop/formal_crop_manifest.csv
SHA256 a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128
```

按预检模式执行：

```bash
if [[ "$(cat "$ROOT/run-a-mode.txt")" == "generate" ]]; then
  python scripts/build_formal_cv3_views.py \
    --manifest data/splits/cv3_airport_proxy_k60_v2.json \
    --config configs/analysis/formal_cv3.yaml \
    --output-dir "$RUN_A/cv3" \
    2>&1 | tee "$ROOT/logs/run-a-cv3.log"

  python scripts/build_formal_crop_manifest.py \
    --exploratory-manifest "$P02_MANIFEST" \
    --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
    --cv3-config configs/analysis/formal_cv3.yaml \
    --config configs/analysis/formal_crop_manifest.yaml \
    --output-dir "$RUN_A/crop" \
    2>&1 | tee "$ROOT/logs/run-a-crop.log"
else
  printf 'RUN_A_VERIFIED_EXISTING_SKIP\n' | tee "$ROOT/logs/run-a-skip.log"
fi
```

## 5. 第二副本确定性复验

`RUN_B` 只用于字节确定性证明，不得配置给 P03/P04：

```bash
if [[ "$(cat "$ROOT/run-b-mode.txt")" == "generate" ]]; then
  python scripts/build_formal_cv3_views.py \
    --manifest data/splits/cv3_airport_proxy_k60_v2.json \
    --config configs/analysis/formal_cv3.yaml \
    --output-dir "$RUN_B/cv3" \
    > "$ROOT/logs/run-b-cv3.log" 2>&1

  python scripts/build_formal_crop_manifest.py \
    --exploratory-manifest "$P02_MANIFEST" \
    --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
    --cv3-config configs/analysis/formal_cv3.yaml \
    --config configs/analysis/formal_crop_manifest.yaml \
    --output-dir "$RUN_B/crop" \
    > "$ROOT/logs/run-b-crop.log" 2>&1
else
  printf 'RUN_B_VERIFIED_EXISTING_SKIP\n' | tee "$ROOT/logs/run-b-skip.log"
fi

for name in \
  formal_cv3_fold0_train.csv formal_cv3_fold0_val.csv \
  formal_cv3_fold1_train.csv formal_cv3_fold1_val.csv \
  formal_cv3_fold2_train.csv formal_cv3_fold2_val.csv; do
  cmp "$RUN_A/cv3/$name" "$RUN_B/cv3/$name"
done
cmp "$RUN_A/crop/formal_crop_manifest.csv" \
    "$RUN_B/crop/formal_crop_manifest.csv"
```

只比较正式 CSV。审计 JSON 写有服务器绝对路径，不要求跨目录字节相同。

## 6. 冻结结果与科学门禁

正式 CSV 的期望 SHA：

```text
93b0cf3782d4c8da3004c7b7a98093b83a57edbe03527d0d950861c413642db5  formal_cv3_fold0_train.csv
f03683689bc17c3bdbecba874f0fa686663527f24e47e347b9f8a2b71107494d  formal_cv3_fold0_val.csv
d3c74835ebbfc4e4b79c8303a83c437560a20817a7552150afe410068a906429  formal_cv3_fold1_train.csv
257598ed0e3ffae4b9a539f395f7a3d9845531ad6f79b6ddb0453ae01d301070  formal_cv3_fold1_val.csv
236aff8d96380eb6e0e7f4cec4cee6b9be328a774a804c8caf29210cfc598286  formal_cv3_fold2_train.csv
1c8c37a2dad0f16ee0f568156559c18fbb374359fe91d527882d08656f71ac78  formal_cv3_fold2_val.csv
a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128  formal_crop_manifest.csv
```

执行最终机器门禁：

```bash
python - <<'PY' 2>&1 | tee "$ROOT/logs/final-gate.json"
import csv
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_A"])
expected_views = {
    "formal_cv3_fold0_train.csv": (2974, "93b0cf3782d4c8da3004c7b7a98093b83a57edbe03527d0d950861c413642db5"),
    "formal_cv3_fold0_val.csv": (1507, "f03683689bc17c3bdbecba874f0fa686663527f24e47e347b9f8a2b71107494d"),
    "formal_cv3_fold1_train.csv": (2868, "d3c74835ebbfc4e4b79c8303a83c437560a20817a7552150afe410068a906429"),
    "formal_cv3_fold1_val.csv": (1613, "257598ed0e3ffae4b9a539f395f7a3d9845531ad6f79b6ddb0453ae01d301070"),
    "formal_cv3_fold2_train.csv": (3120, "236aff8d96380eb6e0e7f4cec4cee6b9be328a774a804c8caf29210cfc598286"),
    "formal_cv3_fold2_val.csv": (1361, "1c8c37a2dad0f16ee0f568156559c18fbb374359fe91d527882d08656f71ac78"),
}
for name, (count, expected_sha) in expected_views.items():
    path = root / "cv3" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert digest == expected_sha, (name, digest)
    assert len(rows) == count, (name, len(rows))

crop = root / "crop/formal_crop_manifest.csv"
crop_sha = hashlib.sha256(crop.read_bytes()).hexdigest()
assert crop_sha == "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
assert crop.stat().st_size == 71995981
with crop.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    fields = set(reader.fieldnames or ())
    rows = list(reader)
assert len(rows) == 62799
assert "historical_p02_fold" in fields and "historical_p02_group_id" in fields
assert "main_split" not in fields
assert {row["assignment_scope"] for row in rows} == {"formal_cv3_fold_only"}
assert len({row["crop_id"] for row in rows}) == 62799
assert len({row["annotation_uid"] for row in rows}) == 20933
assert len({row["source_relative_path"] for row in rows}) == 4481

cv_audit = json.loads((root / "cv3/formal_cv3_consumer_audit.json").read_text())
crop_audit = json.loads((root / "crop/formal_crop_audit.json").read_text())
assert cv_audit["formal_cv3_admission"] is True
assert cv_audit["group_cross_fold_count"] == 0
assert crop_audit["formal_crop_admission"] is True
assert crop_audit["crop_id_preserved"] is True
assert crop_audit["historical_assignment_fields_preserved"] is True
assert crop_audit["geometry_recomputed"] is False
assert crop_audit["pixels_read"] == 0
assert crop_audit["train_val_annotation_overlap"] == 0
assert crop_audit["train_val_source_image_overlap"] == 0
assert crop_audit["train_val_source_group_overlap"] == 0

print(json.dumps({
    "status": "formal_cv3_and_crop_v2_pass",
    "run_a_mode": (Path(os.environ["ROOT"]) / "run-a-mode.txt").read_text().strip(),
    "run_b_mode": (Path(os.environ["ROOT"]) / "run-b-mode.txt").read_text().strip(),
    "formal_cv3_sha256": cv_audit["manifest_sha256"],
    "formal_crop_sha256": crop_sha,
    "formal_crop_rows": len(rows),
    "formal_annotations": 20933,
    "formal_source_images": 4481,
    "pixels_read": 0,
    "deterministic_repeat": True,
}, indent=2))
PY
```

## 7. 回传与最终回报

`RUN_A` 是唯一验收副本，`RUN_B` 只用于确定性证明。打包时包含正式 CSV、
审计、配置、路径记录和日志：

```bash
cd "$WS/results"
tar -czf FORMAL-CV3-CROP-TASK-01-return.tar.gz \
  FORMAL-CV3-CROP-TASK-01/run-a \
  FORMAL-CV3-CROP-TASK-01/p02_manifest_path.txt \
  FORMAL-CV3-CROP-TASK-01/run-a-mode.txt \
  FORMAL-CV3-CROP-TASK-01/run-b-mode.txt \
  FORMAL-CV3-CROP-TASK-01/logs
sha256sum FORMAL-CV3-CROP-TASK-01-return.tar.gz
```

最终回报必须包含：

1. 状态 `formal_cv3_and_crop_v2_pass` 或准确的停止状态；
2. Git commit、代码 SHA 门禁、Python/PyYAML 版本；
3. P0-2 实际物理路径及两个输入 SHA；
4. 6 个视图的行数和 SHA；
5. crop 的 62,799/20,933/4,481 计数、71,995,981 bytes 和 SHA；
6. 三种 crop policy 的逐折对象数 `7350/7179/6404`；
7. 三类 train/val overlap 均为 0；
8. `pixels_read=0`、`geometry_recomputed=false`；
9. 两次正式 CSV 是否逐字节一致；
10. `run_a_mode/run_b_mode` 是 `generate` 还是 `verified_existing_skip`；
11. 回传包大小和 SHA。
