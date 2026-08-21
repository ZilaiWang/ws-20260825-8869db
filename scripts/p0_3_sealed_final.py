"""P0-3: sealed-final 封存(方案5 §二 批次0)。

555 张 sentinel 已被多轮方案反复查看, 已接近"二级开发集"。本脚本从 255 个
source_group 中按类别分层再封存一组新的 sealed-final groups(默认 15%),
只在最终三套方案冻结后打开一次。输出封存清单 + 覆盖校验。

纯本地(基于 formal objects 的 group_id 分布)。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-crop-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seal-frac", type=float, default=0.15, help="封存比例")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    names = {}
    with open(args.formal_crop_manifest, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            names[int(r["class_id"])] = r["class_name"]

    # 每 image 的 group_id + GT 细类
    img2grp = {i: o.group_id for (i, _), o in formal.objects.items()}
    grp_classes = defaultdict(Counter)
    for img, gts in formal.boxes.items():
        g = img2grp.get(img, "?")
        for gt in gts:
            grp_classes[g][int(gt["category_id"])] += 1

    groups = sorted(grp_classes.keys())
    print(f"总 source_group 数: {len(groups)}")

    # 每类的 GT 分布(跨 group)
    class_gt_total = Counter()
    for g in groups:
        for c, n in grp_classes[g].items():
            class_gt_total[c] += n

    # 每个 group 的主类别(GT 最多的类), 用于分层
    grp_major = {}
    for g in groups:
        if not grp_classes[g]:
            grp_major[g] = -1
            continue
        grp_major[g] = grp_classes[g].most_common(1)[0][0]

    # 按主类别分层, 每层随机封存 seal_frac 的 group
    rng = np.random.RandomState(args.seed)
    sealed = set()
    for c in set(grp_major.values()):
        layer = [g for g in groups if grp_major[g] == c]
        n_seal = max(1, int(round(len(layer) * args.seal_frac)))
        rng.shuffle(layer)
        sealed.update(layer[:n_seal])

    # 保证尾类覆盖: 对封存 GT 比例 < 10% 的类, 从未封存 group 中补充含该类的 group
    def sealed_gt_of():
        c = Counter()
        for g in sealed:
            for cc, n in grp_classes[g].items():
                c[cc] += n
        return c

    for _ in range(20):
        sgt = sealed_gt_of()
        need_fix = [c for c in range(25)
                    if class_gt_total[c] > 0 and sgt[c] / class_gt_total[c] < 0.05]
        if not need_fix:
            break
        # 找含最多"缺口类"的未封存 group
        cand = [g for g in groups if g not in sealed]
        if not cand:
            break
        best_g = max(cand, key=lambda g: sum(grp_classes[g].get(c, 0) for c in need_fix))
        if sum(grp_classes[best_g].get(c, 0) for c in need_fix) == 0:
            break
        sealed.add(best_g)
    sealed = sorted(sealed)

    # 覆盖校验
    sealed_gt = Counter()
    for g in sealed:
        for c, n in grp_classes[g].items():
            sealed_gt[c] += n
    total_gt = sum(class_gt_total.values())
    sealed_total = sum(sealed_gt.values())

    print(f"\n封存 {len(sealed)}/{len(groups)} 个 group "
          f"({len(sealed)/len(groups):.1%}), GT {sealed_total}/{total_gt} ({sealed_total/total_gt:.1%})")

    print(f"\n=== 封存集每类覆盖 ===")
    uncovered = []
    for c in range(25):
        total_c = class_gt_total[c]
        seal_c = sealed_gt[c]
        frac = seal_c / total_c if total_c else 0
        if total_c > 0 and seal_c == 0:
            uncovered.append(names[c])
        print(f"  {names[c]:10s} 总GT={total_c:5d} 封存={seal_c:4d} ({frac:.1%})")
    if uncovered:
        print(f"\n⚠️ 封存集未覆盖的类: {uncovered}")

    out = {
        "n_total_groups": len(groups),
        "n_sealed_groups": len(sealed),
        "seal_frac": args.seal_frac,
        "seed": args.seed,
        "sealed_group_ids": sealed,
        "sealed_gt_total": int(sealed_total),
        "total_gt": int(total_gt),
        "per_class": {names[c]: {"total": int(class_gt_total[c]), "sealed": int(sealed_gt[c])}
                      for c in range(25)},
        "uncovered_classes": uncovered,
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写: {args.output}")


if __name__ == "__main__":
    main()
