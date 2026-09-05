"""Train-only capped weak-class image sampling, with unchanged 25-class head."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

TARGETS = (0, 1, 2, 3, 24)


def weak_image_weights(label_sets, *, target_frequency=.10, cap=3.):
    if not label_sets or not 0 < target_frequency <= 1 or not math.isfinite(cap) or cap < 1:
        raise ValueError("invalid sampling inputs")
    sets = [set(labels) for labels in label_sets]
    if any(any(isinstance(c, bool) or int(c) != c or not 0 <= c < 25 for c in labels) for labels in sets):
        raise ValueError("labels must be original 0..24 integer ids")
    counts = Counter(c for labels in sets for c in labels)
    factors = {c: min(cap, max(1., math.sqrt(target_frequency / (counts[c]/len(sets)))))
               if counts[c] else 1. for c in TARGETS}
    weights = [max([1.] + [factors[c] for c in labels if c in factors]) for labels in sets]
    total = sum(weights)
    return weights, {"images": len(sets), "image_counts_by_fine": dict(counts), "weak_factors": factors,
        "target_frequency": target_frequency, "cap": cap, "weight_sum": total,
        "expected_image_draws_by_fine": {c: len(sets)*sum(w for labels, w in zip(sets, weights, strict=True) if c in labels)/total
                                        for c in range(25)},
        "samples_per_epoch": len(sets), "replacement": True,
        "all_images_remain_eligible": True, "validation_or_proposals_used": False}


def weak_rfs_trainer():
    import torch
    from torch.utils.data import WeightedRandomSampler
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class WeakRFSTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1 or self.world_size > 1:
                raise ValueError("weak-rfs-v1 is frozen to a single GPU, not DDP")
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            if dataset.rect or self.args.compile:
                raise ValueError("weak-rfs-v1 requires non-rect, non-compiled training")
            label_sets = [set(int(x) for x in r["cls"].reshape(-1)) for r in dataset.labels]
            weights, audit = weak_image_weights(label_sets)
            paths = [str(Path(p).resolve()) for p in dataset.im_files]
            if len(set(paths)) != len(paths):
                raise ValueError("training source contains duplicate images before sampling")
            audit["ordered_training_paths_sha256"] = hashlib.sha256("\n".join(paths).encode()).hexdigest()
            audit["train_files"] = paths
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "weak_rfs_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
            generator = torch.Generator().manual_seed(int(self.args.seed))
            sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(dataset),
                                            replacement=True, generator=generator)
            batch = min(batch_size, len(dataset))
            nw = min(os.cpu_count() or 1, self.args.workers, math.ceil(len(dataset)/batch))
            loader_generator = torch.Generator().manual_seed(6148914691236517204)
            return InfiniteDataLoader(dataset=dataset, batch_size=batch, shuffle=False,
                num_workers=nw, sampler=sampler, prefetch_factor=4 if nw else None,
                pin_memory=torch.cuda.device_count() > 0, collate_fn=dataset.collate_fn,
                worker_init_fn=seed_worker, generator=loader_generator, drop_last=False)

    return WeakRFSTrainer
