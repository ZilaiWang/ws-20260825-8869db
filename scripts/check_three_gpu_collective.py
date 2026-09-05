#!/usr/bin/env python3
"""Bounded NCCL and device-health preflight; no model/data changes."""

import datetime
import json
import os
import time

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60))
tensor = torch.ones(10_000_000, device=f"cuda:{rank}")
dist.all_reduce(tensor)
torch.cuda.synchronize()
assert float(tensor[0]) == 3.0
start = time.perf_counter()
for _ in range(10):
    tensor.fill_(1.0)
    dist.all_reduce(tensor)
torch.cuda.synchronize()
print(json.dumps({"rank": rank, "status": "pass", "allreduce_40mb_ms":
                  (time.perf_counter() - start) * 100, "device": torch.cuda.get_device_name(rank)}))
dist.destroy_process_group()
