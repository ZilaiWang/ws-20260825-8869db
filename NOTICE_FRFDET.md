# FRFDet-derived component notice

`src/rsdet/models/ibs_sampling.py` adapts the expansion-compression and
spatial-reorganization design from:

- FRFDet repository: <https://github.com/HZAI-ZJNU/FRFDet>
- frozen reference commit: `d424df831da98f0184a8316e73b545add2b0f7a5`
- upstream file: `nn/FRFDet/FRFDet.py`
- upstream repository `LICENSE`: Apache License 2.0 at the frozen commit

The adaptation removes the upstream dependency on its forked Ultralytics and
Einops packages, uses plain PyTorch, adds strict input validation, and limits
the experiment to one symmetric P2-neck up/down pair. It is not presented as
the complete FRFDet architecture.
