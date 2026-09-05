"""Real PyTorch CPU tests; deliberately not model-quality measurements."""

import copy
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from rsdet.data.xh_dataset import FINE_NAMES  # noqa: E402
from rsdet.experiments.bn_recalibration import (  # noqa: E402
    bn_buffer_names,
    copy_bn_buffers,
    reestimate_bn,
    verify_only_bn_changed,
)
from rsdet.experiments.paired_trend import read, sha, write  # noqa: E402
from scripts import run_paired_bn_recalibration as runner  # noqa: E402


def model():
    return nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Dropout2d(.9))


def test_bn_only_update_restores_modes_and_disables_dropout():
    m = model().train()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    observed = []
    hook = m[2].register_forward_pre_hook(lambda module, _: observed.append(module.training))
    audit = reestimate_bn(m, [torch.rand(2, 3, 8, 8) for _ in range(3)],
                         batch_size=2, expected_batches=3)
    hook.remove()
    assert observed == [False] * 3
    assert m.training and m[2].training and m[1].momentum == .1
    assert all(p.requires_grad for p in m.parameters())
    assert audit["all_non_bn_state_bitwise_equal"] and audit["images"] == 6
    assert int(m[1].num_batches_tracked) == 3
    assert audit["changed_bn_buffers"]
    verify_only_bn_changed(before, m.state_dict(), bn_buffer_names(m))


def test_estimator_is_cumulative_equal_batch_not_stale_ema():
    bn = nn.BatchNorm2d(3).eval()
    bn.running_mean.fill_(123)
    batches = [torch.full((2, 3, 2, 2), .2), torch.full((2, 3, 2, 2), .6)]
    reestimate_bn(bn, batches, batch_size=2, expected_batches=2)
    assert torch.allclose(bn.running_mean, torch.full((3,), .4))
    assert torch.allclose(bn.running_var, torch.zeros(3))
    assert not bn.training and bn.momentum == .1


@pytest.mark.parametrize("bad", ["short", "extra", "nan", "range", "half", "tail"])
def test_failure_restores_model_state(bad):
    m = model().eval()
    original = copy.deepcopy(m.state_dict())
    batches = [torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)]
    if bad == "short":
        batches.pop()
    elif bad == "extra":
        batches.append(batches[0])
    elif bad == "nan":
        batches[1].fill_(float("nan"))
    elif bad == "range":
        batches[1].fill_(1.1)
    elif bad == "half":
        batches[1] = batches[1].half()
    else:
        batches[1] = batches[1][:1]
    with pytest.raises(ValueError):
        reestimate_bn(m, batches, batch_size=2, expected_batches=2)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in original.items())
    assert not m.training and m[1].momentum == .1


def test_fused_or_stateless_bn_rejected():
    for m in (nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3, track_running_stats=False)):
        with pytest.raises(ValueError):
            bn_buffer_names(m)


def test_parameter_mutation_and_overflow_rejected():
    m = model().half()
    before = copy.deepcopy(m.state_dict())
    m[0].weight.data.add_(1)
    with pytest.raises(ValueError, match="non-BN"):
        verify_only_bn_changed(before, m.state_dict(), bn_buffer_names(m))
    source = copy.deepcopy(m).float()
    before = copy.deepcopy(m.state_dict())
    source[1].running_var.fill_(1e20)
    with pytest.raises(ValueError, match="nonfinite"):
        copy_bn_buffers(source, m)
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())


def test_image_preprocessing_rgb_padding_and_sha(tmp_path):
    import numpy as np
    from PIL import Image

    p = tmp_path / "red.png"
    Image.fromarray(np.full((4, 8, 3), [255, 0, 0], dtype=np.uint8)).save(p)
    row = {"relative_path": p.name, "image_sha256": sha(p)}
    x = next(runner.image_batches([row, row], tmp_path, {"imgsz": 32, "batch": 2}, "cpu"))
    assert x.shape == (2, 3, 32, 32) and x.dtype == torch.float32
    assert x[0, :, 16, 16].tolist() == [1., 0., 0.]
    assert torch.allclose(x[0, :, 0, 0], torch.full((3,), 114 / 255))
    row["image_sha256"] = "wrong"
    with pytest.raises(ValueError, match="image changed"):
        next(runner.image_batches([row], tmp_path, {"imgsz": 32, "batch": 1}, "cpu"))


def test_calibration_saves_modified_ema_not_shadow_model(tmp_path, monkeypatch):
    from PIL import Image

    parent = tmp_path / "parent.pt"
    ema = model().half().eval()
    ema.names = dict(enumerate(FINE_NAMES))
    stale = copy.deepcopy(ema)
    stale[0].weight.data.add_(2)
    torch.save({"model": stale, "ema": ema, "train_args": {}}, parent)
    original_sha = sha(parent)
    img = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (200, 10, 30)).save(img)
    samples = [{"relative_path": img.name, "image_sha256": sha(img)}] * 2
    plan = {"recipe": {"batch": 2, "imgsz": 32}, "training_samples": samples, "expected_batches": 1}
    out = tmp_path / "candidate"
    write(out / "plan.json", plan)
    lineage = {"checkpoint_sha256": sha(parent), "train_image_ids": [1, 2], "ancestors": []}
    monkeypatch.setattr(runner, "build_plan", lambda *_: plan)
    monkeypatch.setattr(runner, "validate_bundle", lambda *_: {})
    monkeypatch.setattr(runner, "verify_parent", lambda *_: (parent, lineage))
    result = runner.calibrate(tmp_path, tmp_path, tmp_path, out, "cpu")
    cp = torch.load(out / "calibration/bn_last.pt", weights_only=False)
    assert cp["ema"] is None and cp["optimizer"] is None
    assert cp["model"][0].weight.dtype == torch.float16
    assert torch.equal(cp["model"][0].weight, ema[0].weight)
    assert not torch.equal(cp["model"][1].running_mean, ema[1].running_mean)
    assert result["source_model_key"] == "ema" and sha(parent) == original_sha
    assert read(out / "lineage.json")["ancestors"][-1]["kind"] == "bn_recalibrated_on_frozen_train"
    with pytest.raises(FileExistsError):
        runner.calibrate(tmp_path, tmp_path, tmp_path, out, "cpu")


def test_real_yolo_graph_cpu_smoke():
    path = os.environ.get("PAIRED_BN_REAL_WEIGHT")
    if not path:
        pytest.skip("explicit local trusted YOLO weight required; no network downloads")
    from ultralytics import YOLO

    torch.set_num_threads(1)
    parent = YOLO(str(Path(path).resolve())).model.cpu().eval()
    work = copy.deepcopy(parent).float()
    audit = reestimate_bn(work, [torch.rand(2, 3, 64, 64)], batch_size=2, expected_batches=1)
    copied = copy_bn_buffers(work, parent)
    assert audit["bn_layers"] > 20 and copied["all_non_bn_state_bitwise_equal"]
