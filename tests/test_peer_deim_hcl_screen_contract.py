from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_peer_deim_hcl_screen_is_a_single_factor_paired_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/deim_hcl_m_fold0_40ep.yml").read_text()
    )

    assert config["num_classes"] == 25
    assert config["epoches"] == 40
    assert config["DEIM"]["decoder"] == "BHCLDFINETransformer"
    decoder = config["BHCLDFINETransformer"]
    assert decoder["num_layers"] == 4
    assert decoder["decouple_queries"] is True
    assert decoder["bhcl_mode"] == "hcl"
    criterion = config["BHCLDEIMCriterion"]
    assert criterion["losses"] == ["mal", "boxes", "local", "bhcl"]
    assert criterion["weight_dict"]["loss_bhcl"] == 0.6
    assert config["train_dataloader"]["total_batch_size"] == 4


def test_peer_driver_pins_both_public_revisions_and_formal_fold_assets() -> None:
    driver = (ROOT / "scripts/server/run_peer_deim_hcl_m_fold0_screen.sh").read_text()

    assert "d23ef57ea5e3ea80ec71e883776718a8c3c1510a" in driver
    assert "09d35d53d39ee3145a1e61e3a989b28b9468d1dd" in driver
    assert "41e93416083ad39cd8b665b53be6613f81d9d9d6c1d052da1809b7e71d5686ef" in driver
    assert "2641d3bb15388b9a19812ab514b993d5f68ef90d7a59fb02834bf7903e585977" in driver
    assert "--research-only-unlicensed-reference" in driver
    metadata = (
        ROOT / "research/peer_runtime/xh_detect-0.1.0.dist-info/METADATA"
    ).read_text()
    assert "Name: xh-detect" in metadata
    assert "Version: 0.1.0" in metadata


def test_deim_inference_materializes_bhcl_decoder_before_checkpoint_load() -> None:
    for relative_path in (
        "scripts/infer_deim_coco.py",
        "scripts/infer_deim_tiled_coco.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        materialize = source.index('decoder_name == "BHCLDFINETransformer"')
        load_checkpoint = source.index("checkpoint = torch.load")
        load_state = source.index("cfg.model.load_state_dict(state)")
        assert materialize < load_checkpoint < load_state
        assert "initialize_after_tuning" in source[materialize:load_checkpoint]
        assert "decoupled_ready" in source[materialize:load_checkpoint]


def test_posttrain_recovery_cannot_restart_training_or_change_the_gate() -> None:
    recovery = (
        ROOT / "scripts/server/run_peer_deim_hcl_m_posttrain_recovery.sh"
    ).read_text(encoding="utf-8")
    assert "test \"$(wc -l <\"${OUT}/training/log.txt\")\" -eq 40" in recovery
    assert "training_restarted\": False" in recovery
    assert "train.py \\" not in recovery
    assert "--imgsz 1024 --batch-size 4 --score-floor 0.001" in recovery
    assert "decide_peer_normal_screen.py" in recovery
    assert "--research-only-unlicensed-reference" in recovery
    assert "run_peer_deim_hcl_fixed_benchmarks.sh" in recovery


def test_non_tiled_deim_inference_drops_and_audits_degenerate_boxes() -> None:
    source = (ROOT / "scripts/infer_deim_coco.py").read_text(encoding="utf-8")
    assert "dropped_degenerate_boxes = 0" in source
    assert "dropped_degenerate_boxes += 1\n                        continue" in source
    assert '"dropped_degenerate_boxes": dropped_degenerate_boxes' in source
