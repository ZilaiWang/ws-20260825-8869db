from rsdet.external.transfer import model_architecture_sha256


def test_architecture_hash_ignores_class_metadata_but_not_topology() -> None:
    source = {"nc": 4, "names": ["a"], "backbone": [[-1, 1, "Conv", [16]]], "head": []}
    target = {"nc": 25, "names": ["b"], "backbone": [[-1, 1, "Conv", [16]]], "head": []}
    changed = {"nc": 25, "backbone": [[-1, 1, "Conv", [32]]], "head": []}
    assert model_architecture_sha256(source) == model_architecture_sha256(target)
    assert model_architecture_sha256(source) != model_architecture_sha256(changed)
