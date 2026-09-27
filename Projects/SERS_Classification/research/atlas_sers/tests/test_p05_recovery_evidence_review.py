"""Supervisor regression checks of saved-model re-authentication logic."""

from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_recovery_evidence as evidence


def test_expected_dirs_allows_root_manifest():
    assert evidence._expected_dirs({"manifest.json": {}, "executions/id/best.pt": {}}) == {
        "executions",
        "executions/id",
    }


def test_full_unit_requires_every_slot_in_completed_prefix():
    slots = [{"slot_id": "never-started"}]
    with pytest.raises(evidence.RecoveryEvidenceError):
        evidence._completed_slots(
            slots, {"reused_original_slot_ids": [], "interrupted_slot_id": "interrupted"}, True
        )


def _item(seed, recipe):
    digest = f"{seed:064x}"
    return {
        "seed": seed,
        "slot": {"recipe_id": recipe},
        "result": SimpleNamespace(
            seed=seed,
            initial_backbone_digest=digest,
            history=[
                {"sampling_digest": digest, "augmentation_digest": digest, "pair_digest": digest}
            ],
        ),
    }


def test_partial_equivalences_keep_seed_groups_separate(monkeypatch):
    items = [_item(seed, recipe) for recipe in ("D0-M", "D1") for seed in (1, 2, 3)]
    items += [_item(seed, "D2") for seed in (1, 2)]
    pairs = []

    def equivalent(left, right):
        assert left.seed == right.seed
        pairs.append(left.seed)

    monkeypatch.setattr(evidence.pilot, "_check_equivalent_results", equivalent)
    evidence._partial_cross_checks(
        items, {"auxiliary_support": {"cross_instrument_master_pairs": 0}}
    )
    assert pairs == [1, 2]


@pytest.mark.parametrize(
    "field", ["sampling_digest", "augmentation_digest", "pair_digest", "initial_backbone_digest"]
)
def test_partial_digests_must_be_real_sha256(field):
    item = _item(1, "D0-M")
    if field == "initial_backbone_digest":
        item["result"].initial_backbone_digest = ""
    else:
        item["result"].history[0][field] = "not-a-hash"
    with pytest.raises(evidence.RecoveryEvidenceError):
        evidence._partial_cross_checks([item], {})
