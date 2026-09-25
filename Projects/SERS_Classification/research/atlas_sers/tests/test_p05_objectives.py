"""Synthetic tests for P05 objectives and the acquisition adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402

from atlas_sers.evaluation.p05_objectives import (  # noqa: E402
    is_known_family,
    normalized_family,
    paired_consistency,
    positive_weight_between,
    supervised_contrastive,
    weighted_cross_entropy,
)
from atlas_sers.evaluation.p05_sampling import Observation  # noqa: E402
from atlas_sers.models.acquisition import (  # noqa: E402
    AcquisitionClassifier,
    acquisition_audit,
)


def obs(uid, master, *, station="S1", target="A", instrument="I1", substrate=""):
    return Observation(
        uid=uid,
        master=master,
        station=station,
        target=target,
        instrument=instrument,
        substrate=substrate,
    )


def test_weighted_cross_entropy_matches_manual_weighted_sum():
    torch.manual_seed(0)
    logits = torch.randn(4, 3, dtype=torch.float64)
    labels = torch.tensor([0, 2, 1, 0])
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    log_probabilities = F.log_softmax(logits, dim=-1)
    picked = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    expected = -(picked * (weights / weights.sum())).sum()
    result = weighted_cross_entropy(logits, labels, weights)
    assert result.dim() == 0
    assert torch.allclose(result, expected)


def test_weighted_cross_entropy_uniform_weights_is_mean_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(5, 4, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 1])
    weights = torch.ones(5, dtype=torch.float64)
    expected = F.cross_entropy(logits, labels, reduction="mean")
    assert torch.allclose(weighted_cross_entropy(logits, labels, weights), expected)


def test_weighted_cross_entropy_rejects_bad_inputs():
    logits = torch.zeros(3, 2, dtype=torch.float64)
    labels = torch.tensor([0, 1, 1])
    good = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        weighted_cross_entropy(torch.full_like(logits, float("nan")), labels, good)
    with pytest.raises(ValueError):
        weighted_cross_entropy(logits, labels, torch.tensor([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_cross_entropy(logits, labels, torch.tensor([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_cross_entropy(logits, labels, torch.ones(2))
    with pytest.raises(ValueError):
        weighted_cross_entropy(logits, torch.tensor([0, 1, 5]), good)


def test_normalized_family_casefolds_and_flags_unknown():
    assert normalized_family("  Pills ") == "pills"
    assert normalized_family("NOT_APPLICABLE") == "not_applicable"
    for token in ("", "na", "n/a", "none", "unknown", "not_applicable", "unspecified"):
        assert not is_known_family(token)
    assert is_known_family("Gold")


def test_positive_weight_between_relations_and_families():
    base = obs("a", "M1", instrument="I1")
    same_master = obs("b", "M1", instrument="I2")
    other_master_same_instrument = obs("c", "M2", instrument="I1")
    other_master_other_instrument = obs("d", "M2", instrument="I2")
    assert positive_weight_between(base, same_master) == pytest.approx(2.0)
    assert positive_weight_between(base, other_master_other_instrument) == pytest.approx(1.5)
    assert positive_weight_between(base, other_master_same_instrument) == pytest.approx(1.0)
    known_a = obs("e", "M3", instrument="I3", substrate="Gold")
    known_b = obs("f", "M4", instrument="I4", substrate="pills")
    assert positive_weight_between(known_a, known_b) == pytest.approx(1.5 * 1.25)
    same_master_known = obs("g", "M3", instrument="I5", substrate="silver")
    assert positive_weight_between(known_a, same_master_known) == pytest.approx(2.5)
    unknown = obs("h", "M5", instrument="I6", substrate="unknown")
    assert positive_weight_between(known_a, unknown) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        positive_weight_between(base, obs("i", "M1", instrument="I1"))


def test_supervised_contrastive_matches_manual_three_row_example():
    torch.manual_seed(2)
    projections = torch.randn(3, 5, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
    ]
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    temperature = 0.1
    result = supervised_contrastive(projections, rows, weights, temperature=temperature)
    assert result.available
    normalized = F.normalize(projections, dim=-1, eps=1e-8)
    similarity = normalized @ normalized.t() / temperature
    anchor_zero = -(similarity[0, 1] - torch.logsumexp(similarity[0, [1, 2]], dim=0))
    anchor_one = -(similarity[1, 0] - torch.logsumexp(similarity[1, [0, 2]], dim=0))
    expected = (1.0 * anchor_zero + 2.0 * anchor_one) / 3.0
    assert torch.allclose(result.loss, expected)
    assert result.counts == {
        "eligible_anchors": 2,
        "zero_positive_anchors": 1,
        "positive_pairs": 2,
        "negative_pairs": 4,
    }


def test_supervised_contrastive_permutation_invariance():
    torch.manual_seed(3)
    projections = torch.randn(4, 6, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
        obs("u4", "M4", target="B", instrument="I4"),
    ]
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    base = supervised_contrastive(projections, rows, weights)
    order = [3, 1, 0, 2]
    permuted = supervised_contrastive(
        projections[order], [rows[i] for i in order], weights[order]
    )
    assert torch.allclose(base.loss, permuted.loss)
    assert base.counts == permuted.counts


def test_unknown_family_labels_do_not_change_supcon_loss():
    torch.manual_seed(6)
    projections = torch.randn(4, 5, dtype=torch.float64)
    weights = torch.ones(4, dtype=torch.float64)
    baseline_rows = [
        obs("u1", "M1", target="A", instrument="I1", substrate=""),
        obs("u2", "M2", target="A", instrument="I2", substrate=""),
        obs("u3", "M3", target="A", instrument="I3", substrate=""),
        obs("u4", "M4", target="B", instrument="I4", substrate=""),
    ]
    mixed_rows = [
        obs("u1", "M1", target="A", instrument="I1", substrate=""),
        obs("u2", "M2", target="A", instrument="I2", substrate=""),
        obs("u3", "M3", target="A", instrument="I3", substrate="pills"),
        obs("u4", "M4", target="B", instrument="I4", substrate=""),
    ]
    known_rows = [
        obs("u1", "M1", target="A", instrument="I1", substrate="gold"),
        obs("u2", "M2", target="A", instrument="I2", substrate="gold"),
        obs("u3", "M3", target="A", instrument="I3", substrate="pills"),
        obs("u4", "M4", target="B", instrument="I4", substrate=""),
    ]
    baseline = supervised_contrastive(projections, baseline_rows, weights)
    mixed = supervised_contrastive(projections, mixed_rows, weights)
    known = supervised_contrastive(projections, known_rows, weights)
    assert torch.allclose(baseline.loss, mixed.loss)
    assert not torch.allclose(baseline.loss, known.loss)


def test_supervised_contrastive_no_negatives_returns_differentiable_zero():
    torch.manual_seed(4)
    projections = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="A", instrument="I3"),
    ]
    result = supervised_contrastive(projections, rows, torch.ones(3, dtype=torch.float64))
    assert not result.available
    assert result.reason == "no_negatives"
    assert result.counts == {
        "eligible_anchors": 3,
        "zero_positive_anchors": 0,
        "positive_pairs": 6,
        "negative_pairs": 0,
    }
    result.loss.backward()
    assert int(torch.count_nonzero(projections.grad)) == 0


def test_supervised_contrastive_single_row_has_zero_positive():
    projections = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
    rows = [obs("u1", "M1", target="A", instrument="I1")]
    result = supervised_contrastive(projections, rows, torch.ones(1, dtype=torch.float64))
    assert not result.available
    assert result.reason == "no_positives"
    assert result.counts == {
        "eligible_anchors": 0,
        "zero_positive_anchors": 1,
        "positive_pairs": 0,
        "negative_pairs": 0,
    }
    result.loss.backward()
    assert int(torch.count_nonzero(projections.grad)) == 0


def test_supervised_contrastive_gradients_reach_both_views():
    torch.manual_seed(5)
    projections = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
    ]
    result = supervised_contrastive(projections, rows, torch.ones(3, dtype=torch.float64))
    result.loss.backward()
    assert torch.isfinite(projections.grad).all()
    assert projections.grad[0].abs().sum() > 0
    assert projections.grad[1].abs().sum() > 0


def test_supervised_contrastive_handles_extreme_finite_values():
    projections = torch.tensor(
        [[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 0.0], [0.0, 0.0, 1000.0]],
        dtype=torch.float64,
    )
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
    ]
    result = supervised_contrastive(projections, rows, torch.ones(3, dtype=torch.float64))
    assert result.available
    assert torch.isfinite(result.loss)


def test_objectives_reject_same_master_same_instrument_duplicates():
    projections = torch.randn(2, 4, dtype=torch.float64)
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64)
    rows = [obs("u1", "M1", instrument="I1"), obs("u2", "M1", instrument="I1")]
    with pytest.raises(ValueError):
        supervised_contrastive(projections, rows, torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        paired_consistency(logits, embeddings, rows)


def test_objectives_reject_cross_station_rows():
    projections = torch.randn(2, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", station="S1", instrument="I1"),
        obs("u2", "M2", station="S2", instrument="I2"),
    ]
    with pytest.raises(ValueError):
        supervised_contrastive(projections, rows, torch.ones(2, dtype=torch.float64))


def test_paired_consistency_matches_manual_symmetric_kl_and_cosine():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    assert result.available
    assert result.counts == {"eligible_masters": 1, "pairs": 1}
    first = F.log_softmax(logits[0], dim=-1)
    second = F.log_softmax(logits[1], dim=-1)
    first_probability = first.exp()
    second_probability = second.exp()
    forward_kl = (first_probability * (first - second)).sum()
    reverse_kl = (second_probability * (second - first)).sum()
    symmetric = 0.5 * (forward_kl + reverse_kl)
    first_embedding = F.normalize(embeddings[0], dim=-1, eps=1e-8)
    second_embedding = F.normalize(embeddings[1], dim=-1, eps=1e-8)
    cosine = (first_embedding * second_embedding).sum()
    expected = 0.5 * symmetric + 0.5 * (1.0 - cosine)
    assert torch.allclose(result.loss, expected)


def test_paired_consistency_averages_pairs_within_master():
    torch.manual_seed(10)
    logits = torch.randn(3, 3, dtype=torch.float64)
    embeddings = torch.randn(3, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
        obs("u3", "M1", target="A", instrument="I3"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    assert result.counts == {"eligible_masters": 1, "pairs": 3}
    pair_values = []
    for first_index, second_index in ((0, 1), (0, 2), (1, 2)):
        first = F.log_softmax(logits[first_index], dim=-1)
        second = F.log_softmax(logits[second_index], dim=-1)
        symmetric = 0.5 * (
            (first.exp() * (first - second)).sum()
            + (second.exp() * (second - first)).sum()
        )
        cosine = (
            F.normalize(embeddings[first_index], dim=-1, eps=1e-8)
            * F.normalize(embeddings[second_index], dim=-1, eps=1e-8)
        ).sum()
        pair_values.append(0.5 * symmetric + 0.5 * (1.0 - cosine))
    assert torch.allclose(result.loss, torch.stack(pair_values).mean())


def test_paired_consistency_without_pair_returns_differentiable_zero():
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="B", instrument="I2"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    assert not result.available
    assert result.reason == "no_pairs"
    assert result.counts == {"eligible_masters": 0, "pairs": 0}
    result.loss.backward()
    assert int(torch.count_nonzero(embeddings.grad)) == 0


def test_paired_consistency_gradients_flow_to_both_views():
    torch.manual_seed(8)
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    result.loss.backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad[0].abs().sum() > 0
    assert embeddings.grad[1].abs().sum() > 0


def test_paired_consistency_gradient_matches_finite_difference():
    torch.manual_seed(9)
    logits = torch.randn(2, 3, dtype=torch.float64)
    base = torch.randn(2, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    embeddings = base.clone().requires_grad_(True)
    paired_consistency(logits, embeddings, rows).loss.backward()
    analytic = embeddings.grad.detach().clone()
    step = 1e-6
    for row in range(2):
        for column in range(4):
            plus = base.clone()
            minus = base.clone()
            plus[row, column] += step
            minus[row, column] -= step
            value_plus = paired_consistency(logits, plus, rows).loss
            value_minus = paired_consistency(logits, minus, rows).loss
            numerical = (value_plus - value_minus) / (2 * step)
            assert torch.allclose(analytic[row, column], numerical, atol=1e-7, rtol=1e-5)


def test_paired_consistency_handles_extreme_finite_logits():
    logits = torch.tensor(
        [[10000.0, -10000.0, 0.0], [0.0, 10000.0, -10000.0]], dtype=torch.float64
    )
    embeddings = torch.eye(2, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    assert result.available
    assert torch.isfinite(result.loss)


def test_contradictory_master_labels_rejected_for_both_losses():
    projections = torch.randn(2, 4, dtype=torch.float64)
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="B", instrument="I2"),
    ]
    with pytest.raises(ValueError):
        supervised_contrastive(projections, rows, torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        paired_consistency(logits, embeddings, rows)


def test_supervised_contrastive_multi_positive_weight_manual():
    torch.manual_seed(12)
    projections = torch.randn(4, 5, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
        obs("u3", "M2", target="A", instrument="I3"),
        obs("u4", "M3", target="B", instrument="I4"),
    ]
    weights = torch.ones(4, dtype=torch.float64)
    result = supervised_contrastive(projections, rows, weights, temperature=0.1)
    normalized = F.normalize(projections, dim=-1, eps=1e-8)
    similarity = normalized @ normalized.t() / 0.1
    log_probability = similarity - torch.logsumexp(
        similarity.masked_fill(torch.eye(4, dtype=torch.bool), float("-inf")),
        dim=1,
        keepdim=True,
    )
    loss_one = -(2.0 / 3.5) * log_probability[0, 1] - (1.5 / 3.5) * log_probability[0, 2]
    loss_two = -(2.0 / 3.5) * log_probability[1, 0] - (1.5 / 3.5) * log_probability[1, 2]
    loss_three = -0.5 * log_probability[2, 0] - 0.5 * log_probability[2, 1]
    expected = (loss_one + loss_two + loss_three) / 3.0
    assert torch.allclose(result.loss, expected)
    assert result.counts["eligible_anchors"] == 3
    assert result.counts["zero_positive_anchors"] == 1


def test_supervised_contrastive_gradcheck():
    torch.manual_seed(13)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
        obs("u3", "M2", target="A", instrument="I3"),
        obs("u4", "M3", target="B", instrument="I4"),
    ]
    weights = torch.ones(4, dtype=torch.float64)

    def loss_of(projections):
        return supervised_contrastive(projections, rows, weights).loss

    projections = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        loss_of, (projections,), eps=1e-6, atol=1e-4, rtol=1e-2
    )


def test_paired_consistency_gradcheck_both_inputs():
    torch.manual_seed(14)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]

    def loss_of(logits, embeddings):
        return paired_consistency(logits, embeddings, rows).loss

    logits = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    embeddings = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        loss_of, (logits, embeddings), eps=1e-6, atol=1e-4, rtol=1e-2
    )


def test_paired_consistency_mean_over_two_masters_with_different_pair_counts():
    torch.manual_seed(15)
    logits = torch.randn(5, 3, dtype=torch.float64)
    embeddings = torch.randn(5, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
        obs("u3", "M1", target="A", instrument="I3"),
        obs("u4", "M2", target="A", instrument="I4"),
        obs("u5", "M2", target="A", instrument="I5"),
    ]
    result = paired_consistency(logits, embeddings, rows)
    assert result.counts == {"eligible_masters": 2, "pairs": 4}

    def pair_loss(first_index, second_index):
        first = F.log_softmax(logits[first_index], dim=-1)
        second = F.log_softmax(logits[second_index], dim=-1)
        symmetric = 0.5 * (
            (first.exp() * (first - second)).sum()
            + (second.exp() * (second - first)).sum()
        )
        cosine = (
            F.normalize(embeddings[first_index], dim=-1, eps=1e-8)
            * F.normalize(embeddings[second_index], dim=-1, eps=1e-8)
        ).sum()
        return 0.5 * symmetric + 0.5 * (1.0 - cosine)

    master_one = (pair_loss(0, 1) + pair_loss(0, 2) + pair_loss(1, 2)) / 3.0
    master_two = pair_loss(3, 4)
    expected = (master_one + master_two) / 2.0
    assert torch.allclose(result.loss, expected)


def test_objectives_reject_nonfinite_temperature_and_epsilon():
    projections = torch.randn(3, 4, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
    ]
    weights = torch.ones(3, dtype=torch.float64)
    for temperature in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            supervised_contrastive(projections, rows, weights, temperature=temperature)
    with pytest.raises(ValueError):
        supervised_contrastive(projections, rows, weights, epsilon=float("nan"))
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float64)
    pair_rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    with pytest.raises(ValueError):
        paired_consistency(logits, embeddings, pair_rows, epsilon=float("inf"))


def test_objectives_reject_empty_feature_dimension():
    projections = torch.randn(3, 0, dtype=torch.float64)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
        obs("u3", "M3", target="B", instrument="I3"),
    ]
    with pytest.raises(ValueError):
        supervised_contrastive(projections, rows, torch.ones(3, dtype=torch.float64))


def test_weighted_cross_entropy_rejects_bool_labels():
    logits = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(TypeError):
        weighted_cross_entropy(
            logits, torch.tensor([True, False]), torch.ones(2, dtype=torch.float64)
        )


def test_weighted_cross_entropy_rejects_overflowing_weights():
    logits = torch.zeros(3, 2, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0])
    weights = torch.tensor([1e308, 1e308, 1e308], dtype=torch.float64)
    assert torch.isfinite(weights).all()
    with pytest.raises(ValueError):
        weighted_cross_entropy(logits, labels, weights)


def test_zero_fallbacks_survive_large_finite_inputs():
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="A", instrument="I2"),
    ]
    projections = torch.full((2, 4), 1e30, dtype=torch.float64, requires_grad=True)
    supcon = supervised_contrastive(projections, rows, torch.ones(2, dtype=torch.float64))
    assert not supcon.available
    assert torch.isfinite(supcon.loss)
    supcon.loss.backward()
    assert torch.isfinite(projections.grad).all()

    pair_rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M2", target="B", instrument="I2"),
    ]
    logits = torch.full((2, 3), 1e30, dtype=torch.float64)
    embeddings = torch.full((2, 4), 1e30, dtype=torch.float64, requires_grad=True)
    paired = paired_consistency(logits, embeddings, pair_rows)
    assert not paired.available
    assert torch.isfinite(paired.loss)
    paired.loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_paired_consistency_rejects_mismatched_dtype():
    logits = torch.randn(2, 3, dtype=torch.float64)
    embeddings = torch.randn(2, 4, dtype=torch.float32)
    rows = [
        obs("u1", "M1", target="A", instrument="I1"),
        obs("u2", "M1", target="A", instrument="I2"),
    ]
    with pytest.raises(ValueError):
        paired_consistency(logits, embeddings, rows)


def test_acquisition_rejects_invalid_arguments_and_enforces_ceiling():
    with pytest.raises(TypeError):
        AcquisitionClassifier(True)
    with pytest.raises(TypeError):
        AcquisitionClassifier(3.0)
    with pytest.raises(ValueError):
        AcquisitionClassifier(1)
    with pytest.raises(TypeError):
        AcquisitionClassifier(3, use_projection=1)
    model = AcquisitionClassifier(3, use_projection=True)
    assert model.trainable_parameter_count() == 212851
    assert model.trainable_parameter_count() < 250000


def test_acquisition_parameter_counts_match_protocol():
    without_projection = acquisition_audit(class_count=3, use_projection=False)
    assert without_projection.backbone_parameters == 208691
    assert without_projection.projection_parameters == 0
    assert without_projection.total_parameters == 208691
    assert without_projection.trainable_parameters == 208691
    with_projection = acquisition_audit(class_count=3, use_projection=True)
    assert with_projection.backbone_parameters == 208691
    assert with_projection.projection_parameters == 4160
    assert with_projection.total_parameters == 212851
    assert with_projection.trainable_parameters == 212851
    assert with_projection.total_parameters < 250000
    assert with_projection.batch_normalization_modules == 0


def test_acquisition_forward_shapes_and_classification_source():
    model = AcquisitionClassifier(class_count=3, use_projection=True)
    values = torch.zeros(2, 1, 1401)
    logits, embedding, projection = model(values)
    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 64)
    assert projection is not None
    assert projection.shape == (2, 64)
    assert torch.allclose(
        torch.linalg.vector_norm(projection, dim=-1),
        torch.ones(2, dtype=projection.dtype),
        atol=1e-6,
    )
    assert torch.allclose(logits, model.backbone.classifier(embedding))
    assert model.batch_normalization_modules() == 0


def test_acquisition_without_projection_has_no_auxiliary_head():
    model = AcquisitionClassifier(class_count=3, use_projection=False)
    logits, embedding, projection = model(torch.zeros(1, 1, 1401))
    assert projection is None
    assert model.projection is None
    assert model.projection_parameter_count() == 0
    assert model.trainable_parameter_count() == 208691
    assert logits.shape == (1, 3)
    assert embedding.shape == (1, 64)


def test_acquisition_auxiliary_head_receives_gradients():
    torch.manual_seed(11)
    model = AcquisitionClassifier(class_count=3, use_projection=True)
    logits, embedding, projection = model(torch.randn(2, 1, 1401))
    (logits.sum() + projection.sum() + embedding.sum()).backward()
    assert model.projection.weight.grad is not None
    assert model.projection.bias.grad is not None
    assert int(torch.count_nonzero(model.projection.weight.grad)) > 0
    assert int(torch.count_nonzero(model.projection.bias.grad)) > 0
    backbone_gradients = [parameter.grad for parameter in model.backbone.parameters()]
    assert all(gradient is not None for gradient in backbone_gradients)
    assert any(int(torch.count_nonzero(gradient)) > 0 for gradient in backbone_gradients)
