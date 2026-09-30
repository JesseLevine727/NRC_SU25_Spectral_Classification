import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p06p11_inference import (
    compile_pair,
    positive_weights,
    score_weights,
    summarize_draws,
)

COLUMNS = [
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
    "correct_model",
    "correct_reference",
]
ROWS = [
    ("c1", "d1", "s1", "TIRF", "m1", "u1", "a", 1, 0),
    ("c1", "d1", "s1", "TIRF", "m1", "u2", "a", 1, 1),
    ("c1", "d1", "s1", "TIRF", "m4", "u10", "a", 0, 0),
    ("c1", "d1", "s1", "TIRF", "m3", "u3", "b", 0, 0),
    ("c2", "d2", "s2", "TIRF", "m2", "u4", "a", 1, 1),
    ("c2", "d2", "s2", "TIRF", "m2", "u5", "a", 0, 1),
    ("c2", "d2", "s2", "TIRF", "m2", "u6", "a", 1, 0),
    ("c3", "d3", "s1", "CONF", "m1", "u7", "a", 1, 0),
    ("c3", "d3", "s1", "CONF", "m3", "u8", "b", 0, 1),
]


def _frame(rows=None):
    return pd.DataFrame(list(ROWS) if rows is None else rows, columns=COLUMNS)


def _direct(df, mw, iw, masters, instruments, domains):
    d = df.copy()
    d["diff"] = d["correct_model"].astype(int) - d["correct_reference"].astype(int)
    mi = {m: i for i, m in enumerate(masters)}
    ii = {v: i for i, v in enumerate(instruments)}
    di = {v: i for i, v in enumerate(domains)}
    cells = sorted(set(zip(d["context_id"], d["true_label"], strict=True)))
    delta = np.zeros((mw.shape[0], len(cells)))
    for ci, (ctx, label) in enumerate(cells):
        sub = d[(d["context_id"] == ctx) & (d["true_label"] == label)]
        cnt = np.zeros(len(masters))
        dlt = np.zeros(len(masters))
        for name, group in sub.groupby("master_sample_id"):
            cnt[mi[name]] = len(group)
            dlt[mi[name]] = group["diff"].sum()
        delta[:, ci] = (mw @ dlt) / (mw @ cnt)
    dom_ctx = d.groupby("domain")["context_id"].nunique().to_dict()
    ctx_cls = d.groupby("context_id")["true_label"].nunique().to_dict()
    domain_out = np.zeros((mw.shape[0], len(domains)))
    for ci, (ctx, _) in enumerate(cells):
        dom = d[d["context_id"] == ctx]["domain"].iloc[0]
        domain_out[:, di[dom]] += delta[:, ci] / (ctx_cls[ctx] * dom_ctx[dom])
    weights = np.column_stack(
        [iw[:, ii[d[d["domain"] == dom]["instrument"].iloc[0]]] for dom in domains]
    )
    return (domain_out * weights).sum(axis=1) / weights.sum(axis=1), domain_out


def test_compile_pair_structure_and_fixture_identity():
    df = _frame()
    assert df.groupby("master_sample_id")["station"].nunique().max() == 1
    assert df.groupby("master_sample_id")["true_label"].nunique().max() == 1
    assert df.groupby("unit_id")["instrument"].nunique().max() == 1
    design = compile_pair(df)
    assert design.masters == ("m1", "m2", "m3", "m4")
    assert design.instruments == ("CONF", "TIRF")
    assert design.domains == ("d1", "d2", "d3")
    assert design.cell_keys == (("c1", "a"), ("c1", "b"), ("c2", "a"), ("c3", "a"), ("c3", "b"))
    assert design.master_classes == ("a", "a", "b", "a")
    assert design.master_stations == ("s1", "s2", "s1", "s1")
    assert design.cell_domain.tolist() == [0, 0, 1, 2, 2]
    assert design.domain_instrument.tolist() == [1, 1, 0]
    assert design.cell_factor.tolist() == [0.5, 0.5, 1.0, 0.5, 0.5]
    assert design.counts[0].tolist() == [2.0, 0.0, 0.0, 1.0]
    assert design.delta_correct[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert not design.counts.flags.writeable


def test_score_matches_direct_engine():
    design = compile_pair(_frame())
    mw = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 0.5, 3.0, 1.5], [0.3, 0.7, 1.1, 2.2]])
    iw = np.array([[1.0, 1.0], [0.4, 2.5], [3.0, 0.5]])
    overall, domain_out = score_weights(design, mw, iw)
    expected_overall, expected_domain = _direct(
        _frame(), mw, iw, design.masters, design.instruments, design.domains
    )
    np.testing.assert_allclose(overall, expected_overall)
    np.testing.assert_allclose(domain_out, expected_domain)


def test_unit_weights_equal_balanced_accuracy():
    overall, domain_out = score_weights(compile_pair(_frame()), np.ones((1, 4)), np.ones((1, 2)))
    np.testing.assert_allclose(domain_out[0], [1.0 / 6.0, 0.0, 0.0])
    np.testing.assert_allclose(overall[0], 1.0 / 18.0)


def test_reversed_correctness_negates():
    df = _frame()
    reversed_df = df.copy()
    reversed_df["correct_model"] = df["correct_reference"]
    reversed_df["correct_reference"] = df["correct_model"]
    mw = np.array([[1.0, 2.0, 0.5, 1.0], [0.5, 0.5, 2.0, 1.0]])
    iw = np.array([[1.0, 2.0], [2.0, 1.0]])
    left = score_weights(compile_pair(df), mw, iw)
    right = score_weights(compile_pair(reversed_df), mw, iw)
    np.testing.assert_allclose(left[0], -right[0])
    np.testing.assert_allclose(left[1], -right[1])


def test_identical_correctness_is_zero():
    df = _frame()
    df["correct_reference"] = df["correct_model"]
    overall, domain_out = score_weights(compile_pair(df), np.ones((1, 4)), np.ones((1, 2)))
    np.testing.assert_allclose(overall, 0.0)
    np.testing.assert_allclose(domain_out, 0.0)


def test_shuffle_invariance_and_no_mutation():
    df = _frame()
    snapshot = df.copy(deep=True)
    design = compile_pair(df)
    scores = score_weights(design, np.ones((1, 4)), np.ones((1, 2)))
    shuffled = compile_pair(df.sample(frac=1.0, random_state=7).reset_index(drop=True))
    assert shuffled.cell_keys == design.cell_keys
    assert shuffled.masters == design.masters
    np.testing.assert_allclose(shuffled.counts, design.counts)
    np.testing.assert_allclose(shuffled.delta_correct, design.delta_correct)
    other = score_weights(shuffled, np.ones((1, 4)), np.ones((1, 2)))
    np.testing.assert_allclose(scores[0], other[0])
    np.testing.assert_allclose(scores[1], other[1])
    pd.testing.assert_frame_equal(df, snapshot)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["station"], inplace=True),
        lambda d: d.rename(columns={"domain": "context_id"}, inplace=True),
        lambda d: d.__setitem__("station", " s1"),
        lambda d: d.__setitem__("domain", 7),
        lambda d: d.__setitem__("unit_id", "u1"),
        lambda d: d.__setitem__("correct_model", d["correct_model"].astype(str)),
        lambda d: d.__setitem__("correct_reference", np.inf),
        lambda d: d.__setitem__("correct_reference", 2),
        lambda d: d.__setitem__("correct_model", d["correct_model"].astype(complex)),
        lambda d: d.at.__setitem__((1, "domain"), "d2"),
        lambda d: d.at.__setitem__((3, "true_label"), "a"),
        lambda d: d.at.__setitem__((0, "instrument"), "CONF"),
    ],
)
def test_compile_pair_rejects_bad_frames(mutate):
    df = _frame()
    mutate(df)
    with pytest.raises((TypeError, ValueError)):
        compile_pair(df)


@pytest.mark.parametrize(
    "mw",
    [
        np.ones(4),
        np.ones((1, 3)),
        np.ones((2, 4)),
        np.ones((0, 4)),
        np.array([[1.0, 0.0, 1.0, 1.0]]),
        np.array([[-1.0, 1.0, 1.0, 1.0]]),
        np.array([[np.nan, 1.0, 1.0, 1.0]]),
        np.array([["1", "1", "1", "1"]]),
        np.array([[True, True, True, True]]),
    ],
)
def test_score_weights_rejects_bad_master_weights(mw):
    with pytest.raises((TypeError, ValueError)):
        score_weights(compile_pair(_frame()), mw, np.ones((1, 2)))


def test_score_weights_rejects_bad_instrument_weights():
    with pytest.raises(ValueError):
        score_weights(compile_pair(_frame()), np.ones((1, 4)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        score_weights(compile_pair(_frame()), np.ones((1, 4)), np.ones((1, 3)))


def test_positive_weights_deterministic_and_isolated():
    before = np.random.get_state()
    first = positive_weights(["m1", "m2", "m3"], ["i1", "i2"], draws=5)
    second = positive_weights(["m1", "m2", "m3"], ["i1", "i2"], draws=5)
    after = np.random.get_state()
    assert first[0].shape == (5, 3) and first[1].shape == (5, 2)
    assert np.all(first[0] > 0) and np.all(first[1] > 0)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"draws": 0},
        {"draws": 10001},
        {"draws": True},
        {"master_seed": -1},
        {"instrument_seed": True},
    ],
)
def test_positive_weights_rejects_bad_args(kwargs):
    with pytest.raises((TypeError, ValueError)):
        positive_weights(["m1"], ["i1"], **kwargs)


def test_positive_weights_rejects_bad_identities():
    with pytest.raises(ValueError):
        positive_weights(["m1", "m1"], ["i1"])
    with pytest.raises(ValueError):
        positive_weights([], ["i1"])
    with pytest.raises(ValueError):
        positive_weights(["m1"], [" i1"])
    with pytest.raises(ValueError):
        positive_weights(["m1"], [1])


def test_summarize_draws_known_quantiles():
    values = np.arange(1.0, 41.0)
    out = summarize_draws(values)
    assert out["planned_draws"] == 40 and out["defined_draws"] == 40 and out["undefined_draws"] == 0
    assert out["lower"] == pytest.approx(np.quantile(values, 0.025))
    assert out["upper"] == pytest.approx(np.quantile(values, 0.975))
    assert out["reason_code"] == "ok"


def test_summarize_draws_missing_is_undefined():
    out = summarize_draws([1.0, np.nan, 3.0, np.nan])
    assert out["planned_draws"] == 4 and out["defined_draws"] == 2 and out["undefined_draws"] == 2
    assert out["lower"] is None and out["upper"] is None
    assert out["reason_code"] == "hierarchical_fixed_support_undefined"


def test_summarize_draws_degenerate_distribution():
    out = summarize_draws([2.5, 2.5, 2.5])
    assert out["lower"] == 2.5 and out["upper"] == 2.5
    assert out["reason_code"] == "degenerate_distribution"


@pytest.mark.parametrize("values", [[], [np.inf], [True, False], ["1"], [[1.0, 2.0]]])
def test_summarize_draws_rejects_bad_values(values):
    with pytest.raises((TypeError, ValueError)):
        summarize_draws(values)


def test_positive_weights_rejects_unsorted_identities():
    with pytest.raises(ValueError):
        positive_weights(["m2", "m1"], ["i1"])
    with pytest.raises(ValueError):
        positive_weights(["m1"], ["i2", "i1"])


def test_positive_weights_seed_independence():
    masters = ["m1", "m2", "m3"]
    instruments = ["i1", "i2"]
    first = positive_weights(
        masters, instruments, draws=7, master_seed=1, instrument_seed=2
    )
    second = positive_weights(
        masters, instruments, draws=7, master_seed=99, instrument_seed=2
    )
    np.testing.assert_array_equal(first[1], second[1])
    third = positive_weights(
        masters, instruments, draws=7, master_seed=1, instrument_seed=77
    )
    np.testing.assert_array_equal(first[0], third[0])


def test_identity_errors_do_not_leak_private_ids():
    sentinel = "SENTINEL_PRIVATE_7f3a"
    inconsistent = _frame()
    inconsistent.at[4, "domain"] = sentinel
    with pytest.raises(ValueError) as excinfo:
        compile_pair(inconsistent)
    assert sentinel not in str(excinfo.value)

    duplicate = _frame()
    duplicate.at[0, "unit_id"] = sentinel
    duplicate.at[1, "unit_id"] = sentinel
    with pytest.raises(ValueError) as excinfo:
        compile_pair(duplicate)
    assert sentinel not in str(excinfo.value)


def test_compile_pair_accepts_mixed_object_binary():
    df = _frame()
    model = np.array([True, 0, 1, False, 1, 0, True, False, 1.0], dtype=object)
    reference = np.array([0, 1, 0, False, 1, 1, 0, True, 0.0], dtype=object)
    df["correct_model"] = model
    df["correct_reference"] = reference
    design = compile_pair(df)
    assert design.counts.shape == (5, 4)


@pytest.mark.parametrize("bad", [np.nan, "1", 1 + 0j, None])
def test_compile_pair_rejects_bad_object_binary(bad):
    df = _frame()
    values = np.array([True, 0, 1, False, 1, 0, True, False, 1], dtype=object)
    values[3] = bad
    df["correct_model"] = values
    with pytest.raises((TypeError, ValueError)):
        compile_pair(df)


def test_cell_factor_distinguishes_contexts_per_domain():
    rows = list(ROWS) + [("c4", "d1", "s1", "TIRF", "m1", "u9", "a", 1, 0)]
    design = compile_pair(_frame(rows))
    assert design.domains == ("d1", "d2", "d3")
    assert design.cell_keys == (
        ("c1", "a"),
        ("c1", "b"),
        ("c2", "a"),
        ("c3", "a"),
        ("c3", "b"),
        ("c4", "a"),
    )
    factors = dict(zip(design.cell_keys, design.cell_factor.tolist(), strict=True))
    assert factors[("c1", "a")] == pytest.approx(1.0 / 4.0)
    assert factors[("c4", "a")] == pytest.approx(1.0 / 2.0)
    assert factors[("c2", "a")] == pytest.approx(1.0)


def test_score_weights_batch_matches_chunked_calls():
    design = compile_pair(_frame())
    rng = np.random.default_rng(2026)
    master_matrix = rng.uniform(0.5, 2.0, size=(257, len(design.masters)))
    instrument_matrix = rng.uniform(0.5, 2.0, size=(257, len(design.instruments)))
    overall, domain_out = score_weights(design, master_matrix, instrument_matrix)
    chunk_overall = []
    chunk_domain = []
    start = 0
    for count in (1, 128, 128):
        stop = start + count
        part_overall, part_domain = score_weights(
            design, master_matrix[start:stop], instrument_matrix[start:stop]
        )
        chunk_overall.append(part_overall)
        chunk_domain.append(part_domain)
        start = stop
    np.testing.assert_allclose(overall, np.concatenate(chunk_overall))
    np.testing.assert_allclose(domain_out, np.concatenate(chunk_domain, axis=0))


def test_score_weights_overflow_raises():
    design = compile_pair(_frame())
    huge = np.full((1, len(design.masters)), 1e308)
    with pytest.raises(ValueError):
        score_weights(design, huge, np.ones((1, len(design.instruments))))


def test_compiled_arrays_are_read_only():
    design = compile_pair(_frame())
    for array in (
        design.counts,
        design.delta_correct,
        design.cell_domain,
        design.domain_instrument,
        design.cell_factor,
    ):
        assert not array.flags.writeable
