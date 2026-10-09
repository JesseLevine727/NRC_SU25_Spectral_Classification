"""Invented CPU-only tests for the P08 universal unit adapter (private draft)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_universal_units as p08
from atlas_sers.evaluation.classical import classification_metrics

VOCAB = ("a", "b", "c")
DOMAIN_INFO = {
    "d1": {"station": "s1", "instrument": "i1"},
    "d2": {"station": "s2", "instrument": "i2"},
}


def make_registered(
    domains=("d1",),
    repeats=(0, 1),
    folds=(0, 1, 2, 3),
    masters_per_fold=2,
    observations_per_master=2,
):
    records = []
    for domain in domains:
        info = DOMAIN_INFO[domain]
        for repeat in repeats:
            for fold in folds:
                context = f"ctx-{domain}-r{repeat}-f{fold}"
                for index in range(masters_per_fold):
                    master = f"m-{domain}-{fold}-{index}"
                    label = VOCAB[index % 3]
                    for measurement in range(observations_per_master):
                        records.append(
                            {
                                "context_id": context,
                                "domain": domain,
                                "station": info["station"],
                                "instrument": info["instrument"],
                                "outer_repeat": repeat,
                                "outer_fold": fold,
                                "observation_uid": (
                                    f"o-{domain}-r{repeat}-f{fold}-m{index}-k{measurement}"
                                ),
                                "master_sample_id": master,
                                "true_label": label,
                                "class_vocabulary": list(VOCAB),
                            }
                        )
    return pd.DataFrame(records, columns=list(p08._REGISTERED_COLUMNS))


def default_probability(label):
    values = np.full(3, 0.1)
    values[VOCAB.index(label)] = 0.8
    return values


def make_predictions(registered, probability=None):
    rows = []
    for record in registered.itertuples(index=False):
        values = (
            probability(record)
            if probability is not None
            else default_probability(record.true_label)
        )
        for policy in p08.POLICIES:
            for model in p08.MODELS:
                rows.append(
                    {
                        "context_id": record.context_id,
                        "policy_id": policy,
                        "model_id": model,
                        "observation_uid": record.observation_uid,
                        "master_sample_id": record.master_sample_id,
                        "instrument": record.instrument,
                        "station": record.station,
                        "true_label": record.true_label,
                        "class_vocabulary": list(record.class_vocabulary),
                        "probability_0": float(values[0]),
                        "probability_1": float(values[1]),
                        "probability_2": float(values[2]),
                    }
                )
    return pd.DataFrame(rows, columns=list(p08._PREDICTION_COLUMNS))


def sparse_registered():
    records = []
    info = DOMAIN_INFO["d1"]
    fold_labels = {0: ["a", "b"], 1: ["a", "a"], 2: ["a", "a"], 3: ["c", "c"]}
    for fold, labels in fold_labels.items():
        context = f"ctx-d1-r0-f{fold}"
        for index, label in enumerate(labels):
            master = f"m-d1-{fold}-{index}"
            for measurement in range(2):
                records.append(
                    {
                        "context_id": context,
                        "domain": "d1",
                        "station": info["station"],
                        "instrument": info["instrument"],
                        "outer_repeat": 0,
                        "outer_fold": fold,
                        "observation_uid": f"o-d1-r0-f{fold}-m{index}-k{measurement}",
                        "master_sample_id": master,
                        "true_label": label,
                        "class_vocabulary": list(VOCAB),
                    }
                )
    return pd.DataFrame(records, columns=list(p08._REGISTERED_COLUMNS))


def sparse_probability(record):
    if record.master_sample_id == "m-d1-0-1":
        return np.array([0.8, 0.1, 0.1])
    return default_probability(record.true_label)


class TestBuildUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = make_registered(domains=("d1", "d2"))
        cls.predictions = make_predictions(cls.registered)
        cls.panel = p08.build_units(cls.predictions, cls.registered)

    def test_structure_and_all_pairs(self):
        for estimand in p08.ESTIMANDS:
            self.assertIn(estimand, self.panel)
            for policy in p08.POLICIES:
                self.assertIn(policy, self.panel[estimand])
                for aggregation in p08.AGGREGATIONS:
                    frame = self.panel[estimand][policy][aggregation]
                    self.assertFalse(frame.empty)
                    self.assertTrue(set(p08._PANEL_COLUMNS) <= set(frame.columns))

    def test_m01_unit_id_is_observation(self):
        frame = self.panel["equal_context"]["PP-U-MIN"]["M01"]
        self.assertEqual(set(frame.unit_id), set(self.registered.observation_uid))

    def test_m06_unit_id_stable_across_repeats(self):
        frame = self.panel["equal_context"]["PP-U-SG"]["M06"]
        subset = frame[frame.model_id.eq("C-RBF-SVM")]
        self.assertTrue(subset.groupby("unit_id").domain.nunique().eq(1).all())
        self.assertEqual(subset.unit_id.nunique(), subset.master_sample_id.nunique())
        self.assertEqual(subset.groupby("master_sample_id").unit_id.nunique().max(), 1)

    def test_m06_averages_probabilities_not_hard_labels(self):
        registered = make_registered(
            domains=("d1",), repeats=(0,), masters_per_fold=2, observations_per_master=4
        )
        weights = {
            "k0": [0.6, 0.25, 0.15],
            "k1": [0.6, 0.25, 0.15],
            "k2": [0.0, 0.55, 0.45],
            "k3": [0.0, 0.45, 0.55],
        }

        def probability(record):
            if record.master_sample_id == "m-d1-0-0":
                return np.array(weights[record.observation_uid.rsplit("-", 1)[1]])
            return default_probability(record.true_label)

        predictions = make_predictions(registered, probability)
        panel = p08.build_units(predictions, registered)
        m01 = panel["equal_context"]["PP-U-MIN"]["M01"]
        m06 = panel["equal_context"]["PP-U-MIN"]["M06"]
        hard = m01[m01.model_id.eq("C-RBF-SVM") & m01.master_sample_id.eq("m-d1-0-0")]
        self.assertEqual(sorted(hard.predicted_label.tolist()), ["a", "a", "b", "c"])
        row = m06[
            m06.model_id.eq("C-RBF-SVM") & m06.master_sample_id.eq("m-d1-0-0")
        ].iloc[0]
        self.assertEqual(row.predicted_label, "b")
        self.assertAlmostEqual(row.probability_0, 0.3)
        self.assertAlmostEqual(row.probability_1, 0.375)
        self.assertAlmostEqual(row.probability_2, 0.325)
        self.assertFalse(bool(row.correct))

    def test_sparse_fold_classes_split_equal_and_pooled(self):
        registered = sparse_registered()
        predictions = make_predictions(registered, sparse_probability)
        panel = p08.build_units(predictions, registered)
        equal = p08.summarize_units(panel["equal_context"]["PP-U-MIN"]["M01"])
        pooled = p08.summarize_units(panel["pooled_four_fold"]["PP-U-MIN"]["M01"])
        equal_ba = (
            equal["model_summary"]
            .set_index("model_id")
            .balanced_accuracy.loc["C-RBF-SVM"]
        )
        pooled_ba = (
            pooled["model_summary"]
            .set_index("model_id")
            .balanced_accuracy.loc["C-RBF-SVM"]
        )
        self.assertAlmostEqual(equal_ba, 0.875)
        self.assertAlmostEqual(pooled_ba, 2.0 / 3.0)
        self.assertNotAlmostEqual(equal_ba, pooled_ba)

    def test_pooled_contexts_and_fold_metadata(self):
        pooled = self.panel["pooled_four_fold"]["PP-U-ARPLS"]["M01"]
        equal = self.panel["equal_context"]["PP-U-ARPLS"]["M01"]
        self.assertTrue(pooled.outer_fold.isna().all())
        self.assertTrue(pooled.outer_fold_count.eq(4).all())
        self.assertFalse(equal.outer_fold.isna().any())
        self.assertEqual(equal.outer_fold.nunique(), 4)
        self.assertEqual(pooled.context_id.nunique(), 4)

    def test_tie_argmax_uses_sorted_vocabulary(self):
        def probability(record):
            if record.observation_uid == "o-d1-r0-f0-m0-k0":
                return np.array([0.4, 0.4, 0.2])
            return default_probability(record.true_label)

        predictions = make_predictions(self.registered, probability)
        panel = p08.build_units(predictions, self.registered)
        frame = panel["equal_context"]["PP-U-MIN"]["M01"]
        row = frame[frame.unit_id.eq("o-d1-r0-f0-m0-k0")].iloc[0]
        self.assertEqual(row.predicted_label, "a")

    def test_row_order_invariance(self):
        shuffled = self.predictions.sample(frac=1.0, random_state=7).reset_index(
            drop=True
        )
        other = p08.build_units(shuffled, self.registered)
        for estimand in p08.ESTIMANDS:
            for policy in p08.POLICIES:
                for aggregation in p08.AGGREGATIONS:
                    left = self.panel[estimand][policy][aggregation].reset_index(
                        drop=True
                    )
                    right = other[estimand][policy][aggregation].reset_index(drop=True)
                    pd.testing.assert_frame_equal(left, right)

    def test_inputs_not_mutated(self):
        registered = self.registered.copy(deep=True)
        predictions = self.predictions.copy(deep=True)
        p08.build_units(self.predictions, self.registered)
        pd.testing.assert_frame_equal(self.registered, registered)
        pd.testing.assert_frame_equal(self.predictions, predictions)


class TestBuildUnitsRejections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered = make_registered(domains=("d1",))
        cls.predictions = make_predictions(cls.registered)

    def test_missing_uid(self):
        subset = self.predictions[
            self.predictions.policy_id.eq("PP-U-MIN")
            & self.predictions.model_id.eq("C-RBF-SVM")
        ]
        broken = self.predictions.drop(subset.index[0])
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)

    def test_extra_uid(self):
        broken = self.predictions.copy()
        broken.loc[broken.index[0], "observation_uid"] = "o-extra"
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)

    def test_duplicate_uid(self):
        broken = pd.concat(
            [self.predictions, self.predictions.iloc[[0]]], ignore_index=True
        )
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)

    def test_unknown_model_policy(self):
        broken = self.predictions.copy()
        broken.loc[broken.index[0], "model_id"] = "NOPE"
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)

    def test_identity_mismatch(self):
        broken = self.predictions.copy()
        broken.loc[broken.index[0], "master_sample_id"] = "m-bad"
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)

    def test_probability_failures(self):
        broken = self.predictions.copy()
        broken.loc[broken.index[0], "probability_0"] = 0.5
        broken.loc[broken.index[0], "probability_1"] = 0.5
        broken.loc[broken.index[0], "probability_2"] = 0.5
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(broken, self.registered)
        negative = self.predictions.copy()
        negative.loc[negative.index[0], "probability_2"] = -0.1
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(negative, self.registered)

    def test_registered_has_model_column(self):
        broken = self.registered.copy()
        broken["policy_id"] = "PP-U-MIN"
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(self.predictions, broken)

    def test_non_boolean_repeat_fold(self):
        boolean = self.registered.copy()
        boolean["outer_repeat"] = boolean["outer_repeat"].astype(object)
        boolean.loc[boolean.index[0], "outer_repeat"] = True
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(self.predictions, boolean)
        fractional = self.registered.copy()
        fractional["outer_fold"] = fractional["outer_fold"].astype(object)
        fractional.loc[fractional.index[0], "outer_fold"] = 0.5
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(self.predictions, fractional)

    def test_master_overlap_across_folds(self):
        overlap = self.registered.copy()
        mask = (
            overlap.domain.eq("d1")
            & overlap.outer_repeat.eq(0)
            & overlap.outer_fold.eq(1)
            & overlap.master_sample_id.eq("m-d1-1-0")
        )
        overlap.loc[mask, "master_sample_id"] = "m-d1-0-0"
        overlap.loc[mask, "true_label"] = "a"
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(self.predictions, overlap)

    def test_vocabulary_normalization_contract(self):
        self.assertEqual(p08._normalize_vocabulary(["a", "b", "c"]), ("a", "b", "c"))
        self.assertEqual(p08._normalize_vocabulary(("a", "b", "c")), ("a", "b", "c"))
        self.assertEqual(p08._normalize_vocabulary('["a", "b", "c"]'), ("a", "b", "c"))
        self.assertEqual(
            p08._normalize_vocabulary(np.array(["a", "b", "c"])), ("a", "b", "c")
        )
        for invalid in (
            ["c", "b", "a"],
            ["a", "c", "b"],
            ["a ", "b", "c"],
            [" a", "b", "c"],
            ["", "b", "c"],
            ["a", "b", "b"],
            ["a", "b"],
            np.array([["a", "b", "c"]]),
            np.array("abc"),
        ):
            with self.assertRaises(p08.P08UnitsError):
                p08._normalize_vocabulary(invalid)

    def test_class_order_json_and_reversed_rejection(self):
        json_predictions = self.predictions.copy()
        json_predictions["class_vocabulary"] = ['["a", "b", "c"]'] * len(
            json_predictions
        )
        panel = p08.build_units(json_predictions, self.registered)
        self.assertIn("equal_context", panel)
        reversed_registered = self.registered.copy()
        reversed_registered["class_vocabulary"] = reversed_registered[
            "class_vocabulary"
        ].map(lambda _: ["c", "b", "a"])
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(self.predictions, reversed_registered)
        reversed_predictions = self.predictions.copy()
        reversed_predictions["class_vocabulary"] = reversed_predictions[
            "class_vocabulary"
        ].map(lambda _: ["c", "b", "a"])
        with self.assertRaises(p08.P08UnitsError):
            p08.build_units(reversed_predictions, self.registered)


class TestSummarizeUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registered = make_registered(domains=("d1", "d2"))
        predictions = make_predictions(registered)
        cls.panel = p08.build_units(predictions, registered)
        cls.summary = p08.summarize_units(cls.panel["equal_context"]["PP-U-MIN"]["M01"])

    def test_metrics_match_classical_helper(self):
        frame = self.panel["equal_context"]["PP-U-MIN"]["M01"]
        row = self.summary["context_metrics"][
            self.summary["context_metrics"].model_id.eq("C-RBF-SVM")
            & self.summary["context_metrics"].context_id.eq("ctx-d1-r0-f0")
        ].iloc[0]
        cell = frame[
            frame.model_id.eq("C-RBF-SVM") & frame.context_id.eq("ctx-d1-r0-f0")
        ]
        direct = classification_metrics(
            cell.true_label.to_numpy(),
            cell.predicted_label.to_numpy(),
            class_vocabulary=["a", "b", "c"],
            probabilities=cell[list(p08.PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        )
        for column in p08._METRIC_COLUMNS:
            self.assertAlmostEqual(float(row[column]), float(direct[column]))

    def test_denominators(self):
        row = self.summary["model_summary"][
            self.summary["model_summary"].model_id.eq("C-RBF-SVM")
        ].iloc[0]
        self.assertEqual(int(row.contexts), 16)
        self.assertEqual(int(row.domains), 2)
        self.assertEqual(int(row.unit_appearances), 64)
        self.assertEqual(int(row.physical_masters), 16)
        self.assertAlmostEqual(float(row.balanced_accuracy), 1.0)

    def test_class_recall_absent_class_is_nan(self):
        recall = self.summary["class_recall"]
        absent = recall[recall.class_label.eq("c")]
        self.assertTrue(absent.support.eq(0).all())
        self.assertTrue(absent.recall.isna().all())
        present = recall[recall.class_label.eq("a")]
        self.assertTrue(present.support.gt(0).all())
        self.assertTrue(np.allclose(present.recall.to_numpy(dtype=float), 1.0))

    def test_confusion_scope_and_fractions(self):
        confusion = self.summary["confusion"]
        self.assertTrue(confusion.scope.eq("pooled_repeated_appearances").all())
        for _, cell in confusion.groupby(["station", "model_id"]):
            self.assertEqual(len(cell), 9)
            self.assertEqual(cell.true_label.nunique(), 3)
            self.assertEqual(cell.predicted_label.nunique(), 3)
        present = confusion[confusion.true_appearances.gt(0)]
        totals = present.groupby(
            ["station", "model_id", "true_label"]
        ).row_fraction.sum()
        self.assertTrue(np.allclose(totals.to_numpy(dtype=float), 1.0))
        absent = confusion[confusion.true_appearances.eq(0)]
        self.assertFalse(absent.empty)
        self.assertTrue(absent["count"].eq(0).all())
        self.assertTrue(absent.row_fraction.isna().all())

    def test_reversed_vocabulary_in_summary_rejected(self):
        frame = self.panel["equal_context"]["PP-U-MIN"]["M01"].head(4).copy()
        frame["class_vocabulary"] = frame["class_vocabulary"].map(
            lambda _: ("c", "b", "a")
        )
        with self.assertRaises(p08.P08UnitsError):
            p08.summarize_units(frame)

    def test_equal_domain_weighting(self):
        registered = make_registered(
            domains=("d1", "d2"), repeats=(0,), folds=(0, 1, 2, 3)
        )

        def probability(record):
            if record.domain == "d2":
                return np.array([0.1, 0.1, 0.8])
            return default_probability(record.true_label)

        predictions = make_predictions(registered, probability)
        panel = p08.build_units(predictions, registered)
        summary = p08.summarize_units(panel["equal_context"]["PP-U-MIN"]["M01"])
        domain = summary["domain_metrics"].set_index(["model_id", "domain"])
        self.assertAlmostEqual(
            domain.loc[("C-RBF-SVM", "d1"), "balanced_accuracy"], 1.0
        )
        self.assertAlmostEqual(
            domain.loc[("C-RBF-SVM", "d2"), "balanced_accuracy"], 0.0
        )
        model = summary["model_summary"].set_index("model_id")
        self.assertAlmostEqual(model.loc["C-RBF-SVM", "balanced_accuracy"], 0.5)


class TestReliability(unittest.TestCase):
    def _manual_units(self):
        rows = [
            {
                "context_id": "c1",
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "master_sample_id": "m1",
                "unit_id": "u1",
                "true_label": "a",
                "model_id": "C-RBF-SVM",
                "class_vocabulary": ("a", "b", "c"),
                "probability_0": 1.0,
                "probability_1": 0.0,
                "probability_2": 0.0,
                "predicted_label": "a",
                "correct": True,
            },
            {
                "context_id": "c1",
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "master_sample_id": "m2",
                "unit_id": "u2",
                "true_label": "a",
                "model_id": "C-RBF-SVM",
                "class_vocabulary": ("a", "b", "c"),
                "probability_0": 0.5,
                "probability_1": 0.3,
                "probability_2": 0.2,
                "predicted_label": "a",
                "correct": True,
            },
        ]
        return pd.DataFrame(rows, columns=list(p08._SUMMARY_REQUIRED))

    def test_fixed_bins_final_and_empty(self):
        summary = p08.summarize_units(self._manual_units())
        bins = summary["reliability_bins"]
        bins = bins[bins.model_id.eq("C-RBF-SVM") & bins.station.eq("s1")]
        self.assertEqual(len(bins), 10)
        final = bins[bins.bin_index.eq(9)].iloc[0]
        self.assertEqual(int(final["count"]), 1)
        self.assertAlmostEqual(float(final.mean_confidence), 1.0)
        self.assertAlmostEqual(float(final.accuracy), 1.0)
        middle = bins[bins.bin_index.eq(5)].iloc[0]
        self.assertEqual(int(middle["count"]), 1)
        empty = bins[bins.bin_index.eq(0)].iloc[0]
        self.assertEqual(int(empty["count"]), 0)
        self.assertTrue(np.isnan(float(empty.mean_confidence)))
        self.assertTrue(np.isnan(float(empty.accuracy)))

    def _two_context_units(self):
        rows = [
            {
                "context_id": "c1",
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "master_sample_id": "m1",
                "unit_id": "u1",
                "true_label": "a",
                "model_id": "C-RBF-SVM",
                "class_vocabulary": ("a", "b", "c"),
                "probability_0": 0.5,
                "probability_1": 0.3,
                "probability_2": 0.2,
                "predicted_label": "a",
                "correct": True,
            },
        ]
        for index, master in enumerate(("m2", "m3", "m4"), start=2):
            rows.append(
                {
                    "context_id": "c2",
                    "domain": "d1",
                    "station": "s1",
                    "instrument": "i1",
                    "master_sample_id": master,
                    "unit_id": f"u{index}",
                    "true_label": "b",
                    "model_id": "C-RBF-SVM",
                    "class_vocabulary": ("a", "b", "c"),
                    "probability_0": 1.0,
                    "probability_1": 0.0,
                    "probability_2": 0.0,
                    "predicted_label": "a",
                    "correct": False,
                }
            )
        return pd.DataFrame(rows, columns=list(p08._SUMMARY_REQUIRED))

    def test_equal_context_reliability_denominator(self):
        summary = p08.summarize_units(self._two_context_units())
        equal = summary["equal_context_reliability"]
        self.assertTrue(equal.scope.eq("equal_context_fixed_bins").all())
        self.assertEqual(int(equal.contexts.iloc[0]), 2)
        mean_of_context = float(equal.equal_context_ece.iloc[0])
        self.assertAlmostEqual(mean_of_context, 0.75)
        bins = summary["reliability_bins"]
        bins = bins[bins["count"].gt(0)]
        total = float(bins["count"].sum())
        pooled = float(
            np.sum(
                bins["count"].to_numpy(dtype=float)
                / total
                * np.abs(
                    bins["accuracy"].to_numpy(dtype=float)
                    - bins["mean_confidence"].to_numpy(dtype=float)
                )
            )
        )
        self.assertAlmostEqual(pooled, 0.875)
        self.assertNotAlmostEqual(mean_of_context, pooled)


if __name__ == "__main__":
    unittest.main()
