"""Tests for the bounded P08-F02 renderer (invented complete-grid records)."""

import hashlib
import json
import unittest

from atlas_sers.visualization import p08_f02_render as render

DOMAINS = [f"DOM{index:02d}" for index in range(1, 14)]


def _canon(obj):
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(estimand, endpoint, model, policy, domain, x, y, station=None, instrument=None):
    return {
        "estimand": estimand,
        "contrast_id": estimand + "-" + endpoint,
        "family_id": "F02",
        "endpoint": endpoint,
        "model_id": model,
        "policy_id": policy,
        "domain": domain,
        "station": station if station is not None else "station-" + domain,
        "instrument": instrument if instrument is not None else "instrument-" + domain,
        "contexts": 3,
        "unit_appearances": 5,
        "physical_masters": 7,
        "distinct_units": 9,
        "x_balanced_accuracy": x,
        "y_balanced_accuracy": y,
        "effect": y - x,
    }


def _semantic(station=None, instrument=None):
    pairs, count = [], 0
    for estimand in render.ESTIMANDS:
        for endpoint in render.ENDPOINTS:
            for model in render.MODELS:
                for policy in render.POLICIES:
                    for domain in DOMAINS:
                        count += 1
                        x = (count % 5) / 10.0
                        y = x if count % 6 == 0 else (count % 9) / 10.0
                        pairs.append(
                            _row(
                                estimand, endpoint, model, policy, domain, x, y, station, instrument
                            )
                        )
    return {
        "f02_pairs": pairs,
        "metadata": {
            "population": dict(
                primary_spectra=598,
                held_spectra=557,
                masters=69,
                instruments=10,
                held_domains=13,
                contexts=260,
            ),
            "independent_unit": "physical_master",
            "selection": "source-only",
            "interval_caption": "intervals elsewhere",
            "interaction_caption": "descriptive only",
            "counts_reference": "original population counts",
            "display": "descriptive",
        },
        "labels": {"endpoints": {"M01": "M01 individual-spectrum predictions"}, "policies": {}},
        "unrelated": {"drop": "me"},
    }


def _prepared(semantic):
    return {
        "semantic": semantic,
        "semantic_sha256": _canon(semantic),
        "manifest": {"state": "prepared"},
    }


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.semantic = _semantic()
        self.result = render.prepare_f02(_prepared(self.semantic))

    def test_complete_grid(self):
        self.assertEqual(len(self.result["panels"]), 20)
        self.assertEqual(len(self.semantic["f02_pairs"]), 520)
        self.assertEqual(len({panel["slug"] for panel in self.result["panels"]}), 20)
        for panel in self.result["panels"]:
            self.assertEqual(set(panel), {"slug", "semantic_sha256", "tex", "html", "csv"})
            self.assertEqual(panel["semantic_sha256"], self.result["semantic_sha256"])
            self.assertEqual(len(panel["csv"].strip().splitlines()), 27)

    def test_hash_and_source_conventions(self):
        sha = self.result["semantic_sha256"]
        for panel in self.result["panels"]:
            self.assertIn(sha, panel["tex"])
            self.assertIn(sha, panel["html"])
            self.assertIn(sha, panel["csv"])
            self.assertNotIn("includegraphics", panel["tex"])
            self.assertNotIn("http", panel["html"].lower())
            self.assertNotIn("<script src", panel["html"])
            self.assertNotIn("<link", panel["html"])
            self.assertNotIn("<img", panel["html"])

    def test_endpoint_wording_and_axes(self):
        html = self.result["panels"][0]["html"]
        for token in (
            "RQ-S01",
            "P08-F02",
            "source-only",
            "individual-spectrum",
            "physical sample",
            "69",
            "y=x",
            "Claim limit",
        ):
            self.assertIn(token, html)
        self.assertEqual(self.result["semantic"]["axes"], render.AXES)
        self.assertEqual(self.result["semantic"]["x_reference"], "PP-U-MIN")

    def test_policy_encodings(self):
        panel = self.result["panels"][0]
        self.assertIn("0072B2", panel["tex"])
        self.assertIn("D55E00", panel["tex"])
        self.assertIn("#0072B2", panel["html"])
        self.assertIn("#D55E00", panel["html"])
        self.assertIn("dash dot", panel["tex"])
        self.assertIn(r"\usepgfplotslibrary{groupplots}", panel["tex"])
        self.assertNotIn(r"\usepackage{groupplots}", panel["tex"])
        self.assertIn('stroke-dasharray="7 3 1 3"', panel["html"])
        self.assertIn("border=0pt", panel["tex"])

    def test_metadata_binding_and_domain_order(self):
        self.assertEqual(self.result["semantic"]["metadata"], self.semantic["metadata"])
        self.assertEqual(self.result["semantic"]["figure_id"], "P08-F02")
        rows = self.result["panels"][0]["csv"].splitlines()[1:14]
        self.assertEqual([row.split(",")[1] for row in rows], [f"D{i}" for i in range(1, 14)])
        malformed = _semantic()
        malformed["metadata"]["selection"] = {"unapproved": "nested data"}
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(malformed))

    def test_frozen_output_not_mutated_with_input(self):
        old = self.result["semantic"]["f02_pairs"][0]["station"]
        self.semantic["f02_pairs"][0]["station"] = "changed"
        self.assertEqual(self.result["semantic"]["f02_pairs"][0]["station"], old)
        self.assertEqual(_canon(self.result["semantic"]), self.result["semantic_sha256"])

    def test_inconsistent_domain_identity_rejected(self):
        malformed = _semantic()
        malformed["f02_pairs"][0]["station"] = "wrong"
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(malformed))

    def test_rejects(self):
        bad = _prepared(_semantic())
        bad["semantic_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            render.prepare_f02(bad)
        incomplete = _semantic()
        incomplete["f02_pairs"] = incomplete["f02_pairs"][:-1]
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(incomplete))
        duplicate = _semantic()
        duplicate["f02_pairs"][1] = dict(duplicate["f02_pairs"][0])
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(duplicate))
        nonfinite = _semantic()
        nonfinite["f02_pairs"][0]["y_balanced_accuracy"] = float("inf")
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(nonfinite))
        mismatch = _semantic()
        mismatch["f02_pairs"][0]["effect"] = 0.5
        with self.assertRaises(ValueError):
            render.prepare_f02(_prepared(mismatch))

    def test_escaping(self):
        self.assertEqual(render._tex("a_b#c%&"), r"a\_b\#c\%\&")
        self.assertEqual(render._html("<b>&"), "&lt;b&gt;&amp;")
        self.assertNotIn("<", render._json_script({"v": "</script><b>"}))
        injected = _semantic()
        injected["labels"]["policies"] = {
            p: "</script><script>alert(1)</script>" for p in render.POLICIES
        }
        panel = render.prepare_f02(_prepared(injected))["panels"][0]
        self.assertNotIn("</script><script>alert(1)</script>", panel["html"])


if __name__ == "__main__":
    unittest.main()
