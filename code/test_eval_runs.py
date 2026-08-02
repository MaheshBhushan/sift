"""Repeated eval passes and the spread they report (issue #33).

The eval is 30 rows, so one flipped row is 3.3% and most changes worth making
produce deltas smaller than run-to-run variance. --runs exists to make that
noise floor visible instead of reporting a point estimate as if it were exact.

Stdlib unittest, no test dependency:

    python -m unittest test_eval_runs -v
"""
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import main

ROWS = [
    {"message_id": "m1", "action": "notify", "message_type": "urgent", "evidence_message_ids": "h1"},
    {"message_id": "m2", "action": "mute", "message_type": "spam", "evidence_message_ids": "none"},
]


def _decision(action, mtype, evidence="h1"):
    return {
        "action": action,
        "message_type": mtype,
        "evidence_message_ids": evidence,
        "reason": "because",
        "_rules_fired": [],
    }


def _results(*decisions):
    return [(row, {}, d) for row, d in zip(ROWS, decisions)]


class Score(unittest.TestCase):
    def test_returns_rates_rather_than_only_printing(self):
        with redirect_stdout(io.StringIO()):
            rates = main.score(_results(_decision("notify", "urgent"), _decision("mute", "spam")), ROWS)

        self.assertEqual(rates["action"], 1.0)
        self.assertEqual(rates["type"], 1.0)
        self.assertEqual(rates["both"], 1.0)

    def test_counts_a_miss(self):
        with redirect_stdout(io.StringIO()):
            rates = main.score(_results(_decision("digest", "urgent"), _decision("mute", "spam")), ROWS)

        self.assertEqual(rates["action"], 0.5)
        self.assertEqual(rates["both"], 0.5)

    def test_quiet_mode_prints_nothing(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main.score(_results(_decision("notify", "urgent"), _decision("mute", "spam")), ROWS, verbose=False)

        self.assertEqual(buf.getvalue(), "")

    def test_evidence_only_counts_scorable_rows(self):
        with redirect_stdout(io.StringIO()):
            rates = main.score(_results(_decision("notify", "urgent"), _decision("mute", "spam")), ROWS)

        # m2's gold evidence is "none", so only m1 is scorable.
        self.assertEqual(rates["evidence"], 1.0)


class EvalRuns(unittest.TestCase):
    """Driven through a stubbed Router: no API key, no network."""

    def setUp(self):
        self.routers = []

        def make_router(model=None, use_cache=True):
            r = mock.Mock(calls=0, asked=0, use_cache=use_cache)
            self.routers.append(r)
            return r

        patcher = mock.patch.object(main.route, "Router", side_effect=make_router)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, runs, sequence):
        """`sequence` is one results list per run."""
        with mock.patch.object(main, "run", side_effect=sequence), redirect_stdout(io.StringIO()):
            return main.eval_runs(ROWS, {}, "test-model", runs)

    def test_one_pass_per_run(self):
        perfect = _results(_decision("notify", "urgent"), _decision("mute", "spam"))
        per_run = self._run(3, [perfect, perfect, perfect])

        self.assertEqual(len(per_run), 3)
        self.assertEqual(len(self.routers), 3)

    def test_repeats_bypass_the_routing_cache(self):
        """Cached, run 2 would replay run 1 and every spread would read as zero."""
        perfect = _results(_decision("notify", "urgent"), _decision("mute", "spam"))
        self._run(3, [perfect, perfect, perfect])

        self.assertTrue(all(r.use_cache is False for r in self.routers))

    def test_variation_between_runs_is_preserved(self):
        good = _results(_decision("notify", "urgent"), _decision("mute", "spam"))
        bad = _results(_decision("digest", "urgent"), _decision("mute", "spam"))
        per_run = self._run(2, [good, bad])

        self.assertEqual([m["action"] for m in per_run], [1.0, 0.5])


class ReportSpread(unittest.TestCase):
    def _output(self, per_run):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main.report_spread(per_run)
        return buf.getvalue()

    def test_reports_mean_and_range(self):
        out = self._output([{"action": 1.0}, {"action": 0.5}, {"action": 0.75}])

        self.assertIn("75%", out)          # mean
        self.assertIn("[50%-100%]", out)   # observed range

    def test_a_stable_metric_reports_zero_spread(self):
        out = self._output([{"action": 1.0}, {"action": 1.0}])

        self.assertIn("spread 0%", out)

    def test_a_metric_missing_from_every_run_is_skipped(self):
        out = self._output([{"action": 1.0}, {"action": 1.0}])

        self.assertNotIn("evidence", out)


if __name__ == "__main__":
    unittest.main()
