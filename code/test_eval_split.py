"""The scored rows must not be the few-shot examples (issue #31).

All 30 labelled rows were both the few-shot source and the eval set, so action
and type accuracy reported 100% and could not go down. --eval now scores only
the 15 rows FEWSHOT was not drawn from.

Stdlib unittest, no test dependency:

    python -m unittest test_eval_split -v
"""
import unittest

import main
import prompt
from common import load


class Split(unittest.TestCase):
    def setUp(self):
        self.rows = load("sample_messages")

    def test_the_split_is_half_and_half(self):
        held = main.eval_rows(self.rows)

        self.assertEqual(len(self.rows), 30)
        self.assertEqual(len(held), 15)
        self.assertEqual(len(main.FEWSHOT_SOURCE_IDS), 15)

    def test_no_scored_row_is_a_fewshot_source(self):
        held = {r["message_id"] for r in main.eval_rows(self.rows)}

        self.assertEqual(held & main.FEWSHOT_SOURCE_IDS, set())

    def test_every_named_source_id_exists_in_the_dataset(self):
        known = {r["message_id"] for r in self.rows}

        self.assertEqual(main.FEWSHOT_SOURCE_IDS - known, set())

    def test_eval_all_opts_back_into_the_contaminated_set(self):
        self.assertEqual(len(main.eval_rows(self.rows, held_out_only=False)), 30)

    def test_the_held_out_half_still_spans_every_action(self):
        held = main.eval_rows(self.rows)

        self.assertEqual({r["action"] for r in held}, {"notify", "digest", "mute"})

    def test_the_held_out_half_has_scorable_evidence(self):
        held = main.eval_rows(self.rows)
        scorable = [
            r for r in held
            if {x for x in r["evidence_message_ids"].split(";") if x and x != "none"}
        ]

        self.assertGreater(len(scorable), 10)


class SourceMappingHasNotDrifted(unittest.TestCase):
    """Pins the 12 examples whose source row is identifiable from gold evidence.

    Editing FEWSHOT to cite a held-out row's evidence would quietly recontaminate
    the eval, and nothing else in the suite would notice.
    """

    def test_every_evidence_citing_example_maps_into_the_source_half(self):
        rows = load("sample_messages")
        by_evidence = {}
        for r in rows:
            for e in (r["evidence_message_ids"] or "").split(";"):
                e = e.strip()
                if e and e != "none":
                    by_evidence.setdefault(e, []).append(r)

        mapped = 0
        for situation, out in prompt.FEWSHOT:
            ids = [
                x.strip()
                for x in str(out.get("evidence_message_ids") or "").split(";")
                if x.strip() and x.strip() != "none"
            ]
            sources = [r for e in ids for r in by_evidence.get(e, [])]
            matching = [
                r for r in sources
                if r["action"] == out["action"] and r["message_type"] == out["message_type"]
            ]
            if not matching:
                continue
            mapped += 1
            with self.subTest(situation=situation[:60]):
                self.assertIn(matching[0]["message_id"], main.FEWSHOT_SOURCE_IDS)

        self.assertEqual(mapped, 12, "the evidence-based mapping changed; re-derive the split")


if __name__ == "__main__":
    unittest.main()
