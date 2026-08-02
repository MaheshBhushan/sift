"""The unseen-eval scaffold (issue #37).

Covers the sampler and the refusal to score unlabelled rows. It does NOT cover
the labels themselves -- those are human input this suite has no business
generating.

Stdlib unittest, no test dependency:

    python -m unittest test_label_sample -v
"""
import unittest
from unittest import mock

import label_sample
from common import load

MESSAGES = [{"message_id": f"m{i:02d}", "message_text": f"text {i}"} for i in range(1, 13)]
# Three cells: one large, two small. Round-robin must not let the large one
# crowd the small ones out.
PREDICTIONS = (
    [{"message_id": f"m{i:02d}", "action": "mute", "message_type": "scam"} for i in range(1, 9)]
    + [{"message_id": "m09", "action": "notify", "message_type": "urgent"}]
    + [{"message_id": "m10", "action": "notify", "message_type": "urgent"}]
    + [{"message_id": "m11", "action": "digest", "message_type": "promotion"}]
    + [{"message_id": "m12", "action": "digest", "message_type": "promotion"}]
)


class StratifiedIds(unittest.TestCase):
    def test_returns_the_requested_size(self):
        self.assertEqual(len(label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=6)), 6)

    def test_is_deterministic(self):
        a = label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=6)
        b = label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=6)

        self.assertEqual(a, b)

    def test_every_cell_is_represented_before_any_is_doubled(self):
        pred = {p["message_id"]: p for p in PREDICTIONS}
        picked = label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=3)
        cells = {(pred[i]["action"], pred[i]["message_type"]) for i in picked}

        self.assertEqual(len(cells), 3, "a rare cell was crowded out by the large one")

    def test_never_returns_duplicates(self):
        picked = label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=12)

        self.assertEqual(len(picked), len(set(picked)))

    def test_caps_at_the_available_rows(self):
        picked = label_sample.stratified_ids(MESSAGES, PREDICTIONS, size=99)

        self.assertEqual(len(picked), len(MESSAGES))

    def test_ignores_messages_with_no_prediction(self):
        extra = MESSAGES + [{"message_id": "m99", "message_text": "unpredicted"}]
        picked = label_sample.stratified_ids(extra, PREDICTIONS, size=12)

        self.assertNotIn("m99", picked)


class RealDataset(unittest.TestCase):
    def test_the_sample_spans_every_action_the_router_predicts(self):
        messages, predictions = load("messages"), load("output")
        picked = set(label_sample.stratified_ids(messages, predictions))
        actions = {p["action"] for p in predictions if p["message_id"] in picked}

        self.assertEqual(actions, {"notify", "digest", "mute"})

    def test_the_sample_is_drawn_only_from_the_unlabelled_110(self):
        known = {m["message_id"] for m in load("messages")}
        picked = label_sample.stratified_ids(load("messages"), load("output"))

        self.assertEqual(set(picked) - known, set())


class LabelledRows(unittest.TestCase):
    def test_an_unlabelled_sheet_yields_nothing(self):
        rows = [{"message_id": "m1", "action": ""}, {"message_id": "m2", "action": "  "}]
        with mock.patch("csv.DictReader", return_value=rows), mock.patch(
            "builtins.open", mock.mock_open()
        ), mock.patch.object(type(label_sample.SHEET), "exists", return_value=True):
            self.assertEqual(label_sample.labelled_rows(), [])

    def test_partially_labelled_sheets_yield_only_the_filled_rows(self):
        rows = [{"message_id": "m1", "action": "notify"}, {"message_id": "m2", "action": ""}]
        with mock.patch("csv.DictReader", return_value=rows), mock.patch(
            "builtins.open", mock.mock_open()
        ), mock.patch.object(type(label_sample.SHEET), "exists", return_value=True):
            got = label_sample.labelled_rows()

        self.assertEqual([r["message_id"] for r in got], ["m1"])

    def test_a_missing_sheet_yields_nothing(self):
        with mock.patch.object(type(label_sample.SHEET), "exists", return_value=False):
            self.assertEqual(label_sample.labelled_rows(), [])


if __name__ == "__main__":
    unittest.main()
