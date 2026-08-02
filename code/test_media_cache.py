"""A vision response that did not parse must not become a permanent answer.

Stdlib unittest, no test dependency:

    python -m unittest test_media_cache -v
"""
import unittest
from unittest import mock

import media

ROW = {"image_id": "img_1", "file_path": "x.jpg"}
GOOD = '{"kind": "poster", "summary": "A sale poster.", "risk_cues": ["urgency"]}'
UNPARSEABLE = "I am sorry, I cannot read this image."


class UsableEntry(unittest.TestCase):
    def test_a_real_description_is_reused(self):
        self.assertTrue(media._usable({"kind": "poster", "risk_cues": ["urgency"]}))

    def test_a_degraded_entry_is_not(self):
        self.assertFalse(media._usable({"kind": "other", "risk_cues": [], "degraded": True}))

    def test_a_missing_entry_is_not(self):
        self.assertFalse(media._usable(None))
        self.assertFalse(media._usable({}))


class DescribeImages(unittest.TestCase):
    """Driven through a stub cache and a stub client: no API key, no network."""

    def setUp(self):
        self.store = {}
        cache = mock.Mock()
        cache.get.side_effect = self.store.get
        cache.set.side_effect = self.store.__setitem__
        cache.data = self.store
        for target, kwargs in (
            ("media.JsonCache", {"return_value": cache}),
            ("media._prepare", {"return_value": ("image/png", b"bytes")}),
        ):
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, text, rows=(ROW,), **kwargs):
        block = mock.Mock(type="text")
        block.text = text
        client = mock.Mock()
        client.messages.create = mock.Mock(return_value=mock.Mock(content=[block]))
        with mock.patch("media._client", return_value=client):
            media.describe_images(list(rows), verbose=False, **kwargs)
        return client

    def test_an_unparseable_response_is_marked_degraded(self):
        self._run(UNPARSEABLE)
        self.assertTrue(self.store["img_1"]["degraded"])
        self.assertEqual(self.store["img_1"]["risk_cues"], [])

    def test_a_degraded_entry_is_re_asked_on_the_next_run(self):
        self._run(UNPARSEABLE)
        client = self._run(GOOD)

        client.messages.create.assert_called_once()
        self.assertEqual(self.store["img_1"]["kind"], "poster")
        self.assertNotIn("degraded", self.store["img_1"])

    def test_a_good_entry_is_not_re_asked(self):
        self._run(GOOD)
        client = self._run(GOOD)

        client.messages.create.assert_not_called()

    def test_use_cache_false_re_asks_even_a_good_entry(self):
        self._run(GOOD)
        block = mock.Mock(type="text")
        block.text = GOOD
        client = mock.Mock()
        client.messages.create = mock.Mock(return_value=mock.Mock(content=[block]))
        with mock.patch("media._client", return_value=client):
            media.describe_images([ROW], verbose=False, use_cache=False)

        client.messages.create.assert_called_once()

    # A refresh must not cost us a description we already paid for (issue #30).
    # --no-cache re-asks every image and writes the answer back; when that answer
    # fails to parse, the stub used to overwrite a good entry, leaving the cache
    # strictly worse than the run found it.

    def test_a_degraded_parse_does_not_overwrite_a_good_entry(self):
        self._run(GOOD)
        self._run(UNPARSEABLE, use_cache=False)

        self.assertEqual(self.store["img_1"]["kind"], "poster")
        self.assertEqual(self.store["img_1"]["risk_cues"], ["urgency"])
        self.assertNotIn("degraded", self.store["img_1"])

    def test_a_degraded_parse_still_fills_an_empty_slot(self):
        self._run(UNPARSEABLE, use_cache=False)

        self.assertTrue(self.store["img_1"]["degraded"])

    def test_a_degraded_parse_replaces_an_earlier_degraded_one(self):
        self._run(UNPARSEABLE)
        self._run("also unreadable", use_cache=False)

        self.assertEqual(self.store["img_1"]["summary"], "also unreadable")

    def test_a_good_parse_still_overwrites_under_no_cache(self):
        self._run(GOOD)
        self._run('{"kind": "flyer", "summary": "Updated.", "risk_cues": []}', use_cache=False)

        self.assertEqual(self.store["img_1"]["kind"], "flyer")


if __name__ == "__main__":
    unittest.main()
