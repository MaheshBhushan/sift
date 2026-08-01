"""--no-cache has to reach voice transcription too (issue #27).

The flag reached describe_images but not transcribe_voice, so cache/audio.json
was replayed on every run and a bad transcript could not be regenerated through
any supported path -- while still feeding the router as message text.

Stdlib unittest, no test dependency:

    python -m unittest test_media_voice_cache -v
"""
import sys
import types
import unittest
from unittest import mock

import media

ROW = {"voice_note_id": "vn_1", "file_path": "vn_1.ogg"}


class _Segment:
    def __init__(self, text):
        self.text = text


class _Info:
    language = "en"
    duration = 4.0


class TranscribeVoice(unittest.TestCase):
    """Driven through a stub cache and a stub whisper: no model download, no audio."""

    def setUp(self):
        self.store = {}
        cache = mock.Mock()
        cache.get.side_effect = self.store.get
        cache.set.side_effect = self.store.__setitem__
        cache.data = self.store
        patcher = mock.patch("media.JsonCache", return_value=cache)
        patcher.start()
        self.addCleanup(patcher.stop)

        # transcribe_voice imports faster_whisper lazily, so stand a fake module
        # in for it rather than requiring the real one to be installed.
        self.model = mock.Mock()
        self.model.transcribe.return_value = ([_Segment("hello there")], _Info())
        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = mock.Mock(return_value=self.model)
        patcher = mock.patch.dict(sys.modules, {"faster_whisper": fake})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, **kwargs):
        media.transcribe_voice([ROW], verbose=False, **kwargs)

    def test_a_transcript_is_cached(self):
        self._run()
        self.assertEqual(self.store["vn_1"]["transcript"], "hello there")

    def test_a_cached_transcript_is_reused(self):
        self._run()
        self.model.transcribe.reset_mock()
        self._run()

        self.model.transcribe.assert_not_called()

    def test_use_cache_false_re_transcribes(self):
        self._run()
        self.model.transcribe.reset_mock()
        self._run(use_cache=False)

        self.model.transcribe.assert_called_once()

    def test_use_cache_false_overwrites_the_stale_entry(self):
        self._run()
        self.model.transcribe.return_value = ([_Segment("corrected text")], _Info())
        self._run(use_cache=False)

        self.assertEqual(self.store["vn_1"]["transcript"], "corrected text")


if __name__ == "__main__":
    unittest.main()
