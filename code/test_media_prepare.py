"""_prepare() format sniffing and its ffmpeg dependency."""
import os
import subprocess
import unittest
from unittest import mock

import media
from common import DATA

IMAGES = DATA / "media" / "images"


class PrepareSniffing(unittest.TestCase):
    def test_extensions_lie_so_the_avif_branch_is_reached_on_every_run(self):
        """img_020.jpg is AVIF, so a full run always needs the transcode path."""
        self.assertEqual(b"ftypavif", (IMAGES / "img_020.jpg").read_bytes()[4:12])

    def test_sniffed_formats_are_returned_without_transcoding(self):
        """Formats the API accepts must never shell out."""
        with mock.patch.object(media.subprocess, "run") as run:
            for name in ("img_001.jpg", "img_002.jpg", "img_003.jpg"):
                media_type, data = media._prepare(IMAGES / name)
                self.assertTrue(media_type.startswith("image/"), name)
                self.assertTrue(data, name)
        run.assert_not_called()


class PrepareTranscodeFailures(unittest.TestCase):
    """These exercise the transcode, so the binary must look present.

    Without stubbing which(), the guard above fires first and these test the
    wrong branch -- and fail outright on a machine that has no ffmpeg, which
    the rest of this suite is careful not to require.
    """

    def setUp(self):
        patcher = mock.patch.object(media.shutil, "which", return_value="/usr/bin/ffmpeg")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_transcode_failure_reports_what_ffmpeg_said(self):
        """CalledProcessError alone says only "non-zero exit status 1"."""
        failure = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"[avif @ 0x1] Invalid data found when processing input\n"
        )
        with mock.patch.object(media.subprocess, "run", side_effect=failure):
            with self.assertRaises(SystemExit) as caught:
                media._prepare(IMAGES / "img_020.jpg")

        message = str(caught.exception)
        self.assertIn("img_020.jpg", message)
        self.assertIn("Invalid data found when processing input", message)

    def test_a_hung_transcode_is_bounded_and_named(self):
        timeout = subprocess.TimeoutExpired("ffmpeg", media.TRANSCODE_TIMEOUT_S)
        with mock.patch.object(media.subprocess, "run", side_effect=timeout):
            with self.assertRaises(SystemExit) as caught:
                media._prepare(IMAGES / "img_020.jpg")

        message = str(caught.exception)
        self.assertIn("img_020.jpg", message)
        self.assertIn(str(media.TRANSCODE_TIMEOUT_S), message)

    def test_the_transcode_is_given_a_timeout(self):
        with mock.patch.object(media.subprocess, "run") as run:
            run.return_value = mock.Mock(stdout=b"\x89PNG\r\n\x1a\n")
            media._prepare(IMAGES / "img_020.jpg")

        self.assertEqual(run.call_args.kwargs["timeout"], media.TRANSCODE_TIMEOUT_S)


class PrepareWithoutFfmpeg(unittest.TestCase):
    def test_missing_ffmpeg_names_the_file_and_the_binary(self):
        with mock.patch.dict(os.environ, {"PATH": ""}):
            with self.assertRaises(SystemExit) as caught:
                media._prepare(IMAGES / "img_020.jpg")
        message = str(caught.exception)
        self.assertIn("img_020.jpg", message)
        self.assertIn("ffmpeg", message)


if __name__ == "__main__":
    unittest.main()
