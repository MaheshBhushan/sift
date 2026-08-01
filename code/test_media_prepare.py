"""_prepare() format sniffing and its ffmpeg dependency."""
import os
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
