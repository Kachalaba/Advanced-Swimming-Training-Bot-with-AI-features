"""Unit tests for backend video upload validation."""

import unittest

from backend.app.api.upload_validation import (
    MAX_VIDEO_UPLOAD_BYTES,
    UploadValidationError,
    copy_upload_with_limit,
    validate_video_upload,
)


class ChunkedFile:
    """Small file-like object that returns configured chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, size=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class Sink:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data


class UploadValidationTests(unittest.TestCase):
    def test_accepts_supported_video_suffix_and_content_type(self):
        suffix = validate_video_upload(
            filename="session.MP4",
            content_type="video/mp4",
            declared_size=1024,
        )

        self.assertEqual(suffix, ".mp4")

    def test_rejects_unsupported_suffix(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_video_upload(
                filename="notes.txt",
                content_type="video/mp4",
                declared_size=100,
            )

        self.assertIn("Unsupported video format", str(ctx.exception))

    def test_rejects_unsupported_content_type(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_video_upload(
                filename="session.mp4",
                content_type="application/pdf",
                declared_size=100,
            )

        self.assertIn("Unsupported content type", str(ctx.exception))

    def test_rejects_declared_size_above_limit(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_video_upload(
                filename="session.mp4",
                content_type="video/mp4",
                declared_size=MAX_VIDEO_UPLOAD_BYTES + 1,
            )

        self.assertIn("too large", str(ctx.exception))

    def test_copy_upload_with_limit_stops_oversized_stream(self):
        sink = Sink()
        source = ChunkedFile([b"abc", b"def"])

        with self.assertRaises(UploadValidationError) as ctx:
            copy_upload_with_limit(source, sink, max_bytes=5)

        self.assertIn("too large", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
