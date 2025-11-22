from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from video_analysis.video_overlay import VideoOverlayGenerator


def test_generate_annotated_video(tmp_path):
    frame_paths = []
    detections = []

    for index in range(2):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        frame_path = tmp_path / f"frame_{index}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))

        detections.append(
            {
                "frame_index": index,
                "bbox": [50, 30, 150, 90],
                "confidence": 0.9,
                "center": [100, 60],
                "lane": 2,
                "frame_path": str(frame_path),
            }
        )

    analysis = {"wall_touches": {"frames": [0, 1], "count": 2}}

    generator = VideoOverlayGenerator(output_dir=str(tmp_path), fps=1.0)
    output_path = generator.generate_annotated_video(
        frame_paths,
        detections,
        analysis=analysis,
    )

    output_file = Path(output_path)
    assert output_file.exists()
    assert output_file.stat().st_size > 0
