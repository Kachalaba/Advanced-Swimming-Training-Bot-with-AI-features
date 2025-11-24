from video_analysis.split_analyzer import SplitAnalyzer


def test_detect_wall_touches_marks_edges():
    analyzer = SplitAnalyzer(pool_length=25.0, fps=2.0)
    detections = [
        {"frame_index": 0, "center": [0.0, 0.0], "bbox": [0, 0, 1, 1]},
        {"frame_index": 1, "center": [2.0, 0.0], "bbox": [0, 0, 1, 1]},
        {"frame_index": 2, "center": [12.0, 0.0], "bbox": [0, 0, 1, 1]},
        {"frame_index": 3, "center": [20.0, 0.0], "bbox": [0, 0, 1, 1]},
        {"frame_index": 4, "center": [24.0, 0.0], "bbox": [0, 0, 1, 1]},
    ]

    touches = analyzer.detect_wall_touches(detections, edge_threshold=0.2)

    assert touches == [0, 4]
