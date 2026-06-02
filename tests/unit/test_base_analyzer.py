"""Unit tests for BaseAnalyzer shared utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_analysis.base_analyzer import BaseAnalyzer


class ConcreteAnalyzer(BaseAnalyzer):
    """Minimal concrete subclass used only in tests."""

    pass


class TestCalculateAngle:
    """Tests for BaseAnalyzer._calculate_angle."""

    def setup_method(self):
        self.a = ConcreteAnalyzer()

    def test_right_angle(self):
        # p1=(1,0) p2=(0,0) p3=(0,1) — 90°
        angle = self.a._calculate_angle((1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 0.01

    def test_straight_line(self):
        # p1=(-1,0) p2=(0,0) p3=(1,0) — 180°
        angle = self.a._calculate_angle((-1, 0), (0, 0), (1, 0))
        assert abs(angle - 180.0) < 0.01

    def test_zero_vector_returns_zero(self):
        # Degenerate: p1 == p2
        angle = self.a._calculate_angle((0, 0), (0, 0), (1, 0))
        assert angle == 0.0

    def test_none_input_returns_zero(self):
        assert self.a._calculate_angle(None, (0, 0), (1, 0)) == 0.0
        assert self.a._calculate_angle((1, 0), None, (0, 0)) == 0.0

    def test_45_degree_angle(self):
        angle = self.a._calculate_angle((1, 0), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 0.5


class TestGetPoint:
    """Tests for BaseAnalyzer._get_point."""

    def setup_method(self):
        self.a = ConcreteAnalyzer()

    def test_tuple_value(self):
        kps = {"left_knee": (0.4, 0.7, 0.9)}
        assert self.a._get_point(kps, "left_knee") == (0.4, 0.7)

    def test_list_value(self):
        kps = {"right_hip": [0.6, 0.5]}
        assert self.a._get_point(kps, "right_hip") == (0.6, 0.5)

    def test_object_with_xy(self):
        class Lm:
            x, y = 0.3, 0.8

        kps = {"nose": Lm()}
        pt = self.a._get_point(kps, "nose")
        assert pt == (0.3, 0.8)

    def test_missing_key_returns_none(self):
        assert self.a._get_point({}, "left_ankle") is None

    def test_alt_names_fallback(self):
        kps = {"L.knee": (0.4, 0.7)}
        alt = {"left_knee": ["L.knee"]}
        assert self.a._get_point(kps, "left_knee", alt) == (0.4, 0.7)


class TestSmooth:
    """Tests for BaseAnalyzer._smooth."""

    def setup_method(self):
        self.a = ConcreteAnalyzer()

    def test_returns_same_length(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.a._smooth(values, window=3)
        assert len(result) == len(values)

    def test_too_short_returns_unchanged(self):
        values = [1.0, 2.0]
        result = self.a._smooth(values, window=5)
        assert result == values

    def test_constant_series_central_values_unchanged(self):
        # np.convolve mode='same' has edge effects on boundary elements;
        # only interior values (away from ends) should equal the constant.
        values = [3.0] * 10
        result = self.a._smooth(values, window=3)
        for v in result[1:-1]:
            assert abs(v - 3.0) < 0.01


class TestEMA:
    """Tests for BaseAnalyzer._ema."""

    def setup_method(self):
        self.a = ConcreteAnalyzer()

    def test_first_call_returns_value(self):
        result = self.a._ema("key", 10.0)
        assert result == 10.0

    def test_smoothing_converges(self):
        for _ in range(50):
            v = self.a._ema("sensor", 100.0)
        assert abs(v - 100.0) < 0.5

    def test_different_keys_independent(self):
        self.a._ema("a", 10.0)
        self.a._ema("b", 20.0)
        assert abs(self.a._ema_state["a"] - 10.0) < 0.01
        assert abs(self.a._ema_state["b"] - 20.0) < 0.01


class TestScoreInRange:
    """Tests for BaseAnalyzer._score_in_range."""

    def setup_method(self):
        self.a = ConcreteAnalyzer()

    def test_in_range_returns_100(self):
        assert self.a._score_in_range(50.0, 40.0, 60.0) == 100.0

    def test_at_boundary_returns_100(self):
        assert self.a._score_in_range(40.0, 40.0, 60.0) == 100.0

    def test_outside_range_penalized(self):
        score = self.a._score_in_range(70.0, 40.0, 60.0, penalty_per_unit=2.0)
        assert score == 80.0  # 100 - 10*2

    def test_score_never_negative(self):
        score = self.a._score_in_range(200.0, 40.0, 60.0, penalty_per_unit=2.0)
        assert score == 0.0
