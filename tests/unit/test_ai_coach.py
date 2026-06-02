"""Unit tests for AICoach — provider fallback chain and offline analysis."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_analysis.ai_coach import AICoach, CoachingAdvice, get_ai_coaching

SAMPLE_BIOMECHANICS = {
    "average_metrics": {
        "average_posture_score": 75.0,
        "average_drag_coefficient": 0.45,
        "average_streamline_score": 72.0,
        "frames_with_pose": 90,
        "total_frames": 100,
        "average_angles": {
            "head_elevation": 15.0,
            "body_streamline": 8.0,
        },
    },
    "recommendations": ["✅ Good posture"],
}

SAMPLE_SWIMMING_POSE = {
    "detection_rate": 0.9,
    "avg_streamline": 75.0,
    "avg_deviation": 5.0,
}


class TestAICoachOfflineMode:
    """Tests that don't require real API keys."""

    def test_offline_returns_coaching_advice(self):
        coach = AICoach(provider="offline")
        result = coach.analyze(
            biomechanics=SAMPLE_BIOMECHANICS,
            swimming_pose=SAMPLE_SWIMMING_POSE,
            athlete_name="Test Athlete",
        )
        assert isinstance(result, CoachingAdvice)
        assert isinstance(result.summary, str)
        assert isinstance(result.strengths, list)
        assert isinstance(result.improvements, list)
        assert isinstance(result.drills, list)
        assert 0 <= result.score <= 100
        assert result.priority in ("technique", "endurance", "speed")

    def test_offline_with_none_inputs(self):
        coach = AICoach(provider="offline")
        result = coach.analyze(biomechanics=None)
        assert isinstance(result, CoachingAdvice)

    def test_offline_with_empty_biomechanics(self):
        coach = AICoach(provider="offline")
        result = coach.analyze(biomechanics={})
        assert isinstance(result, CoachingAdvice)

    def test_offline_improvements_is_non_empty(self):
        """Offline mode should always return at least one improvement."""
        coach = AICoach(provider="offline")
        result = coach.analyze(biomechanics=SAMPLE_BIOMECHANICS)
        assert len(result.improvements) > 0


class TestAICoachProviderFallback:
    """Test that provider auto-detection falls back gracefully."""

    def test_auto_with_no_keys_uses_offline(self):
        """When no env keys are set, provider must be 'offline'."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing API keys from env
            import os

            env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
            with patch.dict("os.environ", env, clear=True):
                coach = AICoach(provider="auto")
                assert coach.provider == "offline"

    def test_anthropic_import_error_falls_back_to_offline(self):
        """If anthropic package is missing, should fall back to offline."""
        with patch.dict("sys.modules", {"anthropic": None}):
            coach = AICoach(provider="anthropic", api_key="fake-key")
            assert coach.provider == "offline"

    def test_openai_import_error_falls_back_to_offline(self):
        """If openai package is missing, should fall back to offline."""
        with patch.dict("sys.modules", {"openai": None}):
            coach = AICoach(provider="openai", api_key="fake-key")
            assert coach.provider == "offline"


class TestAICoachParseResponse:
    """Test LLM response parsing."""

    def test_parse_valid_json_response(self):
        coach = AICoach(provider="offline")
        valid_json = json.dumps(
            {
                "summary": "Good swimmer",
                "strengths": ["Fast", "Streamlined"],
                "improvements": ["Work on kick"],
                "drills": ["Kick drill"],
                "priority": "technique",
                "score": 80,
            }
        )
        result = coach._parse_response(valid_json)
        assert result.summary == "Good swimmer"
        assert result.score == 80
        assert "Fast" in result.strengths

    def test_parse_json_embedded_in_text(self):
        """JSON embedded in prose text should still be extracted."""
        coach = AICoach(provider="offline")
        response = 'Here is my analysis:\n{"summary": "Test", "strengths": [], "improvements": ["Improve kick"], "drills": [], "priority": "speed", "score": 65}\nEnd.'
        result = coach._parse_response(response)
        assert result.score == 65
        assert result.priority == "speed"

    def test_parse_invalid_json_returns_fallback(self):
        coach = AICoach(provider="offline")
        result = coach._parse_response("This is not JSON at all.")
        assert isinstance(result, CoachingAdvice)
        assert result.score == 50  # default fallback score

    def test_parse_partial_json_returns_fallback(self):
        coach = AICoach(provider="offline")
        result = coach._parse_response("{broken json: }")
        assert isinstance(result, CoachingAdvice)


class TestMockedAnthropicProvider:
    """Test Anthropic provider with mocked client."""

    def test_claude_success_path(self):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "summary": "Claude analysis",
                        "strengths": ["Good technique"],
                        "improvements": ["Improve turns"],
                        "drills": ["Wall drill"],
                        "priority": "technique",
                        "score": 78,
                    }
                )
            )
        ]
        mock_client.messages.create.return_value = mock_message

        coach = AICoach(provider="offline")  # Start offline
        coach.provider = "anthropic"  # Manually set
        coach.client = mock_client  # Inject mock

        result = coach.analyze(biomechanics=SAMPLE_BIOMECHANICS)
        assert result.score == 78
        assert result.summary == "Claude analysis"

    def test_claude_api_error_falls_back_to_offline(self):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        coach = AICoach(provider="offline")
        coach.provider = "anthropic"
        coach.client = mock_client

        # Must not raise — should fall back to offline result
        result = coach.analyze(biomechanics=SAMPLE_BIOMECHANICS)
        assert isinstance(result, CoachingAdvice)


class TestGetAICoachingConvenienceFunction:
    """Test the module-level get_ai_coaching() function."""

    def test_returns_coaching_advice(self):
        result = get_ai_coaching(
            biomechanics=SAMPLE_BIOMECHANICS,
            trajectory=None,
            splits=None,
            swimming_pose=SAMPLE_SWIMMING_POSE,
            athlete_name="Test",
        )
        assert isinstance(result, CoachingAdvice)
        assert hasattr(result, "score")
        assert hasattr(result, "summary")
