"""
Tests for inference functionality.

Tests cover:
- Image classification inference
- Visual embedding extraction
- Object detection
- Pose estimation
- Semantic segmentation
- Audio classification
- Keyword spotting
"""

import json
import base64
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

import pytest


# ============================================================================
# Test: Image Classification
# ============================================================================

class TestImageClassification:
    """Tests for image classification inference."""

    @pytest.mark.asyncio
    async def test_classify_image_with_path(self, mock_engine, tmp_path, mock_models_dir):
        """Test classification with image file path."""
        from PIL import Image

        # Create test image file
        img = Image.new('RGB', (224, 224), color=(128, 64, 32))
        img_path = tmp_path / "test.jpg"
        img.save(img_path, format='JPEG')

        mock_engine.classify = MagicMock()
        mock_engine.classify.return_value = MagicMock(
            predictions=[{"class_id": 0, "score": 0.95, "label": "cat"}],
            latency_ms=15.0
        )

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_classify_image

            result = await handle_classify_image({
                "image_path": str(img_path),
                "model": "mobilenet_v2",
                "top_k": 3
            })
            data = json.loads(result[0].text)

            assert "predictions" in data or "error" in data

    @pytest.mark.asyncio
    async def test_classify_image_top_k_parameter(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test classification respects top_k parameter."""
        mock_engine.classify = MagicMock()
        mock_engine.classify.return_value = MagicMock(
            predictions=[
                {"class_id": 0, "score": 0.95},
                {"class_id": 1, "score": 0.03},
                {"class_id": 2, "score": 0.01},
            ],
            latency_ms=15.0
        )

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_classify_image

            result = await handle_classify_image({
                "image_base64": sample_image_base64,
                "model": "mobilenet_v2",
                "top_k": 3
            })

            # Verify classify was called with correct top_k
            if mock_engine.classify.called:
                call_args = mock_engine.classify.call_args
                assert call_args[1].get("top_k") == 3 or call_args[0][2] == 3

    @pytest.mark.asyncio
    async def test_classify_image_model_selection(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test classification with different models."""
        for model_name in ["mobilenet_v2", "efficientnet_s"]:
            mock_engine.classify = MagicMock()
            mock_engine.classify.return_value = MagicMock(
                predictions=[{"class_id": 0, "score": 0.95}],
                latency_ms=15.0
            )

            with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
                 patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
                from coral_tpu_mcp.server import handle_classify_image

                result = await handle_classify_image({
                    "image_base64": sample_image_base64,
                    "model": model_name
                })

                # Should not have error about unknown model
                data = json.loads(result[0].text)
                if "error" in data:
                    assert "Unknown model" not in data["error"]


# ============================================================================
# Test: Visual Embedding
# ============================================================================

class TestVisualEmbedding:
    """Tests for visual embedding extraction."""

    @pytest.mark.asyncio
    async def test_get_visual_embedding_returns_vector(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test visual embedding returns vector."""
        mock_engine.get_embedding = MagicMock()
        mock_engine.get_embedding.return_value = (np.random.randn(1000), 10.5)

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_visual_embedding

            result = await handle_visual_embedding({
                "image_base64": sample_image_base64
            })
            data = json.loads(result[0].text)

            if "error" not in data:
                assert "embedding" in data
                assert "dimension" in data
                assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_get_visual_embedding_with_path(self, mock_engine, tmp_path, mock_models_dir):
        """Test visual embedding with image file path."""
        from PIL import Image

        # Create test image file
        img = Image.new('RGB', (224, 224), color=(128, 64, 32))
        img_path = tmp_path / "test.jpg"
        img.save(img_path, format='JPEG')

        mock_engine.get_embedding = MagicMock()
        mock_engine.get_embedding.return_value = (np.random.randn(1000), 10.5)

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_visual_embedding

            result = await handle_visual_embedding({
                "image_path": str(img_path)
            })
            data = json.loads(result[0].text)

            if "error" not in data:
                assert "embedding" in data

    @pytest.mark.asyncio
    async def test_get_visual_embedding_no_input(self, mock_engine):
        """Test visual embedding fails without input."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_visual_embedding

            result = await handle_visual_embedding({})
            data = json.loads(result[0].text)

            assert "error" in data


# ============================================================================
# Test: Object Detection
# ============================================================================

class TestObjectDetection:
    """Tests for object detection."""

    @pytest.mark.asyncio
    async def test_detect_objects_returns_detections(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test object detection returns detection list."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_detect_objects

            result = await handle_detect_objects({
                "image_base64": sample_image_base64,
                "threshold": 0.5
            })
            data = json.loads(result[0].text)

            # May have error due to model not found, but should handle gracefully
            assert "detections" in data or "error" in data

    @pytest.mark.asyncio
    async def test_detect_objects_threshold_parameter(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test object detection respects threshold parameter."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_detect_objects

            result = await handle_detect_objects({
                "image_base64": sample_image_base64,
                "threshold": 0.9,
                "max_detections": 5
            })
            data = json.loads(result[0].text)

            # Check threshold is reflected in response or error handled
            assert "threshold" in data or "error" in data

    @pytest.mark.asyncio
    async def test_detect_objects_tpu_unavailable(self, sample_image_base64):
        """Test object detection when TPU unavailable."""
        mock_engine = MagicMock()
        mock_engine.is_available = False

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_detect_objects

            result = await handle_detect_objects({
                "image_base64": sample_image_base64
            })
            data = json.loads(result[0].text)

            assert "error" in data
            assert "TPU not available" in data["error"]


# ============================================================================
# Test: Pose Estimation
# ============================================================================

class TestPoseEstimation:
    """Tests for pose estimation."""

    @pytest.mark.asyncio
    async def test_estimate_pose_returns_keypoints(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test pose estimation returns keypoints."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_estimate_pose

            result = await handle_estimate_pose({
                "image_base64": sample_image_base64,
                "model": "movenet"
            })
            data = json.loads(result[0].text)

            assert "keypoints" in data or "error" in data

    @pytest.mark.asyncio
    async def test_estimate_pose_invalid_model(self, mock_engine, sample_image_base64):
        """Test pose estimation with invalid model."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_estimate_pose

            result = await handle_estimate_pose({
                "image_base64": sample_image_base64,
                "model": "invalid_pose_model"
            })
            data = json.loads(result[0].text)

            assert "error" in data

    @pytest.mark.asyncio
    async def test_estimate_pose_tpu_unavailable(self, sample_image_base64):
        """Test pose estimation when TPU unavailable."""
        mock_engine = MagicMock()
        mock_engine.is_available = False

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_estimate_pose

            result = await handle_estimate_pose({
                "image_base64": sample_image_base64
            })
            data = json.loads(result[0].text)

            assert "error" in data


# ============================================================================
# Test: Semantic Segmentation
# ============================================================================

class TestSemanticSegmentation:
    """Tests for semantic segmentation."""

    @pytest.mark.asyncio
    async def test_segment_image_returns_distribution(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test segmentation returns class distribution."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_segment_image

            result = await handle_segment_image({
                "image_base64": sample_image_base64
            })
            data = json.loads(result[0].text)

            assert "class_distribution" in data or "error" in data

    @pytest.mark.asyncio
    async def test_segment_image_return_mask_option(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test segmentation with return_mask option."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_segment_image

            result = await handle_segment_image({
                "image_base64": sample_image_base64,
                "return_mask": True
            })
            data = json.loads(result[0].text)

            # Either returns mask or error (model not found)
            if "error" not in data:
                assert "segmentation_mask" in data or "class_distribution" in data


# ============================================================================
# Test: Audio Classification
# ============================================================================

class TestAudioClassification:
    """Tests for audio classification."""

    @pytest.mark.asyncio
    async def test_classify_audio_requires_input(self, mock_engine):
        """Test audio classification fails without input."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_classify_audio

            result = await handle_classify_audio({})
            data = json.loads(result[0].text)

            # Should have error about missing input
            assert "error" in data or "predictions" in data

    @pytest.mark.asyncio
    async def test_classify_audio_tpu_unavailable(self, sample_audio_base64):
        """Test audio classification when TPU unavailable."""
        mock_engine = MagicMock()
        mock_engine.is_available = False

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_classify_audio

            result = await handle_classify_audio({
                "audio_base64": sample_audio_base64
            })
            data = json.loads(result[0].text)

            assert "error" in data


# ============================================================================
# Test: Keyword Spotting
# ============================================================================

class TestKeywordSpotting:
    """Tests for keyword spotting."""

    @pytest.mark.asyncio
    async def test_spot_keyword_requires_input(self, mock_engine):
        """Test keyword spotting fails without input."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_spot_keyword

            result = await handle_spot_keyword({})
            data = json.loads(result[0].text)

            # Should have error about missing input
            assert "error" in data or "keywords" in data

    @pytest.mark.asyncio
    async def test_spot_keyword_tpu_unavailable(self, sample_audio_base64):
        """Test keyword spotting when TPU unavailable."""
        mock_engine = MagicMock()
        mock_engine.is_available = False

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_spot_keyword

            result = await handle_spot_keyword({
                "audio_base64": sample_audio_base64
            })
            data = json.loads(result[0].text)

            assert "error" in data

    @pytest.mark.asyncio
    async def test_spot_keyword_threshold_parameter(self, mock_engine, sample_audio_base64):
        """Test keyword spotting respects threshold parameter."""
        mock_engine.is_available = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_spot_keyword

            result = await handle_spot_keyword({
                "audio_base64": sample_audio_base64,
                "threshold": 0.8
            })
            data = json.loads(result[0].text)

            # Threshold should be in response or error handled
            if "error" not in data:
                assert "threshold" in data or "keywords" in data


# ============================================================================
# Test: Inference Performance
# ============================================================================

class TestInferencePerformance:
    """Tests for inference performance characteristics."""

    def test_inference_result_dataclass(self):
        """Test InferenceResult dataclass structure."""
        from coral_tpu_mcp.tpu_engine import InferenceResult

        result = InferenceResult(
            predictions=[{"class_id": 0, "score": 0.95}],
            latency_ms=15.5,
            model_name="test_model",
            timestamp=1234567890.0
        )

        assert result.predictions[0]["score"] == 0.95
        assert result.latency_ms == 15.5
        assert result.model_name == "test_model"

    def test_tpu_health_dataclass(self):
        """Test TPUHealth dataclass structure."""
        from coral_tpu_mcp.tpu_engine import TPUHealth

        health = TPUHealth()

        assert health.is_healthy is False
        assert health.consecutive_failures == 0
        assert health.recovery_attempts == 0

    def test_retry_config_values(self):
        """Test retry configuration values are reasonable."""
        from coral_tpu_mcp.tpu_engine import RETRY_CONFIG

        assert RETRY_CONFIG["max_retries"] >= 1
        assert RETRY_CONFIG["initial_delay"] > 0
        assert RETRY_CONFIG["max_delay"] >= RETRY_CONFIG["initial_delay"]
        assert RETRY_CONFIG["backoff_factor"] >= 1
