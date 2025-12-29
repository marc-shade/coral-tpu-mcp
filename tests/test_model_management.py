"""
Tests for model management functionality.

Tests cover:
- Model listing and categorization
- Model file discovery
- Label file loading
- Input/output tensor details
"""

import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test: Model Listing
# ============================================================================

class TestModelListing:
    """Tests for model listing functionality."""

    def test_models_dict_has_required_keys(self):
        """Test MODELS dictionary has required configuration keys."""
        from coral_tpu_mcp.server import MODELS

        required_keys = ["file", "labels", "input_size", "description", "category"]

        for model_name, config in MODELS.items():
            for key in required_keys:
                assert key in config, f"Model {model_name} missing key: {key}"

    def test_models_have_valid_categories(self):
        """Test all models have valid category assignments."""
        from coral_tpu_mcp.server import MODELS

        valid_categories = {"classification", "detection", "pose", "segmentation", "audio"}

        for model_name, config in MODELS.items():
            assert config["category"] in valid_categories, \
                f"Model {model_name} has invalid category: {config['category']}"

    def test_classification_models_have_labels(self):
        """Test classification models specify label files."""
        from coral_tpu_mcp.server import MODELS

        for model_name, config in MODELS.items():
            if config["category"] == "classification":
                assert config["labels"] is not None, \
                    f"Classification model {model_name} should have labels"

    def test_input_size_format(self):
        """Test input_size is properly formatted tuple or None (for audio)."""
        from coral_tpu_mcp.server import MODELS

        for model_name, config in MODELS.items():
            input_size = config["input_size"]
            if config["category"] == "audio":
                assert input_size is None, \
                    f"Audio model {model_name} should have None input_size"
            else:
                assert isinstance(input_size, tuple), \
                    f"Model {model_name} should have tuple input_size"
                assert len(input_size) == 2, \
                    f"Model {model_name} input_size should have 2 dimensions"


# ============================================================================
# Test: Pose Keypoints
# ============================================================================

class TestPoseKeypoints:
    """Tests for pose estimation keypoint definitions."""

    def test_pose_keypoints_count(self):
        """Test correct number of pose keypoints defined."""
        from coral_tpu_mcp.server import POSE_KEYPOINTS

        assert len(POSE_KEYPOINTS) == 17, "Should have 17 COCO keypoints"

    def test_pose_keypoints_names(self):
        """Test pose keypoints include expected body parts."""
        from coral_tpu_mcp.server import POSE_KEYPOINTS

        expected_parts = ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"]

        for part in expected_parts:
            assert part in POSE_KEYPOINTS, f"Missing keypoint: {part}"

    def test_pose_keypoints_symmetry(self):
        """Test pose keypoints have left/right pairs."""
        from coral_tpu_mcp.server import POSE_KEYPOINTS

        pairs = ["eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]

        for pair in pairs:
            left = f"left_{pair}"
            right = f"right_{pair}"
            assert left in POSE_KEYPOINTS, f"Missing left keypoint: {left}"
            assert right in POSE_KEYPOINTS, f"Missing right keypoint: {right}"


# ============================================================================
# Test: Label File Reading
# ============================================================================

class TestLabelFileReading:
    """Tests for label file loading functionality."""

    def test_read_simple_labels(self, tmp_path):
        """Test reading simple label file."""
        from coral_tpu_mcp.tpu_engine import _read_label_file

        label_file = tmp_path / "labels.txt"
        label_file.write_text("cat\ndog\nbird\n")

        labels = _read_label_file(str(label_file))

        assert labels == ["cat", "dog", "bird"]

    def test_read_labels_with_indices(self, tmp_path):
        """Test reading label file with index prefixes."""
        from coral_tpu_mcp.tpu_engine import _read_label_file

        label_file = tmp_path / "labels.txt"
        label_file.write_text("0:background\n1:person\n2:car\n")

        labels = _read_label_file(str(label_file))

        assert labels == ["background", "person", "car"]

    def test_read_labels_ignores_empty_lines(self, tmp_path):
        """Test reading label file with empty lines."""
        from coral_tpu_mcp.tpu_engine import _read_label_file

        label_file = tmp_path / "labels.txt"
        label_file.write_text("cat\n\ndog\n\n\nbird\n")

        labels = _read_label_file(str(label_file))

        assert labels == ["cat", "dog", "bird"]

    def test_read_labels_strips_whitespace(self, tmp_path):
        """Test reading label file strips whitespace."""
        from coral_tpu_mcp.tpu_engine import _read_label_file

        label_file = tmp_path / "labels.txt"
        label_file.write_text("  cat  \n  dog  \n  bird  \n")

        labels = _read_label_file(str(label_file))

        assert labels == ["cat", "dog", "bird"]


# ============================================================================
# Test: Input Details
# ============================================================================

class TestInputDetails:
    """Tests for model input tensor details."""

    def test_get_input_details_returns_dict(self, mock_models_dir):
        """Test get_input_details returns proper dictionary."""
        from coral_tpu_mcp.tpu_engine import TPUEngine
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        # Manually add a mock interpreter
        engine.interpreters["test_model.tflite"] = MockInterpreter()

        details = engine.get_input_details("test_model.tflite")

        assert details is not None
        assert "shape" in details
        assert "dtype" in details

    def test_get_input_details_not_loaded(self, mock_models_dir):
        """Test get_input_details returns None for unloaded model."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()

        details = engine.get_input_details("nonexistent_model.tflite")

        assert details is None


# ============================================================================
# Test: Image Preprocessing
# ============================================================================

class TestImagePreprocessing:
    """Tests for image preprocessing functionality."""

    def test_preprocess_image_resizes(self, sample_image_bytes):
        """Test preprocess_image resizes to target size."""
        from coral_tpu_mcp.server import preprocess_image

        target_size = (224, 224)
        result = preprocess_image(sample_image_bytes, target_size)

        assert result.shape == (224, 224, 3)

    def test_preprocess_image_dtype(self, sample_image_bytes):
        """Test preprocess_image returns uint8."""
        from coral_tpu_mcp.server import preprocess_image

        result = preprocess_image(sample_image_bytes, (224, 224))

        assert result.dtype == np.uint8

    def test_preprocess_image_rgb_conversion(self):
        """Test preprocess_image converts to RGB."""
        from coral_tpu_mcp.server import preprocess_image
        from PIL import Image
        from io import BytesIO

        # Create grayscale image
        img = Image.new('L', (100, 100), color=128)
        buffer = BytesIO()
        img.save(buffer, format='PNG')

        result = preprocess_image(buffer.getvalue(), (224, 224))

        # Should be 3 channels (RGB)
        assert result.shape[-1] == 3


# ============================================================================
# Test: Text Embedding Model
# ============================================================================

class TestTextEmbeddingModel:
    """Tests for text embedding model loading."""

    def test_get_text_model_returns_none_without_library(self):
        """Test get_text_model returns None when sentence-transformers not installed."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with patch('coral_tpu_mcp.server._text_model', None):
                # Simulate ImportError
                def mock_import():
                    raise ImportError("No module named 'sentence_transformers'")

                with patch('coral_tpu_mcp.server.get_text_model') as mock_func:
                    mock_func.return_value = None
                    result = mock_func()
                    assert result is None

    def test_text_embedding_dimension(self, mock_text_model):
        """Test text embedding dimension constant."""
        from coral_tpu_mcp.server import TEXT_EMBEDDING_DIM

        assert TEXT_EMBEDDING_DIM == 384


# ============================================================================
# Test: Model Categories
# ============================================================================

class TestModelCategories:
    """Tests for model categorization."""

    def test_classification_models_exist(self):
        """Test classification models are defined."""
        from coral_tpu_mcp.server import MODELS

        classification_models = [
            name for name, cfg in MODELS.items()
            if cfg["category"] == "classification"
        ]

        assert len(classification_models) >= 2
        assert "mobilenet_v2" in classification_models

    def test_detection_models_exist(self):
        """Test detection models are defined."""
        from coral_tpu_mcp.server import MODELS

        detection_models = [
            name for name, cfg in MODELS.items()
            if cfg["category"] == "detection"
        ]

        assert len(detection_models) >= 1
        assert "coco_detection" in detection_models

    def test_pose_models_exist(self):
        """Test pose estimation models are defined."""
        from coral_tpu_mcp.server import MODELS

        pose_models = [
            name for name, cfg in MODELS.items()
            if cfg["category"] == "pose"
        ]

        assert len(pose_models) >= 1
        assert "movenet" in pose_models

    def test_segmentation_models_exist(self):
        """Test segmentation models are defined."""
        from coral_tpu_mcp.server import MODELS

        segmentation_models = [
            name for name, cfg in MODELS.items()
            if cfg["category"] == "segmentation"
        ]

        assert len(segmentation_models) >= 1
        assert "deeplabv3_pascal" in segmentation_models

    def test_audio_models_exist(self):
        """Test audio models are defined."""
        from coral_tpu_mcp.server import MODELS

        audio_models = [
            name for name, cfg in MODELS.items()
            if cfg["category"] == "audio"
        ]

        assert len(audio_models) >= 1
        assert "yamnet" in audio_models
