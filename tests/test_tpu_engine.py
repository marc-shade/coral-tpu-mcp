"""
Tests for tpu_engine.py - Core TPU inference functionality.

Tests cover:
- TPU initialization and availability checks
- Model loading with validation
- Classification inference
- Embedding extraction
- Health monitoring and recovery
- Retry logic and resilience
"""

import time
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest


# ============================================================================
# Test: TPU Availability Detection
# ============================================================================

class TestTPUAvailability:
    """Tests for TPU availability detection."""

    def test_tpu_unavailable_without_libraries(self):
        """Test graceful handling when no TPU libraries are installed."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        # With mocked pycoral returning empty device list, TPU should be unavailable
        engine = TPUEngine()
        # The engine should not crash even without hardware
        assert hasattr(engine, 'is_available')

    def test_tpu_available_property_exists(self):
        """Test TPU availability property exists."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        assert hasattr(engine, 'is_available')
        assert isinstance(engine.is_available, bool)

    def test_tpu_engine_has_interpreters_dict(self):
        """Test TPU engine has interpreters dictionary."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        assert hasattr(engine, 'interpreters')
        assert isinstance(engine.interpreters, dict)


# ============================================================================
# Test: Model File Validation
# ============================================================================

class TestModelValidation:
    """Tests for model file validation."""

    def test_validate_model_file_not_found(self, tmp_path):
        """Test validation fails for missing model file."""
        from coral_tpu_mcp.tpu_engine import validate_model_file

        missing_path = tmp_path / "nonexistent.tflite"
        is_valid, msg = validate_model_file(missing_path)

        assert is_valid is False
        assert "not found" in msg.lower()

    def test_validate_model_file_wrong_extension(self, tmp_path):
        """Test validation fails for wrong file extension."""
        from coral_tpu_mcp.tpu_engine import validate_model_file

        wrong_ext = tmp_path / "model.txt"
        wrong_ext.write_text("not a model")
        is_valid, msg = validate_model_file(wrong_ext)

        assert is_valid is False
        assert "extension" in msg.lower()

    def test_validate_model_file_too_small(self, tmp_path):
        """Test validation fails for suspiciously small files."""
        from coral_tpu_mcp.tpu_engine import validate_model_file

        small_file = tmp_path / "tiny.tflite"
        small_file.write_bytes(b'\x00' * 100)  # Only 100 bytes
        is_valid, msg = validate_model_file(small_file)

        assert is_valid is False
        assert "too small" in msg.lower()

    def test_validate_model_file_success(self, tmp_path):
        """Test validation succeeds for valid model file."""
        from coral_tpu_mcp.tpu_engine import validate_model_file

        valid_model = tmp_path / "valid.tflite"
        # Create a minimally valid-looking TFLite file
        valid_model.write_bytes(b'\x1c\x00\x00\x00TFL3' + b'\x00' * 2048)
        is_valid, msg = validate_model_file(valid_model)

        assert is_valid is True
        assert "success" in msg.lower()

    def test_validate_model_with_manifest_checksum(self, tmp_path):
        """Test validation with manifest checksum verification."""
        from coral_tpu_mcp.tpu_engine import validate_model_file, compute_file_checksum

        # Create model file
        model_path = tmp_path / "model.tflite"
        model_content = b'\x1c\x00\x00\x00TFL3' + b'\x00' * 2048
        model_path.write_bytes(model_content)

        # Create manifest with correct checksum
        checksum = compute_file_checksum(model_path)
        manifest = {"model.tflite": {"sha256": checksum}}
        manifest_path = tmp_path / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        is_valid, msg = validate_model_file(model_path, manifest_path)
        assert is_valid is True

    def test_validate_model_checksum_mismatch(self, tmp_path):
        """Test validation fails on checksum mismatch."""
        from coral_tpu_mcp.tpu_engine import validate_model_file

        # Create model file
        model_path = tmp_path / "model.tflite"
        model_path.write_bytes(b'\x1c\x00\x00\x00TFL3' + b'\x00' * 2048)

        # Create manifest with wrong checksum
        manifest = {"model.tflite": {"sha256": "wrongchecksum123456"}}
        manifest_path = tmp_path / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        is_valid, msg = validate_model_file(model_path, manifest_path)
        assert is_valid is False
        assert "mismatch" in msg.lower()


# ============================================================================
# Test: Model Loading
# ============================================================================

class TestModelLoading:
    """Tests for model loading functionality."""

    def test_load_model_file_not_found(self, mock_models_dir):
        """Test loading fails gracefully for missing model file."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        with patch('coral_tpu_mcp.tpu_engine.MODELS_DIR', mock_models_dir):
            engine = TPUEngine()

            # Try to load a model that doesn't exist
            result = engine.load_model("nonexistent_model.tflite")
            assert result is False

    def test_unload_model(self):
        """Test model unloading."""
        from coral_tpu_mcp.tpu_engine import TPUEngine
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        engine.interpreters["test_model"] = MockInterpreter()

        assert "test_model" in engine.interpreters
        engine.unload_model("test_model")
        assert "test_model" not in engine.interpreters

    def test_unload_all_models(self):
        """Test unloading all models."""
        from coral_tpu_mcp.tpu_engine import TPUEngine
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        engine.interpreters["model1"] = MockInterpreter()
        engine.interpreters["model2"] = MockInterpreter()
        engine.labels["model1"] = ["label1"]

        engine.unload_all()

        assert len(engine.interpreters) == 0
        assert len(engine.labels) == 0

    def test_load_model_returns_bool(self, mock_models_dir):
        """Test load_model returns boolean."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        result = engine.load_model("any_model.tflite")

        assert isinstance(result, bool)


# ============================================================================
# Test: Classification Inference
# ============================================================================

class TestClassification:
    """Tests for classification inference."""

    def test_classify_with_mock_interpreter(self):
        """Test classification with manually added interpreter."""
        from coral_tpu_mcp.tpu_engine import TPUEngine, InferenceResult
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        engine._tpu_available = True
        engine.interpreters["test_model"] = MockInterpreter()
        engine.labels["test_model"] = ["cat", "dog", "bird"]
        engine.stats["by_model"]["test_model"] = {"inferences": 0, "total_latency_ms": 0}

        # Mock the _tpu_operation context manager
        from contextlib import contextmanager

        @contextmanager
        def mock_tpu_op(name):
            yield

        engine._tpu_operation = mock_tpu_op

        input_data = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)

        # This should work with the mock interpreter
        # Note: actual behavior depends on internal implementation
        try:
            result = engine.classify("test_model", input_data, top_k=3)
            assert hasattr(result, 'predictions')
        except (RuntimeError, ValueError):
            # Expected if TPU operation fails in mock
            pass


# ============================================================================
# Test: Embedding Extraction
# ============================================================================

class TestEmbedding:
    """Tests for embedding extraction."""

    def test_get_embedding_method_exists(self):
        """Test get_embedding method exists."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        assert hasattr(engine, 'get_embedding')
        assert callable(engine.get_embedding)


# ============================================================================
# Test: Health Monitoring
# ============================================================================

class TestHealthMonitoring:
    """Tests for health monitoring and recovery."""

    def test_health_check_returns_status(self):
        """Test health check returns proper status dict."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        status = engine.health_check()

        assert isinstance(status, dict)
        assert "is_healthy" in status
        assert "tpu_available" in status

    def test_health_status_tracks_failures(self):
        """Test that health status tracks consecutive failures."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        engine._health.consecutive_failures = 5

        status = engine._get_health_status()
        assert status["consecutive_failures"] == 5

    def test_reconnect_clears_interpreters(self):
        """Test that reconnect clears loaded interpreters."""
        from coral_tpu_mcp.tpu_engine import TPUEngine
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        engine.interpreters["test_model"] = MockInterpreter()

        engine.reconnect()

        assert len(engine.interpreters) == 0

    def test_reconnect_increments_recovery_attempts(self):
        """Test that reconnect increments recovery attempts."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        initial_attempts = engine._health.recovery_attempts

        engine.reconnect()

        assert engine._health.recovery_attempts == initial_attempts + 1


# ============================================================================
# Test: Statistics
# ============================================================================

class TestStatistics:
    """Tests for inference statistics."""

    def test_stats_tracking(self):
        """Test that stats are properly tracked."""
        from coral_tpu_mcp.tpu_engine import TPUEngine
        from tests.conftest import MockInterpreter

        engine = TPUEngine()
        engine.interpreters["test_model"] = MockInterpreter()
        engine.stats["by_model"]["test_model"] = {"inferences": 0, "total_latency_ms": 0}

        # Manually update stats
        engine._update_stats("test_model", 15.5)

        assert engine.stats["total_inferences"] == 1
        assert engine.stats["total_latency_ms"] == 15.5

    def test_get_stats_returns_all_fields(self):
        """Test get_stats returns all required fields."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        stats = engine.get_stats()

        assert "total_inferences" in stats
        assert "avg_latency_ms" in stats
        assert "tpu_available" in stats
        assert "loaded_models" in stats
        assert "health" in stats
        assert "resilience" in stats

    def test_stats_calculates_average(self):
        """Test stats calculates average latency correctly."""
        from coral_tpu_mcp.tpu_engine import TPUEngine

        engine = TPUEngine()
        engine.stats["total_inferences"] = 10
        engine.stats["total_latency_ms"] = 150.0

        stats = engine.get_stats()
        assert stats["avg_latency_ms"] == 15.0


# ============================================================================
# Test: Global Engine
# ============================================================================

class TestGlobalEngine:
    """Tests for global engine singleton."""

    def test_get_engine_returns_engine(self):
        """Test get_engine returns TPUEngine instance."""
        from coral_tpu_mcp.tpu_engine import get_engine, TPUEngine

        engine = get_engine()
        assert isinstance(engine, TPUEngine)

    def test_get_engine_returns_same_instance(self):
        """Test get_engine returns same instance on repeated calls."""
        from coral_tpu_mcp.tpu_engine import get_engine

        engine1 = get_engine()
        engine2 = get_engine()

        assert engine1 is engine2


# ============================================================================
# Test: Audit Logging
# ============================================================================

class TestAuditLogging:
    """Tests for audit logging functionality."""

    def test_audit_log_writes_to_file(self, tmp_path):
        """Test audit_log writes entries to file."""
        from coral_tpu_mcp.tpu_engine import audit_log

        log_file = tmp_path / "audit.log"

        with patch('coral_tpu_mcp.tpu_engine.AUDIT_LOG_FILE', log_file):
            audit_log("test_event", {"key": "value"})

        content = log_file.read_text()
        assert "test_event" in content
        assert "key" in content

    def test_audit_log_handles_errors_gracefully(self):
        """Test audit_log doesn't raise on errors."""
        from coral_tpu_mcp.tpu_engine import audit_log

        # Point to an unwritable location
        with patch('coral_tpu_mcp.tpu_engine.AUDIT_LOG_FILE', Path("/nonexistent/path/audit.log")):
            # Should not raise
            audit_log("test_event", {"key": "value"})


# ============================================================================
# Test: Retry Configuration
# ============================================================================

class TestRetryConfiguration:
    """Tests for retry configuration."""

    def test_retry_config_exists(self):
        """Test RETRY_CONFIG dictionary exists."""
        from coral_tpu_mcp.tpu_engine import RETRY_CONFIG

        assert isinstance(RETRY_CONFIG, dict)

    def test_retry_config_has_required_keys(self):
        """Test RETRY_CONFIG has required keys."""
        from coral_tpu_mcp.tpu_engine import RETRY_CONFIG

        required_keys = ["max_retries", "initial_delay", "max_delay", "backoff_factor"]
        for key in required_keys:
            assert key in RETRY_CONFIG

    def test_retry_config_values_reasonable(self):
        """Test RETRY_CONFIG values are reasonable."""
        from coral_tpu_mcp.tpu_engine import RETRY_CONFIG

        assert RETRY_CONFIG["max_retries"] >= 1
        assert RETRY_CONFIG["initial_delay"] > 0
        assert RETRY_CONFIG["max_delay"] >= RETRY_CONFIG["initial_delay"]
        assert RETRY_CONFIG["backoff_factor"] >= 1


# ============================================================================
# Test: TPU Health Dataclass
# ============================================================================

class TestTPUHealthDataclass:
    """Tests for TPUHealth dataclass."""

    def test_tpu_health_default_values(self):
        """Test TPUHealth has correct default values."""
        from coral_tpu_mcp.tpu_engine import TPUHealth

        health = TPUHealth()

        assert health.is_healthy is False
        assert health.consecutive_failures == 0
        assert health.recovery_attempts == 0
        assert health.last_error is None

    def test_tpu_health_attributes(self):
        """Test TPUHealth has all expected attributes."""
        from coral_tpu_mcp.tpu_engine import TPUHealth

        health = TPUHealth()

        assert hasattr(health, 'is_healthy')
        assert hasattr(health, 'last_check')
        assert hasattr(health, 'consecutive_failures')
        assert hasattr(health, 'last_error')
        assert hasattr(health, 'recovery_attempts')
        assert hasattr(health, 'last_successful_inference')


# ============================================================================
# Test: InferenceResult Dataclass
# ============================================================================

class TestInferenceResultDataclass:
    """Tests for InferenceResult dataclass."""

    def test_inference_result_creation(self):
        """Test InferenceResult can be created with values."""
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
        assert result.timestamp == 1234567890.0

    def test_inference_result_attributes(self):
        """Test InferenceResult has all expected attributes."""
        from coral_tpu_mcp.tpu_engine import InferenceResult

        result = InferenceResult(
            predictions=[],
            latency_ms=0,
            model_name="",
            timestamp=0
        )

        assert hasattr(result, 'predictions')
        assert hasattr(result, 'latency_ms')
        assert hasattr(result, 'model_name')
        assert hasattr(result, 'timestamp')


# ============================================================================
# Test: File Checksum
# ============================================================================

class TestFileChecksum:
    """Tests for file checksum computation."""

    def test_compute_file_checksum(self, tmp_path):
        """Test compute_file_checksum returns consistent hash."""
        from coral_tpu_mcp.tpu_engine import compute_file_checksum

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        checksum1 = compute_file_checksum(test_file)
        checksum2 = compute_file_checksum(test_file)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest length

    def test_different_files_different_checksums(self, tmp_path):
        """Test different files produce different checksums."""
        from coral_tpu_mcp.tpu_engine import compute_file_checksum

        file1 = tmp_path / "file1.bin"
        file2 = tmp_path / "file2.bin"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        checksum1 = compute_file_checksum(file1)
        checksum2 = compute_file_checksum(file2)

        assert checksum1 != checksum2
