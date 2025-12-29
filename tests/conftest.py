"""
Pytest fixtures for coral-tpu-mcp tests.

Provides comprehensive mocking for:
- Coral TPU hardware (pycoral, tflite_runtime)
- Model files and labels
- Text embedding models (sentence-transformers)
"""

import sys
import os

# ============================================================================
# Early mocking of external dependencies
# This must happen BEFORE any coral_tpu_mcp imports
# ============================================================================

# Mock external dependencies that may not be available or have issues
from unittest.mock import MagicMock

# Mock tpu_monitor to avoid external file syntax errors
sys.modules['tpu_monitor'] = MagicMock()

# Mock xrg_tpu_stats
sys.modules['xrg_tpu_stats'] = MagicMock()

# Mock pycoral and related modules
_mock_edgetpu = MagicMock()
_mock_edgetpu.list_edge_tpus = MagicMock(return_value=[])
_mock_edgetpu.make_interpreter = MagicMock()

_mock_common = MagicMock()
_mock_classify = MagicMock()
_mock_detect = MagicMock()
_mock_dataset = MagicMock()

sys.modules['pycoral'] = MagicMock()
sys.modules['pycoral.utils'] = MagicMock()
sys.modules['pycoral.utils.edgetpu'] = _mock_edgetpu
sys.modules['pycoral.utils.dataset'] = _mock_dataset
sys.modules['pycoral.adapters'] = MagicMock()
sys.modules['pycoral.adapters.common'] = _mock_common
sys.modules['pycoral.adapters.classify'] = _mock_classify
sys.modules['pycoral.adapters.detect'] = _mock_detect

# Mock ai_edge_litert
sys.modules['ai_edge_litert'] = MagicMock()
sys.modules['ai_edge_litert.interpreter'] = MagicMock()

# ============================================================================
# Now import the rest
# ============================================================================

import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import List, Any

import pytest


# ============================================================================
# Mock Classes for TPU Hardware
# ============================================================================

class MockInterpreter:
    """Mock TFLite interpreter for testing without hardware."""

    def __init__(self, model_path=None, experimental_delegates=None):
        self.model_path = model_path
        self._allocated = False
        self._input_tensor = None

        # Default tensor shapes for different model types
        self._input_details = [{
            "index": 0,
            "shape": np.array([1, 224, 224, 3]),
            "dtype": np.uint8,
            "quantization": (0.00784313771873713, 128),
        }]
        self._output_details = [{
            "index": 1,
            "shape": np.array([1, 1001]),
            "dtype": np.uint8,
            "quantization": (0.00390625, 0),
        }]

        # Simulate output tensor
        self._output_tensor = np.random.randint(0, 255, (1, 1001), dtype=np.uint8)

    def allocate_tensors(self):
        self._allocated = True

    def get_input_details(self):
        return self._input_details

    def get_output_details(self):
        return self._output_details

    def set_tensor(self, index, data):
        self._input_tensor = data

    def invoke(self):
        if not self._allocated:
            raise RuntimeError("Tensors not allocated")

    def get_tensor(self, index):
        return self._output_tensor


class MockEdgeTPUDelegate:
    """Mock EdgeTPU delegate."""
    pass


@dataclass
class MockClass:
    """Mock classification result from pycoral."""
    id: int
    score: float


@dataclass
class MockBBox:
    """Mock bounding box for detection."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class MockDetection:
    """Mock detection result."""
    id: int
    score: float
    bbox: MockBBox


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_models_dir(tmp_path):
    """Create a temporary models directory with mock model files."""
    models_dir = tmp_path / "models" / "coral"
    models_dir.mkdir(parents=True)

    # Create mock model files (minimal valid files)
    model_files = [
        "mobilenet_v2_edgetpu.tflite",
        "efficientnet_s_edgetpu.tflite",
        "ssdlite_mobiledet_coco_edgetpu.tflite",
        "movenet_single_pose_lightning_edgetpu.tflite",
        "deeplabv3_pascal_edgetpu.tflite",
        "yamnet_edgetpu.tflite",
        "keyword_spotter_edgetpu.tflite",
    ]

    for model_file in model_files:
        model_path = models_dir / model_file
        # Create a minimal valid-looking file (TFLite magic bytes + padding)
        model_path.write_bytes(b'\x1c\x00\x00\x00TFL3' + b'\x00' * 2048)

    # Create label files
    imagenet_labels = models_dir / "imagenet_labels.txt"
    imagenet_labels.write_text("\n".join([f"class_{i}" for i in range(1001)]))

    coco_labels = models_dir / "coco_labels.txt"
    coco_labels.write_text("\n".join([f"object_{i}" for i in range(91)]))

    pascal_labels = models_dir / "pascal_voc_labels.txt"
    pascal_labels.write_text("\n".join([
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]))

    # Create YamNet CSV labels
    yamnet_labels = models_dir / "yamnet_class_map.csv"
    yamnet_labels.write_text("index,mid,display_name\n" +
        "\n".join([f"{i},/m/test{i},sound_{i}" for i in range(521)]))

    return models_dir


@pytest.fixture
def mock_tpu_available():
    """Mock TPU as available."""
    with patch.dict(sys.modules, {
        'pycoral': MagicMock(),
        'pycoral.utils': MagicMock(),
        'pycoral.utils.edgetpu': MagicMock(),
        'pycoral.utils.dataset': MagicMock(),
        'pycoral.adapters': MagicMock(),
        'pycoral.adapters.common': MagicMock(),
        'pycoral.adapters.classify': MagicMock(),
        'pycoral.adapters.detect': MagicMock(),
    }):
        # Configure the mock edgetpu module
        mock_edgetpu = sys.modules['pycoral.utils.edgetpu']
        mock_edgetpu.list_edge_tpus.return_value = [{'type': 'usb', 'path': '/dev/bus/usb/001/002'}]
        mock_edgetpu.make_interpreter.return_value = MockInterpreter()

        # Configure classify adapter
        mock_classify = sys.modules['pycoral.adapters.classify']
        mock_classify.get_classes.return_value = [
            MockClass(id=0, score=0.95),
            MockClass(id=1, score=0.03),
            MockClass(id=2, score=0.01),
        ]

        # Configure common adapter
        mock_common = sys.modules['pycoral.adapters.common']
        mock_common.set_input = MagicMock()

        # Configure detect adapter
        mock_detect = sys.modules['pycoral.adapters.detect']
        mock_detect.get_objects.return_value = [
            MockDetection(id=0, score=0.85, bbox=MockBBox(0.1, 0.1, 0.5, 0.5)),
        ]

        yield mock_edgetpu


@pytest.fixture
def mock_tpu_unavailable():
    """Mock TPU as unavailable."""
    with patch.dict(sys.modules, {
        'pycoral': MagicMock(),
        'pycoral.utils': MagicMock(),
        'pycoral.utils.edgetpu': MagicMock(),
        'pycoral.utils.dataset': MagicMock(),
        'pycoral.adapters': MagicMock(),
        'pycoral.adapters.common': MagicMock(),
        'pycoral.adapters.classify': MagicMock(),
    }):
        mock_edgetpu = sys.modules['pycoral.utils.edgetpu']
        mock_edgetpu.list_edge_tpus.return_value = []
        yield mock_edgetpu


@pytest.fixture
def mock_text_model():
    """Mock sentence-transformers model for text embedding tests."""
    mock_model = MagicMock()

    def mock_encode(texts):
        if isinstance(texts, str):
            return np.random.randn(384).astype(np.float32)
        return np.random.randn(len(texts), 384).astype(np.float32)

    mock_model.encode = mock_encode
    return mock_model


@pytest.fixture
def sample_image_bytes():
    """Generate a sample image as bytes for testing."""
    from io import BytesIO
    from PIL import Image

    # Create a simple test image
    img = Image.new('RGB', (224, 224), color=(128, 64, 32))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_image_base64(sample_image_bytes):
    """Return sample image as base64 string."""
    import base64
    return base64.b64encode(sample_image_bytes).decode('utf-8')


@pytest.fixture
def sample_audio_bytes():
    """Generate sample audio data for testing."""
    # 2 seconds of 16kHz mono audio (silence)
    duration = 2.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    audio = np.zeros(samples, dtype=np.int16)
    return audio.tobytes()


@pytest.fixture
def sample_audio_base64(sample_audio_bytes):
    """Return sample audio as base64 string."""
    import base64
    return base64.b64encode(sample_audio_bytes).decode('utf-8')


@pytest.fixture
def mock_engine(mock_models_dir):
    """Create a mock TPUEngine for testing."""
    from unittest.mock import MagicMock

    engine = MagicMock()
    engine.is_available = True
    engine.interpreters = {}
    engine.labels = {}
    engine.stats = {
        "total_inferences": 10,
        "total_latency_ms": 150.0,
        "by_model": {},
        "retries": 0,
        "recoveries": 0,
    }

    def mock_load_model(model_name, labels_file=None):
        engine.interpreters[model_name] = MockInterpreter()
        if labels_file:
            label_path = mock_models_dir / labels_file
            if label_path.exists():
                engine.labels[model_name] = label_path.read_text().strip().split('\n')
        return True

    engine.load_model = mock_load_model

    def mock_get_stats():
        return {
            **engine.stats,
            "avg_latency_ms": 15.0,
            "tpu_available": True,
            "loaded_models": list(engine.interpreters.keys()),
            "health": {"is_healthy": True, "consecutive_failures": 0},
            "resilience": {"retries": 0, "recoveries": 0, "is_healthy": True},
        }

    engine.get_stats = mock_get_stats
    engine.health_check.return_value = {"is_healthy": True, "tpu_available": True}
    engine.reconnect.return_value = True
    engine._get_health_status.return_value = {"is_healthy": True}
    engine._update_stats = MagicMock()
    engine.get_input_details.return_value = {"shape": [1, 224, 224, 3], "dtype": "uint8"}

    return engine


@pytest.fixture
def mock_inference_result():
    """Create a mock InferenceResult."""
    from dataclasses import dataclass
    from typing import List, Dict, Any

    @dataclass
    class InferenceResult:
        predictions: List[Dict[str, Any]]
        latency_ms: float
        model_name: str
        timestamp: float

    return InferenceResult(
        predictions=[
            {"class_id": 0, "score": 0.95, "label": "cat"},
            {"class_id": 1, "score": 0.03, "label": "dog"},
        ],
        latency_ms=15.5,
        model_name="mobilenet_v2_edgetpu.tflite",
        timestamp=1234567890.0
    )
