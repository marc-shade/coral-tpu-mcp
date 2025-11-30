"""
TPU Inference Engine - Core inference functionality for Coral Edge TPU.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

# PyCoral imports
from pycoral.utils import edgetpu

# Shared stats file for cross-process metrics (XRG monitors this file)
TPU_STATS_FILE = Path("/tmp/xrg-coral-tpu-stats.json")

# Also try to use xrg_tpu_stats module for richer integration
try:
    from xrg_tpu_stats import record_inference as _xrg_record
    _HAS_XRG_STATS = True
except ImportError:
    _HAS_XRG_STATS = False
    _xrg_record = None

# TPU monitor for historical usage tracking
try:
    import sys as _sys
    _hooks_path = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "/mnt/agentic-system"), "scripts/hooks")
    if _hooks_path not in _sys.path:
        _sys.path.insert(0, _hooks_path)
    from tpu_monitor import record_tpu_usage as _record_tpu_usage
    _HAS_TPU_MONITOR = True
except ImportError:
    _HAS_TPU_MONITOR = False
    _record_tpu_usage = None
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import common, classify

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "/mnt/agentic-system"), "models/coral"))


@dataclass
class InferenceResult:
    """Result of TPU inference."""
    predictions: List[Dict[str, Any]]
    latency_ms: float
    model_name: str
    timestamp: float


class TPUEngine:
    """
    Manages Coral Edge TPU inference.

    Handles model loading, inference, and statistics tracking.
    """

    def __init__(self):
        self.interpreters: Dict[str, Any] = {}
        self.labels: Dict[str, List[str]] = {}
        self.stats = {
            "total_inferences": 0,
            "total_latency_ms": 0,
            "by_model": {}
        }
        self._tpu_available = False
        self._check_tpu()

    def _check_tpu(self) -> bool:
        """Check if TPU is available."""
        try:
            devices = edgetpu.list_edge_tpus()
            self._tpu_available = len(devices) > 0
            if self._tpu_available:
                logger.info(f"Found {len(devices)} Edge TPU device(s): {devices}")
            else:
                logger.warning("No Edge TPU devices found")
            return self._tpu_available
        except Exception as e:
            logger.error(f"Error checking TPU: {e}")
            self._tpu_available = False
            return False

    @property
    def is_available(self) -> bool:
        """Check if TPU is available."""
        return self._tpu_available

    def load_model(self, model_name: str, labels_file: Optional[str] = None) -> bool:
        """
        Load a model onto the TPU.

        Args:
            model_name: Name of the model file (without path)
            labels_file: Optional labels file name

        Returns:
            True if model loaded successfully
        """
        if model_name in self.interpreters:
            logger.info(f"Model {model_name} already loaded")
            return True

        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        try:
            # Create interpreter with Edge TPU delegate
            interpreter = edgetpu.make_interpreter(str(model_path))
            interpreter.allocate_tensors()

            self.interpreters[model_name] = interpreter

            # Load labels if provided
            if labels_file:
                labels_path = MODELS_DIR / labels_file
                if labels_path.exists():
                    self.labels[model_name] = read_label_file(str(labels_path))
                    logger.info(f"Loaded {len(self.labels[model_name])} labels for {model_name}")

            # Initialize stats
            self.stats["by_model"][model_name] = {
                "inferences": 0,
                "total_latency_ms": 0
            }

            logger.info(f"Loaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def get_input_details(self, model_name: str) -> Optional[Dict]:
        """Get input tensor details for a model."""
        if model_name not in self.interpreters:
            return None

        interpreter = self.interpreters[model_name]
        input_details = interpreter.get_input_details()[0]
        return {
            "shape": input_details["shape"].tolist(),
            "dtype": str(input_details["dtype"]),
            "quantization": input_details.get("quantization", None)
        }

    def classify(
        self,
        model_name: str,
        input_data: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> InferenceResult:
        """
        Run classification inference.

        Args:
            model_name: Model to use
            input_data: Preprocessed input tensor
            top_k: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            InferenceResult with predictions
        """
        if model_name not in self.interpreters:
            raise ValueError(f"Model not loaded: {model_name}")

        interpreter = self.interpreters[model_name]

        # Set input
        common.set_input(interpreter, input_data)

        # Run inference with timing
        start = time.perf_counter()
        interpreter.invoke()
        latency_ms = (time.perf_counter() - start) * 1000

        # Get results
        classes = classify.get_classes(interpreter, top_k, threshold)

        # Format predictions
        labels = self.labels.get(model_name, [])
        predictions = []
        for c in classes:
            pred = {
                "class_id": int(c.id),  # Convert numpy int64 to Python int for JSON
                "score": float(c.score)
            }
            if labels and int(c.id) < len(labels):
                pred["label"] = labels[int(c.id)]
            predictions.append(pred)

        # Update stats
        self._update_stats(model_name, latency_ms, operation="classification")

        return InferenceResult(
            predictions=predictions,
            latency_ms=latency_ms,
            model_name=model_name,
            timestamp=time.time()
        )

    def get_embedding(
        self,
        model_name: str,
        input_data: np.ndarray,
        layer_index: int = -2  # Second to last layer usually has embeddings
    ) -> Tuple[np.ndarray, float]:
        """
        Extract embeddings from a model's intermediate layer.

        For image models, this returns visual feature embeddings.

        Args:
            model_name: Model to use
            input_data: Preprocessed input tensor
            layer_index: Which output layer to use (-2 for second to last)

        Returns:
            Tuple of (embedding array, latency_ms)
        """
        if model_name not in self.interpreters:
            raise ValueError(f"Model not loaded: {model_name}")

        interpreter = self.interpreters[model_name]

        # Set input
        common.set_input(interpreter, input_data)

        # Run inference
        start = time.perf_counter()
        interpreter.invoke()
        latency_ms = (time.perf_counter() - start) * 1000

        # Get output tensor (embeddings are usually in output)
        output_details = interpreter.get_output_details()

        # Use the specified layer or last layer
        if abs(layer_index) <= len(output_details):
            output_detail = output_details[layer_index]
        else:
            output_detail = output_details[-1]

        embedding = interpreter.get_tensor(output_detail["index"])

        # Update stats
        self._update_stats(model_name, latency_ms, operation="embedding")

        return embedding.flatten(), latency_ms

    def _update_stats(self, model_name: str, latency_ms: float, operation: str = "inference"):
        """Update inference statistics."""
        self.stats["total_inferences"] += 1
        self.stats["total_latency_ms"] += latency_ms

        if model_name in self.stats["by_model"]:
            self.stats["by_model"][model_name]["inferences"] += 1
            self.stats["by_model"][model_name]["total_latency_ms"] += latency_ms

        # Use xrg_tpu_stats module if available (writes in XRG-expected format)
        if _HAS_XRG_STATS and _xrg_record:
            _xrg_record(latency_ms=latency_ms, model_name=model_name)
        else:
            # Fallback: persist stats directly for cross-process access
            self._quick_persist(model_name=model_name, latency_ms=latency_ms)

        # Record to TPU monitor for historical tracking
        if _HAS_TPU_MONITOR and _record_tpu_usage:
            try:
                _record_tpu_usage(
                    operation=operation,
                    latency_ms=latency_ms,
                    model=model_name,
                    success=True,
                    source="coral-tpu-mcp"
                )
            except Exception:
                pass  # Don't let monitoring failures affect inference

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = dict(self.stats)

        # Calculate averages
        if stats["total_inferences"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_inferences"]
        else:
            stats["avg_latency_ms"] = 0

        for model, model_stats in stats["by_model"].items():
            if model_stats["inferences"] > 0:
                model_stats["avg_latency_ms"] = (
                    model_stats["total_latency_ms"] / model_stats["inferences"]
                )
            else:
                model_stats["avg_latency_ms"] = 0

        stats["tpu_available"] = self._tpu_available
        stats["loaded_models"] = list(self.interpreters.keys())

        # Persist stats for cross-process access
        self._persist_stats(stats)

        return stats

    def _persist_stats(self, stats: Dict[str, Any]):
        """Write stats to shared file for exporter access."""
        try:
            stats_copy = dict(stats)
            stats_copy["timestamp"] = time.time()
            TPU_STATS_FILE.write_text(json.dumps(stats_copy))
        except Exception as e:
            logger.debug(f"Failed to persist stats: {e}")

    def _quick_persist(self, model_name: str = "", latency_ms: float = 0.0):
        """Quickly persist current stats in XRG-expected format."""
        try:
            total = self.stats["total_inferences"]
            avg_latency = self.stats["total_latency_ms"] / total if total > 0 else 0

            # Write in XRG-expected format (matches xrg_tpu_stats.py)
            stats = {
                "total_inferences": total,
                "last_latency_ms": round(latency_ms, 2) if latency_ms > 0 else round(avg_latency, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "model_name": model_name or (list(self.interpreters.keys())[0] if self.interpreters else ""),
                "timestamp": time.time()
            }
            TPU_STATS_FILE.write_text(json.dumps(stats))
        except Exception as e:
            logger.debug(f"Failed to quick persist: {e}")

    def unload_model(self, model_name: str):
        """Unload a model from TPU."""
        if model_name in self.interpreters:
            del self.interpreters[model_name]
            logger.info(f"Unloaded model: {model_name}")

    def unload_all(self):
        """Unload all models."""
        self.interpreters.clear()
        self.labels.clear()
        logger.info("Unloaded all models")


# Global engine instance
_engine: Optional[TPUEngine] = None


def get_engine() -> TPUEngine:
    """Get or create the global TPU engine."""
    global _engine
    if _engine is None:
        _engine = TPUEngine()
    return _engine
