"""
TPU Inference Engine - Core inference functionality for Coral Edge TPU.

Resilient design:
- Retry logic with exponential backoff
- Health monitoring and auto-reconnect
- Graceful degradation when TPU unavailable
- Lock detection and recovery
"""

import os
import sys
import io
import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
import asyncio
from contextlib import contextmanager

# Suppress TensorFlow/TFLite logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

@contextmanager
def _suppress_tflite_output():
    """Context manager to suppress stdout/stderr (for TFLite XNNPACK messages).

    Uses OS-level file descriptor redirection to catch native library output.
    """
    # Save original file descriptors
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        # Redirect stdout and stderr to /dev/null at OS level
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

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
    _hooks_path = os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "scripts/hooks")
    if _hooks_path not in _sys.path:
        _sys.path.insert(0, _hooks_path)
    from tpu_monitor import record_tpu_usage as _record_tpu_usage
    _HAS_TPU_MONITOR = True
except ImportError:
    _HAS_TPU_MONITOR = False
    _record_tpu_usage = None

try:
    from pycoral.utils import edgetpu
    from pycoral.utils.dataset import read_label_file
    from pycoral.adapters import common, classify
    _PYCORAL_AVAILABLE = True
except ImportError:
    _PYCORAL_AVAILABLE = False
    edgetpu = None
    read_label_file = None
    common = None
    classify = None

logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.path.join(os.environ.get("AGENTIC_SYSTEM_PATH", "${AGENTIC_SYSTEM_PATH:-/opt/agentic}"), "models/coral"))

# Resilience configuration
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 0.5,  # seconds
    "max_delay": 10.0,
    "backoff_factor": 2.0,
}

HEALTH_CHECK_INTERVAL = 30  # seconds


@dataclass
class TPUHealth:
    """Track TPU health status."""
    is_healthy: bool = False
    last_check: float = 0.0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    last_successful_inference: float = 0.0


@dataclass
class InferenceResult:
    """Result of TPU inference."""
    predictions: List[Dict[str, Any]]
    latency_ms: float
    model_name: str
    timestamp: float


class TPUEngine:
    """
    Manages Coral Edge TPU inference with resilience.

    Features:
    - Retry logic with exponential backoff
    - Health monitoring and auto-recovery
    - Graceful degradation when TPU busy
    - Thread-safe operations
    """

    def __init__(self):
        self.interpreters: Dict[str, Any] = {}
        self.labels: Dict[str, List[str]] = {}
        self.stats = {
            "total_inferences": 0,
            "total_latency_ms": 0,
            "by_model": {},
            "retries": 0,
            "recoveries": 0,
        }
        self._tpu_available = False
        self._health = TPUHealth()
        self._lock = threading.RLock()
        self._init_lock = threading.Lock()
        self._initializing = False
        self._check_tpu_with_retry()

    def _check_tpu_with_retry(self) -> bool:
        """Check TPU availability with retry logic."""
        if not _PYCORAL_AVAILABLE:
            self._health.is_healthy = False
            self._health.last_error = "pycoral library not installed"
            logger.warning("TPU disabled: pycoral library not found")
            return False

        for attempt in range(RETRY_CONFIG["max_retries"]):
            if self._check_tpu():
                self._health.is_healthy = True
                self._health.last_check = time.time()
                self._health.consecutive_failures = 0
                return True

            if attempt < RETRY_CONFIG["max_retries"] - 1:
                delay = min(
                    RETRY_CONFIG["initial_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt),
                    RETRY_CONFIG["max_delay"]
                )
                logger.info(f"TPU check failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{RETRY_CONFIG['max_retries']})")
                time.sleep(delay)
                self.stats["retries"] += 1

        logger.warning("TPU initialization failed after all retries - running in degraded mode")
        self._health.is_healthy = False
        self._health.last_error = "Failed to initialize TPU after retries"
        return False

    def _check_tpu(self) -> bool:
        """Check if TPU is available (single attempt)."""
        if not _PYCORAL_AVAILABLE:
            self._tpu_available = False
            self._health.last_error = "pycoral library not installed"
            return False

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
            self._health.last_error = str(e)
            return False

    def reconnect(self) -> bool:
        """Attempt to reconnect to TPU after failure."""
        with self._init_lock:
            if self._initializing:
                return False
            self._initializing = True

        try:
            logger.info("Attempting TPU reconnection...")
            self._health.recovery_attempts += 1

            # Unload all models first
            self.interpreters.clear()

            # Wait a moment for any lock to release
            time.sleep(1.0)

            # Try to reinitialize
            if self._check_tpu_with_retry():
                self.stats["recoveries"] += 1
                logger.info("TPU reconnection successful!")
                return True
            else:
                logger.warning("TPU reconnection failed")
                return False
        finally:
            self._initializing = False

    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        now = time.time()

        # Skip if checked recently
        if now - self._health.last_check < HEALTH_CHECK_INTERVAL:
            return self._get_health_status()

        self._health.last_check = now

        # Try a quick TPU check
        try:
            devices = edgetpu.list_edge_tpus()
            if devices:
                self._health.is_healthy = True
                self._health.consecutive_failures = 0
            else:
                self._health.is_healthy = False
                self._health.consecutive_failures += 1
        except Exception as e:
            self._health.is_healthy = False
            self._health.consecutive_failures += 1
            self._health.last_error = str(e)

        # Auto-reconnect if unhealthy
        if not self._health.is_healthy and self._health.consecutive_failures >= 3:
            logger.warning(f"TPU unhealthy after {self._health.consecutive_failures} failures, attempting recovery")
            self.reconnect()

        return self._get_health_status()

    def _get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "is_healthy": self._health.is_healthy,
            "tpu_available": self._tpu_available,
            "consecutive_failures": self._health.consecutive_failures,
            "last_error": self._health.last_error,
            "recovery_attempts": self._health.recovery_attempts,
            "last_check": self._health.last_check,
            "last_successful_inference": self._health.last_successful_inference,
        }

    @contextmanager
    def _tpu_operation(self, operation_name: str = "inference"):
        """Context manager for TPU operations with error handling."""
        if not self._tpu_available:
            # Try reconnection if TPU not available
            if not self.reconnect():
                raise RuntimeError("TPU not available and reconnection failed")

        try:
            yield
            self._health.last_successful_inference = time.time()
            self._health.consecutive_failures = 0
        except Exception as e:
            self._health.consecutive_failures += 1
            self._health.last_error = str(e)

            # Check if it's a device error that might be recoverable
            error_str = str(e).lower()
            if "device" in error_str or "usb" in error_str or "delegate" in error_str:
                logger.warning(f"TPU device error during {operation_name}, will attempt recovery: {e}")
                self._tpu_available = False
            raise

    @property
    def is_available(self) -> bool:
        """Check if TPU is available."""
        return self._tpu_available

    def load_model(self, model_name: str, labels_file: Optional[str] = None) -> bool:
        """
        Load a model onto the TPU with retry logic.

        Args:
            model_name: Name of the model file (without path)
            labels_file: Optional labels file name

        Returns:
            True if model loaded successfully
        """
        if not _PYCORAL_AVAILABLE:
            return False

        with self._lock:
            if model_name in self.interpreters:
                logger.debug(f"Model {model_name} already loaded")
                return True

            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False

            # Retry loop for model loading
            last_error = None
            for attempt in range(RETRY_CONFIG["max_retries"]):
                try:
                    # Create interpreter with Edge TPU delegate
                    # Suppress stdout/stderr to prevent TFLite XNNPACK messages from confusing MCP clients
                    with _suppress_tflite_output():
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
                    self._health.is_healthy = True
                    return True

                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load model {model_name} (attempt {attempt + 1}): {e}")

                    if attempt < RETRY_CONFIG["max_retries"] - 1:
                        delay = min(
                            RETRY_CONFIG["initial_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt),
                            RETRY_CONFIG["max_delay"]
                        )
                        logger.info(f"Retrying model load in {delay:.1f}s...")
                        time.sleep(delay)
                        self.stats["retries"] += 1

                        # Try reconnecting if it looks like a device issue
                        error_str = str(e).lower()
                        if "device" in error_str or "delegate" in error_str:
                            self.reconnect()

            logger.error(f"Failed to load model {model_name} after {RETRY_CONFIG['max_retries']} attempts: {last_error}")
            self._health.last_error = str(last_error)
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
        Run classification inference with resilience.

        Args:
            model_name: Model to use
            input_data: Preprocessed input tensor
            top_k: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            InferenceResult with predictions
        """
        if model_name not in self.interpreters:
            # Try loading the model with retry
            if not self.load_model(model_name):
                raise ValueError(f"Model not loaded and couldn't be loaded: {model_name}")

        # Retry loop for inference
        last_error = None
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                with self._tpu_operation("classification"):
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
                            "class_id": int(c.id),
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

            except Exception as e:
                last_error = e
                logger.warning(f"Classification failed (attempt {attempt + 1}): {e}")

                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    delay = RETRY_CONFIG["initial_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt)
                    time.sleep(min(delay, RETRY_CONFIG["max_delay"]))
                    self.stats["retries"] += 1

                    # Try to reload the model
                    if model_name in self.interpreters:
                        del self.interpreters[model_name]
                    self.load_model(model_name)

        raise RuntimeError(f"Classification failed after {RETRY_CONFIG['max_retries']} attempts: {last_error}")

    def get_embedding(
        self,
        model_name: str,
        input_data: np.ndarray,
        layer_index: int = -2  # Second to last layer usually has embeddings
    ) -> Tuple[np.ndarray, float]:
        """
        Extract embeddings from a model's intermediate layer with resilience.

        For image models, this returns visual feature embeddings.

        Args:
            model_name: Model to use
            input_data: Preprocessed input tensor
            layer_index: Which output layer to use (-2 for second to last)

        Returns:
            Tuple of (embedding array, latency_ms)
        """
        if model_name not in self.interpreters:
            if not self.load_model(model_name):
                raise ValueError(f"Model not loaded and couldn't be loaded: {model_name}")

        # Retry loop for embedding extraction
        last_error = None
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                with self._tpu_operation("embedding"):
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

            except Exception as e:
                last_error = e
                logger.warning(f"Embedding extraction failed (attempt {attempt + 1}): {e}")

                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    delay = RETRY_CONFIG["initial_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt)
                    time.sleep(min(delay, RETRY_CONFIG["max_delay"]))
                    self.stats["retries"] += 1

                    # Try to reload the model
                    if model_name in self.interpreters:
                        del self.interpreters[model_name]
                    self.load_model(model_name)

        raise RuntimeError(f"Embedding extraction failed after {RETRY_CONFIG['max_retries']} attempts: {last_error}")

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
        """Get inference statistics with health information."""
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

        # Include health status
        stats["health"] = self._get_health_status()

        # Resilience stats
        stats["resilience"] = {
            "retries": self.stats.get("retries", 0),
            "recoveries": self.stats.get("recoveries", 0),
            "consecutive_failures": self._health.consecutive_failures,
            "is_healthy": self._health.is_healthy,
        }

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
