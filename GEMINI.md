# GEMINI.md - Coral TPU MCP Server

## Project Overview

The **Coral TPU MCP Server** is a Model Context Protocol (MCP) server designed to provide fast, local Machine Learning inference for agentic systems. It leverages the **Google Coral Edge TPU** (USB or PCIe) to accelerate tasks like image classification, object detection, pose estimation, segmentation, and audio classification. It also provides CPU-based fallbacks for text embeddings.

This project acts as a bridge between AI agents (like Claude or Gemini) and local hardware acceleration, enabling agents to "see" and "hear" with low latency and without relying on external APIs for these specific tasks.

### Key Features
*   **Image Classification**: MobileNet V2, EfficientNet-S (1000 ImageNet classes).
*   **Object Detection**: SSD MobileDet (90 COCO classes).
*   **Pose Estimation**: MoveNet (fast single pose, 17 keypoints).
*   **Semantic Segmentation**: DeepLab v3 (21 Pascal VOC classes).
*   **Audio Classification**: YamNet (520+ sound classes).
*   **Keyword Spotting**: Detects specific voice commands.
*   **Embeddings**:
    *   **Visual**: EfficientNet-S embeddings for image similarity.
    *   **Text**: `all-MiniLM-L6-v2` (CPU-based) for semantic search.
*   **Utilities**: Anomaly detection, importance scoring, intent classification.

## Architecture

The system is built on the `mcp` python SDK and `pycoral`/`tflite-runtime`.

*   **`src/coral_tpu_mcp/server.py`**: The main entry point. Defines the MCP tools and handles requests. It contains the `MODELS` configuration dictionary which defines available models and their properties.
*   **`src/coral_tpu_mcp/tpu_engine.py`**: The core inference engine.
    *   Manages the lifecycle of the Edge TPU connection.
    *   Handles model loading (lazy loading strategies).
    *   Implements resilience patterns: retry logic, exponential backoff, and auto-reconnection.
    *   Collects and persists inference statistics.
*   **`pyproject.toml`**: Defines project metadata, dependencies, and the `coral-tpu-mcp` CLI entry point.

## Building and Running

### Prerequisites
*   **Hardware**: Google Coral Edge TPU (USB Accelerator or PCIe card).
*   **Drivers**: Coral TPU drivers must be installed on the host system.
*   **Python**: 3.10+.

### Installation
1.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```
3.  **Important**: `pycoral` and `tflite-runtime` often require specific installation steps or wheels depending on your platform (Linux/Mac/Windows) and Python version. You may need to install them separately if `pip install -e .` fails to resolve them.

### Running the Server
Start the MCP server:
```bash
python -m coral_tpu_mcp.server
# OR using the installed script
coral-tpu-mcp
```

### Testing
Run the test suite:
```bash
pytest
```

## Development Conventions

*   **Code Style**: Follows PEP 8. Type hints are used throughout.
*   **Async/Sync**: The MCP server uses `asyncio`. Tool handlers are `async` functions. The underlying `tpu_engine` uses synchronous calls for TPU operations (as `pycoral` is blocking), but these are wrapped effectively within the async handlers.
*   **Error Handling**: The `TPUEngine` class in `tpu_engine.py` is designed to be resilient. It catches TPU-related errors, attempts retries, and monitors health.
*   **Logging**: Logging is configured to `WARNING` level by default to keep the standard output clean for MCP communication (which uses JSON-RPC over stdio).
*   **Model Management**: Models are defined in the `MODELS` dict in `server.py`. New models should be added there with their filename, input size, and description. Model files are expected to be in the directory pointed to by `MODELS_DIR` (defaulting to an agentic system path or local assets).

## Key Tools (MCP)

*   `classify_image`: Classify an image.
*   `detect_objects`: Find objects in an image.
*   `estimate_pose`: Detect human body keypoints.
*   `segment_image`: Pixel-level image segmentation.
*   `classify_audio`: Identify sounds.
*   `spot_keyword`: Listen for specific command words.
*   `get_visual_embedding`: Get a vector representation of an image.
*   `embed_text`: Get a vector representation of text.
*   `tpu_health_check`: Check system status.
