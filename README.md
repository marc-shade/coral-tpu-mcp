# Coral TPU MCP Server

Fast local ML inference for agentic systems using Google Coral Edge TPU.

## Features

- **Image Classification**: 1000 ImageNet classes with MobileNet/EfficientNet
- **Visual Embeddings**: Generate embeddings for image similarity
- **Text Embeddings**: CPU-based text embeddings with small models
- **Importance Scoring**: Score memory importance for prioritization
- **Anomaly Detection**: Detect anomalies in data patterns
- **Face Detection**: Detect faces in images

## MCP Tools

| Tool | Description |
|------|-------------|
| `classify_image` | Classify image using MobileNet V2 |
| `get_visual_embedding` | Generate visual embedding for an image |
| `get_text_embedding` | Generate text embedding (CPU) |
| `score_importance` | Score memory importance (0-1) |
| `detect_anomaly` | Detect anomalies in data |
| `detect_faces` | Detect faces in an image |

## Models

- `mobilenet_v2` - Image classification (224x224 input)
- `efficientnet_s` - Visual embeddings and classification
- `face_detection` - Face detection model

## Requirements

- Python 3.10+
- Google Coral Edge TPU USB Accelerator
- pycoral and tflite-runtime
- mcp SDK

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m coral_tpu_mcp.server
```

## Hardware

Requires a Google Coral Edge TPU (USB or PCIe) for accelerated inference. Falls back to CPU for text embeddings.

## Integration

Provides fast local inference for:
- Memory importance scoring
- Visual episode encoding
- Image-based context understanding

## License

MIT
