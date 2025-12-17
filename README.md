# Coral TPU MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python-3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Part of Agentic System](https://img.shields.io/badge/Part_of-Agentic_System-brightgreen)](https://github.com/marc-shade/agentic-system-oss)

> **Google Coral TPU integration for edge ML inference.**

Part of the [Agentic System](https://github.com/marc-shade/agentic-system-oss) - a 24/7 autonomous AI framework with persistent memory.

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
---

## Part of the MCP Ecosystem

This server integrates with other MCP servers for comprehensive AGI capabilities:

| Server | Purpose |
|--------|---------|
| [enhanced-memory-mcp](https://github.com/marc-shade/enhanced-memory-mcp) | 4-tier persistent memory with semantic search |
| [agent-runtime-mcp](https://github.com/marc-shade/agent-runtime-mcp) | Persistent task queues and goal decomposition |
| [agi-mcp](https://github.com/marc-shade/agi-mcp) | Full AGI orchestration with 21 tools |
| [cluster-execution-mcp](https://github.com/marc-shade/cluster-execution-mcp) | Distributed task routing across nodes |
| [node-chat-mcp](https://github.com/marc-shade/node-chat-mcp) | Inter-node AI communication |
| [ember-mcp](https://github.com/marc-shade/ember-mcp) | Production-only policy enforcement |

See [agentic-system-oss](https://github.com/marc-shade/agentic-system-oss) for the complete framework.
