"""
Coral TPU MCP Server - Fast local inference for agentic systems.

Exposes TPU-accelerated ML tools:
- Image classification and visual embeddings
- Text embeddings (CPU with small models)
- Importance scoring for memory systems
- Anomaly detection

Run with: python -m coral_tpu_mcp.server
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tpu_engine import get_engine, TPUEngine, MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations - Comprehensive TPU model library
MODELS = {
    # === IMAGE CLASSIFICATION ===
    "mobilenet_v2": {
        "file": "mobilenet_v2_edgetpu.tflite",
        "labels": "imagenet_labels.txt",
        "input_size": (224, 224),
        "description": "Image classification (1000 ImageNet classes)",
        "category": "classification"
    },
    "efficientnet_s": {
        "file": "efficientnet_s_edgetpu.tflite",
        "labels": "imagenet_labels.txt",
        "input_size": (224, 224),
        "description": "Visual embeddings and classification",
        "category": "classification"
    },

    # === OBJECT DETECTION ===
    "face_detection": {
        "file": "face_detection_edgetpu.tflite",
        "labels": None,
        "input_size": (320, 320),
        "description": "Face detection",
        "category": "detection"
    },
    "coco_detection": {
        "file": "ssdlite_mobiledet_coco_edgetpu.tflite",
        "labels": "coco_labels.txt",
        "input_size": (320, 320),
        "description": "Object detection (90 COCO classes: person, car, dog, etc.)",
        "category": "detection"
    },

    # === POSE ESTIMATION ===
    "posenet_353": {
        "file": "posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite",
        "labels": None,
        "input_size": (353, 481),
        "description": "Human pose estimation - 17 keypoints (353x481 input)",
        "category": "pose"
    },
    "posenet_481": {
        "file": "posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite",
        "labels": None,
        "input_size": (481, 641),
        "description": "Human pose estimation - 17 keypoints (481x641 input)",
        "category": "pose"
    },
    "posenet_721": {
        "file": "posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite",
        "labels": None,
        "input_size": (721, 1281),
        "description": "Human pose estimation - 17 keypoints (721x1281 input, highest accuracy)",
        "category": "pose"
    },
    "movenet": {
        "file": "movenet_single_pose_lightning_edgetpu.tflite",
        "labels": None,
        "input_size": (192, 192),
        "description": "MoveNet Lightning - fast single pose estimation (~30ms)",
        "category": "pose"
    },

    # === SEGMENTATION ===
    "deeplabv3_pascal": {
        "file": "deeplabv3_pascal_edgetpu.tflite",
        "labels": "pascal_voc_labels.txt",
        "input_size": (513, 513),
        "description": "Semantic segmentation (21 Pascal VOC classes)",
        "category": "segmentation"
    },
    "bodypix": {
        "file": "bodypix_640x480_edgetpu.tflite",
        "labels": None,
        "input_size": (640, 480),
        "description": "Body segmentation (24 body parts)",
        "category": "segmentation"
    },

    # === AUDIO ===
    "keyword_spotter": {
        "file": "keyword_spotter_edgetpu.tflite",
        "labels": None,
        "input_size": None,  # Audio model
        "description": "Voice command keyword spotting (12 keywords)",
        "category": "audio"
    },
    "keyword_spotter_140": {
        "file": "keyword_spotter_140_edgetpu.tflite",
        "labels": "keyword_spotter_labels.txt",
        "input_size": None,  # Audio model
        "description": "Enhanced keyword spotting (140+ phrases)",
        "category": "audio"
    },
    "yamnet": {
        "file": "yamnet_edgetpu.tflite",
        "labels": "yamnet_class_map.csv",
        "input_size": None,  # Audio model
        "description": "Audio classification (520+ sound classes)",
        "category": "audio"
    }
}

# Pose keypoint names (17 COCO keypoints)
POSE_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Text embedding model (CPU-based for now)
TEXT_EMBEDDING_DIM = 384  # MiniLM dimension
_text_model = None


def get_text_model():
    """Lazy load text embedding model (CPU-based)."""
    global _text_model
    if _text_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Small, fast model - runs well on CPU
            _text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded text embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not installed, text embedding disabled")
            _text_model = False
    return _text_model if _text_model else None


def preprocess_image(image_data: bytes, target_size: tuple) -> np.ndarray:
    """Preprocess image for model input."""
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    # Convert to numpy and normalize for uint8 quantized models
    arr = np.array(image, dtype=np.uint8)
    return arr


# Initialize MCP server
server = Server("coral-tpu-mcp")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available TPU tools."""
    return [
        Tool(
            name="tpu_status",
            description="Get TPU status and inference statistics",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="classify_image",
            description="Classify an image using TPU. Returns top-k class predictions with confidence scores. Fast (~15ms) local inference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data (JPEG/PNG)"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file (alternative to base64)"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["mobilenet_v2", "efficientnet_s"],
                        "default": "mobilenet_v2",
                        "description": "Model to use for classification"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of top predictions to return"
                    }
                }
            }
        ),
        Tool(
            name="get_visual_embedding",
            description="Extract visual feature embedding from an image using TPU. Returns a vector that can be used for similarity search or clustering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file"
                    }
                }
            }
        ),
        Tool(
            name="embed_text",
            description="Generate semantic embedding for text. Uses fast CPU model (MiniLM-L6). Returns 384-dim vector for similarity search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to embed"
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple texts to embed (batch)"
                    }
                }
            }
        ),
        Tool(
            name="score_importance",
            description="Score the importance/salience of content for memory prioritization. Uses semantic analysis to determine if content should be prioritized for storage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to score"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context about current task/goals"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="classify_intent",
            description="Classify user intent for command routing. Determines if input is a question, command, statement, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "User input text"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="compute_similarity",
            description="Compute semantic similarity between two texts. Returns cosine similarity score 0.0-1.0.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text1": {
                        "type": "string",
                        "description": "First text"
                    },
                    "text2": {
                        "type": "string",
                        "description": "Second text"
                    }
                },
                "required": ["text1", "text2"]
            }
        ),
        Tool(
            name="detect_anomaly",
            description="Detect if content/pattern is anomalous compared to baseline. Uses embedding distance for detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to check"
                    },
                    "baseline_embeddings": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Baseline embeddings to compare against"
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.7,
                        "description": "Anomaly threshold (lower = more anomalies detected)"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="spot_keyword",
            description="Detect keywords/phrases in audio using TPU. Supports ~140 phrases like 'start', 'stop', 'next', 'previous'. Returns detected keywords with confidence.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_base64": {
                        "type": "string",
                        "description": "Base64-encoded raw audio data (16-bit PCM, 16kHz mono)"
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "Path to WAV file (16kHz mono preferred)"
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Minimum confidence threshold for detection"
                    }
                }
            }
        ),
        # === NEW TOOLS ===
        Tool(
            name="estimate_pose",
            description="Estimate human pose from image using TPU. Returns 17 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) with confidence scores. Fast (~15-30ms).",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["movenet", "posenet_353", "posenet_481", "posenet_721"],
                        "default": "movenet",
                        "description": "Pose model (movenet=fastest, posenet_721=most accurate)"
                    }
                }
            }
        ),
        Tool(
            name="detect_objects",
            description="Detect objects in image using TPU. Returns bounding boxes for 90 COCO classes (person, car, dog, chair, etc.) with confidence scores. Fast (~15ms).",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file"
                    },
                    "threshold": {
                        "type": "number",
                        "default": 0.4,
                        "description": "Minimum confidence threshold"
                    },
                    "max_detections": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of detections to return"
                    }
                }
            }
        ),
        Tool(
            name="segment_image",
            description="Perform semantic segmentation on image using TPU. Returns per-pixel class labels for 21 Pascal VOC classes (person, car, bicycle, etc.). Fast (~25ms).",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data"
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file"
                    },
                    "return_mask": {
                        "type": "boolean",
                        "default": False,
                        "description": "Return full segmentation mask (large data)"
                    }
                }
            }
        ),
        Tool(
            name="classify_audio",
            description="Classify audio/sounds using YamNet TPU model. Recognizes 520+ sound classes (speech, music, dogs, cars, alarms, etc.). Useful for environment context detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_base64": {
                        "type": "string",
                        "description": "Base64-encoded raw audio (16-bit PCM)"
                    },
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file"
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of top predictions to return"
                    }
                }
            }
        ),
        Tool(
            name="list_models",
            description="List all available TPU models grouped by category (classification, detection, pose, segmentation, audio).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="tpu_health_check",
            description="Check TPU health status and trigger recovery if needed. Returns health metrics, retry counts, and recovery status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "force_reconnect": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force a reconnection attempt even if healthy"
                    }
                }
            }
        ),
        Tool(
            name="tpu_reconnect",
            description="Manually trigger TPU reconnection. Use when TPU is unresponsive or after device reset.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""

    try:
        if name == "tpu_status":
            return await handle_tpu_status()

        elif name == "classify_image":
            return await handle_classify_image(arguments)

        elif name == "get_visual_embedding":
            return await handle_visual_embedding(arguments)

        elif name == "embed_text":
            return await handle_embed_text(arguments)

        elif name == "score_importance":
            return await handle_score_importance(arguments)

        elif name == "classify_intent":
            return await handle_classify_intent(arguments)

        elif name == "compute_similarity":
            return await handle_compute_similarity(arguments)

        elif name == "detect_anomaly":
            return await handle_detect_anomaly(arguments)

        elif name == "spot_keyword":
            return await handle_spot_keyword(arguments)

        elif name == "estimate_pose":
            return await handle_estimate_pose(arguments)

        elif name == "detect_objects":
            return await handle_detect_objects(arguments)

        elif name == "segment_image":
            return await handle_segment_image(arguments)

        elif name == "classify_audio":
            return await handle_classify_audio(arguments)

        elif name == "list_models":
            return await handle_list_models(arguments)

        elif name == "tpu_health_check":
            return await handle_tpu_health_check(arguments)

        elif name == "tpu_reconnect":
            return await handle_tpu_reconnect(arguments)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "tool": name
        }))]


async def handle_tpu_status() -> List[TextContent]:
    """Get TPU status and stats."""
    engine = get_engine()
    stats = engine.get_stats()

    # Add model info
    stats["available_models"] = {
        name: config["description"]
        for name, config in MODELS.items()
    }

    # Check text model
    text_model = get_text_model()
    stats["text_embedding_available"] = text_model is not None

    return [TextContent(type="text", text=json.dumps(stats, indent=2))]


async def handle_classify_image(args: Dict) -> List[TextContent]:
    """Classify image using TPU."""
    engine = get_engine()

    # Get model config
    model_key = args.get("model", "mobilenet_v2")
    if model_key not in MODELS:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown model: {model_key}"
        }))]

    config = MODELS[model_key]

    # Load model if needed
    if not engine.load_model(config["file"], config["labels"]):
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to load model: {config['file']}"
        }))]

    # Get image data
    if "image_base64" in args:
        image_data = base64.b64decode(args["image_base64"])
    elif "image_path" in args:
        with open(args["image_path"], "rb") as f:
            image_data = f.read()
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide image_base64 or image_path"
        }))]

    # Preprocess
    input_data = preprocess_image(image_data, config["input_size"])

    # Run inference
    result = engine.classify(
        config["file"],
        input_data,
        top_k=args.get("top_k", 5)
    )

    return [TextContent(type="text", text=json.dumps({
        "predictions": result.predictions,
        "latency_ms": result.latency_ms,
        "model": model_key
    }, indent=2))]


async def handle_visual_embedding(args: Dict) -> List[TextContent]:
    """Extract visual embedding using TPU."""
    engine = get_engine()

    config = MODELS["efficientnet_s"]

    # Load model
    if not engine.load_model(config["file"], config["labels"]):
        return [TextContent(type="text", text=json.dumps({
            "error": "Failed to load EfficientNet model"
        }))]

    # Get image data
    if "image_base64" in args:
        image_data = base64.b64decode(args["image_base64"])
    elif "image_path" in args:
        with open(args["image_path"], "rb") as f:
            image_data = f.read()
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide image_base64 or image_path"
        }))]

    # Preprocess
    input_data = preprocess_image(image_data, config["input_size"])

    # Get embedding
    embedding, latency_ms = engine.get_embedding(config["file"], input_data)

    return [TextContent(type="text", text=json.dumps({
        "embedding": embedding.tolist(),
        "dimension": len(embedding),
        "latency_ms": latency_ms
    }))]


async def handle_embed_text(args: Dict) -> List[TextContent]:
    """Generate text embedding using CPU model."""
    model = get_text_model()
    if model is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Text embedding model not available. Install sentence-transformers."
        }))]

    start = time.perf_counter()

    if "texts" in args:
        texts = args["texts"]
        embeddings = model.encode(texts)
        latency_ms = (time.perf_counter() - start) * 1000

        return [TextContent(type="text", text=json.dumps({
            "embeddings": [e.tolist() for e in embeddings],
            "count": len(texts),
            "dimension": TEXT_EMBEDDING_DIM,
            "latency_ms": latency_ms
        }))]
    elif "text" in args:
        embedding = model.encode(args["text"])
        latency_ms = (time.perf_counter() - start) * 1000

        return [TextContent(type="text", text=json.dumps({
            "embedding": embedding.tolist(),
            "dimension": TEXT_EMBEDDING_DIM,
            "latency_ms": latency_ms
        }))]
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide text or texts"
        }))]


async def handle_score_importance(args: Dict) -> List[TextContent]:
    """Score content importance for memory prioritization."""
    model = get_text_model()
    if model is None:
        # Fallback: simple heuristic scoring
        content = args["content"]
        score = min(1.0, len(content) / 500)  # Longer = more important (rough heuristic)
        return [TextContent(type="text", text=json.dumps({
            "importance_score": score,
            "method": "heuristic",
            "note": "Install sentence-transformers for semantic scoring"
        }))]

    content = args["content"]
    context = args.get("context", "")

    # Importance indicators
    importance_phrases = [
        "critical", "important", "key insight", "remember",
        "breakthrough", "discovery", "error", "bug", "fix",
        "decision", "conclusion", "learned", "pattern",
        "algorithm", "architecture", "security", "performance"
    ]

    start = time.perf_counter()

    # Embed content and importance phrases
    all_texts = [content] + importance_phrases
    embeddings = model.encode(all_texts)

    content_emb = embeddings[0]
    phrase_embs = embeddings[1:]

    # Calculate max similarity to importance phrases
    similarities = []
    for phrase_emb in phrase_embs:
        sim = np.dot(content_emb, phrase_emb) / (
            np.linalg.norm(content_emb) * np.linalg.norm(phrase_emb)
        )
        similarities.append(sim)

    # Importance score based on max similarity
    importance_score = float(max(similarities))

    # Boost for longer content (more substance)
    length_factor = min(1.0, len(content) / 200)
    importance_score = (importance_score + length_factor) / 2

    latency_ms = (time.perf_counter() - start) * 1000

    return [TextContent(type="text", text=json.dumps({
        "importance_score": importance_score,
        "method": "semantic",
        "latency_ms": latency_ms
    }))]


async def handle_classify_intent(args: Dict) -> List[TextContent]:
    """Classify user intent for routing."""
    model = get_text_model()

    text = args["text"].lower().strip()

    # Intent patterns
    intents = {
        "question": ["what", "how", "why", "when", "where", "who", "can you", "could you", "is it", "are there", "?"],
        "command": ["do", "make", "create", "build", "run", "execute", "start", "stop", "install", "remove", "delete"],
        "search": ["find", "search", "look for", "locate", "show me", "list"],
        "explain": ["explain", "describe", "tell me about", "what is", "define"],
        "fix": ["fix", "debug", "solve", "repair", "troubleshoot", "error", "bug", "issue"],
        "create": ["create", "write", "generate", "make", "build", "implement", "add"],
        "modify": ["change", "update", "edit", "modify", "refactor", "improve"]
    }

    if model is None:
        # Simple keyword matching
        scores = {}
        for intent, keywords in intents.items():
            score = sum(1 for kw in keywords if kw in text)
            scores[intent] = score / len(keywords)

        top_intent = max(scores, key=scores.get)
        confidence = scores[top_intent]

        return [TextContent(type="text", text=json.dumps({
            "intent": top_intent,
            "confidence": confidence,
            "all_scores": scores,
            "method": "keyword"
        }))]

    start = time.perf_counter()

    # Semantic matching against intent exemplars
    exemplars = {
        "question": "What is the answer? How does this work?",
        "command": "Execute this action. Run the command.",
        "search": "Find files matching this pattern. Search for code.",
        "explain": "Explain how this works. Describe the process.",
        "fix": "Fix this bug. Debug the error.",
        "create": "Create a new file. Write this function.",
        "modify": "Change this code. Update the configuration."
    }

    # Embed text and exemplars
    all_texts = [text] + list(exemplars.values())
    embeddings = model.encode(all_texts)

    text_emb = embeddings[0]
    scores = {}

    for i, (intent, _) in enumerate(exemplars.items()):
        exemplar_emb = embeddings[i + 1]
        sim = float(np.dot(text_emb, exemplar_emb) / (
            np.linalg.norm(text_emb) * np.linalg.norm(exemplar_emb)
        ))
        scores[intent] = sim

    top_intent = max(scores, key=scores.get)
    latency_ms = (time.perf_counter() - start) * 1000

    return [TextContent(type="text", text=json.dumps({
        "intent": top_intent,
        "confidence": scores[top_intent],
        "all_scores": scores,
        "method": "semantic",
        "latency_ms": latency_ms
    }))]


async def handle_compute_similarity(args: Dict) -> List[TextContent]:
    """Compute semantic similarity between texts."""
    model = get_text_model()
    if model is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Text model not available"
        }))]

    start = time.perf_counter()

    embeddings = model.encode([args["text1"], args["text2"]])
    emb1, emb2 = embeddings[0], embeddings[1]

    similarity = float(np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    ))

    latency_ms = (time.perf_counter() - start) * 1000

    return [TextContent(type="text", text=json.dumps({
        "similarity": similarity,
        "latency_ms": latency_ms
    }))]


async def handle_detect_anomaly(args: Dict) -> List[TextContent]:
    """Detect anomalous patterns."""
    model = get_text_model()
    if model is None:
        return [TextContent(type="text", text=json.dumps({
            "error": "Text model not available"
        }))]

    content = args["content"]
    threshold = args.get("threshold", 0.7)
    baseline = args.get("baseline_embeddings", [])

    start = time.perf_counter()

    # Embed content
    content_emb = model.encode(content)

    if not baseline:
        # No baseline - can't detect anomaly
        return [TextContent(type="text", text=json.dumps({
            "is_anomaly": False,
            "note": "No baseline provided for comparison",
            "embedding": content_emb.tolist()
        }))]

    # Calculate similarity to baseline
    baseline_arr = np.array(baseline)
    similarities = []

    for base_emb in baseline_arr:
        sim = float(np.dot(content_emb, base_emb) / (
            np.linalg.norm(content_emb) * np.linalg.norm(base_emb)
        ))
        similarities.append(sim)

    max_similarity = max(similarities)
    avg_similarity = sum(similarities) / len(similarities)

    # Anomaly if max similarity is below threshold
    is_anomaly = max_similarity < threshold

    latency_ms = (time.perf_counter() - start) * 1000

    return [TextContent(type="text", text=json.dumps({
        "is_anomaly": is_anomaly,
        "max_similarity": max_similarity,
        "avg_similarity": avg_similarity,
        "threshold": threshold,
        "latency_ms": latency_ms
    }))]


async def handle_spot_keyword(args: Dict) -> List[TextContent]:
    """Detect keywords in audio using TPU keyword spotter model."""
    engine = get_engine()

    if not engine.is_available:
        return [TextContent(type="text", text=json.dumps({
            "error": "TPU not available",
            "keywords": []
        }))]

    threshold = args.get("threshold", 0.5)

    try:
        import librosa
        import scipy.io.wavfile as wavfile

        # Load audio data
        if "audio_base64" in args:
            audio_bytes = base64.b64decode(args["audio_base64"])
            # Assume 16-bit PCM 16kHz
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
        elif "audio_path" in args:
            audio_path = args["audio_path"]
            if audio_path.endswith('.wav'):
                sr_orig, audio_raw = wavfile.read(audio_path)
                audio_data = audio_raw.astype(np.float32) / 32768.0
                if sr_orig != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sr_orig, target_sr=16000)
                sr = 16000
            else:
                audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": "Must provide audio_base64 or audio_path",
                "keywords": []
            }))]

        # Ensure we have 2 seconds of audio (pad or trim)
        target_length = 2 * sr  # 2 seconds at 16kHz = 32000 samples
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        else:
            audio_data = audio_data[:target_length]

        # Generate mel spectrogram (32 mel bins, 10ms hop)
        hop_length = int(0.01 * sr)  # 10ms
        n_fft = 1024
        n_mels = 32

        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to uint8 for quantized model
        log_mel_norm = ((log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8) * 255).astype(np.uint8)

        # Transpose to (time, mel) and reshape for model
        features = log_mel_norm.T  # Shape: (time_steps, 32)

        start = time.perf_counter()

        # Load keyword spotter model
        config = MODELS["keyword_spotter"]
        if not engine.load_model(config["file"]):
            return [TextContent(type="text", text=json.dumps({
                "error": "Failed to load keyword spotter model",
                "keywords": []
            }))]

        # Get input shape and resize features if needed
        input_details = engine.get_input_details(config["file"])
        if input_details:
            expected_shape = input_details["shape"]
            # Reshape features to match expected input (typically [1, time, mels])
            if len(expected_shape) == 3:
                # Resize time dimension if needed
                target_time = expected_shape[1]
                if features.shape[0] != target_time:
                    # Resample time dimension
                    indices = np.linspace(0, features.shape[0]-1, target_time).astype(int)
                    features = features[indices]
                features = features.reshape(1, target_time, n_mels)

        # Run inference
        result = engine.classify(config["file"], features, top_k=5, threshold=threshold)

        latency_ms = (time.perf_counter() - start) * 1000

        # Filter by threshold and format results
        keywords = []
        for pred in result.predictions:
            if pred["score"] >= threshold:
                keywords.append({
                    "keyword": pred.get("label", f"class_{pred['class_id']}"),
                    "confidence": pred["score"],
                    "class_id": pred["class_id"]
                })

        return [TextContent(type="text", text=json.dumps({
            "keywords": keywords,
            "latency_ms": latency_ms,
            "audio_duration_sec": len(audio_data) / sr,
            "threshold": threshold,
            "tpu_used": True
        }))]

    except ImportError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Missing audio library: {e}. Install librosa: pip install librosa",
            "keywords": []
        }))]
    except Exception as e:
        logger.exception("Keyword spotting error")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "keywords": []
        }))]


async def handle_estimate_pose(args: Dict) -> List[TextContent]:
    """Estimate human pose from image using TPU PoseNet/MoveNet models."""
    engine = get_engine()

    if not engine.is_available:
        return [TextContent(type="text", text=json.dumps({
            "error": "TPU not available",
            "keypoints": []
        }))]

    # Get model
    model_key = args.get("model", "movenet")
    if model_key not in ["movenet", "posenet_353", "posenet_481", "posenet_721"]:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown pose model: {model_key}. Use movenet, posenet_353, posenet_481, or posenet_721"
        }))]

    config = MODELS[model_key]

    # Check if model file exists
    model_path = MODELS_DIR / config["file"]
    if not model_path.exists():
        return [TextContent(type="text", text=json.dumps({
            "error": f"Model file not found: {config['file']}. Download from coral.ai/models"
        }))]

    # Load model
    if not engine.load_model(config["file"]):
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to load pose model: {config['file']}"
        }))]

    # Get image data
    if "image_base64" in args:
        image_data = base64.b64decode(args["image_base64"])
    elif "image_path" in args:
        with open(args["image_path"], "rb") as f:
            image_data = f.read()
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide image_base64 or image_path"
        }))]

    try:
        # Preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size

        # Resize to model input size
        input_size = config["input_size"]
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        input_data = np.array(image_resized, dtype=np.uint8)

        # Run inference
        start = time.perf_counter()
        interpreter = engine.interpreters[config["file"]]

        from pycoral.adapters import common
        common.set_input(interpreter, input_data)
        interpreter.invoke()

        latency_ms = (time.perf_counter() - start) * 1000

        # Get output tensors
        output_details = interpreter.get_output_details()

        # Parse pose output based on model type
        keypoints = []

        if model_key == "movenet":
            # MoveNet outputs [1, 1, 17, 3] - (batch, num_poses, keypoints, [y, x, score])
            output = interpreter.get_tensor(output_details[0]["index"])
            poses = output[0, 0]  # First pose

            for i, kp_name in enumerate(POSE_KEYPOINTS):
                y, x, score = poses[i]
                # Convert normalized coords to image coords
                keypoints.append({
                    "name": kp_name,
                    "x": float(x) * original_size[0],
                    "y": float(y) * original_size[1],
                    "confidence": float(score)
                })
        else:
            # PoseNet - different output format
            # Output typically has heatmaps and offsets
            if len(output_details) >= 4:
                # Standard PoseNet outputs
                heatmaps = interpreter.get_tensor(output_details[0]["index"])
                offsets = interpreter.get_tensor(output_details[1]["index"])

                # Simple peak detection from heatmaps
                for i, kp_name in enumerate(POSE_KEYPOINTS):
                    if i < heatmaps.shape[-1]:
                        hm = heatmaps[0, :, :, i]
                        max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                        score = float(hm[max_idx])

                        # Convert to image coordinates
                        y = (max_idx[0] / hm.shape[0]) * original_size[1]
                        x = (max_idx[1] / hm.shape[1]) * original_size[0]

                        keypoints.append({
                            "name": kp_name,
                            "x": float(x),
                            "y": float(y),
                            "confidence": score
                        })
            else:
                # Fallback - raw output
                output = interpreter.get_tensor(output_details[0]["index"]).flatten()
                for i, kp_name in enumerate(POSE_KEYPOINTS[:min(17, len(output)//3)]):
                    idx = i * 3
                    keypoints.append({
                        "name": kp_name,
                        "x": float(output[idx]) * original_size[0],
                        "y": float(output[idx + 1]) * original_size[1],
                        "confidence": float(output[idx + 2]) if idx + 2 < len(output) else 0.5
                    })

        # Update stats
        engine._update_stats(config["file"], latency_ms, operation="pose_estimation")

        return [TextContent(type="text", text=json.dumps({
            "keypoints": keypoints,
            "num_keypoints": len(keypoints),
            "model": model_key,
            "latency_ms": latency_ms,
            "image_size": original_size,
            "tpu_used": True
        }, indent=2))]

    except Exception as e:
        logger.exception("Pose estimation error")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "keypoints": []
        }))]


async def handle_detect_objects(args: Dict) -> List[TextContent]:
    """Detect objects in image using TPU SSD model (90 COCO classes)."""
    engine = get_engine()

    if not engine.is_available:
        return [TextContent(type="text", text=json.dumps({
            "error": "TPU not available",
            "detections": []
        }))]

    config = MODELS["coco_detection"]
    threshold = args.get("threshold", 0.4)
    max_detections = args.get("max_detections", 10)

    # Check if model exists
    model_path = MODELS_DIR / config["file"]
    if not model_path.exists():
        return [TextContent(type="text", text=json.dumps({
            "error": f"Model not found: {config['file']}. Download ssdlite_mobiledet_coco_edgetpu.tflite from coral.ai"
        }))]

    # Load model
    if not engine.load_model(config["file"], config["labels"]):
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to load detection model: {config['file']}"
        }))]

    # Get image data
    if "image_base64" in args:
        image_data = base64.b64decode(args["image_base64"])
    elif "image_path" in args:
        with open(args["image_path"], "rb") as f:
            image_data = f.read()
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide image_base64 or image_path"
        }))]

    try:
        # Preprocess
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size
        input_size = config["input_size"]
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        input_data = np.array(image_resized, dtype=np.uint8)

        # Run inference
        start = time.perf_counter()
        interpreter = engine.interpreters[config["file"]]

        from pycoral.adapters import common, detect
        common.set_input(interpreter, input_data)
        interpreter.invoke()

        latency_ms = (time.perf_counter() - start) * 1000

        # Get detections using pycoral detect adapter
        try:
            objs = detect.get_objects(interpreter, threshold)
        except Exception:
            # Fallback: manual parsing
            output_details = interpreter.get_output_details()
            objs = []

            # SSD outputs: boxes, classes, scores, count
            boxes = interpreter.get_tensor(output_details[0]["index"])[0]
            classes = interpreter.get_tensor(output_details[1]["index"])[0]
            scores = interpreter.get_tensor(output_details[2]["index"])[0]
            count = int(interpreter.get_tensor(output_details[3]["index"])[0])

            class Object:
                def __init__(self, id, score, bbox):
                    self.id = id
                    self.score = score
                    self.bbox = bbox

            class BBox:
                def __init__(self, ymin, xmin, ymax, xmax):
                    self.ymin, self.xmin = ymin, xmin
                    self.ymax, self.xmax = ymax, xmax

            for i in range(min(count, len(scores))):
                if scores[i] >= threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    objs.append(Object(int(classes[i]), float(scores[i]),
                                      BBox(ymin, xmin, ymax, xmax)))

        # Load labels
        labels = engine.labels.get(config["file"], [])

        # Format detections
        detections = []
        for obj in objs[:max_detections]:
            if obj.score >= threshold:
                # Convert normalized bbox to image coordinates
                bbox = {
                    "xmin": int(obj.bbox.xmin * original_size[0]),
                    "ymin": int(obj.bbox.ymin * original_size[1]),
                    "xmax": int(obj.bbox.xmax * original_size[0]),
                    "ymax": int(obj.bbox.ymax * original_size[1])
                }
                label = labels[obj.id] if obj.id < len(labels) else f"class_{obj.id}"
                detections.append({
                    "label": label,
                    "class_id": obj.id,
                    "confidence": float(obj.score),
                    "bbox": bbox
                })

        # Update stats
        engine._update_stats(config["file"], latency_ms, operation="object_detection")

        return [TextContent(type="text", text=json.dumps({
            "detections": detections,
            "num_detections": len(detections),
            "latency_ms": latency_ms,
            "image_size": original_size,
            "threshold": threshold,
            "tpu_used": True
        }, indent=2))]

    except Exception as e:
        logger.exception("Object detection error")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "detections": []
        }))]


async def handle_segment_image(args: Dict) -> List[TextContent]:
    """Perform semantic segmentation using DeepLab v3 TPU model."""
    engine = get_engine()

    if not engine.is_available:
        return [TextContent(type="text", text=json.dumps({
            "error": "TPU not available"
        }))]

    config = MODELS["deeplabv3_pascal"]
    return_mask = args.get("return_mask", False)

    # Check if model exists
    model_path = MODELS_DIR / config["file"]
    if not model_path.exists():
        return [TextContent(type="text", text=json.dumps({
            "error": f"Model not found: {config['file']}. Download deeplabv3_pascal_edgetpu.tflite from coral.ai"
        }))]

    # Load model
    if not engine.load_model(config["file"], config["labels"]):
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to load segmentation model: {config['file']}"
        }))]

    # Get image data
    if "image_base64" in args:
        image_data = base64.b64decode(args["image_base64"])
    elif "image_path" in args:
        with open(args["image_path"], "rb") as f:
            image_data = f.read()
    else:
        return [TextContent(type="text", text=json.dumps({
            "error": "Must provide image_base64 or image_path"
        }))]

    try:
        # Pascal VOC class names
        pascal_classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        # Preprocess
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size
        input_size = config["input_size"]
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        input_data = np.array(image_resized, dtype=np.uint8)

        # Run inference
        start = time.perf_counter()
        interpreter = engine.interpreters[config["file"]]

        from pycoral.adapters import common
        common.set_input(interpreter, input_data)
        interpreter.invoke()

        latency_ms = (time.perf_counter() - start) * 1000

        # Get segmentation output
        output_details = interpreter.get_output_details()
        seg_output = interpreter.get_tensor(output_details[0]["index"])

        # Get class per pixel (argmax)
        if len(seg_output.shape) == 4:
            seg_map = np.argmax(seg_output[0], axis=-1)
        else:
            seg_map = seg_output[0] if len(seg_output.shape) == 3 else seg_output

        # Count pixels per class
        unique, counts = np.unique(seg_map, return_counts=True)
        total_pixels = seg_map.size

        class_distribution = {}
        for cls_id, count in zip(unique, counts):
            cls_id = int(cls_id)
            label = pascal_classes[cls_id] if cls_id < len(pascal_classes) else f"class_{cls_id}"
            percentage = float(count) / total_pixels * 100
            if percentage > 0.5:  # Only include classes with >0.5% coverage
                class_distribution[label] = {
                    "class_id": cls_id,
                    "pixel_count": int(count),
                    "percentage": round(percentage, 2)
                }

        # Update stats
        engine._update_stats(config["file"], latency_ms, operation="segmentation")

        result = {
            "class_distribution": class_distribution,
            "num_classes_detected": len(class_distribution),
            "latency_ms": latency_ms,
            "image_size": original_size,
            "segmentation_size": list(seg_map.shape),
            "tpu_used": True
        }

        # Optionally include full mask (can be large)
        if return_mask:
            result["segmentation_mask"] = seg_map.tolist()

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.exception("Segmentation error")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e)
        }))]


async def handle_classify_audio(args: Dict) -> List[TextContent]:
    """Classify audio/sounds using YamNet TPU model (520+ classes)."""
    engine = get_engine()

    if not engine.is_available:
        return [TextContent(type="text", text=json.dumps({
            "error": "TPU not available",
            "predictions": []
        }))]

    config = MODELS["yamnet"]
    top_k = args.get("top_k", 5)

    # Check if model exists
    model_path = MODELS_DIR / config["file"]
    if not model_path.exists():
        return [TextContent(type="text", text=json.dumps({
            "error": f"Model not found: {config['file']}. Download yamnet_edgetpu.tflite from coral.ai"
        }))]

    # Load model
    if not engine.load_model(config["file"]):
        return [TextContent(type="text", text=json.dumps({
            "error": f"Failed to load YamNet model: {config['file']}"
        }))]

    try:
        import librosa
        import scipy.io.wavfile as wavfile

        # Load audio data
        if "audio_base64" in args:
            audio_bytes = base64.b64decode(args["audio_base64"])
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000
        elif "audio_path" in args:
            audio_path = args["audio_path"]
            if audio_path.endswith('.wav'):
                sr_orig, audio_raw = wavfile.read(audio_path)
                if audio_raw.dtype == np.int16:
                    audio_data = audio_raw.astype(np.float32) / 32768.0
                elif audio_raw.dtype == np.float32:
                    audio_data = audio_raw
                else:
                    audio_data = audio_raw.astype(np.float32) / np.max(np.abs(audio_raw))
                if sr_orig != 16000:
                    audio_data = librosa.resample(audio_data, orig_sr=sr_orig, target_sr=16000)
                sr = 16000
            else:
                audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": "Must provide audio_base64 or audio_path"
            }))]

        # YamNet expects 0.975s windows at 16kHz
        window_length = int(0.975 * sr)  # ~15600 samples
        hop_length = int(0.975 * sr)  # Non-overlapping

        # Ensure we have enough audio
        if len(audio_data) < window_length:
            audio_data = np.pad(audio_data, (0, window_length - len(audio_data)))

        # Process in windows and aggregate predictions
        all_predictions = []
        interpreter = engine.interpreters[config["file"]]
        input_details = interpreter.get_input_details()

        start = time.perf_counter()

        # Get expected input shape
        expected_shape = input_details[0]["shape"]

        # Process first window (or aggregate multiple)
        window = audio_data[:window_length]

        # YamNet typically expects mel spectrogram features
        # Generate 96 mel bands log spectrogram
        n_fft = 1024
        hop = 160  # 10ms at 16kHz
        n_mels = 96

        mel_spec = librosa.feature.melspectrogram(
            y=window, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to match model input
        log_mel_norm = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

        # Reshape for model - YamNet expects [1, frames, mels] or similar
        if len(expected_shape) == 3:
            target_frames = expected_shape[1]
            target_mels = expected_shape[2]

            # Resize to match expected shape
            features = log_mel_norm.T  # Transpose to (time, mels)
            if features.shape[0] != target_frames:
                indices = np.linspace(0, features.shape[0]-1, target_frames).astype(int)
                features = features[indices]
            if features.shape[1] != target_mels:
                # Resize mel dimension
                features = np.array([np.interp(np.linspace(0, len(f)-1, target_mels),
                                               np.arange(len(f)), f) for f in features])

            features = features.reshape(1, target_frames, target_mels).astype(np.float32)

            # Quantize if needed
            if input_details[0]["dtype"] == np.uint8:
                features = ((features - features.min()) / (features.max() - features.min() + 1e-8) * 255).astype(np.uint8)
        else:
            # Flatten for simpler input
            features = log_mel_norm.flatten().astype(np.float32)
            features = features[:np.prod(expected_shape[1:])].reshape(expected_shape)

        # Run inference
        from pycoral.adapters import common
        common.set_input(interpreter, features)
        interpreter.invoke()

        latency_ms = (time.perf_counter() - start) * 1000

        # Get predictions
        output_details = interpreter.get_output_details()
        scores = interpreter.get_tensor(output_details[0]["index"]).flatten()

        # Load YamNet labels from CSV
        labels = []
        labels_path = MODELS_DIR / config["labels"]
        if labels_path.exists():
            import csv
            with open(labels_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        labels.append(row[2])  # Display name is usually column 3
                    elif len(row) >= 1:
                        labels.append(row[0])

        # Get top-k predictions
        top_indices = np.argsort(scores)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            label = labels[idx] if idx < len(labels) else f"sound_class_{idx}"
            predictions.append({
                "label": label,
                "class_id": int(idx),
                "confidence": float(scores[idx])
            })

        # Update stats
        engine._update_stats(config["file"], latency_ms, operation="audio_classification")

        return [TextContent(type="text", text=json.dumps({
            "predictions": predictions,
            "audio_duration_sec": len(audio_data) / sr,
            "latency_ms": latency_ms,
            "num_classes": len(labels) if labels else len(scores),
            "tpu_used": True
        }, indent=2))]

    except ImportError as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Missing audio library: {e}. Install librosa: pip install librosa",
            "predictions": []
        }))]
    except Exception as e:
        logger.exception("Audio classification error")
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "predictions": []
        }))]


async def handle_list_models(args: Dict) -> List[TextContent]:
    """List all available TPU models grouped by category."""
    engine = get_engine()

    # Group models by category
    by_category = {}
    for name, config in MODELS.items():
        category = config.get("category", "other")
        if category not in by_category:
            by_category[category] = []

        # Check if model file exists
        model_path = MODELS_DIR / config["file"]
        exists = model_path.exists()

        by_category[category].append({
            "name": name,
            "file": config["file"],
            "description": config["description"],
            "input_size": config["input_size"],
            "installed": exists,
            "loaded": config["file"] in engine.interpreters
        })

    # Summary stats
    total_models = len(MODELS)
    installed = sum(1 for name, cfg in MODELS.items() if (MODELS_DIR / cfg["file"]).exists())
    loaded = len(engine.interpreters)

    return [TextContent(type="text", text=json.dumps({
        "models_by_category": by_category,
        "summary": {
            "total_models": total_models,
            "installed": installed,
            "loaded": loaded,
            "tpu_available": engine.is_available
        },
        "models_directory": str(MODELS_DIR)
    }, indent=2))]


async def handle_tpu_health_check(args: Dict) -> List[TextContent]:
    """Check TPU health status and optionally trigger recovery."""
    engine = get_engine()
    force_reconnect = args.get("force_reconnect", False)

    # Get current health status
    health_status = engine.health_check()

    # Force reconnect if requested
    if force_reconnect:
        logger.info("Force reconnection requested via health check")
        reconnect_success = engine.reconnect()
        health_status["force_reconnect_requested"] = True
        health_status["reconnect_result"] = "success" if reconnect_success else "failed"

        # Get updated health after reconnect
        if reconnect_success:
            health_status.update(engine.health_check())

    # Add recommendations
    recommendations = []
    if not health_status.get("is_healthy", False):
        if health_status.get("consecutive_failures", 0) > 3:
            recommendations.append("Multiple failures detected - consider restarting TPU device")
        if health_status.get("last_error"):
            recommendations.append(f"Last error: {health_status['last_error']}")
        if not health_status.get("tpu_available", False):
            recommendations.append("TPU not detected - check USB connection")

    health_status["recommendations"] = recommendations

    return [TextContent(type="text", text=json.dumps(health_status, indent=2))]


async def handle_tpu_reconnect(args: Dict) -> List[TextContent]:
    """Manually trigger TPU reconnection."""
    engine = get_engine()

    logger.info("Manual TPU reconnection triggered")

    # Get state before reconnect
    before_available = engine.is_available
    before_health = engine._get_health_status()

    # Attempt reconnection
    success = engine.reconnect()

    # Get state after reconnect
    after_available = engine.is_available
    after_health = engine._get_health_status()

    result = {
        "success": success,
        "before": {
            "tpu_available": before_available,
            "was_healthy": before_health.get("is_healthy", False)
        },
        "after": {
            "tpu_available": after_available,
            "is_healthy": after_health.get("is_healthy", False)
        },
        "recovery_attempts": after_health.get("recovery_attempts", 0),
        "message": "TPU reconnected successfully" if success else "TPU reconnection failed - check device connection"
    }

    # Add loaded models info
    result["loaded_models"] = list(engine.interpreters.keys())

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _load_models_background():
    """Load models in background after server starts."""
    await asyncio.sleep(0.1)  # Let server start first
    engine = get_engine()
    logger.info(f"TPU available: {engine.is_available}")

    # Load models lazily
    for model_key, config in MODELS.items():
        if (MODELS_DIR / config["file"]).exists():
            engine.load_model(config["file"], config.get("labels"))
            await asyncio.sleep(0)  # Yield to allow MCP messages

    logger.info("All models loaded")


async def main():
    """Run the MCP server."""
    logger.info("Starting Coral TPU MCP Server...")

    # Start model loading in background (don't block MCP init)
    asyncio.create_task(_load_models_background())

    # Run server immediately (models load in background)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    """Synchronous entry point for console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
