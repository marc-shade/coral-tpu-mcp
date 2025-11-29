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

# Model configurations
MODELS = {
    "mobilenet_v2": {
        "file": "mobilenet_v2_edgetpu.tflite",
        "labels": "imagenet_labels.txt",
        "input_size": (224, 224),
        "description": "Image classification (1000 ImageNet classes)"
    },
    "efficientnet_s": {
        "file": "efficientnet_s_edgetpu.tflite",
        "labels": "imagenet_labels.txt",
        "input_size": (224, 224),
        "description": "Visual embeddings and classification"
    },
    "face_detection": {
        "file": "face_detection_edgetpu.tflite",
        "labels": None,
        "input_size": (320, 320),
        "description": "Face detection"
    },
    "keyword_spotter": {
        "file": "keyword_spotter_edgetpu.tflite",
        "labels": None,
        "input_size": None,  # Audio model
        "description": "Voice command keyword spotting"
    }
}

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


if __name__ == "__main__":
    asyncio.run(main())
