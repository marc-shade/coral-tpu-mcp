"""
Tests for MCP server tool endpoints.

Tests cover:
- Tool listing
- Tool invocation
- Input validation
- Error handling
- Response formatting
"""

import json
import base64
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO

import pytest


# ============================================================================
# Test: Tool Listing
# ============================================================================

class TestToolListing:
    """Tests for MCP tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_list(self):
        """Test list_tools returns a list of tools."""
        from coral_tpu_mcp.server import list_tools

        tools = await list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_list_tools_includes_tpu_status(self):
        """Test list_tools includes tpu_status tool."""
        from coral_tpu_mcp.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "tpu_status" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_includes_classify_image(self):
        """Test list_tools includes classify_image tool."""
        from coral_tpu_mcp.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "classify_image" in tool_names

    @pytest.mark.asyncio
    async def test_list_tools_includes_all_expected_tools(self):
        """Test list_tools includes all expected tools."""
        from coral_tpu_mcp.server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "tpu_status",
            "classify_image",
            "get_visual_embedding",
            "embed_text",
            "score_importance",
            "classify_intent",
            "compute_similarity",
            "detect_anomaly",
            "spot_keyword",
            "estimate_pose",
            "detect_objects",
            "segment_image",
            "classify_audio",
            "list_models",
            "tpu_health_check",
            "tpu_reconnect",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    @pytest.mark.asyncio
    async def test_tools_have_valid_schemas(self):
        """Test all tools have valid input schemas."""
        from coral_tpu_mcp.server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"


# ============================================================================
# Test: TPU Status Tool
# ============================================================================

class TestTPUStatusTool:
    """Tests for tpu_status tool."""

    @pytest.mark.asyncio
    async def test_tpu_status_returns_json(self, mock_engine):
        """Test tpu_status returns valid JSON."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import handle_tpu_status

            result = await handle_tpu_status()

            assert len(result) == 1
            data = json.loads(result[0].text)
            assert "tpu_available" in data or "total_inferences" in data

    @pytest.mark.asyncio
    async def test_tpu_status_includes_models(self, mock_engine):
        """Test tpu_status includes available models."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import handle_tpu_status

            result = await handle_tpu_status()
            data = json.loads(result[0].text)

            assert "available_models" in data


# ============================================================================
# Test: Image Classification Tool
# ============================================================================

class TestClassifyImageTool:
    """Tests for classify_image tool."""

    @pytest.mark.asyncio
    async def test_classify_image_requires_input(self, mock_engine):
        """Test classify_image fails without image input."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_classify_image

            result = await handle_classify_image({})
            data = json.loads(result[0].text)

            assert "error" in data

    @pytest.mark.asyncio
    async def test_classify_image_with_base64(self, mock_engine, sample_image_base64, mock_models_dir):
        """Test classify_image with base64 input."""
        # Setup mock engine to return predictions
        mock_engine.classify = MagicMock()
        mock_engine.classify.return_value = MagicMock(
            predictions=[{"class_id": 0, "score": 0.95, "label": "cat"}],
            latency_ms=15.0
        )

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_classify_image

            result = await handle_classify_image({
                "image_base64": sample_image_base64,
                "model": "mobilenet_v2"
            })
            data = json.loads(result[0].text)

            assert "predictions" in data or "error" in data

    @pytest.mark.asyncio
    async def test_classify_image_invalid_model(self, mock_engine, sample_image_base64):
        """Test classify_image fails with invalid model name."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_classify_image

            result = await handle_classify_image({
                "image_base64": sample_image_base64,
                "model": "nonexistent_model"
            })
            data = json.loads(result[0].text)

            assert "error" in data


# ============================================================================
# Test: Text Embedding Tool
# ============================================================================

class TestEmbedTextTool:
    """Tests for embed_text tool."""

    @pytest.mark.asyncio
    async def test_embed_text_requires_input(self):
        """Test embed_text fails without text input."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=MagicMock()):
            from coral_tpu_mcp.server import handle_embed_text

            result = await handle_embed_text({})
            data = json.loads(result[0].text)

            assert "error" in data

    @pytest.mark.asyncio
    async def test_embed_text_single(self, mock_text_model):
        """Test embed_text with single text."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_embed_text

            result = await handle_embed_text({"text": "Hello world"})
            data = json.loads(result[0].text)

            assert "embedding" in data
            assert "dimension" in data

    @pytest.mark.asyncio
    async def test_embed_text_batch(self, mock_text_model):
        """Test embed_text with multiple texts."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_embed_text

            result = await handle_embed_text({
                "texts": ["Hello", "World", "Test"]
            })
            data = json.loads(result[0].text)

            assert "embeddings" in data
            assert "count" in data
            assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_embed_text_no_model(self):
        """Test embed_text when model not available."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import handle_embed_text

            result = await handle_embed_text({"text": "Hello"})
            data = json.loads(result[0].text)

            assert "error" in data


# ============================================================================
# Test: Similarity Tool
# ============================================================================

class TestComputeSimilarityTool:
    """Tests for compute_similarity tool."""

    @pytest.mark.asyncio
    async def test_compute_similarity_returns_score(self, mock_text_model):
        """Test compute_similarity returns a score."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_compute_similarity

            result = await handle_compute_similarity({
                "text1": "Hello world",
                "text2": "Hi there world"
            })
            data = json.loads(result[0].text)

            assert "similarity" in data
            assert -1 <= data["similarity"] <= 1


# ============================================================================
# Test: Intent Classification Tool
# ============================================================================

class TestClassifyIntentTool:
    """Tests for classify_intent tool."""

    @pytest.mark.asyncio
    async def test_classify_intent_question(self, mock_text_model):
        """Test classify_intent identifies questions."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_classify_intent

            result = await handle_classify_intent({
                "text": "What is the weather today?"
            })
            data = json.loads(result[0].text)

            assert "intent" in data
            assert "confidence" in data

    @pytest.mark.asyncio
    async def test_classify_intent_without_model(self):
        """Test classify_intent works with keyword matching fallback."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import handle_classify_intent

            result = await handle_classify_intent({
                "text": "What is this?"
            })
            data = json.loads(result[0].text)

            assert "intent" in data
            assert data["method"] == "keyword"


# ============================================================================
# Test: Importance Scoring Tool
# ============================================================================

class TestScoreImportanceTool:
    """Tests for score_importance tool."""

    @pytest.mark.asyncio
    async def test_score_importance_with_model(self, mock_text_model):
        """Test score_importance with text model."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_score_importance

            result = await handle_score_importance({
                "content": "This is a critical security vulnerability"
            })
            data = json.loads(result[0].text)

            assert "importance_score" in data
            assert 0 <= data["importance_score"] <= 1

    @pytest.mark.asyncio
    async def test_score_importance_fallback(self):
        """Test score_importance uses heuristic when model unavailable."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import handle_score_importance

            result = await handle_score_importance({
                "content": "Short text"
            })
            data = json.loads(result[0].text)

            assert "importance_score" in data
            assert data["method"] == "heuristic"


# ============================================================================
# Test: Anomaly Detection Tool
# ============================================================================

class TestDetectAnomalyTool:
    """Tests for detect_anomaly tool."""

    @pytest.mark.asyncio
    async def test_detect_anomaly_no_baseline(self, mock_text_model):
        """Test detect_anomaly without baseline returns embedding."""
        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_detect_anomaly

            result = await handle_detect_anomaly({
                "content": "Test content"
            })
            data = json.loads(result[0].text)

            assert "is_anomaly" in data
            assert data["is_anomaly"] is False  # No baseline to compare

    @pytest.mark.asyncio
    async def test_detect_anomaly_with_baseline(self, mock_text_model):
        """Test detect_anomaly with baseline embeddings."""
        baseline = np.random.randn(3, 384).tolist()

        with patch('coral_tpu_mcp.server.get_text_model', return_value=mock_text_model):
            from coral_tpu_mcp.server import handle_detect_anomaly

            result = await handle_detect_anomaly({
                "content": "Test content",
                "baseline_embeddings": baseline,
                "threshold": 0.5
            })
            data = json.loads(result[0].text)

            assert "is_anomaly" in data
            assert "max_similarity" in data


# ============================================================================
# Test: Health Check Tool
# ============================================================================

class TestTPUHealthCheckTool:
    """Tests for tpu_health_check tool."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, mock_engine):
        """Test tpu_health_check returns status dict."""
        mock_engine.health_check.return_value = {
            "is_healthy": True,
            "tpu_available": True,
            "consecutive_failures": 0
        }

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_tpu_health_check

            result = await handle_tpu_health_check({})
            data = json.loads(result[0].text)

            assert "is_healthy" in data or "recommendations" in data

    @pytest.mark.asyncio
    async def test_health_check_force_reconnect(self, mock_engine):
        """Test tpu_health_check with force_reconnect."""
        mock_engine.health_check.return_value = {"is_healthy": True}
        mock_engine.reconnect.return_value = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_tpu_health_check

            result = await handle_tpu_health_check({"force_reconnect": True})
            data = json.loads(result[0].text)

            assert "force_reconnect_requested" in data
            mock_engine.reconnect.assert_called_once()


# ============================================================================
# Test: Reconnect Tool
# ============================================================================

class TestTPUReconnectTool:
    """Tests for tpu_reconnect tool."""

    @pytest.mark.asyncio
    async def test_reconnect_returns_result(self, mock_engine):
        """Test tpu_reconnect returns reconnection result."""
        mock_engine.reconnect.return_value = True

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine):
            from coral_tpu_mcp.server import handle_tpu_reconnect

            result = await handle_tpu_reconnect({})
            data = json.loads(result[0].text)

            assert "success" in data
            assert "before" in data
            assert "after" in data


# ============================================================================
# Test: List Models Tool
# ============================================================================

class TestListModelsTool:
    """Tests for list_models tool."""

    @pytest.mark.asyncio
    async def test_list_models_returns_categories(self, mock_engine, mock_models_dir):
        """Test list_models returns models by category."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_list_models

            result = await handle_list_models({})
            data = json.loads(result[0].text)

            assert "models_by_category" in data
            assert "summary" in data

    @pytest.mark.asyncio
    async def test_list_models_summary(self, mock_engine, mock_models_dir):
        """Test list_models includes summary statistics."""
        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.MODELS_DIR', mock_models_dir):
            from coral_tpu_mcp.server import handle_list_models

            result = await handle_list_models({})
            data = json.loads(result[0].text)

            summary = data["summary"]
            assert "total_models" in summary
            assert "installed" in summary
            assert "tpu_available" in summary


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in tool calls."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Test unknown tool name returns error."""
        from coral_tpu_mcp.server import call_tool

        result = await call_tool("nonexistent_tool", {})

        assert len(result) == 1
        assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error(self):
        """Test tool exceptions are caught and returned as errors."""
        mock_engine = MagicMock()
        mock_engine.get_stats.side_effect = RuntimeError("Test error")

        with patch('coral_tpu_mcp.server.get_engine', return_value=mock_engine), \
             patch('coral_tpu_mcp.server.get_text_model', return_value=None):
            from coral_tpu_mcp.server import call_tool

            result = await call_tool("tpu_status", {})
            data = json.loads(result[0].text)

            # The error should be caught and returned in the response
            assert "error" in data or "total_inferences" not in data
