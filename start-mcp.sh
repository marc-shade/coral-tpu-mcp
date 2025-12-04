#!/bin/bash
# Wrapper script for coral-tpu MCP server
# Ensures clean environment and suppresses all unwanted output

# Suppress all TensorFlow/TFLite logging
export TF_CPP_MIN_LOG_LEVEL=3
export ABSL_MIN_LOG_LEVEL=3

# Set Python path
export PYTHONPATH="${AGENTIC_SYSTEM_PATH:-/opt/agentic}/mcp-servers/coral-tpu-mcp/src"

# Run the server
exec ${AGENTIC_SYSTEM_PATH:-/opt/agentic}/coral-venv/bin/python -m coral_tpu_mcp.server
