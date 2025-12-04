#!/bin/bash
# Coral TPU MCP Server Startup Script
# NOTE: No stderr output - Claude Code treats stderr as errors

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${AGENTIC_SYSTEM_PATH:-/opt/agentic}/coral-venv"
LOG_FILE="${SCRIPT_DIR}/startup.log"

# Suppress all TensorFlow/TFLite logging
export TF_CPP_MIN_LOG_LEVEL=3
export ABSL_MIN_LOG_LEVEL=3

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}/src"

# Activate venv
source "$VENV_PATH/bin/activate"

# Start the server (stderr goes to log, stdout is MCP protocol)
echo "[$(date)] Starting coral-tpu MCP server..." >> "$LOG_FILE"
exec python -m coral_tpu_mcp.server 2>> "$LOG_FILE"
