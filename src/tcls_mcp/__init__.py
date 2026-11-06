"""MCP server that exposes a fine-tuned BERTurk classifier as tools.

Modules:
- ``config``: settings from environment variables (MODEL_PATH, MAX_BATCH, DEVICE, ...)
- ``model``: loading, tokenization/truncation and batched inference
- ``explain``: word-level occlusion explanations
- ``card``: the ``model://card`` resource
- ``schemas``: Pydantic output models (they become the tools' output schemas)
- ``server``: the MCPServer instance, lifespan and tool definitions
"""

__version__ = "0.1.0"
