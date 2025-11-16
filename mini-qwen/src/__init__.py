"""
Mini-Qwen: A Learning-Focused Implementation of Modern LLM Architecture

This package implements the Qwen-7B architecture at educational scale (~50M params).

Main components:
- tokenizer: BPE tokenization
- model: Transformer architecture (RoPE, RMSNorm, SwiGLU, Multi-Head Attention)
- training: Training pipeline (AdamW, cosine schedule, mixed precision)
- inference: Text generation (greedy, top-k, top-p, KV cache)
- utils: Utilities for logging, checkpointing, metrics

Author: Mini-Qwen Learning Project
Version: 0.1.0 (Phase 1 in progress)
"""

__version__ = "0.1.0"
__author__ = "Mini-Qwen Learning Project"
