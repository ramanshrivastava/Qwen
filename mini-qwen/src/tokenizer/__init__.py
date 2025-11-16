"""
Tokenizer Module

This module implements Byte-Pair Encoding (BPE) tokenization,
following the approach used by GPT-2, GPT-3, LLaMA, and Qwen.

Main Components:
- BPETokenizer: Main tokenizer class (byte-level BPE)
- Utility functions: Pair counting, merging, etc.

Usage:
    from src.tokenizer import BPETokenizer

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.train(corpus)

    # Encode text
    ids = tokenizer.encode("Hello world")

    # Decode back
    text = tokenizer.decode(ids)

Historical Context:
- 2016: BPE introduced for NMT (Sennrich et al.)
- 2019: GPT-2 uses byte-level BPE
- 2019: OpenAI releases tiktoken (Rust-based)
- 2023: Qwen uses tiktoken with 151,851 vocab

See Also:
- docs/adrs/ADR-007-bpe-tokenization.md (decision record)
- Qwen: /home/user/Qwen/tokenization_note.md
"""

from .bpe import BPETokenizer, get_pair_frequencies, merge_pair

__all__ = ['BPETokenizer', 'get_pair_frequencies', 'merge_pair']
