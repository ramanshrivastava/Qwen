"""
Byte-Pair Encoding (BPE) Tokenizer

This module implements a simplified BPE tokenizer for educational purposes.
BPE was introduced for neural machine translation (Sennrich et al., 2016)
and became the standard for language models (GPT-2, GPT-3, LLaMA, Qwen).

Key Concepts:
- Operates on UTF-8 bytes (not characters)
- Iteratively merges most frequent byte pairs
- Creates a vocabulary of subword units
- Balances vocabulary size vs sequence length

Historical Context:
- 2016: BPE introduced for NMT (Sennrich et al.)
- 2019: GPT-2 uses BPE with byte-level encoding
- 2019: tiktoken library (Rust-based, used by OpenAI)
- 2023: Qwen uses tiktoken with 151,851 vocab

Our Implementation:
- Pure Python (not Rust like tiktoken)
- Educational focus (clarity over speed)
- ~10,000 vocab (vs Qwen's 151,851)
- Follows GPT-2/Qwen approach

Reference:
- Paper: "Neural Machine Translation of Rare Words with Subword Units"
  https://arxiv.org/abs/1508.07909
- Qwen: /home/user/Qwen/tokenization_note.md
"""

import regex as re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer (byte-level).

    This tokenizer operates on UTF-8 bytes rather than Unicode characters,
    which allows it to handle any text without unknown tokens.

    Architecture:
        Text → UTF-8 bytes → BPE encoding → Token IDs
        Token IDs → BPE decoding → UTF-8 bytes → Text

    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(["Hello world", "Hello there"], vocab_size=300)
        >>> ids = tokenizer.encode("Hello world")
        >>> text = tokenizer.decode(ids)
        >>> text == "Hello world"
        True
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (includes special tokens)
        """
        self.vocab_size = vocab_size

        # Special tokens (following Qwen/GPT-2 convention)
        self.special_tokens = {
            '<|endoftext|>': 0,   # End of document/sequence
            '<|pad|>': 1,          # Padding token
            '<|unk|>': 2,          # Unknown token (rarely used with BPE)
        }

        # Reverse mapping: ID → token
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # BPE merge rules (learned during training)
        # Format: (byte_pair) → merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}

        # Vocabulary: token → id
        # Initialized with all 256 possible bytes + special tokens
        self.vocab: Dict[bytes, int] = {}

        # Cache for encoding (speeds up repeated encodings)
        self._cache: Dict[str, List[int]] = {}

        # Initialize with byte vocabulary (0-255)
        self._initialize_byte_vocab()

    def _initialize_byte_vocab(self):
        """
        Initialize vocabulary with all 256 possible byte values.

        This is the base vocabulary before any BPE merges are learned.
        Each byte (0-255) gets its own token ID.

        Special tokens occupy IDs 0-2, so bytes start at ID 3.
        """
        # Reserve first few IDs for special tokens
        next_id = len(self.special_tokens)

        # Add all 256 bytes to vocabulary
        for byte_val in range(256):
            byte_token = bytes([byte_val])
            self.vocab[byte_token] = next_id
            self.id_to_token[next_id] = byte_token
            next_id += 1

        # Verify we have 259 tokens (3 special + 256 bytes)
        assert len(self.vocab) == 256
        assert len(self.id_to_token) == 256 + len(self.special_tokens)

    def train(self, corpus: List[str], vocab_size: Optional[int] = None):
        """
        Train BPE tokenizer on a corpus of text.

        This learns which byte pairs to merge based on frequency.
        The algorithm:
        1. Start with byte-level vocabulary (256 tokens)
        2. Count all adjacent byte-pair frequencies
        3. Merge the most frequent pair
        4. Repeat until vocab_size is reached

        Args:
            corpus: List of text documents to train on
            vocab_size: Target vocabulary size (overrides __init__ value)

        Example:
            >>> tokenizer = BPETokenizer()
            >>> corpus = ["hello world", "hello there", "world peace"]
            >>> tokenizer.train(corpus, vocab_size=300)

        Note:
            Training can be slow for large corpora. For production,
            use tiktoken (Rust-based) or HuggingFace tokenizers.
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        # Reset to base vocabulary
        self._initialize_byte_vocab()
        self.merges = {}

        print(f"Training BPE tokenizer...")
        print(f"  Corpus size: {len(corpus)} documents")
        print(f"  Target vocab size: {self.vocab_size}")
        print(f"  Starting vocab: {len(self.vocab)} (base bytes)")

        # Convert corpus to byte sequences
        byte_sequences = [text.encode('utf-8') for text in corpus]

        # Current vocabulary size (special tokens + bytes + learned merges)
        current_vocab_size = len(self.special_tokens) + 256

        # Learn merges until we reach target vocab size
        num_merges = self.vocab_size - current_vocab_size

        if num_merges <= 0:
            print(f"  Vocab size {self.vocab_size} is too small!")
            print(f"  Minimum required: {current_vocab_size} (special tokens + bytes)")
            return

        print(f"  Learning {num_merges} merges...")

        # This will be implemented in Phase 1.2
        # For now, just the structure
        print(f"  [TODO Phase 1.2: Implement merge learning algorithm]")

        print(f"Training complete!")
        print(f"  Final vocab size: {len(self.vocab) + len(self.special_tokens)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using BPE.

        Process:
        1. Convert text to UTF-8 bytes
        2. Apply BPE merges (learned during training)
        3. Convert to token IDs

        Args:
            text: Input text string

        Returns:
            List of token IDs

        Example:
            >>> tokenizer.encode("Hello world")
            [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]

        # Handle empty string
        if not text:
            return []

        # Convert to UTF-8 bytes
        byte_sequence = text.encode('utf-8')

        # For now, just return byte-level encoding
        # BPE merging will be added in Phase 1.2
        token_ids = [self.vocab[bytes([b])] for b in byte_sequence]

        # Cache result
        self._cache[text] = token_ids

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Process:
        1. Convert token IDs to bytes
        2. Concatenate byte sequences
        3. Decode UTF-8 to string

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string

        Example:
            >>> ids = [72, 101, 108, 108, 111]
            >>> tokenizer.decode(ids)
            'Hello'

        Note:
            If decoding encounters invalid UTF-8, it will use
            the replacement character '�' by default.
            Use decode(..., errors='ignore') to skip invalid bytes.
        """
        # Handle empty list
        if not token_ids:
            return ""

        # Collect byte sequences
        byte_list = []

        for token_id in token_ids:
            # Check for special tokens
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]

                # Skip special tokens during decoding
                if isinstance(token, str):
                    # It's a special token like '<|endoftext|>'
                    # In practice, we might want to handle these specially
                    continue
                else:
                    # It's a byte sequence
                    byte_list.append(token)
            else:
                # Unknown token ID - this shouldn't happen with proper BPE
                # Use replacement character
                byte_list.append(b'?')

        # Concatenate all bytes
        byte_sequence = b''.join(byte_list)

        # Decode UTF-8 (with error handling)
        try:
            text = byte_sequence.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Decoding error: {e}")
            text = ""

        return text

    def save(self, path: str):
        """
        Save tokenizer vocabulary and merges to file.

        Args:
            path: Output file path (e.g., "tokenizer.json")

        TODO: Implement serialization (Phase 1.2)
        """
        print(f"[TODO Phase 1.2: Implement save to {path}]")

    def load(self, path: str):
        """
        Load tokenizer vocabulary and merges from file.

        Args:
            path: Input file path

        TODO: Implement deserialization (Phase 1.2)
        """
        print(f"[TODO Phase 1.2: Implement load from {path}]")

    def __len__(self) -> int:
        """Return current vocabulary size."""
        return len(self.vocab) + len(self.special_tokens)

    def __repr__(self) -> str:
        """String representation of tokenizer."""
        return (f"BPETokenizer(vocab_size={len(self)}, "
                f"num_merges={len(self.merges)}, "
                f"special_tokens={len(self.special_tokens)})")


# Utility functions for BPE algorithm
# These will be used in Phase 1.2

def get_pair_frequencies(byte_sequences: List[List[int]]) -> Counter:
    """
    Count frequencies of all adjacent byte pairs in sequences.

    Args:
        byte_sequences: List of byte ID sequences

    Returns:
        Counter mapping (byte1, byte2) → frequency

    Example:
        >>> seqs = [[1, 2, 3], [1, 2, 4]]
        >>> get_pair_frequencies(seqs)
        Counter({(1, 2): 2, (2, 3): 1, (2, 4): 1})
    """
    pair_counts = Counter()

    for seq in byte_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1

    return pair_counts


def merge_pair(byte_sequence: List[int], pair: Tuple[int, int],
               new_token_id: int) -> List[int]:
    """
    Merge all occurrences of a byte pair in a sequence.

    Args:
        byte_sequence: Sequence of byte IDs
        pair: Pair to merge (byte1, byte2)
        new_token_id: ID for the merged token

    Returns:
        New sequence with pairs merged

    Example:
        >>> merge_pair([1, 2, 3, 1, 2], (1, 2), 99)
        [99, 3, 99]
    """
    if len(byte_sequence) < 2:
        return byte_sequence

    merged = []
    i = 0

    while i < len(byte_sequence):
        # Check if current position matches the pair
        if (i < len(byte_sequence) - 1 and
            byte_sequence[i] == pair[0] and
            byte_sequence[i + 1] == pair[1]):
            # Merge the pair
            merged.append(new_token_id)
            i += 2  # Skip both bytes
        else:
            # Keep the byte as-is
            merged.append(byte_sequence[i])
            i += 1

    return merged


if __name__ == "__main__":
    # Simple test
    print("=" * 60)
    print("BPE Tokenizer - Phase 1.1 (Foundation)")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_size=300)
    print(f"\n{tokenizer}")

    # Test basic encoding/decoding (byte-level only for now)
    test_text = "Hello, world! 你好世界"
    print(f"\nTest text: {test_text}")

    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Encoded: {token_ids[:20]}... ({len(token_ids)} tokens)")

    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")

    # Verify round-trip
    assert decoded == test_text, "Round-trip failed!"
    print("✓ Round-trip encoding/decoding works!")

    # Show special tokens
    print(f"\nSpecial tokens:")
    for token, token_id in tokenizer.special_tokens.items():
        print(f"  {token}: {token_id}")

    # Show first few byte tokens
    print(f"\nFirst 10 byte tokens:")
    for i in range(3, 13):  # Start after special tokens
        byte_val = i - 3
        print(f"  Byte {byte_val:3d} ({chr(byte_val) if 32 <= byte_val < 127 else '?'}): ID {i}")

    print("\n" + "=" * 60)
    print("Phase 1.1 Complete! Next: Phase 1.2 (BPE training)")
    print("=" * 60)
