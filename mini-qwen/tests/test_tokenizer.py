"""
Unit tests for BPE Tokenizer

Tests Phase 1.1 functionality:
- Initialization
- Basic encoding/decoding (byte-level)
- Special tokens
- Round-trip conversion
- Edge cases

Phase 1.2 will add:
- Training algorithm tests
- Merge learning tests
- Serialization tests
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tokenizer import BPETokenizer


class TestTokenizerInitialization:
    """Test tokenizer initialization and setup."""

    def test_init_default(self):
        """Test default initialization."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 10000
        assert len(tokenizer.special_tokens) == 3
        assert len(tokenizer.vocab) == 256  # All bytes

    def test_init_custom_vocab_size(self):
        """Test custom vocabulary size."""
        tokenizer = BPETokenizer(vocab_size=5000)
        assert tokenizer.vocab_size == 5000

    def test_special_tokens_exist(self):
        """Test special tokens are defined."""
        tokenizer = BPETokenizer()

        # Check special tokens
        assert '<|endoftext|>' in tokenizer.special_tokens
        assert '<|pad|>' in tokenizer.special_tokens
        assert '<|unk|>' in tokenizer.special_tokens

        # Check IDs
        assert tokenizer.special_tokens['<|endoftext|>'] == 0
        assert tokenizer.special_tokens['<|pad|>'] == 1
        assert tokenizer.special_tokens['<|unk|>'] == 2

    def test_byte_vocabulary_initialized(self):
        """Test all 256 bytes are in vocabulary."""
        tokenizer = BPETokenizer()

        # Should have 256 byte tokens
        assert len(tokenizer.vocab) == 256

        # Check some specific bytes
        assert bytes([0]) in tokenizer.vocab
        assert bytes([65]) in tokenizer.vocab  # 'A'
        assert bytes([255]) in tokenizer.vocab

    def test_id_to_token_mapping(self):
        """Test reverse mapping is correct."""
        tokenizer = BPETokenizer()

        # Special tokens
        assert tokenizer.id_to_token[0] == '<|endoftext|>'
        assert tokenizer.id_to_token[1] == '<|pad|>'

        # Byte tokens (start at ID 3)
        assert tokenizer.id_to_token[3] == bytes([0])
        assert tokenizer.id_to_token[4] == bytes([1])


class TestBasicEncoding:
    """Test basic encoding functionality (byte-level)."""

    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = BPETokenizer()
        assert tokenizer.encode("") == []

    def test_encode_ascii(self):
        """Test encoding ASCII text."""
        tokenizer = BPETokenizer()

        # "Hi" = [72, 105] in ASCII
        # After special tokens (3) and byte offset: [75, 108]
        ids = tokenizer.encode("Hi")

        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)

    def test_encode_unicode(self):
        """Test encoding Unicode text (UTF-8)."""
        tokenizer = BPETokenizer()

        # Chinese characters are multi-byte in UTF-8
        ids = tokenizer.encode("你好")

        # Should have multiple tokens (3 bytes per Chinese char)
        assert len(ids) == 6  # 2 chars × 3 bytes each

    def test_encode_mixed(self):
        """Test encoding mixed ASCII and Unicode."""
        tokenizer = BPETokenizer()

        ids = tokenizer.encode("Hello 世界")

        # "Hello " (6 ASCII) + "世界" (6 bytes UTF-8) = 12 bytes
        assert len(ids) == 12

    def test_encode_caching(self):
        """Test encoding cache works."""
        tokenizer = BPETokenizer()

        # First encoding
        ids1 = tokenizer.encode("test")

        # Second encoding (should use cache)
        ids2 = tokenizer.encode("test")

        assert ids1 == ids2
        assert "test" in tokenizer._cache


class TestBasicDecoding:
    """Test basic decoding functionality."""

    def test_decode_empty_list(self):
        """Test decoding empty list."""
        tokenizer = BPETokenizer()
        assert tokenizer.decode([]) == ""

    def test_decode_ascii(self):
        """Test decoding ASCII text."""
        tokenizer = BPETokenizer()

        # Encode then decode
        original = "Hello"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)

        assert decoded == original

    def test_decode_unicode(self):
        """Test decoding Unicode text."""
        tokenizer = BPETokenizer()

        original = "你好世界"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)

        assert decoded == original

    def test_decode_mixed(self):
        """Test decoding mixed text."""
        tokenizer = BPETokenizer()

        original = "Hello 世界! 123"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)

        assert decoded == original


class TestRoundTrip:
    """Test encode/decode round-trip conversion."""

    @pytest.mark.parametrize("text", [
        "Hello, world!",
        "The quick brown fox",
        "123 456 789",
        "你好世界",
        "Привет мир",  # Russian
        "مرحبا بالعالم",  # Arabic
        "😀🎉🚀",  # Emoji
        "",  # Empty
        " ",  # Space
        "\n\t",  # Whitespace
    ])
    def test_roundtrip(self, text):
        """Test round-trip encoding/decoding for various texts."""
        tokenizer = BPETokenizer()

        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text, f"Round-trip failed for: {text!r}"

    def test_roundtrip_long_text(self):
        """Test round-trip for longer text."""
        tokenizer = BPETokenizer()

        text = """
        This is a longer piece of text that spans multiple lines.
        It includes various characters, numbers like 123, and even
        some Unicode: 你好世界! Let's see if it round-trips correctly.
        """

        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_special_characters(self):
        """Test encoding special characters."""
        tokenizer = BPETokenizer()

        # Various special characters
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text

    def test_null_byte(self):
        """Test encoding null byte."""
        tokenizer = BPETokenizer()

        text = "before\x00after"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert decoded == text

    def test_very_long_text(self):
        """Test encoding very long text."""
        tokenizer = BPETokenizer()

        # 10,000 character text
        text = "a" * 10000
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        assert len(ids) == 10000  # One byte per 'a'
        assert decoded == text

    def test_repr(self):
        """Test string representation."""
        tokenizer = BPETokenizer(vocab_size=500)
        repr_str = repr(tokenizer)

        assert "BPETokenizer" in repr_str
        assert "vocab_size" in repr_str
        assert "259" in repr_str  # 3 special + 256 bytes

    def test_len(self):
        """Test __len__ returns correct vocab size."""
        tokenizer = BPETokenizer()

        # 3 special tokens + 256 byte tokens
        assert len(tokenizer) == 259


class TestUtilityFunctions:
    """Test utility functions for BPE algorithm."""

    def test_get_pair_frequencies(self):
        """Test pair frequency counting."""
        from tokenizer import get_pair_frequencies

        # Simple sequences
        sequences = [
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
        ]

        freqs = get_pair_frequencies(sequences)

        # (1, 2) appears 3 times
        assert freqs[(1, 2)] == 3

        # (2, 3) appears 2 times
        assert freqs[(2, 3)] == 2

        # (2, 4) appears 1 time
        assert freqs[(2, 4)] == 1

    def test_merge_pair(self):
        """Test merging a byte pair."""
        from tokenizer import merge_pair

        # Merge (1, 2) → 99
        sequence = [1, 2, 3, 1, 2]
        merged = merge_pair(sequence, (1, 2), 99)

        assert merged == [99, 3, 99]

    def test_merge_pair_no_match(self):
        """Test merging when pair doesn't exist."""
        from tokenizer import merge_pair

        sequence = [1, 3, 5, 7]
        merged = merge_pair(sequence, (2, 4), 99)

        # Should remain unchanged
        assert merged == sequence

    def test_merge_pair_empty(self):
        """Test merging empty sequence."""
        from tokenizer import merge_pair

        merged = merge_pair([], (1, 2), 99)
        assert merged == []

    def test_merge_pair_single_element(self):
        """Test merging single-element sequence."""
        from tokenizer import merge_pair

        merged = merge_pair([1], (1, 2), 99)
        assert merged == [1]


# Phase 1.2 tests (TODO)
class TestTraining:
    """Test BPE training algorithm (Phase 1.2)."""

    @pytest.mark.skip(reason="Phase 1.2 not implemented yet")
    def test_train_basic(self):
        """Test basic training."""
        tokenizer = BPETokenizer(vocab_size=300)

        corpus = [
            "hello world",
            "hello there",
            "world peace",
        ]

        tokenizer.train(corpus)

        # Should learn some merges
        assert len(tokenizer.merges) > 0
        assert len(tokenizer) == 300

    @pytest.mark.skip(reason="Phase 1.2 not implemented yet")
    def test_train_improves_compression(self):
        """Test that training improves compression."""
        tokenizer = BPETokenizer(vocab_size=500)

        text = "hello " * 100  # Repetitive text

        # Before training (byte-level)
        ids_before = tokenizer.encode(text)

        # Train
        tokenizer.train([text])

        # After training (with merges)
        ids_after = tokenizer.encode(text)

        # Should use fewer tokens after learning merges
        assert len(ids_after) < len(ids_before)


class TestSerialization:
    """Test saving/loading tokenizer (Phase 1.2)."""

    @pytest.mark.skip(reason="Phase 1.2 not implemented yet")
    def test_save_load(self, tmp_path):
        """Test saving and loading tokenizer."""
        # Create and train tokenizer
        tokenizer1 = BPETokenizer(vocab_size=300)
        tokenizer1.train(["hello world"])

        # Save
        save_path = tmp_path / "tokenizer.json"
        tokenizer1.save(str(save_path))

        # Load
        tokenizer2 = BPETokenizer()
        tokenizer2.load(str(save_path))

        # Should have same vocabulary
        assert len(tokenizer2) == len(tokenizer1)
        assert tokenizer2.merges == tokenizer1.merges

        # Should encode identically
        text = "hello world"
        assert tokenizer2.encode(text) == tokenizer1.encode(text)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
