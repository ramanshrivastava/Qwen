# Tokenizer Module

**Phase**: 1.1, 1.2
**Status**: 🚧 Foundation complete, training algorithm pending

---

## Overview

This module implements **Byte-Pair Encoding (BPE)** tokenization, the standard approach for modern language models.

### Why Tokenization?

Neural networks operate on numbers, not text. Tokenization converts text → numbers (token IDs).

**Naive approaches**:
- **Character-level**: 1 char = 1 token → Very long sequences, hard to learn
- **Word-level**: 1 word = 1 token → Huge vocabulary, can't handle new words

**BPE approach** (best of both):
- Start with bytes (256 base tokens)
- Learn subword units based on frequency
- Balance vocabulary size vs sequence length
- Handle any text (no unknown tokens)

---

## Algorithm: Byte-Pair Encoding

### Training (Phase 1.2)

```
Input: Corpus of text, target vocab_size
Output: Merge rules

1. Initialize vocabulary with all 256 bytes
2. Convert corpus to byte sequences
3. Repeat until vocab_size reached:
   a. Count all adjacent byte-pair frequencies
   b. Find most frequent pair (p1, p2)
   c. Create new token for this pair
   d. Merge all occurrences of (p1, p2) → new_token
   e. Add to vocabulary
4. Save merge rules
```

**Example**:
```
Text: "hello hello world"
Bytes: [104, 101, 108, 108, 111, 32, 104, 101, ...]

Iteration 1:
  Most frequent pair: (108, 108) "ll" → appears 2x
  Create token 256 for "ll"
  After merge: [104, 101, 256, 111, 32, 104, 101, 256, 111, ...]

Iteration 2:
  Most frequent pair: (104, 101) "he" → appears 2x
  Create token 257 for "he"
  After merge: [257, 256, 111, 32, 257, 256, 111, ...]

... continue until vocab_size reached
```

### Encoding

```
Input: Text string
Output: List of token IDs

1. Convert text to UTF-8 bytes
2. Apply learned merge rules (in order)
3. Convert to token IDs
```

### Decoding

```
Input: List of token IDs
Output: Text string

1. Convert token IDs to bytes
2. Concatenate byte sequences
3. Decode UTF-8 to string
```

---

## Implementation Details

### Class: `BPETokenizer`

**Attributes**:
- `vocab`: Mapping from byte sequences to token IDs
- `merges`: Learned merge rules (pair → new token)
- `special_tokens`: Special tokens (EOS, PAD, etc.)
- `_cache`: Encoding cache for speed

**Methods**:
- `train(corpus, vocab_size)`: Learn BPE merges
- `encode(text)`: Text → token IDs
- `decode(token_ids)`: Token IDs → text
- `save(path)` / `load(path)`: Serialization

### Special Tokens

Following Qwen/GPT-2 convention:
- `<|endoftext|>` (ID 0): End of document
- `<|pad|>` (ID 1): Padding for batching
- `<|unk|>` (ID 2): Unknown token (rarely used with BPE)

### Byte-Level Encoding

**Why bytes, not characters?**

1. **Universal**: Works for any language (Chinese, Arabic, emoji, etc.)
2. **No unknown tokens**: All UTF-8 text is representable
3. **Fixed base**: 256 possible bytes (vs millions of characters)

**Trade-off**: Longer sequences for non-ASCII text

**Example**:
```python
# English (ASCII): ~1 byte per character
"Hello" → [72, 101, 108, 108, 111]  # 5 bytes

# Chinese (UTF-8): ~3 bytes per character
"你好" → [228, 189, 160, 229, 165, 189]  # 6 bytes (2 chars)
```

---

## Comparison with Qwen

| Aspect | Qwen-7B | Mini-Qwen | Reason |
|--------|---------|-----------|--------|
| Library | tiktoken (Rust) | Pure Python | Learning focus |
| Vocab Size | 151,851 | 10,000 | Simpler, faster training |
| Algorithm | BPE | BPE | Same |
| Byte-level | ✅ | ✅ | Same |
| Special Tokens | 3 main + 205 extra | 3 main | Simplified |
| Multilingual | Optimized | English-focused | Simplified |

**Qwen's Advantages**:
- **Large vocab**: Better compression, fewer tokens per text
- **Multilingual**: Efficient for 100+ languages
- **Fast**: Rust implementation (tiktoken)
- **Numbers**: Special handling (digit-by-digit)

**Our Focus**:
- **Clarity**: Understand the algorithm
- **Simplicity**: Pure Python, readable code
- **Educational**: Comments, examples, tests

---

## Usage Examples

### Training

```python
from src.tokenizer import BPETokenizer

# Create tokenizer
tokenizer = BPETokenizer(vocab_size=10000)

# Prepare corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! This is a test.",
    # ... more documents
]

# Train (learn merge rules)
tokenizer.train(corpus, vocab_size=10000)

# Save for later
tokenizer.save("tokenizer.json")
```

### Encoding

```python
# Encode text to token IDs
text = "Hello, how are you?"
token_ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Tokens: {token_ids}")
print(f"Num tokens: {len(token_ids)}")
```

### Decoding

```python
# Decode token IDs back to text
decoded = tokenizer.decode(token_ids)

assert decoded == text  # Should round-trip perfectly
```

### Inspecting Vocabulary

```python
# Show vocabulary size
print(f"Vocab size: {len(tokenizer)}")

# Show some tokens
for i in range(10):
    token_id = i + 3  # Skip special tokens
    token_bytes = tokenizer.id_to_token[token_id]
    print(f"ID {token_id}: {token_bytes}")
```

---

## Testing

### Unit Tests

```bash
# Run tokenizer tests
pytest tests/test_tokenizer.py -v

# Specific tests
pytest tests/test_tokenizer.py::test_encode_decode_roundtrip
pytest tests/test_tokenizer.py::test_special_tokens
pytest tests/test_tokenizer.py::test_multilingual
```

### Manual Testing

```bash
# Run the module directly
python -m src.tokenizer.bpe

# Should show:
# - Tokenizer initialization
# - Round-trip encoding/decoding
# - Special tokens
# - Byte tokens
```

---

## Performance

### Expected Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Training time | 1-10 min | Depends on corpus size |
| Encoding speed | ~1K tokens/sec | Pure Python |
| Memory | ~10 MB | For 10K vocab |
| Vocab size | 10,000 | vs Qwen's 151,851 |

**Bottlenecks**:
- Training: Pair counting and merging
- Encoding: Merge application (no caching yet)

**Optimizations** (Phase 2+):
- Encoding cache (already implemented)
- Parallel processing for large corpora
- Numba/Cython compilation
- Pre-tokenization (split on whitespace first)

---

## Common Issues

### Issue 1: UTF-8 Decoding Errors

**Symptom**: Replacement character '�' in decoded text

**Cause**: BPE can split multi-byte UTF-8 characters

**Solution**:
```python
# Use error handling
decoded = tokenizer.decode(ids, errors='replace')  # Default
decoded = tokenizer.decode(ids, errors='ignore')   # Skip invalid
```

### Issue 2: Slow Training

**Symptom**: Training takes too long

**Solution**:
- Reduce corpus size (sample documents)
- Reduce vocab_size (fewer merges needed)
- Use smaller documents for testing

### Issue 3: Long Sequences

**Symptom**: Too many tokens for short text

**Cause**: Vocab too small, not enough merges learned

**Solution**:
- Increase vocab_size (more merges = better compression)
- Check training corpus is representative

---

## Files

```
src/tokenizer/
├── __init__.py           # Module exports
├── bpe.py                # BPETokenizer class (Phase 1.1, 1.2)
├── README.md             # This file
└── (future)
    ├── trainer.py        # Training utilities (Phase 1.2)
    └── utils.py          # Helper functions
```

---

## Learning Resources

### Papers
- **BPE**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
  https://arxiv.org/abs/1508.07909

- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
  https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

### Blog Posts
- **Let's build the GPT Tokenizer** (Andrej Karpathy)
  https://www.youtube.com/watch?v=zduSFxRajkE

- **Understanding BPE** (HuggingFace)
  https://huggingface.co/learn/nlp-course/chapter6/5

### Code References
- **Qwen**: `/home/user/Qwen/tokenization_note.md`
- **tiktoken**: https://github.com/openai/tiktoken
- **HuggingFace**: https://github.com/huggingface/tokenizers

---

## Next Steps

### Phase 1.2: Complete Training Algorithm

Implement the merge learning loop:
- [ ] Pair frequency counting
- [ ] Iterative merging
- [ ] Vocabulary building
- [ ] Serialization (save/load)

### Phase 2+: Optimizations

- [ ] Pre-tokenization (split on whitespace)
- [ ] Parallel training
- [ ] Efficient encoding with caching
- [ ] Batch encoding/decoding

---

**Status**: Foundation complete ✅
**Next**: Implement BPE training algorithm (Phase 1.2)
