# ADR-001: Rotary Position Embeddings (RoPE)

**Status**: Accepted
**Date**: 2025-01-16
**Commit**: Phase 1.3, 1.5
**Qwen Reference**: modeling_qwen.py (in HF model checkpoint), uses RoPE with theta=10000
**Related Papers**:
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
**Phase**: Phase 1.3 (basics), Phase 1.5 (integration)

---

## Context

### Problem Statement
Language models need to understand **word order** and **relative positions** in sequences. The original Transformer (2017) used sinusoidal positional encodings, and GPT-2 (2019) used learned positional embeddings. We need to choose a positional encoding method for Mini-Qwen.

### Constraints
- Must work for variable-length sequences
- Should extrapolate beyond training sequence lengths
- Must preserve relative position information
- Should be computationally efficient
- Learning goal: Understand modern position encoding

### Current Situation
In 2023-2024, **RoPE (Rotary Position Embeddings)** has become the de facto standard for decoder-only LLMs:
- Used by: LLaMA, Qwen, GPT-NeoX, Mistral, Llama-3, etc.
- Replaced learned positional embeddings
- Enables context extension via interpolation

---

## Decision

### What We're Implementing

**Rotary Position Embeddings (RoPE)** applied to Query and Key vectors in attention.

### Code Example

```python
# Our Mini-Qwen implementation
import torch
import math

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute complex exponentials for RoPE.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation (10000 in Qwen/LLaMA)

    Returns:
        freqs_cis: Complex tensor of shape (max_seq_len, dim/2)
    """
    # Compute frequencies: 1 / (theta^(2i/dim)) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # Outer product: (max_seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to queries and keys.

    Args:
        xq: Queries (batch, seq_len, n_heads, head_dim)
        xk: Keys (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed frequencies (seq_len, head_dim/2)

    Returns:
        xq_out, xk_out: Rotated queries and keys
    """
    # Reshape to complex numbers: (batch, seq_len, n_heads, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting: (1, seq_len, 1, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**Usage in Attention**:
```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, freqs_cis):
        q, k, v = self.compute_qkv(x)  # (batch, seq_len, n_heads, head_dim)

        # Apply RoPE to queries and keys (NOT values)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Continue with standard attention...
```

---

## Rationale

### Why RoPE?

1. **Relative Position Encoding**: RoPE naturally encodes relative positions between tokens
   - The dot product q_m · k_n depends on (m - n), not absolute positions m, n
   - This is exactly what we want for language understanding!

2. **Infinite Extrapolation**: Works on sequences longer than training length
   - No learned parameters → no overfitting to training length
   - Can extend context via interpolation (NTK-aware scaling)

3. **No Additional Parameters**: Pure mathematical transformation
   - Zero parameters vs learned embeddings (max_seq_len × hidden_dim parameters)
   - Saves memory and parameters

4. **Industry Standard (2023-2024)**: All modern LLMs use RoPE
   - LLaMA, Qwen, Mistral, Llama-3, GPT-NeoX, etc.
   - Battle-tested at scale

5. **Efficient**: Computed once, reused across layers
   - O(max_seq_len × head_dim) precomputation
   - O(seq_len × head_dim) per layer application

### Alternatives Considered

#### Alternative A: Learned Positional Embeddings (GPT-2 style)
**Description**: Add learned embedding for each position
```python
pos_emb = nn.Embedding(max_seq_len, hidden_dim)
x = token_emb + pos_emb
```

**Pros**:
- Simple to implement
- Worked well for GPT-2

**Cons**:
- Limited to max_seq_len (no extrapolation)
- Encodes absolute positions (not relative)
- max_seq_len × hidden_dim parameters
- Doesn't generalize to longer sequences

**Why Rejected**: Cannot extend context length without retraining

#### Alternative B: Sinusoidal Positional Encodings (Original Transformer)
**Description**: Add fixed sin/cos positional encodings
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Pros**:
- No parameters
- Works for any length
- Used in original Transformer

**Cons**:
- Additive (not multiplicative like RoPE)
- Encodes absolute positions primarily
- Less effective than RoPE empirically
- Outdated (2017 approach)

**Why Rejected**: RoPE has superior empirical performance

#### Alternative C: ALiBi (Attention with Linear Biases)
**Description**: Add position-dependent bias to attention scores
```python
scores = scores + alibi_bias(m - n)
```

**Pros**:
- Very simple
- Good extrapolation
- Used by BLOOM

**Cons**:
- Less widely adopted than RoPE
- Requires bias term in attention
- Not compatible with Flash Attention (initially)

**Why Rejected**: RoPE is more standard and well-supported

---

## Qwen Comparison

### What Qwen-7B Does

**File**: `modeling_qwen.py` (in HuggingFace model checkpoint)

**Implementation**:
Qwen uses RoPE with:
- `theta = 10000` (base frequency)
- Applied to all 32 layers
- Supports context extension via **NTK-aware scaling** and **LogN attention scaling**

```python
# Qwen's RoPE (simplified from modeling_qwen.py)
class QWenAttention(nn.Module):
    def __init__(self, config):
        self.rotary_emb = RotaryEmbedding(
            head_dim,
            base=10000,  # theta
            scale_base=config.scale_base  # For NTK scaling
        )

    def forward(self, hidden_states, position_ids):
        q, k, v = self.c_attn(hidden_states).split(...)

        # Apply RoPE
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Continue with attention...
```

**Key Details**:
- Uses real-valued (cos/sin) representation (alternative to complex)
- Supports dynamic position IDs for context extension
- Integrates with LogN scaling for long context

### What We're Doing Differently

| Aspect | Qwen-7B | Mini-Qwen | Reason for Difference |
|--------|---------|-----------|----------------------|
| Implementation | Real-valued (cos/sin) | Complex-valued | Complex math is more elegant for learning |
| Theta | 10000 | 10000 | Same (standard value) |
| NTK Scaling | ✅ Dynamic | ⚠️ Phase 6 | Advanced feature, add later |
| LogN Scaling | ✅ For long context | ⚠️ Phase 6 | Not needed for 512 context |
| Layers | 32 | 8 | Scaled down model |

### Simplifications

1. **Complex-valued implementation**: Easier to understand mathematically
   - Qwen uses cos/sin (real-valued) for optimization
   - We use complex numbers (torch.polar, view_as_complex) for clarity

2. **No dynamic scaling (initially)**: Fixed theta=10000
   - Qwen supports NTK-aware scaling for context extension
   - We'll add this in Phase 6 (Extended Context)

3. **Static position IDs**: Always [0, 1, 2, ..., seq_len-1]
   - Qwen supports custom position IDs for advanced use cases
   - Not needed for basic learning

---

## Historical Evolution

### Timeline of Positional Encoding

| Year | Milestone | Description | Impact |
|------|-----------|-------------|--------|
| 2017 | Original Transformer | Sinusoidal encodings | First solution to position encoding |
| 2018 | GPT-1 | Learned positional embeddings | Trainable positions |
| 2019 | Transformer-XL | Relative positional encodings | Focus on relative positions |
| 2020 | DeBERTa | Disentangled attention | Separate content/position |
| 2021 | **RoFormer** | **Rotary embeddings** | **Game changer: relative + extrapolation** |
| 2021 | ALiBi | Linear biases | Alternative approach (BLOOM) |
| 2021 | GPT-NeoX-20B | RoPE adoption | First major open LLM with RoPE |
| 2023 | **LLaMA** | RoPE standardized | **RoPE becomes default** |
| 2023 | **Qwen-7B** | RoPE + NTK scaling | Extended context support |
| 2024 | Llama-3, Mistral | RoPE everywhere | Industry standard |

### Detailed Evolution

#### Original Transformer (2017)
**Sinusoidal Positional Encodings**:
- Added to input embeddings
- Fixed frequencies based on position
- Encodes absolute positions primarily

**Problem**: Doesn't capture relative positions well

#### GPT-2 (2019)
**Learned Positional Embeddings**:
- Trainable embedding table
- One vector per position (up to max_seq_len)

**Problem**: Can't extrapolate beyond max_seq_len

#### RoFormer (2021) - The Breakthrough
**Rotary Position Embeddings**:
- Apply rotation to Q and K based on position
- Relative position naturally encoded in dot product
- No parameters, infinite extrapolation

**Math**: For positions m and n:
```
q_m^T k_n = q^T R_m^T R_n k = q^T R_{m-n} k
```
Where R is a rotation matrix. The dot product only depends on (m - n)!

#### LLaMA (2023) - Standardization
- Adopted RoPE with theta=10000
- All open models followed suit
- RoPE became the default

#### Qwen (2023) - Extended Context
- RoPE + NTK-aware scaling
- Extends context from 2K to 32K
- LogN attention scaling for stability

#### Our Implementation (2024)
- Clean RoPE implementation for learning
- Complex-valued for mathematical clarity
- Foundation for context extension (Phase 6)

---

## Trade-offs

### Benefits
✅ **Relative Position Encoding**: Naturally captures token relationships
✅ **Infinite Extrapolation**: Works on any sequence length
✅ **Zero Parameters**: No additional model parameters
✅ **Industry Standard**: Compatible with all modern LLMs
✅ **Efficient**: O(seq_len × head_dim) complexity
✅ **Mathematically Elegant**: Based on rotation matrices

### Limitations
❌ **Complex Math**: Harder to understand than learned embeddings
❌ **Fixed Frequencies**: Can't adapt frequencies during training
❌ **Requires Even Dimensions**: head_dim must be divisible by 2
❌ **Not Optimal for Very Long Context**: Needs NTK scaling for >2K tokens

### Performance Implications

| Metric | Impact | Notes |
|--------|--------|-------|
| Speed | Same as learned | O(seq_len × dim) precomputation, amortized |
| Memory | Less than learned | No parameters vs max_seq_len × hidden_dim |
| Quality | Better | Empirically superior in LLMs |
| Extrapolation | Much better | Works beyond training length |

---

## Learning Outcomes

After implementing RoPE, you should understand:

### Conceptual Understanding
1. **Relative vs Absolute Positions**: Why relative positions matter for language
2. **Rotation Matrices**: How 2D rotations encode positions
3. **Complex Number Representation**: Using complex math for rotations
4. **Frequency Decomposition**: Why different dimensions use different frequencies

### Practical Skills
1. **Complex Tensor Operations**: torch.view_as_complex, torch.polar
2. **Broadcasting**: Handling different shaped tensors in PyTorch
3. **Precomputation**: Computing once, reusing across batches
4. **Integration**: Applying RoPE within attention mechanism

### Comparative Analysis
1. **RoPE vs Learned**: When to use each approach
2. **RoPE vs Sinusoidal**: Evolution of positional encoding
3. **RoPE vs ALiBi**: Different modern approaches

---

## Implementation Details

### Prerequisites
- [x] Understand multi-head attention
- [x] Familiar with PyTorch tensor operations
- [x] Read RoFormer paper (at least Section 3)
- [ ] Optional: Complex number math review

### Testing Strategy

**Unit Tests**:
```python
def test_rope_rotations():
    """Verify RoPE applies different rotations to different positions"""
    dim = 64
    seq_len = 128
    freqs_cis = precompute_freqs_cis(dim, seq_len)

    # Create sample Q, K
    q = torch.randn(1, seq_len, 8, dim)
    k = torch.randn(1, seq_len, 8, dim)

    # Apply RoPE
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Verify shapes preserved
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # Verify rotations differ across positions
    assert not torch.allclose(q_rot[:, 0], q_rot[:, 1])

def test_rope_relative_positions():
    """Verify RoPE encodes relative positions"""
    dim = 64
    seq_len = 10
    freqs_cis = precompute_freqs_cis(dim, seq_len)

    q = torch.randn(1, seq_len, 1, dim)
    k = torch.randn(1, seq_len, 1, dim)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Dot product should depend on relative distance
    # q_m · k_n ≈ q_{m+d} · k_{n+d} for small d
    dot_01 = (q_rot[0, 0] * k_rot[0, 1]).sum()
    dot_23 = (q_rot[0, 2] * k_rot[0, 3]).sum()

    # Should be similar (relative distance = 1 in both)
    assert torch.abs(dot_01 - dot_23) < 0.1 * torch.abs(dot_01)
```

**Integration Tests**:
- Test with real attention computation
- Verify gradients flow backwards
- Test with different sequence lengths

### Expected Behavior
- Q and K are rotated by position-dependent angles
- V is NOT rotated (only Q and K)
- Attention scores depend on relative positions
- Works for any sequence length ≤ max_seq_len

### Common Pitfalls

1. **Applying RoPE to Values**
   - **Symptom**: Poor model performance
   - **Fix**: Only apply to queries and keys, NOT values

2. **Dimension Mismatch**
   - **Symptom**: Runtime error "expected even dimension"
   - **Fix**: Ensure head_dim is even (e.g., 64, 128, not 65)

3. **Wrong Frequency Computation**
   - **Symptom**: Model doesn't learn position patterns
   - **Fix**: Verify theta=10000, frequencies decrease with dimension

4. **Forgetting to Detach Freqs**
   - **Symptom**: Memory leak, slow training
   - **Fix**: Precompute freqs_cis, don't recompute every forward

5. **Complex/Real Type Mismatch**
   - **Symptom**: Type errors in multiplication
   - **Fix**: Use .type_as(xq) to match input dtype

---

## References

### Primary Sources
- **Paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- **LLaMA Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)
- **Blog**: [Rotary Embeddings Explained (EleutherAI)](https://blog.eleuther.ai/rotary-embeddings/)

### Qwen Codebase
- **File**: `modeling_qwen.py` (in HuggingFace model checkpoint)
- **Documentation**: `/home/user/Qwen/tech_memo.md` (mentions RoPE)

### Additional Reading
- [Illustrated RoPE](https://nn.labml.ai/transformers/rope/index.html) - Visual explanation
- [RoPE Implementation Analysis](https://github.com/meta-llama/llama/blob/main/llama/model.py#L84) - LLaMA code

### Related ADRs
- ADR-002: Multi-Head Attention
- ADR-003: SwiGLU MLP
- ADR-010: NTK-aware RoPE Scaling (Phase 6)

---

## Exercises

### Exercise 1: Visualize Rotations
**Difficulty**: ⭐⭐☆☆☆
**Estimated Time**: 30 minutes
**Learning Goal**: Understand how RoPE rotates vectors

**Task**: Create a visualization showing how a 2D vector is rotated by RoPE at different positions

```python
import matplotlib.pyplot as plt

def visualize_rope_2d(theta=10000, max_pos=10):
    """Visualize RoPE rotations for first 2 dimensions"""
    # Compute rotation angles
    freq = 1.0 / theta
    angles = [pos * freq for pos in range(max_pos)]

    # Plot unit vectors rotated by each angle
    # TODO: Complete this visualization
```

**Hints**:
- Use `plt.quiver` to plot vectors
- Show positions 0, 1, 2, ..., 9
- Observe how rotation angle increases linearly

### Exercise 2: Implement Real-Valued RoPE
**Difficulty**: ⭐⭐⭐☆☆
**Estimated Time**: 60 minutes
**Learning Goal**: Understand cos/sin representation used by Qwen

**Task**: Implement RoPE using real-valued cos/sin (like Qwen) instead of complex numbers

```python
def apply_rotary_emb_real(xq, xk, cos, sin):
    """
    Apply RoPE using real-valued representation.

    Hint: For 2D rotation by angle θ:
    x' = x * cos(θ) - y * sin(θ)
    y' = x * sin(θ) + y * cos(θ)
    """
    # TODO: Implement using cos/sin
    pass
```

**Hints**:
- Reshape last dimension as pairs: (..., dim) → (..., dim/2, 2)
- Apply rotation formula to each pair
- Verify results match complex implementation

**Solution**: See `src/model/rope_variants.py`

### Exercise 3: Debugging Challenge
**Task**: We've introduced a bug where RoPE is applied to Values (V) in addition to Queries and Keys. Find and fix it.

**Bug Location**: `src/model/attention.py:45`

**Symptom**: Model trains but achieves poor perplexity (~100 vs expected ~30)

**Root Cause**: Applying RoPE to V breaks the attention mechanism. V should remain unrotated, only Q and K are rotated.

**Fix**: Remove RoPE application from V:
```python
# WRONG:
q, k, v = apply_rotary_emb(q, k, v, freqs_cis)

# CORRECT:
q, k = apply_rotary_emb(q, k, freqs_cis)
# v is not rotated!
```

---

## Verification Checklist

After implementing RoPE:

- [ ] Code runs without errors
- [ ] `test_rope_rotations` passes
- [ ] `test_rope_relative_positions` passes
- [ ] Integrated into attention mechanism
- [ ] Model trains with RoPE (loss decreases)
- [ ] Can explain why only Q and K are rotated
- [ ] Completed Exercise 1 (visualization)
- [ ] Attempted Exercise 2 (real-valued implementation)
- [ ] Compared with Qwen's RoPE implementation

---

## Future Enhancements

**Phase 6: Extended Context**
- ADR-010: NTK-aware RoPE scaling
- Dynamic theta adjustment for longer context
- LogN attention scaling integration

**Advanced (Optional)**
- YaRN (Yet another RoPE extensioN)
- Position interpolation techniques
- Efficient long-context attention patterns

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2025-01-16 | Mini-Qwen Team | Initial version |

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Reviewers**: None (first ADR)
