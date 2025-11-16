# Mini-Qwen Learning Guide
**Building a Modern Language Model from Scratch**
Based on Qwen-7B Codebase Analysis

---

## Quick Reference

### What's in Qwen-7B
- **Total Code**: ~7,000 lines Python (deployment toolkit)
- **Model Architecture**: Distributed via HuggingFace (modeling_qwen.py ~2,000 lines)
- **Main Components**: 4 (Tokenizer, Transformer, Training, Inference)
- **Vocabulary Size**: 151,851 tokens
- **Model Layers**: 32 transformer blocks
- **Parameters**: ~7 billion
- **Context Length**: 2,048 (base) → 32,768 (extended)

### What Your Mini Version Should Have
- **Target Size**: 8,000-12,000 lines of Python
- **Core Components**: 5 (Tokenizer, Embedding, Attention, MLP, Training Loop)
- **Model Layers**: 4-8 (configurable)
- **Vocabulary Size**: 5,000-10,000 tokens
- **Hidden Dimension**: 256-512
- **Attention Heads**: 4-8
- **Context Length**: 128-512 tokens
- **Parameters**: ~10-50 million (manageable on CPU/single GPU)

---

## Architecture Overview

```
Text Input
    ↓
TOKENIZER (BPE)              [500-800 lines]
    ↓
Token IDs
    ↓
EMBEDDING LAYER              [200-300 lines]
├─ Token Embeddings
└─ (Position handled by RoPE)
    ↓
TRANSFORMER BLOCKS (4-8x)    [2,000-3,000 lines]
├─ RMSNorm
├─ Multi-Head Attention
│   ├─ Q/K/V Projections
│   ├─ Rotary Position Embedding (RoPE)
│   ├─ Scaled Dot-Product Attention
│   └─ Output Projection
├─ Residual Connection
├─ RMSNorm
├─ MLP (SwiGLU)
│   ├─ Gate Projection
│   ├─ Up Projection
│   ├─ SiLU Activation
│   └─ Down Projection
└─ Residual Connection
    ↓
OUTPUT LAYER                 [200-300 lines]
├─ Final RMSNorm
└─ LM Head (vocab projection)
    ↓
TRAINING ENGINE              [2,000-3,000 lines]
├─ Data Loader
├─ Optimizer (AdamW)
├─ Learning Rate Scheduler
├─ Loss Computation
└─ Gradient Accumulation
    ↓
INFERENCE ENGINE             [1,000-1,500 lines]
├─ Greedy Decoding
├─ Top-k/Top-p Sampling
├─ Temperature Scaling
├─ KV Cache
└─ Batch Generation
```

---

## Component Design Details

### 1. TOKENIZER (500-800 lines)

**Purpose**: Convert text → token IDs using Byte-Pair Encoding (BPE)

**Algorithm**: BPE (Byte-Pair Encoding) on UTF-8 bytes

**Key Components**:

```python
class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}           # token -> id
        self.merges = []          # ordered merge rules
        self.special_tokens = {
            '<|endoftext|>': 0,
            '<|pad|>': 1,
            '<|unk|>': 2,
        }

    def train(self, corpus):
        """Learn BPE merges from corpus"""
        pass

    def encode(self, text):
        """text -> list of token IDs"""
        pass

    def decode(self, ids):
        """list of token IDs -> text"""
        pass
```

**Training Process**:
1. Initialize vocabulary with all bytes (256 tokens)
2. Count all adjacent byte-pair frequencies
3. Merge most frequent pair, add to vocabulary
4. Repeat until vocab_size reached

**Simplified vs Qwen**:
- **Qwen**: Uses tiktoken library (Rust-based, highly optimized)
- **Mini**: Pure Python BPE implementation
- **Qwen vocab**: 151,851 tokens
- **Mini vocab**: 5,000-10,000 tokens

**Reference**: `/home/user/Qwen/tokenization_note.md`

---

### 2. EMBEDDING LAYER (200-300 lines)

**Purpose**: Convert token IDs → dense vectors

```python
class Embeddings:
    def __init__(self, vocab_size, hidden_dim):
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        # Note: No positional embeddings (RoPE handles this)

    def forward(self, input_ids):
        return self.token_embeddings(input_ids)
```

**Key Points**:
- **Untied embeddings**: Input embedding ≠ Output projection (as in Qwen)
- **No learned positional embeddings**: RoPE provides position info
- **Embedding dimension**: Same as hidden_dim (256-512 for mini)

**Reference**: Qwen uses untied embeddings for better performance

---

### 3. ROTARY POSITION EMBEDDING - RoPE (300-500 lines)

**Purpose**: Inject positional information via rotation matrices

**Why RoPE?**
- No learned position embeddings needed
- Naturally extends to longer sequences
- Preserves relative position information
- Used by: LLaMA, GPT-NeoX, PaLM, Qwen, etc.

**Implementation**:

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    Precompute complex exponentials for RoPE
    dim: head dimension (must be even)
    theta: base for frequency computation
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to queries and keys
    xq, xk: (batch, seq_len, n_heads, head_dim)
    freqs_cis: (seq_len, head_dim/2)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(1)  # (seq_len, 1, head_dim/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**Historical Context**:
- **Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **Adoption**: Became standard after LLaMA (2023) showed strong results
- **Qwen choice**: Uses RoPE with theta=10000, same as LLaMA

**Reference**: Qwen technical report mentions RoPE as key architectural choice

---

### 4. RMS NORMALIZATION (100-200 lines)

**Purpose**: Normalize activations for stable training

**Why RMSNorm instead of LayerNorm?**
- Simpler: No mean centering, only scale
- Faster: ~5-10% speedup
- Equally effective for LLMs
- Used by: LLaMA, Qwen, PaLM, Mistral

**Implementation**:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, seq_len, dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return self.weight * x_normed
```

**Comparison**:
- **LayerNorm**: `(x - mean) / std * gamma + beta`
- **RMSNorm**: `x / rms * gamma`

**Historical Context**:
- **Paper**: "Root Mean Square Layer Normalization" (2019)
- **Adoption**: GPT-NeoX, Gopher, Chinchilla, then became standard
- **Qwen**: Uses RMSNorm for all 32 layers

**Reference**: tech_memo.md mentions RMSNorm as architectural choice

---

### 5. MULTI-HEAD ATTENTION (600-1,000 lines)

**Purpose**: Allow model to attend to different positions

**Architecture**:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # QKV projections (with bias, as in Qwen)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output projection (no bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            self.head_dim, max_seq_len
        )

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # QKV projections
        q = self.q_proj(x)  # (batch, seq_len, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:seq_len])

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)
```

**Key Details**:
- **Qwen peculiarity**: Biases on QKV projections only
- **Scaling**: `1/sqrt(head_dim)` prevents softmax saturation
- **Masking**: Causal mask for autoregressive generation

**Optional Enhancement**: Flash Attention (Phase 2+)

**Reference**: Qwen uses flash attention for training acceleration

---

### 6. SWIGLU MLP (400-600 lines)

**Purpose**: Non-linear transformation with gated activation

**Why SwiGLU?**
- Better than ReLU/GELU for language models
- Gating mechanism provides more expressiveness
- Used by: LLaMA, PaLM, Qwen, Mistral

**Implementation**:

```python
class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Intermediate dimension: typically 4x hidden (Qwen uses 11008 for hidden=4096)
        self.intermediate_dim = int(hidden_dim * 8 / 3)  # ~2.67x for SwiGLU

        # Gate and Up projections
        self.gate_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=False)

        # Down projection
        self.down_proj = nn.Linear(self.intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

**Comparison**:
- **Standard FFN**: `W2(ReLU(W1(x)))`
- **GELU FFN**: `W2(GELU(W1(x)))`
- **GLU**: `(W1(x) * σ(W2(x))) @ W3`
- **SwiGLU**: `(swish(W_gate(x)) * W_up(x)) @ W_down`

**Historical Context**:
- **Paper**: "GLU Variants Improve Transformer" (Shazeer, 2020)
- **Adoption**: PaLM (2022), LLaMA (2023), became standard
- **Cost**: Requires ~50% more parameters than standard FFN

**Reference**: Qwen uses SwiGLU with intermediate_dim = 11008 (hidden=4096)

---

### 7. TRANSFORMER BLOCK (300-500 lines)

**Purpose**: Complete transformer layer with pre-norm architecture

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len):
        super().__init__()

        # Pre-normalization (as in Qwen)
        self.attn_norm = RMSNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, n_heads, max_seq_len)

        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp = SwiGLU_MLP(hidden_dim)

    def forward(self, x, mask=None):
        # Pre-norm + Attention + Residual
        x = x + self.attention(self.attn_norm(x), mask)

        # Pre-norm + MLP + Residual
        x = x + self.mlp(self.mlp_norm(x))

        return x
```

**Pre-Norm vs Post-Norm**:
- **Post-Norm** (Original Transformer): `x + F(LayerNorm(x))`
- **Pre-Norm** (GPT-2, LLaMA, Qwen): `x + F(x)` with LayerNorm before F
- **Advantage**: More stable training, especially for deep models

**Reference**: Qwen uses pre-norm architecture across all 32 layers

---

### 8. COMPLETE LANGUAGE MODEL (500-800 lines)

```python
class MiniQwen(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Embeddings
        self.embeddings = Embeddings(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_dim,
                config.n_heads,
                config.max_seq_len
            ) for _ in range(config.n_layers)
        ])

        # Output
        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Untied embeddings (don't share weights)
        # self.lm_head.weight = self.embeddings.token_embeddings.weight  # TIED
        # Leave separate for UNTIED (as in Qwen)

    def forward(self, input_ids):
        # Embed tokens
        x = self.embeddings(input_ids)

        # Create causal mask
        seq_len = input_ids.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm + projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits
```

**Configuration**:

```python
@dataclass
class MiniQwenConfig:
    vocab_size: int = 10000
    hidden_dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 512
    intermediate_dim: int = None  # Auto: ~2.67x hidden_dim

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = int(self.hidden_dim * 8 / 3)
```

---

### 9. TRAINING ENGINE (2,000-3,000 lines)

**Purpose**: Train the model on text data

**Key Components**:

```python
class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )

    def train_step(self, batch):
        input_ids, labels = batch

        # Forward pass
        logits = self.model(input_ids)

        # Compute loss (cross-entropy)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()
```

**Training Details (from Qwen tech_memo.md)**:
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, ε=10⁻⁶)
- **Learning Rate**: 3×10⁻⁴ (peak), warmup 2000 steps, cosine decay
- **Batch Size**: 2048 sequences (Qwen), 32-128 for mini
- **Sequence Length**: 2048 (Qwen), 128-512 for mini
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0
- **Mixed Precision**: bfloat16 (Qwen), optional for mini

**Data Preparation**:

```python
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize and chunk
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            # Chunk into sequences
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1])
        labels = torch.tensor(tokens[1:])  # Next token prediction
        return input_ids, labels
```

**Reference**: `/home/user/Qwen/finetune.py` - training loop implementation

---

### 10. INFERENCE ENGINE (1,000-1,500 lines)

**Purpose**: Generate text from the trained model

**Greedy Decoding**:

```python
def generate_greedy(model, tokenizer, prompt, max_length=100):
    model.eval()
    tokens = tokenizer.encode(prompt)

    for _ in range(max_length):
        # Forward pass
        input_ids = torch.tensor([tokens])
        logits = model(input_ids)

        # Get next token (greedy)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()

        # Stop if EOS
        if next_token == tokenizer.eos_token_id:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)
```

**Top-k / Top-p Sampling**:

```python
def sample_top_p(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling
    """
    # Apply temperature
    logits = logits / temperature

    # Sort probabilities
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability > p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set removed indices to 0 probability
    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample
    next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
    return next_token
```

**KV Cache Optimization** (Phase 2+):

```python
class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, hidden_dim, n_heads, max_seq_len):
        super().__init__()
        # ... same as before ...

        # KV cache
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        q, k, v = self.compute_qkv(x)

        if use_cache and self.k_cache is not None:
            # Concatenate with cached keys/values
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)

        # Update cache
        if use_cache:
            self.k_cache = k
            self.v_cache = v

        # Compute attention as before...
```

**Reference**: `/home/user/Qwen/cli_demo.py` - streaming generation

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2, ~2,500 lines)

**Goal**: Build basic tokenizer and model architecture

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 1.1 | Project setup + BPE tokenizer foundation | 300 | BPE algorithm, byte-level encoding |
| 1.2 | Complete BPE training and encoding | 400 | Merge rules, vocabulary building |
| 1.3 | RMSNorm + RoPE implementation | 300 | Modern normalization, rotary embeddings |
| 1.4 | Multi-head attention (no RoPE yet) | 500 | Attention mechanism, Q/K/V projections |
| 1.5 | Integrate RoPE into attention | 200 | Complex number rotations |
| 1.6 | SwiGLU MLP implementation | 300 | Gated activations, GLU variants |
| 1.7 | TransformerBlock assembly | 200 | Pre-norm architecture, residual connections |
| 1.8 | Complete MiniQwen model | 300 | End-to-end forward pass |

**Checkpoint 1**: Forward pass on random input, verify shapes

---

### Phase 2: Training Pipeline (Week 2-3, ~2,500 lines)

**Goal**: Train the model on small text corpus

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 2.1 | Text dataset preparation | 400 | Chunking, next-token prediction |
| 2.2 | AdamW optimizer setup | 300 | Adam variants, weight decay |
| 2.3 | Learning rate scheduling | 300 | Warmup, cosine decay |
| 2.4 | Training loop with loss computation | 500 | Cross-entropy, gradient flow |
| 2.5 | Gradient accumulation + clipping | 300 | Stability techniques |
| 2.6 | Logging and checkpointing | 400 | Experiment tracking |
| 2.7 | Mixed precision training (optional) | 300 | FP16/BF16, speed vs stability |

**Checkpoint 2**: Train on small dataset (Shakespeare, Wikipedia), monitor loss

---

### Phase 3: Inference & Generation (Week 3-4, ~1,500 lines)

**Goal**: Generate text with various decoding strategies

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 3.1 | Greedy decoding | 200 | Argmax generation |
| 3.2 | Temperature sampling | 200 | Controlling randomness |
| 3.3 | Top-k sampling | 300 | Truncated distributions |
| 3.4 | Top-p (nucleus) sampling | 300 | Dynamic truncation |
| 3.5 | KV cache for fast generation | 500 | Inference optimization |

**Checkpoint 3**: Generate coherent text, compare decoding strategies

---

### Phase 4: Advanced Features (Week 4-5, ~2,000 lines)

**Goal**: Add production features

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 4.1 | Batch generation | 400 | Padding, attention masks |
| 4.2 | Beam search | 500 | Search algorithms |
| 4.3 | Flash Attention integration | 400 | Memory-efficient attention |
| 4.4 | Gradient checkpointing | 300 | Memory-compute tradeoff |
| 4.5 | LoRA fine-tuning | 400 | Parameter-efficient training |

**Checkpoint 4**: Fine-tune on custom dataset, measure perplexity

---

### Phase 5: Evaluation & Optimization (Week 5-6, ~1,500 lines)

**Goal**: Benchmark and optimize

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 5.1 | Perplexity evaluation | 300 | Language model metrics |
| 5.2 | Few-shot prompting | 400 | In-context learning |
| 5.3 | Model quantization (8-bit) | 400 | Post-training quantization |
| 5.4 | Performance profiling | 400 | Bottleneck identification |

**Checkpoint 5**: Compare with Qwen-7B on benchmarks (scaled appropriately)

---

### Phase 6: Extended Context (Optional, ~1,000 lines)

**Goal**: Support longer sequences

| Commit | Feature | Lines | Learning Focus |
|--------|---------|-------|----------------|
| 6.1 | Dynamic NTK scaling | 300 | RoPE interpolation |
| 6.2 | LogN attention scaling | 300 | Context length extrapolation |
| 6.3 | Efficient long-context attention | 400 | Sparse attention patterns |

---

## Key Design Decisions for Mini Version

### 1. **Model Scale**

| Aspect | Qwen-7B | Mini-Qwen | Reasoning |
|--------|---------|-----------|-----------|
| Layers | 32 | 8 | Faster training, easier debugging |
| Hidden Dim | 4,096 | 512 | ~64x fewer parameters |
| Heads | 32 | 8 | Maintains head_dim = 64 |
| Vocab Size | 151,851 | 10,000 | Simpler tokenizer, faster softmax |
| Context | 2,048 | 512 | Reduced memory footprint |
| Parameters | 7B | ~50M | CPU/single GPU trainable |

### 2. **Tokenization**

| Aspect | Qwen | Mini-Qwen |
|--------|------|-----------|
| Library | tiktoken (Rust) | Pure Python BPE |
| Vocab Size | 151,851 | 10,000 |
| Special Tokens | ChatML format | Simple EOS/PAD |
| Multilingual | 100+ languages | English-focused |

**Reasoning**: Pure Python for transparency and learning

### 3. **Architecture Choices**

| Component | Qwen | Mini-Qwen | Keep or Simplify? |
|-----------|------|-----------|-------------------|
| RoPE | ✅ | ✅ | **KEEP** - Core to modern LLMs |
| RMSNorm | ✅ | ✅ | **KEEP** - Simple to implement |
| SwiGLU | ✅ | ✅ | **KEEP** - Industry standard |
| Flash Attention | ✅ | ⚠️ Optional | **PHASE 2** - Optimize later |
| Untied Embeddings | ✅ | ✅ | **KEEP** - Better performance |
| KV Cache Quant | ✅ 8-bit | ⚠️ Optional | **PHASE 2** - Advanced |
| Bias in QKV | ✅ | ✅ | **KEEP** - Qwen peculiarity |

### 4. **Training**

| Aspect | Qwen | Mini-Qwen |
|--------|------|-----------|
| Dataset | 2.2T tokens | 10M-100M tokens |
| Batch Size | 2048 seqs | 32-128 seqs |
| Precision | BF16 | FP32 → BF16 (optional) |
| Distributed | Multi-node | Single GPU/CPU |
| Duration | Weeks | Hours-days |

### 5. **Optimization**

**Phase 1 Focus**: Correctness
**Phase 2 Focus**: Speed (Flash Attention, KV cache)
**Phase 3 Focus**: Memory (Gradient checkpointing, quantization)

---

## File Organization

```
mini-qwen/
├── README.md                          # Project overview
├── LEARNING_GUIDE.md                  # This document
├── HISTORICAL_TIMELINE.md             # LLM evolution timeline
│
├── docs/
│   ├── adrs/                          # Architecture Decision Records
│   │   ├── 001-why-rope.md
│   │   ├── 002-swiglu-vs-relu.md
│   │   ├── 003-rmsnorm-vs-layernorm.md
│   │   ├── 004-untied-embeddings.md
│   │   ├── 005-bpe-tokenization.md
│   │   └── ...
│   │
│   ├── comparisons/                   # Mini vs Real Qwen
│   │   ├── architecture-comparison.md
│   │   ├── performance-comparison.md
│   │   └── tokenizer-comparison.md
│   │
│   ├── diagrams/                      # Visual aids
│   │   ├── transformer-block.svg
│   │   ├── attention-mechanism.svg
│   │   ├── rope-visualization.svg
│   │   └── training-pipeline.svg
│   │
│   ├── checkpoints/                   # Learning milestones
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   └── ...
│   │
│   └── references/                    # External resources
│       ├── papers.md                  # Key papers
│       ├── qwen-references.md         # Qwen codebase links
│       └── blog-posts.md              # Learning resources
│
├── src/
│   ├── tokenizer/                     # Phase 1
│   │   ├── __init__.py
│   │   ├── bpe.py                     # BPE implementation
│   │   ├── trainer.py                 # Vocabulary training
│   │   └── README.md
│   │
│   ├── model/                         # Phase 1-2
│   │   ├── __init__.py
│   │   ├── config.py                  # Model configuration
│   │   ├── embeddings.py              # Token embeddings
│   │   ├── rope.py                    # Rotary embeddings
│   │   ├── normalization.py           # RMSNorm
│   │   ├── attention.py               # Multi-head attention
│   │   ├── mlp.py                     # SwiGLU MLP
│   │   ├── transformer.py             # Transformer block
│   │   ├── model.py                   # Complete model
│   │   └── README.md
│   │
│   ├── training/                      # Phase 2
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Data loading
│   │   ├── trainer.py                 # Training loop
│   │   ├── optimizer.py               # AdamW setup
│   │   ├── scheduler.py               # LR scheduling
│   │   └── README.md
│   │
│   ├── inference/                     # Phase 3
│   │   ├── __init__.py
│   │   ├── generator.py               # Text generation
│   │   ├── sampling.py                # Sampling strategies
│   │   ├── kv_cache.py                # KV cache
│   │   └── README.md
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── checkpoint.py
│       └── metrics.py
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_generation.py
│
├── examples/
│   ├── 01_train_tokenizer.py
│   ├── 02_train_model.py
│   ├── 03_generate_text.py
│   ├── 04_finetune_lora.py
│   └── datasets/
│       ├── shakespeare.txt
│       ├── wikipedia_sample.txt
│       └── code_sample.txt
│
├── notebooks/
│   ├── 01_tokenization_demo.ipynb
│   ├── 02_attention_visualization.ipynb
│   ├── 03_training_walkthrough.ipynb
│   └── 04_generation_comparison.ipynb
│
├── tools/
│   ├── visualize_attention.py
│   ├── compare_models.py
│   ├── profile_performance.py
│   └── export_onnx.py
│
├── configs/
│   ├── mini_config.yaml              # 50M params
│   ├── tiny_config.yaml              # 10M params
│   └── medium_config.yaml            # 100M params
│
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Historical Timeline: Modern LLM Evolution

This project maps to the evolution of transformer-based language models:

### Era 1: Transformer Foundation (2017-2019)

| Year | Milestone | Relevance to Mini-Qwen |
|------|-----------|-------------------------|
| 2017 | "Attention Is All You Need" (Vaswani et al.) | **Core architecture**: Multi-head attention, feed-forward networks |
| 2018 | GPT-1 (OpenAI) | **Decoder-only** architecture for language modeling |
| 2019 | GPT-2 (OpenAI) | Scaling laws, **pre-norm** architecture |
| 2019 | RMSNorm paper (Zhang & Sennrich) | Simpler normalization we use |

**Commits mapping to this era**: 1.3 (RMSNorm), 1.4 (Attention), 1.7 (Transformer Block)

---

### Era 2: Efficiency Innovations (2020-2021)

| Year | Milestone | Relevance |
|------|-----------|-----------|
| 2020 | "GLU Variants Improve Transformer" (Shazeer) | **SwiGLU** activation |
| 2021 | RoFormer (Su et al.) | **RoPE** positional encoding |
| 2021 | Switch Transformer (Google) | Sparse models, scaling |
| 2021 | GPT-NeoX (EleutherAI) | Open-source architecture with RoPE |

**Commits mapping to this era**: 1.5 (RoPE), 1.6 (SwiGLU)

---

### Era 3: Open LLM Renaissance (2022-2023)

| Year | Milestone | Relevance |
|------|-----------|-----------|
| 2022 | PaLM (Google) | SwiGLU + RMSNorm combination |
| 2022 | Flash Attention (Dao et al.) | Memory-efficient attention |
| 2023 | LLaMA (Meta) | **Direct inspiration**: RoPE + RMSNorm + SwiGLU |
| 2023 | Qwen-7B (Alibaba) | **Our reference model** |
| 2023 | Mistral-7B | Further optimizations |

**Commits mapping to this era**: 1.8 (Complete model), 4.3 (Flash Attention)

---

### Era 4: Long Context & Efficiency (2023-2024)

| Year | Milestone | Relevance |
|------|-----------|-----------|
| 2023 | NTK-aware RoPE | Context extension techniques |
| 2023 | LoRA (Hu et al.) | Parameter-efficient fine-tuning |
| 2024 | Qwen2, Llama 3 | Multi-lingual, extended context |

**Commits mapping to this era**: 4.5 (LoRA), 6.1-6.3 (Extended context)

---

## ADR Template

Each commit includes an Architecture Decision Record:

```markdown
# ADR-XXX: [Decision Title]

**Status**: Accepted
**Date**: 2025-01-XX
**Commit**: [hash]
**Qwen Reference**: [file/line or description]
**Related Papers**: [citations]

---

## Context

What problem are we solving?
What are the constraints?

---

## Decision

What approach did we choose?

### Code Example

```python
# Our Mini-Qwen implementation
[code snippet]
```

---

## Rationale

### Why This Approach?

1. Reason 1
2. Reason 2

### Alternatives Considered

- **Alternative A**: [why rejected]
- **Alternative B**: [why rejected]

---

## Qwen Comparison

### What Qwen Does

[Explanation + code references from /home/user/Qwen/]

### What We're Doing Differently

[Simplifications + rationale]

---

## Historical Evolution

- **Original Transformer (2017)**: [original approach]
- **GPT-2 (2019)**: [evolution]
- **LLaMA (2023)**: [modern approach]
- **Qwen (2023)**: [current state]

---

## Trade-offs

### Benefits
✅ Benefit 1
✅ Benefit 2

### Limitations
❌ Limitation 1
❌ Limitation 2

---

## Learning Outcomes

After this commit, you should understand:

1. [Concept 1]
2. [Concept 2]
3. [Concept 3]

---

## References

- **Qwen**: `/home/user/Qwen/[file]`
- **Paper**: [citation with link]
- **Blog**: [optional]

---

## Exercises

1. **Exercise 1**: [hands-on task]
2. **Extension**: [challenge]
3. **Debugging**: [intentional bug to find]
```

---

## Learning Checkpoint Template

```markdown
# Phase X Checkpoint: [Phase Name]

## Self-Assessment Quiz

### Conceptual Understanding

1. **Q**: Why did we choose RoPE over learned positional embeddings?
   **A**: [expected answer]
   **Reference**: ADR-001

2. **Q**: What is the advantage of SwiGLU over ReLU?
   **A**: [expected answer]
   **Reference**: ADR-002

### Code Comprehension

1. **Q**: What does this code do?
   ```python
   xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
   xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
   ```
   **A**: Applies rotary embeddings using complex number multiplication

---

## Hands-On Exercises

### Exercise 1: Modify Attention Heads
**Task**: Change number of attention heads and observe impact
**Difficulty**: ⭐⭐☆☆☆
**Estimated Time**: 30 minutes
**Learning Goal**: Understand multi-head attention mechanics

**Hints**:
- Consider what happens when `hidden_dim % n_heads != 0`
- Try 4, 8, 16 heads
- Measure parameters and speed

### Exercise 2: Implement Flash Attention
**Task**: Replace standard attention with flash attention
**Difficulty**: ⭐⭐⭐⭐☆
**Estimated Time**: 2 hours
**Learning Goal**: Memory-efficient attention computation

---

## Comparative Analysis

### Mini-Qwen vs Qwen-7B

| Aspect | Mini-Qwen | Qwen-7B | Why Different? |
|--------|-----------|---------|----------------|
| Layers | 8 | 32 | Faster training |
| Params | 50M | 7B | ~140x smaller |
| Vocab | 10K | 152K | Simpler tokenizer |
| Context | 512 | 2048 | Memory constraints |

---

## Performance Benchmark

Run `python tools/benchmark.py` and compare:

**Expected Results**:
- Training: ~X tokens/sec
- Inference: ~Y tokens/sec
- Memory: ~Z GB

**Bottleneck**: [attention vs MLP vs embeddings]

**Qwen's Solution**: [flash attention, quantization, etc.]

---

## Next Steps

Before moving to Phase X+1, ensure you can:

- [ ] Explain [concept] to someone else
- [ ] Modify the code to add [feature]
- [ ] Identify the performance bottleneck
- [ ] Read equivalent sections of Qwen codebase
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_rope.py
def test_rope_rotations():
    """Verify RoPE applies correct rotations"""
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
```

### Integration Tests

```python
# tests/test_model.py
def test_forward_pass():
    """Verify end-to-end forward pass"""
    config = MiniQwenConfig(
        vocab_size=1000,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=128
    )

    model = MiniQwen(config)

    # Random input
    input_ids = torch.randint(0, 1000, (2, 64))  # batch=2, seq_len=64

    # Forward pass
    logits = model(input_ids)

    # Verify output shape
    assert logits.shape == (2, 64, 1000)  # (batch, seq_len, vocab_size)
```

### Training Tests

```python
# tests/test_training.py
def test_overfitting_single_batch():
    """Model should overfit a single batch (sanity check)"""
    # ... setup model and data ...

    initial_loss = compute_loss()

    # Train for 100 steps
    for _ in range(100):
        train_step()

    final_loss = compute_loss()

    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.1
```

---

## Performance Expectations

### Training Speed

| Configuration | Tokens/sec (CPU) | Tokens/sec (GPU) |
|---------------|------------------|------------------|
| Tiny (10M params) | 100-500 | 5K-10K |
| Mini (50M params) | 50-200 | 2K-5K |
| Medium (100M params) | 20-100 | 1K-3K |

### Memory Usage

| Configuration | Training (GB) | Inference (GB) |
|---------------|---------------|----------------|
| Tiny | 2-4 | 0.5-1 |
| Mini | 8-12 | 1-2 |
| Medium | 16-24 | 2-4 |

### Generation Quality

After training on 100M tokens:
- **Perplexity**: 30-50 (validation set)
- **Coherence**: 2-3 sentence continuity
- **Comparison**: Much worse than Qwen-7B (expected!)

**Focus**: Understanding > Performance

---

## Common Pitfalls to Avoid

### 1. ❌ Incorrect RoPE Implementation
**Symptom**: Model doesn't learn positional patterns
**Cause**: Complex number operations wrong
**Solution**: Test on synthetic position-dependent task

### 2. ❌ Gradient Explosion
**Symptom**: Loss becomes NaN
**Cause**: Missing gradient clipping or wrong learning rate
**Solution**: Clip at 1.0, reduce LR

### 3. ❌ Causal Mask Errors
**Symptom**: Model "cheats" by seeing future tokens
**Cause**: Incorrect mask in attention
**Solution**: Verify mask is lower-triangular

### 4. ❌ Tokenizer Bugs
**Symptom**: Strange generation outputs
**Cause**: encode/decode mismatch
**Solution**: Test `decode(encode(text)) == text`

### 5. ❌ KV Cache Staleness
**Symptom**: Repetitive generation
**Cause**: Not clearing cache between generations
**Solution**: Reset cache at start of each generation

---

## Tools for Learning

### 1. Attention Visualizer

```bash
python tools/visualize_attention.py \
  --checkpoint checkpoints/step_1000.pt \
  --text "The quick brown fox"
```

Outputs heatmap of attention weights across heads and layers.

### 2. Model Comparator

```bash
python tools/compare_models.py \
  --model1 mini-qwen \
  --model2 qwen-7b \
  --prompt "Once upon a time"
```

Side-by-side generation comparison.

### 3. Performance Profiler

```bash
python tools/profile_performance.py \
  --config configs/mini_config.yaml
```

Identifies bottlenecks (attention vs MLP vs embeddings).

---

## References

### Qwen Codebase
- **README**: `/home/user/Qwen/README.md`
- **Technical Memo**: `/home/user/Qwen/tech_memo.md`
- **Tokenization**: `/home/user/Qwen/tokenization_note.md`
- **Training**: `/home/user/Qwen/finetune.py`
- **Inference**: `/home/user/Qwen/cli_demo.py`

### Papers (Chronological)
1. **Attention Is All You Need** (Vaswani et al., 2017)
   https://arxiv.org/abs/1706.03762

2. **Language Models are Unsupervised Multitask Learners** (GPT-2, 2019)
   https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)
   https://arxiv.org/abs/1910.07467

4. **GLU Variants Improve Transformer** (Shazeer, 2020)
   https://arxiv.org/abs/2002.05202

5. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
   https://arxiv.org/abs/2104.09864

6. **Flash Attention** (Dao et al., 2022)
   https://arxiv.org/abs/2205.14135

7. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
   https://arxiv.org/abs/2302.13971

8. **Qwen Technical Report** (Bai et al., 2023)
   https://arxiv.org/abs/2309.16609

### Blog Posts & Tutorials
- **The Illustrated Transformer** (Jay Alammar)
  https://jalammar.github.io/illustrated-transformer/

- **RoPE Explained** (EleutherAI)
  https://blog.eleuther.ai/rotary-embeddings/

- **LLaMA Paper Analysis** (Nathan Lambert)
  https://www.interconnects.ai/p/llama-analysis

---

## Commit Message Format

```
[Phase X.Y] Title - Historical Context

Brief description of what this commit implements.

Qwen Reference: [file/description]
Paper Reference: [citation]
Historical Note: [how this relates to LLM evolution]

Design Decisions:
- Decision 1: [rationale]
- Decision 2: [rationale]

Trade-offs:
- Simplified [X] because [Y]
- Kept [A] to preserve [B]

Learning Outcomes:
1. Understand [concept]
2. See how [feature] works
3. Compare [approach A] vs [approach B]

Testing:
- Unit tests: tests/test_[component].py
- Integration: tests/test_integration.py

See docs/adrs/ADR-XXX.md for full decision record.
```

---

## Success Criteria

### Phase 1: Foundation ✅
- [ ] BPE tokenizer trains on corpus
- [ ] Model forward pass completes
- [ ] Shapes verified at each layer
- [ ] Gradient flows backwards

### Phase 2: Training ✅
- [ ] Loss decreases on training set
- [ ] Model overfits single batch
- [ ] Perplexity < 100 on validation
- [ ] Checkpoints saved/loaded correctly

### Phase 3: Inference ✅
- [ ] Generates coherent 2-3 sentences
- [ ] Top-p sampling works
- [ ] KV cache speeds up generation
- [ ] No repetition loops

### Phase 4: Advanced ✅
- [ ] LoRA fine-tuning works
- [ ] Flash attention integrated
- [ ] Batch generation efficient

### Phase 5: Evaluation ✅
- [ ] Perplexity measured on benchmark
- [ ] Compared with baseline (GPT-2 small)
- [ ] Profiling identifies bottlenecks

---

## Next Steps After Completion

1. **Scale Up**: Train 100M-500M param model on larger corpus
2. **Multi-Modal**: Add vision encoder (like Qwen-VL)
3. **Alignment**: Implement RLHF/DPO for chat
4. **Quantization**: 4-bit/8-bit inference
5. **Serving**: Deploy with vLLM or TensorRT

---

## Final Thoughts

Building a language model from scratch is the **best way to understand modern AI**. This project prioritizes:

1. **Understanding over Performance**
2. **Reasoning over Rote Implementation**
3. **Historical Context over Isolated Code**
4. **Iterative Learning over Big Bang Rewrites**

Each commit is a lesson. Each ADR is a deep dive. Each checkpoint is a milestone.

**Welcome to the journey of building Mini-Qwen!** 🚀

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Maintainer**: Mini-Qwen Learning Project
