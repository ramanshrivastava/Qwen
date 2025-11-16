# Historical Timeline: Mini-Qwen Development ↔ LLM Evolution

This document maps each phase and commit of Mini-Qwen to the historical evolution of language models and transformers.

---

## Timeline Overview

```
1990s: RNNs & LSTMs
   ↓
2017: Transformer Revolution
   ↓
2018-2019: GPT Era Begins
   ↓
2020-2021: Efficiency Innovations
   ↓
2022-2023: Open LLM Renaissance
   ↓
2023-2024: Long Context & Scaling
   ↓
2024-Present: Mini-Qwen Learning Project
```

---

## Detailed Timeline

### Pre-Transformer Era (1990s-2016)

| Year | Milestone | Architecture | Impact |
|------|-----------|--------------|--------|
| 1997 | LSTM (Hochreiter & Schmidhuber) | Recurrent | Solved vanishing gradients |
| 2013 | Word2Vec (Mikolov et al.) | Embeddings | Distributed word representations |
| 2014 | Seq2Seq (Sutskever et al.) | LSTM encoder-decoder | Neural machine translation |
| 2015 | Attention Mechanism (Bahdanau et al.) | Attention over RNN | Alignment in translation |
| 2016 | Neural Machine Translation (Google) | LSTM + Attention | Production deployment |

**Mini-Qwen Connection**: We skip RNNs entirely, going straight to transformers (2017+)

---

### Era 1: The Transformer Revolution (2017)

| Date | Milestone | Key Innovation | Mini-Qwen Phase |
|------|-----------|----------------|-----------------|
| Jun 2017 | **"Attention Is All You Need"** (Vaswani et al.) | • Multi-head attention<br>• Position encodings<br>• Feed-forward networks<br>• Encoder-decoder architecture | **Phase 1.4**: Multi-head attention<br>**Phase 1.7**: Transformer block |

**Key Innovations**:
- Parallelizable (vs sequential RNNs)
- Scaled dot-product attention
- Positional encodings (sinusoidal)
- Layer normalization (post-norm)

**Original Architecture**:
```
Input → Embedding + Positional Encoding
      → Encoder Blocks (6x)
          ├─ Multi-Head Self-Attention
          ├─ Add & LayerNorm
          ├─ Feed-Forward Network (ReLU)
          └─ Add & LayerNorm
      → Decoder Blocks (6x)
      → Output
```

**Mini-Qwen Evolution**: We use decoder-only (like GPT), not encoder-decoder

---

### Era 2: GPT - Decoder-Only Language Models (2018-2019)

| Date | Milestone | Key Innovation | Mini-Qwen Phase |
|------|-----------|----------------|-----------------|
| Jun 2018 | **GPT-1** (OpenAI, Radford et al.) | • Decoder-only architecture<br>• Pre-training + fine-tuning<br>• 117M parameters | **Phase 1.8**: Complete model<br>**Phase 2**: Training |
| Feb 2019 | **GPT-2** (OpenAI) | • Scaled to 1.5B params<br>• Pre-norm instead of post-norm<br>• Zero-shot learning | **Phase 1.7**: Pre-norm architecture |
| Oct 2019 | **T5** (Google) | • Text-to-text framework<br>• Relative position bias | N/A (we use RoPE instead) |
| Oct 2019 | **ALBERT** (Google) | • Parameter sharing<br>• Factorized embeddings | N/A (not our focus) |

**GPT-2 Architecture Evolution**:
```
Original Transformer:      GPT-2 (Pre-norm):
x → Attention              x → LayerNorm → Attention
  → Add & Norm                → Add (with x)
  → FFN                       → LayerNorm → FFN
  → Add & Norm                → Add (with x)
```

**Why Pre-norm?**
- More stable training for deep models (100+ layers)
- Allows higher learning rates
- Became standard for LLMs

**Mini-Qwen Adoption**: ✅ We use pre-norm (Phase 1.7)

---

### Era 3: Efficiency Innovations (2019-2021)

| Date | Milestone | Key Innovation | Mini-Qwen Phase |
|------|-----------|----------------|-----------------|
| Oct 2019 | **RMSNorm** (Zhang & Sennrich) | • Simpler than LayerNorm<br>• No mean centering<br>• 5-10% faster | **Phase 1.3**: RMSNorm |
| Feb 2020 | **GLU Variants** (Shazeer) | • SwiGLU activation<br>• Gated linear units<br>• Better than ReLU/GELU | **Phase 1.6**: SwiGLU MLP |
| Apr 2021 | **RoFormer** (Su et al.) | • Rotary Position Embeddings<br>• Relative position encoding<br>• Infinite extrapolation | **Phase 1.3, 1.5**: RoPE |
| Jun 2021 | **Switch Transformer** (Google) | • Sparse mixture of experts<br>• Trillion-parameter models | N/A (too complex) |
| Oct 2021 | **GPT-NeoX-20B** (EleutherAI) | • Open-source LLM<br>• RoPE adoption<br>• Parallelizable | Reference for our architecture |

**Technical Deep Dive: Why These Innovations Matter**

#### RMSNorm vs LayerNorm

**LayerNorm** (2016):
```python
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)
output = gamma * x_norm + beta
```

**RMSNorm** (2019):
```python
rms = sqrt(mean(x^2) + eps)
x_norm = x / rms
output = gamma * x_norm
```

**Advantages**:
- No mean computation → faster
- No beta parameter → fewer params
- Empirically works as well for LLMs

**Mini-Qwen**: ✅ RMSNorm (Phase 1.3)

---

#### SwiGLU vs ReLU

**Standard FFN** (Original Transformer):
```python
FFN(x) = W2(ReLU(W1(x)))
```

**GELU FFN** (BERT, GPT-2):
```python
FFN(x) = W2(GELU(W1(x)))
```

**SwiGLU** (2020, used in PaLM, LLaMA, Qwen):
```python
FFN(x) = (Swish(W_gate(x)) ⊙ W_up(x)) @ W_down
where Swish(x) = x * sigmoid(x)  # Also called SiLU
```

**Why SwiGLU?**
- Gating mechanism → more expressiveness
- Empirically better for language modeling
- Used by all modern LLMs (PaLM, LLaMA, Qwen)

**Cost**: ~50% more parameters (3 matrices vs 2)

**Mini-Qwen**: ✅ SwiGLU (Phase 1.6)

---

#### RoPE: Rotary Position Embeddings

**Learned Positional Embeddings** (Original Transformer):
```python
pos_emb = nn.Embedding(max_seq_len, hidden_dim)
x = token_emb + pos_emb
```

**Limitations**:
- Fixed maximum length
- Doesn't generalize beyond training length
- Absolute positions

**Sinusoidal Positional Encodings** (Original Transformer):
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**RoPE** (2021, used in GPT-NeoX, LLaMA, Qwen):
- Encodes relative positions via rotation matrices
- Applied to Q and K in attention
- Naturally extrapolates to longer sequences

**Math**: Rotates query/key vectors by angle proportional to position
```
q_m = R_m @ q
k_n = R_n @ k

where R_θ is a rotation matrix based on position
```

**Advantages**:
- Relative position encoding
- Extrapolates to unseen lengths
- No learned parameters
- Decays attention with distance

**Mini-Qwen**: ✅ RoPE (Phase 1.3, 1.5)

---

### Era 4: Open LLM Renaissance (2022-2023)

| Date | Milestone | Key Innovation | Parameters | Mini-Qwen Phase |
|------|-----------|----------------|------------|-----------------|
| Apr 2022 | **PaLM** (Google) | • RMSNorm + SwiGLU combo<br>• 540B params<br>• Pathways system | 540B | Architecture inspiration |
| May 2022 | **OPT** (Meta) | • Open pre-trained models<br>• 175B params<br>• Replicating GPT-3 | 175B | N/A |
| Jun 2022 | **Flash Attention** (Dao et al.) | • Memory-efficient attention<br>• IO-aware algorithm<br>• 2-4x speedup | N/A | **Phase 4.3**: Flash Attention |
| Nov 2022 | **BLOOM** (BigScience) | • Multilingual (46 languages)<br>• 176B params<br>• Open weights | 176B | Multilingual inspiration |
| Feb 2023 | **LLaMA** (Meta) | • **Direct inspiration for Qwen**<br>• RoPE + RMSNorm + SwiGLU<br>• Efficient training<br>• 7B, 13B, 33B, 65B | 7-65B | **Primary reference**<br>All phases |
| Mar 2023 | **GPT-4** (OpenAI) | • Multimodal<br>• Rumored 1.8T params<br>• MoE architecture | ~1.8T | N/A (closed) |
| Jul 2023 | **LLaMA-2** (Meta) | • Commercial license<br>• 70B model<br>• RLHF for chat | 7-70B | Training inspiration |
| Aug 2023 | **Qwen-7B** (Alibaba) | 🎯 **OUR REFERENCE MODEL**<br>• LLaMA-based architecture<br>• 151K vocab (multilingual)<br>• Extended context (32K)<br>• Tool use | 7B | **All phases**<br>Direct comparison |
| Sep 2023 | **Mistral-7B** (Mistral AI) | • Sliding window attention<br>• Grouped-query attention<br>• Best 7B model | 7B | GQA inspiration (Phase 6+) |
| Dec 2023 | **Mixtral 8x7B** (Mistral AI) | • Sparse mixture of experts<br>• 47B params, 13B active<br>• SOTA efficiency | 47B | N/A (too advanced) |

**LLaMA Architecture** (Direct Qwen Ancestor):
```python
LLaMA Architecture (2023):
├─ Token Embeddings (untied)
├─ RoPE (no learned positional)
├─ 32 Transformer Blocks:
│   ├─ RMSNorm (pre-norm)
│   ├─ Multi-Head Attention
│   │   └─ RoPE on Q, K
│   ├─ Residual
│   ├─ RMSNorm
│   ├─ SwiGLU MLP
│   └─ Residual
├─ Final RMSNorm
└─ LM Head (untied)

Qwen-7B = LLaMA + Multilingual Vocab + Extended Context
```

**Mini-Qwen**: We replicate this architecture at smaller scale

---

### Era 5: Long Context & Advanced Techniques (2023-2024)

| Date | Milestone | Key Innovation | Mini-Qwen Phase |
|------|-----------|----------------|-----------------|
| Jun 2023 | **NTK-aware RoPE** (emozilla) | • Dynamic position scaling<br>• Extends context without retraining | **Phase 6.1**: Context extension |
| Jul 2023 | **Llama-2-Long** (Meta) | • Extended to 32K context<br>• Position interpolation | **Phase 6.1-6.3** |
| Aug 2023 | **Code Llama** (Meta) | • Specialized for code<br>• Infilling support | N/A |
| Sep 2023 | **Qwen-14B/72B** (Alibaba) | • Scaled Qwen family<br>• 32K context standard | Reference architecture |
| Nov 2023 | **Qwen-1.8B** (Alibaba) | • Small efficient model<br>• Mobile deployment | Closest to our scale! |
| Feb 2024 | **Gemini 1.5** (Google) | • 1M context length<br>• Multimodal | N/A (closed) |
| Apr 2024 | **Llama-3** (Meta) | • 8B, 70B models<br>• 128K vocab<br>• Grouped-Query Attention | GQA reference |
| Jun 2024 | **Qwen2** (Alibaba) | • Improved architecture<br>• Better multilingual<br>• Qwen2-0.5B to Qwen2-72B | Latest reference |

**Context Extension Techniques**:

1. **Position Interpolation** (Meta):
   - Scale position indices: `pos' = pos * (L_train / L_inference)`
   - Simple but requires fine-tuning

2. **NTK-aware RoPE** (emozilla):
   - Dynamically adjust rotation frequency
   - No fine-tuning needed
   - Used by Qwen for 32K context

3. **YaRN** (Peng et al.):
   - Advanced interpolation
   - Temperature scaling
   - SOTA long context

**Mini-Qwen**: Implements NTK-aware scaling (Phase 6.1)

---

## Mini-Qwen Development Timeline

### Phase 1: Foundation (Weeks 1-2)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 1.1 | BPE Tokenizer foundation | Byte-Pair Encoding (Sennrich et al.) | 2016 | 300 |
| 1.2 | Complete BPE training | GPT-2 tokenizer (tiktoken) | 2019 | 400 |
| 1.3 | RMSNorm + RoPE basics | RMSNorm paper + RoFormer | 2019-2021 | 300 |
| 1.4 | Multi-head attention | Original Transformer | 2017 | 500 |
| 1.5 | RoPE integration | GPT-NeoX, LLaMA adoption | 2021-2023 | 200 |
| 1.6 | SwiGLU MLP | GLU Variants paper → PaLM | 2020-2022 | 300 |
| 1.7 | Transformer block (pre-norm) | GPT-2 pre-norm architecture | 2019 | 200 |
| 1.8 | Complete model | LLaMA/Qwen architecture | 2023 | 300 |

**Total**: ~2,500 lines | **Historical span**: 2016-2023

---

### Phase 2: Training (Weeks 2-3)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 2.1 | Dataset preparation | Language modeling tradition | 2018+ | 400 |
| 2.2 | AdamW optimizer | "Decoupled Weight Decay" (Loshchilov) | 2019 | 300 |
| 2.3 | Cosine LR scheduling | GPT-3, LLaMA training | 2020-2023 | 300 |
| 2.4 | Training loop | Standard LLM training | 2018+ | 500 |
| 2.5 | Gradient accumulation + clipping | Stability techniques | 2019+ | 300 |
| 2.6 | Logging & checkpointing | Production training | 2020+ | 400 |
| 2.7 | Mixed precision (BF16) | Tensor Cores, A100 GPUs | 2020+ | 300 |

**Total**: ~2,500 lines | **Historical span**: 2018-2023

---

### Phase 3: Inference (Weeks 3-4)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 3.1 | Greedy decoding | Basic autoregressive generation | 2018 | 200 |
| 3.2 | Temperature sampling | Controlled randomness | 2019 | 200 |
| 3.3 | Top-k sampling | "Hierarchical Neural Story Generation" | 2018 | 300 |
| 3.4 | Top-p (nucleus) sampling | "The Curious Case of Neural Text Degeneration" | 2019 | 300 |
| 3.5 | KV cache | Fast autoregressive decoding | 2020+ | 500 |

**Total**: ~1,500 lines | **Historical span**: 2018-2020

---

### Phase 4: Advanced Features (Weeks 4-5)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 4.1 | Batch generation | Production inference | 2020+ | 400 |
| 4.2 | Beam search | Classical NLP → Transformers | 2017+ | 500 |
| 4.3 | Flash Attention | "Flash Attention" paper (Dao et al.) | 2022 | 400 |
| 4.4 | Gradient checkpointing | "Training Deep Nets with Sublinear Memory" | 2016 | 300 |
| 4.5 | LoRA fine-tuning | "LoRA: Low-Rank Adaptation" (Hu et al.) | 2021 | 400 |

**Total**: ~2,000 lines | **Historical span**: 2016-2022

---

### Phase 5: Evaluation (Weeks 5-6)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 5.1 | Perplexity evaluation | Classical LM metric | 1990s+ | 300 |
| 5.2 | Few-shot prompting | GPT-3 in-context learning | 2020 | 400 |
| 5.3 | 8-bit quantization | "LLM.int8()" (Dettmers et al.) | 2022 | 400 |
| 5.4 | Performance profiling | Production deployment | 2023+ | 400 |

**Total**: ~1,500 lines | **Historical span**: 1990s-2023

---

### Phase 6: Extended Context (Optional)

| Commit | Feature | Historical Parallel | Year | Lines |
|--------|---------|---------------------|------|-------|
| 6.1 | Dynamic NTK scaling | NTK-aware RoPE (community) | 2023 | 300 |
| 6.2 | LogN attention scaling | Qwen extended context | 2023 | 300 |
| 6.3 | Efficient long-context | Sliding window, sparse patterns | 2023 | 400 |

**Total**: ~1,000 lines | **Historical span**: 2023

---

## Key Papers by Phase

### Phase 1: Foundation
- ✅ **Attention Is All You Need** (Vaswani et al., 2017) - Transformer architecture
- ✅ **RoFormer** (Su et al., 2021) - Rotary position embeddings
- ✅ **RMSNorm** (Zhang & Sennrich, 2019) - Normalization
- ✅ **GLU Variants** (Shazeer, 2020) - SwiGLU activation

### Phase 2: Training
- ✅ **AdamW** (Loshchilov & Hutter, 2019) - Optimizer
- ✅ **GPT-2** (Radford et al., 2019) - Training methodology
- ✅ **LLaMA** (Touvron et al., 2023) - Efficient training

### Phase 3: Inference
- ✅ **Top-p Sampling** (Holtzman et al., 2019) - Nucleus sampling
- ✅ **KV Cache** - Standard technique (2020+)

### Phase 4: Advanced
- ✅ **Flash Attention** (Dao et al., 2022) - Efficient attention
- ✅ **LoRA** (Hu et al., 2021) - Parameter-efficient fine-tuning
- ✅ **Gradient Checkpointing** (Chen et al., 2016) - Memory efficiency

### Phase 5: Evaluation
- ✅ **Few-shot Learning** (Brown et al., 2020) - GPT-3 paper
- ✅ **LLM.int8()** (Dettmers et al., 2022) - Quantization

### Phase 6: Extended Context
- ✅ **NTK-aware RoPE** (emozilla, 2023) - Context extension
- ✅ **Position Interpolation** (Chen et al., 2023) - Long context

---

## Architectural Evolution Summary

```
2017: Original Transformer
├─ Encoder-Decoder
├─ Post-norm
├─ Learned positional embeddings
├─ LayerNorm
└─ ReLU activation

↓

2019: GPT-2
├─ Decoder-only
├─ Pre-norm ✅
├─ Learned positional embeddings
├─ LayerNorm
└─ GELU activation

↓

2021: GPT-NeoX
├─ Decoder-only
├─ Pre-norm ✅
├─ RoPE ✅
├─ LayerNorm
└─ GELU activation

↓

2022: PaLM
├─ Decoder-only
├─ Pre-norm ✅
├─ RoPE ✅
├─ RMSNorm ✅
└─ SwiGLU ✅

↓

2023: LLaMA (Qwen's base)
├─ Decoder-only
├─ Pre-norm ✅
├─ RoPE ✅
├─ RMSNorm ✅
├─ SwiGLU ✅
└─ Untied embeddings ✅

↓

2023: Qwen-7B
├─ Decoder-only
├─ Pre-norm ✅
├─ RoPE ✅ (with NTK scaling)
├─ RMSNorm ✅
├─ SwiGLU ✅
├─ Untied embeddings ✅
├─ Bias in QKV only ✅
├─ 151K multilingual vocab
└─ Extended context (32K)

↓

2024: Mini-Qwen (Our Implementation)
├─ Decoder-only ✅
├─ Pre-norm ✅
├─ RoPE ✅
├─ RMSNorm ✅
├─ SwiGLU ✅
├─ Untied embeddings ✅
├─ Bias in QKV only ✅
├─ 10K vocab (simplified)
└─ 512 context (scaled down)
```

---

## Timeline Visualization

```
1990    2000    2010    2017    2019    2021    2023    2024
  |       |       |       |       |       |       |       |
  |       |       |   Transformer   GPT-2   RoPE    LLaMA   Mini-Qwen
  |       |       |       |       |       |       |       |
RNN     LSTM   Word2Vec  |    Pre-norm  SwiGLU  Qwen-7B  ↓
                Seq2Seq   |               |       |    Implementation
                Attention |          RMSNorm     |
                          |                       |
                     Multi-head           Flash Attention
                     Attention            LoRA
                                         Extended Context

[Phase 1] ←→ 2017-2021: Core architecture
[Phase 2] ←→ 2018-2023: Training techniques
[Phase 3] ←→ 2018-2020: Inference methods
[Phase 4] ←→ 2016-2022: Advanced optimization
[Phase 5] ←→ 2020-2023: Evaluation & quantization
[Phase 6] ←→ 2023: Long context extension
```

---

## Mini-Qwen's Place in History

**Mini-Qwen (2024)** is a **learning-focused re-implementation** of the **modern LLM architecture** that evolved from 2017-2023:

- **Architecture**: State-of-the-art (RoPE, RMSNorm, SwiGLU)
- **Scale**: Educational (~50M params vs 7B)
- **Purpose**: Understanding > Performance
- **Approach**: Reasoning-based, historically-grounded learning

We stand on the shoulders of giants:
- Vaswani et al. (Transformer, 2017)
- Radford et al. (GPT series, 2018-2023)
- Touvron et al. (LLaMA, 2023)
- Bai et al. (Qwen, 2023)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-16
**Historical Span**: 1997-2024
