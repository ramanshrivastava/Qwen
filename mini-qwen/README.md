# Mini-Qwen: Build a Modern Language Model from Scratch

A **reasoning-based, historically-grounded learning project** to understand modern LLMs by implementing Qwen-7B architecture at educational scale.

---

## 🎯 Project Goals

1. **Understand** modern LLM architecture deeply (not just use it)
2. **Learn** by implementing from first principles
3. **Trace** the historical evolution of each design decision
4. **Compare** our implementation with production Qwen-7B
5. **Build** a working language model that generates coherent text

---

## 📊 Quick Stats

| Metric | Qwen-7B | Mini-Qwen | Ratio |
|--------|---------|-----------|-------|
| Parameters | 7 billion | ~50 million | 140x smaller |
| Layers | 32 | 8 | 4x fewer |
| Hidden Dim | 4,096 | 512 | 8x smaller |
| Vocab Size | 151,851 | 10,000 | 15x smaller |
| Context Length | 2,048 → 32K | 512 | 4-64x smaller |
| Training Data | 2.2T tokens | 10M-100M tokens | ~20,000x less |
| Training Time | Weeks (multi-node) | Hours-days (single GPU/CPU) | - |

---

## 🏗️ Architecture

Mini-Qwen implements the **modern LLM stack** (2023 state-of-the-art):

- ✅ **RoPE** (Rotary Position Embeddings) - from RoFormer (2021)
- ✅ **RMSNorm** - from RMSNorm paper (2019)
- ✅ **SwiGLU** activation - from GLU Variants (2020)
- ✅ **Pre-norm** architecture - from GPT-2 (2019)
- ✅ **Untied embeddings** - modern practice
- ✅ **Multi-head attention** - from Transformer (2017)
- ✅ **BPE tokenization** - from Neural MT (2016)

Same architecture as: **LLaMA**, **Qwen**, **Mistral**, **Llama-3**

---

## 📚 Learning Approach

This project uses a **unique learning methodology**:

### 1. Architecture Decision Records (ADRs)
Every design choice is documented with:
- **Context**: What problem are we solving?
- **Decision**: What did we choose?
- **Rationale**: Why this over alternatives?
- **Historical Evolution**: How did this evolve?
- **Trade-offs**: What did we gain/lose?

### 2. Historical Timeline
Each commit maps to LLM evolution:
- Commit 1.3: RMSNorm (2019) + RoPE (2021)
- Commit 1.6: SwiGLU (2020) → PaLM (2022) → LLaMA (2023)
- Commit 4.3: Flash Attention (2022)

### 3. Learning Checkpoints
After each phase:
- **Quiz**: Test conceptual understanding
- **Exercises**: Extend the code
- **Comparison**: Mini-Qwen vs Qwen-7B
- **Benchmarks**: Measure performance

---

## 🗺️ Implementation Roadmap

### Phase 1: Foundation (Week 1-2, ~2,500 lines)
Build tokenizer and model architecture
- [x] BPE tokenizer (Commits 1.1-1.2)
- [x] RMSNorm + RoPE (Commit 1.3)
- [x] Multi-head attention (Commits 1.4-1.5)
- [x] SwiGLU MLP (Commit 1.6)
- [x] Transformer block (Commit 1.7)
- [x] Complete model (Commit 1.8)

**Checkpoint**: Forward pass works, shapes verified

### Phase 2: Training (Week 2-3, ~2,500 lines)
Train the model on text
- [ ] Dataset preparation (Commit 2.1)
- [ ] AdamW optimizer (Commit 2.2)
- [ ] Learning rate scheduling (Commit 2.3)
- [ ] Training loop (Commit 2.4)
- [ ] Gradient techniques (Commit 2.5)
- [ ] Logging & checkpointing (Commit 2.6)
- [ ] Mixed precision (Commit 2.7)

**Checkpoint**: Loss decreases, model overfits small dataset

### Phase 3: Inference (Week 3-4, ~1,500 lines)
Generate text
- [ ] Greedy decoding (Commit 3.1)
- [ ] Temperature sampling (Commit 3.2)
- [ ] Top-k sampling (Commit 3.3)
- [ ] Top-p sampling (Commit 3.4)
- [ ] KV cache (Commit 3.5)

**Checkpoint**: Generates coherent 2-3 sentences

### Phase 4: Advanced (Week 4-5, ~2,000 lines)
Production features
- [ ] Batch generation (Commit 4.1)
- [ ] Beam search (Commit 4.2)
- [ ] Flash Attention (Commit 4.3)
- [ ] Gradient checkpointing (Commit 4.4)
- [ ] LoRA fine-tuning (Commit 4.5)

**Checkpoint**: Fine-tuned model performs well

### Phase 5: Evaluation (Week 5-6, ~1,500 lines)
Benchmark and optimize
- [ ] Perplexity evaluation (Commit 5.1)
- [ ] Few-shot prompting (Commit 5.2)
- [ ] 8-bit quantization (Commit 5.3)
- [ ] Performance profiling (Commit 5.4)

**Checkpoint**: Compared with baselines

### Phase 6: Extended Context (Optional, ~1,000 lines)
Support longer sequences
- [ ] NTK scaling (Commit 6.1)
- [ ] LogN attention (Commit 6.2)
- [ ] Sparse attention (Commit 6.3)

---

## 📖 Documentation

- **[Learning Guide](../MINI_QWEN_LEARNING_GUIDE.md)**: Complete implementation guide
- **[Historical Timeline](../HISTORICAL_TIMELINE.md)**: Maps commits to LLM evolution
- **[ADRs](docs/adrs/)**: Architecture decision records
- **[Comparisons](docs/comparisons/)**: Mini-Qwen vs Qwen-7B
- **[Checkpoints](docs/checkpoints/)**: Learning milestones

---

## 🚀 Quick Start

### Installation

```bash
cd mini-qwen
pip install -r requirements.txt
```

### Train a Tokenizer

```bash
python examples/01_train_tokenizer.py \
  --corpus examples/datasets/shakespeare.txt \
  --vocab-size 10000 \
  --output tokenizer.json
```

### Train the Model

```bash
python examples/02_train_model.py \
  --config configs/mini_config.yaml \
  --data examples/datasets/shakespeare.txt \
  --output checkpoints/
```

### Generate Text

```bash
python examples/03_generate_text.py \
  --checkpoint checkpoints/step_10000.pt \
  --prompt "Once upon a time" \
  --max-length 100
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/test_tokenizer.py
pytest tests/test_attention.py
pytest tests/test_model.py

# Integration tests
pytest tests/test_training.py
pytest tests/test_generation.py

# All tests
pytest tests/
```

---

## 🔬 Tools

### Attention Visualizer
```bash
python tools/visualize_attention.py \
  --checkpoint checkpoints/step_1000.pt \
  --text "The quick brown fox"
```

### Model Comparator
```bash
python tools/compare_models.py \
  --model1 mini-qwen \
  --model2 qwen-7b \
  --prompt "Explain quantum computing"
```

### Performance Profiler
```bash
python tools/profile_performance.py \
  --config configs/mini_config.yaml
```

---

## 📊 Expected Results

After training on 100M tokens:

| Metric | Mini-Qwen | Notes |
|--------|-----------|-------|
| Perplexity (val) | 30-50 | Lower is better |
| Tokens/sec (training) | 1K-5K | Single GPU |
| Tokens/sec (inference) | 100-500 | With KV cache |
| Memory (training) | 8-12 GB | FP32, batch=32 |
| Memory (inference) | 1-2 GB | FP32 |

**Generation Quality**: 2-3 sentence coherence (not production-ready, but educational!)

---

## 🎓 Learning Resources

### Papers (Chronological)
1. **Attention Is All You Need** (2017) - Transformer
2. **GPT-2** (2019) - Pre-norm, scaling
3. **RMSNorm** (2019) - Normalization
4. **GLU Variants** (2020) - SwiGLU
5. **RoFormer** (2021) - RoPE
6. **Flash Attention** (2022) - Efficient attention
7. **LLaMA** (2023) - Direct inspiration
8. **Qwen Technical Report** (2023) - Our reference

### Blog Posts
- The Illustrated Transformer (Jay Alammar)
- RoPE Explained (EleutherAI)
- LLaMA Paper Analysis (Nathan Lambert)

### Code References
- **Qwen codebase**: `/home/user/Qwen/`
- **LLaMA implementation**: Various open-source repos
- **HuggingFace Transformers**: Reference implementations

---

## 🤝 Contributing

This is a **learning project**. Contributions that enhance educational value are welcome:

- 📝 Improve documentation/ADRs
- 🐛 Fix bugs with explanations
- 📊 Add visualization tools
- 🧪 Add tests with learning exercises
- 📖 Add references to papers/blogs

**Not welcome**: Performance optimizations that obscure learning

---

## 📜 License

MIT License - See LICENSE file

This project is for **educational purposes**. The architecture is inspired by:
- LLaMA (Meta AI)
- Qwen (Alibaba Cloud)
- Various academic papers

---

## 🙏 Acknowledgments

- **Qwen team** (Alibaba Cloud) - Reference architecture
- **LLaMA team** (Meta AI) - Pioneering open LLMs
- **Transformer authors** (Google) - Original architecture
- **EleutherAI** - GPT-NeoX and open research
- **HuggingFace** - Transformers library

---

## 📞 Contact

Questions? Issues? Want to discuss?

- Open an issue on GitHub
- Read the [FAQ](docs/FAQ.md)
- Check [Learning Guide](../MINI_QWEN_LEARNING_GUIDE.md)

---

**Status**: 🚧 Under active development
**Version**: 0.1.0 (Phase 1 in progress)
**Last Updated**: 2025-01-16

---

*"The best way to understand AI is to build it from scratch."*
