# Architecture Decision Records (ADRs)

This directory contains **Architecture Decision Records** for Mini-Qwen. Each ADR documents a significant design decision made during implementation.

---

## What is an ADR?

An **Architecture Decision Record** captures:
- **Context**: What problem are we solving?
- **Decision**: What did we choose?
- **Rationale**: Why this over alternatives?
- **Consequences**: What are the trade-offs?
- **Historical Evolution**: How did this evolve in the field?

ADRs help you understand the **"why"** behind each design choice, not just the "what".

---

## ADR Index

### Phase 1: Foundation

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| [ADR-001](ADR-001-rope-positional-encoding.md) | Rotary Position Embeddings (RoPE) | ✅ Accepted | 1.3, 1.5 |
| ADR-002 | Multi-Head Attention | 🚧 Pending | 1.4 |
| ADR-003 | RMSNorm vs LayerNorm | 🚧 Pending | 1.3 |
| ADR-004 | SwiGLU Activation | 🚧 Pending | 1.6 |
| ADR-005 | Pre-Norm Architecture | 🚧 Pending | 1.7 |
| ADR-006 | Untied Embeddings | 🚧 Pending | 1.8 |
| ADR-007 | BPE Tokenization | 🚧 Pending | 1.1, 1.2 |

### Phase 2: Training

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| ADR-008 | AdamW Optimizer | 🚧 Pending | 2.2 |
| ADR-009 | Cosine Learning Rate Schedule | 🚧 Pending | 2.3 |
| ADR-010 | Gradient Clipping | 🚧 Pending | 2.5 |
| ADR-011 | Mixed Precision Training | 🚧 Pending | 2.7 |

### Phase 3: Inference

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| ADR-012 | Top-p (Nucleus) Sampling | 🚧 Pending | 3.4 |
| ADR-013 | KV Cache Optimization | 🚧 Pending | 3.5 |

### Phase 4: Advanced

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| ADR-014 | Flash Attention | 🚧 Pending | 4.3 |
| ADR-015 | LoRA Fine-Tuning | 🚧 Pending | 4.5 |
| ADR-016 | Gradient Checkpointing | 🚧 Pending | 4.4 |

### Phase 5: Evaluation

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| ADR-017 | 8-bit Quantization | 🚧 Pending | 5.3 |

### Phase 6: Extended Context

| ADR | Decision | Status | Commits |
|-----|----------|--------|---------|
| ADR-018 | NTK-aware RoPE Scaling | 🚧 Pending | 6.1 |
| ADR-019 | LogN Attention Scaling | 🚧 Pending | 6.2 |

---

## How to Use ADRs

### For Learning
1. **Before implementing**: Read the relevant ADR
2. **Understand context**: Why is this decision needed?
3. **Compare alternatives**: What else was considered?
4. **Trace history**: How did this evolve in the field?
5. **After implementing**: Complete the exercises

### For Reference
- **Lookup**: Find decisions by topic or commit
- **Compare**: See how our approach differs from Qwen
- **Extend**: Use as foundation for enhancements

---

## ADR Template

Use [`ADR-TEMPLATE.md`](ADR-TEMPLATE.md) when creating new ADRs.

Key sections:
- **Context** and **Decision** (the "what")
- **Rationale** and **Alternatives** (the "why")
- **Historical Evolution** (the "when/how")
- **Qwen Comparison** (the "difference")
- **Learning Outcomes** (the "education")
- **Exercises** (the "practice")

---

## ADR Lifecycle

```
Proposed → Accepted → Implemented → Verified
              ↓
         (sometimes)
              ↓
         Deprecated → Superseded by ADR-XXX
```

**Statuses**:
- **Proposed**: Under consideration
- **Accepted**: Decision made, ready to implement
- **Implemented**: Code written
- **Verified**: Tests pass, exercises completed
- **Deprecated**: No longer recommended
- **Superseded**: Replaced by newer decision

---

## Example: ADR-001 (RoPE)

**Purpose**: Understand why we use Rotary Position Embeddings

**Key Learnings**:
- RoPE encodes relative positions via rotation matrices
- Superior to learned embeddings for extrapolation
- Industry standard (LLaMA, Qwen, Mistral)
- Complex number implementation for elegance

**Exercises**:
- Visualize 2D rotations
- Implement real-valued variant
- Debug common pitfalls

---

## Contributing ADRs

When adding a new decision:

1. **Copy template**: `cp ADR-TEMPLATE.md ADR-XXX-title.md`
2. **Fill sections**: Especially Context, Decision, Rationale
3. **Add comparisons**: How does Qwen do this?
4. **Include history**: Evolution timeline
5. **Write exercises**: Hands-on learning
6. **Update index**: Add to this README

---

## Reading Order

**For systematic learning**, read ADRs in chronological order:

**Phase 1** (Foundation):
1. ADR-007: BPE Tokenization → Understand text-to-tokens
2. ADR-003: RMSNorm → Modern normalization
3. ADR-001: RoPE → Position encoding
4. ADR-002: Multi-Head Attention → Core mechanism
5. ADR-004: SwiGLU → Modern activation
6. ADR-005: Pre-Norm → Architecture choice
7. ADR-006: Untied Embeddings → Output layer

**Then continue phase by phase...**

---

## FAQ

### Q: Do I need to read all ADRs before starting?
**A**: No! Read them just-in-time, before implementing each component.

### Q: What if I disagree with a decision?
**A**: Great! That means you're thinking critically. Try implementing the alternative and compare results.

### Q: Can I skip the historical evolution sections?
**A**: You can, but you'll miss valuable context. Understanding "why" is as important as "how".

### Q: How detailed should my answers to exercises be?
**A**: Enough to demonstrate understanding. Code + brief explanation.

---

## Related Documentation

- **[Learning Guide](../../MINI_QWEN_LEARNING_GUIDE.md)**: Overall project structure
- **[Historical Timeline](../../HISTORICAL_TIMELINE.md)**: Evolution of LLMs
- **[Checkpoints](../checkpoints/)**: Phase completion tests
- **[Comparisons](../comparisons/)**: Mini-Qwen vs Qwen-7B

---

**Last Updated**: 2025-01-16
**Total ADRs**: 19 (planned)
**Completed**: 1 (ADR-001)
