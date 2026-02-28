# Model Selection Guide — ollama-chain

Optimal model combinations for maximum answer accuracy on this system.

---

## System Profile

| Component | Specification |
|-----------|--------------|
| **CPU** | AMD Ryzen AI MAX+ 395 — 16 cores / 32 threads, 5.19 GHz boost |
| **Memory** | 128 GB unified (shared between CPU and iGPU) |
| **GPU** | Radeon 8060S (gfx1151) — 40 CUs, 2.9 GHz, integrated |
| **GPU Backend** | Vulkan (HIP/ROCm disabled via `HIP_VISIBLE_DEVICES=-1`) |
| **VRAM** | 128.5 GiB visible to Ollama (unified memory aperture) |
| **Ollama Config** | `MAX_LOADED_MODELS=1`, `KEEP_ALIVE=24h`, `FLASH_ATTENTION=true` |

### Memory Budget

With `OLLAMA_MAX_LOADED_MODELS=1`, only **one model is loaded at a time**.
The cascade unloads the previous model before loading the next, so the
constraint is that any single model must fit within available memory:

| Budget | Amount |
|--------|--------|
| Total unified memory | 128 GB |
| OS + services overhead | ~17 GB |
| Available for Ollama (model weights + KV cache) | ~111 GB |
| Largest safe model size (with 8K context KV cache) | ~100 GB |
| Largest safe model size (with 128K context KV cache) | ~80 GB |

> **KV cache cost**: Each model allocates a KV cache proportional to
> `num_layers × context_length × kv_head_dim`. A 70B model at 131K
> context uses ~45 GB for KV cache alone. Reducing context length
> (via `OLLAMA_CONTEXT_LENGTH`) dramatically reduces memory pressure.

---

## Installed Models Audit

Models sorted by parameter count (cascade order):

| # | Model | Params | Quant | Disk | Capabilities | Strengths | Weaknesses |
|---|-------|--------|-------|------|-------------|-----------|------------|
| 1 | `qwen3:8b` | 8.2B | Q4_K_M | 5.2 GB | tools, thinking | Fast inference, search query generation, routing decisions | Limited knowledge breadth, weaker on complex reasoning |
| 2 | `qwen3:30b-a3b` | 30.5B MoE (~3B active) | Q4_K_M | 18 GB | tools, thinking | **MoE speed** — nearly as fast as 8b with much broader knowledge; 262K context | Active parameters still small (~3B); Q4_K_M quantization |
| 3 | `deepseek-r1:32b` | 32.8B (distilled) | Q4_K_M | 19 GB | thinking | **Cross-architecture reviewer** — different biases from Qwen family; strong reasoning | **No tool support** — cannot execute tools; based on qwen2 arch but DeepSeek training |
| 4 | `qwen3:32b-q8_0` | 32.8B | **Q8_0** | 35 GB | tools, thinking | **High-fidelity quantization**, excellent review quality, strong reasoning | Slower to load than Q4_K_M variants |
| 5 | `deepseek-r1:70b` | 70.6B | Q4_K_M | 42 GB | thinking | **Exceptional chain-of-thought reasoning**, mathematical proofs | **No tool support** — unusable in agent mode, large KV cache at full context |
| 6 | `qwen3.5:122b` | 125.1B MoE | Q4_K_M | 81 GB | tools, thinking, **vision** | Broadest knowledge (MoE: 125B params, ~24B active), vision understanding, 262K context | Slow to load (~40s), Q4_K_M quantization on this scale |

### Key Observations

1. **`qwen3:32b-q8_0` is the highest-quality mid-range model** — Q8_0
   quantization preserves significantly more precision than Q4_K_M. At 32B
   parameters this is the sweet spot between speed and accuracy for review
   tasks.

2. **`qwen3:30b-a3b` is the optimal drafter** — as a MoE model with only
   ~3B active parameters, it runs nearly as fast as the 8b model but draws
   on 30B total parameters for broader knowledge. Supports tools + thinking.

3. **`deepseek-r1:32b` provides cross-architecture diversity** — although
   based on qwen2 architecture, it was trained by DeepSeek with different
   data and reasoning techniques. It catches biases that the Qwen3 family
   models share. No tool support limits it to reviewer role only.

4. **`deepseek-r1:70b` lacks tool support** — the cascade uses tools for
   search query generation and agent mode. deepseek-r1 can only contribute
   as a pure reasoning reviewer, not as a drafter or agent executor.

5. **`qwen3.5:122b` is the best final-stage model** — it supports all
   capabilities (tools + thinking + vision), has the broadest knowledge
   as a Mixture-of-Experts model, and the longest context window (262K).

6. **Model swap cost is significant** — loading a 81 GB model takes ~40s,
   a 42 GB model ~20s, and a 5 GB model ~2s. Each model in the chain
   adds swap overhead. Fewer, better models beat many mediocre ones.

---

## Recommended Chain Configurations

### Chain A — Maximum Accuracy (Recommended Default)

```
qwen3:30b-a3b → qwen3:32b-q8_0 → deepseek-r1:32b → qwen3.5:122b
```

| Role | Model | Why |
|------|-------|-----|
| **Drafter** | `qwen3:30b-a3b` | MoE drafter — fast inference (~3B active) with broad 30B knowledge; tools + thinking for search queries |
| **Reviewer 1** | `qwen3:32b-q8_0` | Q8_0 quantization catches subtle errors Q4_K_M models miss; thinking-enabled cross-verification |
| **Reviewer 2** | `deepseek-r1:32b` | Cross-architecture check — different training data and reasoning catches Qwen-family biases |
| **Final** | `qwen3.5:122b` | Broadest knowledge (125B MoE), all capabilities, produces authoritative final answer |

- **Swap time**: ~8s + ~15s + ~10s + ~40s = ~73s overhead
- **Total disk**: 153 GB (only one loaded at a time)
- **Capabilities**: Full pipeline (tools on drafter/reviewer1/final; thinking on all four)
- **Accuracy profile**: Highest possible — three-architecture cross-verification
  with Q8_0 precision review and MoE final model

> This is the recommended configuration for `cascade`, `auto`, `verify`,
> and `consensus` modes. The combination of Q8_0 precision (reviewer 1) and
> cross-architecture diversity (reviewer 2) catches the widest range of errors.

### Chain B — Accuracy + Speed Balance

```
qwen3:30b-a3b → qwen3:32b-q8_0 → qwen3.5:122b
```

| Role | Model | Why |
|------|-------|-----|
| **Drafter** | `qwen3:30b-a3b` | MoE draft + search |
| **Reviewer** | `qwen3:32b-q8_0` | Q8_0 precision review |
| **Final** | `qwen3.5:122b` | Authoritative final answer |

- **Swap time**: ~8s + ~15s + ~40s = ~63s overhead
- **Accuracy profile**: Very good — skips cross-architecture review but
  Q8_0 reviewer + 122B MoE final still catch most errors
- **Best for**: `verify`, `cascade`, `auto` (moderate complexity)

### Chain C — Deep Reasoning

```
qwen3:8b → deepseek-r1:70b
```

| Role | Model | Why |
|------|-------|-----|
| **Drafter** | `qwen3:8b` | Search queries, initial draft |
| **Final** | `deepseek-r1:70b` | Superior chain-of-thought reasoning, mathematical and logical proofs |

- **Swap time**: ~2s + ~20s = ~22s overhead
- **Accuracy profile**: Best for reasoning-heavy questions (math, logic,
  code analysis, proofs); weaker on factual recall vs. MoE models
- **Limitations**: No tool support — cannot be used in `agent` mode;
  reduce context length to avoid 45 GB KV cache at 131K context
- **Best for**: `verify`, `pipeline` modes with complex analytical queries

### Chain D — Fast + Quality

```
qwen3:30b-a3b → qwen3:32b-q8_0
```

| Role | Model | Why |
|------|-------|-----|
| **Drafter** | `qwen3:30b-a3b` | MoE draft — fast + broad |
| **Final** | `qwen3:32b-q8_0` | Q8_0 precision, fast loading, good reasoning |

- **Swap time**: ~8s + ~15s = ~23s overhead
- **Accuracy profile**: Good for moderate complexity; Q8_0 quality
  exceeds larger Q4_K_M models on precision-sensitive tasks
- **Best for**: `fast`, `route`, `auto` (simple/moderate queries)

### Chain E — Full Diversity (Maximum Cross-Verification)

```
qwen3:8b → qwen3:30b-a3b → qwen3:32b-q8_0 → deepseek-r1:32b → deepseek-r1:70b → qwen3.5:122b
```

All 6 installed models in cascade. Uses every model for maximum
cross-verification at the cost of ~95s swap overhead. Only worth it
for high-stakes queries where accuracy is paramount and time is not
a concern.

---

## Models Removed

| Model | Reason |
|-------|--------|
| `qwen3:14b` | Removed — marginal accuracy gain between 8b and 32b-q8_0; adds a swap for minimal benefit. The 32b-q8_0 subsumes its role with higher quality. |
| `mistral-large:123b` | Removed — no thinking support makes it unable to perform chain-of-thought verification. At 73 GB it was expensive to load for a model that can only do single-pass generation. Redundant with qwen3.5:122b which is larger (MoE) and supports thinking. |

---

## Suggested Additional Models to Pull

Models that would improve chain accuracy if added to the system:

### Installed (pulled)

| Model | Size | Arch | Capabilities | Role |
|-------|------|------|-------------|------|
| `qwen3:30b-a3b` | 18 GB | qwen3moe (30.5B, ~3B active) | tools, thinking | Fast MoE drafter — better quality than qwen3:8b at comparable speed; 262K context |
| `deepseek-r1:32b` | 19 GB | qwen2 (distilled, 32.8B) | thinking | Cross-architecture reasoning reviewer; catches Qwen-family biases; 131K context |

> **Note**: `qwen3.5:32b` is not yet available in the Ollama registry.
> Qwen 3.5 currently ships only as the 122B MoE variant.

### Medium Priority (not yet pulled)

| Model | Size | Why |
|-------|------|-----|
| `gemma3:27b` | ~17 GB | Google architecture provides a third "opinion" family alongside Qwen and DeepSeek; strong on factual accuracy |
| `phi4:14b` | ~9 GB | Microsoft's Phi-4 excels at code and STEM; useful as a specialized reviewer for technical queries |

### Aspirational (If Available)

| Model | Size | Why |
|-------|------|-----|
| `qwen3.5:122b-q8_0` | ~125 GB | Q8_0 of the MoE model — would push final answer quality significantly; tight fit at 128 GB but possible with reduced context |
| `deepseek-r1:70b-q8_0` | ~70 GB | Q8_0 reasoning model; massive quality improvement over Q4_K_M for mathematical and logical tasks |

### Optimal Chain with Current Models

With the newly pulled models, the best chain is:

```
qwen3:30b-a3b → qwen3:32b-q8_0 → deepseek-r1:32b → qwen3.5:122b
```

- **Drafter**: MoE 30B (3B active) — fast + high quality drafting (tools + thinking)
- **Reviewer 1**: Qwen 32B Q8_0 — precision review with high-fidelity quantization
- **Reviewer 2**: DeepSeek-R1 32B — cross-architecture reasoning check (different training data and biases)
- **Final**: Qwen 3.5 122B MoE — authoritative answer from broadest model

This chain provides three different model architectures (qwen3moe, qwen3 dense,
qwen2/deepseek distill, qwen35moe) for cross-verification while keeping swap
times reasonable (18 GB + 35 GB + 19 GB + 81 GB).

---

## Ollama Configuration Tuning

### Current Settings Analysis

```
OLLAMA_MAX_LOADED_MODELS=1     # One model at a time (sequential cascade)
OLLAMA_KEEP_ALIVE=24h          # Models stay loaded 24 hours
OLLAMA_FLASH_ATTENTION=true    # Efficient attention computation
OLLAMA_VULKAN=true             # Vulkan GPU backend
HIP_VISIBLE_DEVICES=-1         # ROCm/HIP disabled
OLLAMA_CONTEXT_LENGTH=0        # Auto (uses model default, up to 262144)
```

### Recommended Changes

| Setting | Current | Recommended | Impact |
|---------|---------|-------------|--------|
| `OLLAMA_KEEP_ALIVE` | `24h` | `15m` | Frees memory between sessions; the code already sets `keep_alive=15m` per request, but the server default holds models longer than needed |
| `OLLAMA_CONTEXT_LENGTH` | `0` (auto) | `8192` or `16384` | Reduces KV cache from ~45 GB (131K ctx) to ~3 GB (8K ctx) for the 70B model; **biggest single performance and memory improvement** |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | `2` | Allows keeping the 8b drafter loaded while swapping review/final models; saves ~2s per cascade step |
| `OLLAMA_KV_CACHE_TYPE` | (default f16) | `q8_0` or `q4_0` | Reduces KV cache memory by 2-4x with minimal quality loss; requires Ollama 0.6+ |

### Context Length vs. KV Cache Memory

KV cache is the hidden memory cost. For a 70B model (88 layers):

| Context Length | KV Cache Size | Total Memory (weights + KV) |
|----------------|--------------|----------------------------|
| 4,096 | ~1.4 GB | ~43 GB |
| 8,192 | ~2.8 GB | ~45 GB |
| 16,384 | ~5.6 GB | ~48 GB |
| 32,768 | ~11.2 GB | ~53 GB |
| 65,536 | ~22.4 GB | ~64 GB |
| 131,072 | ~44.8 GB | ~87 GB |

> **For ollama-chain, 8K–16K context is sufficient.** The cascade
> passes prompts of 2K–8K tokens. Setting `OLLAMA_CONTEXT_LENGTH=16384`
> frees 30–40 GB of memory and allows models to load faster.

---

## Per-Chain-Mode Recommendations

| Mode | Recommended Models | Reasoning |
|------|-------------------|-----------|
| **cascade** | `qwen3:30b-a3b → qwen3:32b-q8_0 → deepseek-r1:32b → qwen3.5:122b` | Chain A: MoE draft + Q8_0 review + cross-arch check + MoE final |
| **auto** | All of Chain A; router selects subset | Router picks fast-only for simple, full cascade for complex |
| **verify** | `qwen3:30b-a3b → qwen3.5:122b` | Two-step verify benefits from the gap between drafter and verifier |
| **consensus** | `qwen3:30b-a3b, qwen3:32b-q8_0, deepseek-r1:32b, qwen3.5:122b` | Four independent answers from diverse architectures |
| **search** | `qwen3:8b` (queries) + `qwen3.5:122b` (synthesis) | Search-first needs fastest query gen + strong synthesis |
| **pipeline** | `qwen3:30b-a3b` (extract) + `qwen3.5:122b` (reason) | MoE extraction + strong reasoning |
| **route** | `qwen3:8b` (classify) + `qwen3.5:122b` (complex) | Binary routing: fast or strong |
| **fast** | `qwen3:8b` | Single model, fastest response |
| **strong** | `qwen3.5:122b` | Single model, highest quality |
| **agent** | `qwen3:30b-a3b → qwen3:32b-q8_0 → qwen3.5:122b` | Agent needs tool support (excludes both deepseek-r1 variants) |
| **hack** | `qwen3:30b-a3b → qwen3:32b-q8_0 → qwen3.5:122b` | Pentest agent needs tools + thinking for planning/execution |

---

## Accuracy Factors Ranked by Impact

What contributes most to answer accuracy, in order:

| Rank | Factor | Impact | Status |
|------|--------|--------|--------|
| 1 | **Web search grounding** | Prevents stale training-data answers; critical for time-sensitive queries | Implemented (`_GROUNDING_INSTRUCTION`, `_extract_key_facts_from_search`) |
| 2 | **Final model quality** | The strongest model produces the final answer; MoE 122B has broadest knowledge | qwen3.5:122b is the best available |
| 3 | **Quantization quality** | Q8_0 preserves 99.5%+ of model quality vs. ~97% for Q4_K_M | qwen3:32b-q8_0 as reviewer is critical |
| 4 | **Cross-architecture verification** | Different model families have different biases; cross-checking catches family-specific errors | Add deepseek-r1:32b for cost-effective cross-architecture review |
| 5 | **Temperature control** | Low temperature (0.2) for grounded answers prevents creative hallucination | Implemented (`_TEMP_GROUNDED = 0.2`) |
| 6 | **Context length** | More context = more search results included | Adequate at 8K–16K for most queries |
| 7 | **Chain length** | More review steps = more error correction | 3 models is the sweet spot (diminishing returns after) |
| 8 | **News search** | Recent articles for time-sensitive queries | Implemented (`web_search_news` for time-sensitive) |

---

## Quick Start

### Current Model Inventory

All Chain A models are installed. The current cascade order is:

```bash
ollama-chain --list-models
# Expected output (sorted by parameter_size):
# 1  qwen3:8b           8.2B    Q4_K_M   qwen3      5.2 GB
# 2  qwen3:30b-a3b      30.5B   Q4_K_M   qwen3moe   18 GB
# 3  deepseek-r1:32b    32.8B   Q4_K_M   qwen2      19 GB
# 4  qwen3:32b-q8_0     32.8B   Q8_0     qwen3      35 GB
# 5  deepseek-r1:70b    70.6B   Q4_K_M   llama      42 GB
# 6  qwen3.5:122b       125.1B  Q4_K_M   qwen35moe  81 GB
#
# Cascade order: qwen3:8b → qwen3:30b-a3b → deepseek-r1:32b → qwen3:32b-q8_0 → deepseek-r1:70b → qwen3.5:122b
```

> **Note**: The cascade sorts by `parameter_size`. Since `deepseek-r1:32b`
> and `qwen3:32b-q8_0` are both 32.8B, their order depends on discovery
> order. Both serve as effective reviewers regardless of ordering.

### Reduce Context Length for Faster Loading

```bash
# Add to your Ollama service environment (e.g., /etc/systemd/system/ollama.service)
Environment="OLLAMA_CONTEXT_LENGTH=16384"

# Restart Ollama
sudo systemctl restart ollama
```

---

## All 6 Models vs. Chain A (4 Models)

Running all 6 installed models provides maximum cross-verification at the
cost of additional swap time. Every model in the current set contributes
unique value:

| Models | Unique contribution |
|--------|-------------------|
| `qwen3:8b` | Fastest for routing and search query generation |
| `qwen3:30b-a3b` | MoE breadth for drafting (30B params, ~3B active) |
| `deepseek-r1:32b` | Cross-architecture reasoning (DeepSeek training) |
| `qwen3:32b-q8_0` | Highest quantization fidelity (Q8_0) |
| `deepseek-r1:70b` | Strongest pure reasoning (70B + thinking) |
| `qwen3.5:122b` | Broadest knowledge, all capabilities (125B MoE) |

**Chain A (4 models, ~73s swap overhead)** achieves ~95% of maximum accuracy
by covering all three critical roles: fast drafter, precision reviewer,
cross-architecture check, and authoritative final.

**Chain E (all 6, ~95s swap overhead)** adds the 8b for faster routing and
the 70b for deeper reasoning. Worth it for high-stakes queries; overkill
for routine questions.

The `auto` mode router handles this tradeoff automatically — simple queries
use 1–2 models, complex queries use the full cascade.

---

## Image Generation

Ollama supports experimental text-to-image generation via diffusion models.
These models use a separate pipeline from the text LLMs and have the
`image` capability flag.

> **Platform support**: Image generation currently requires the MLX runner.
> As of Ollama 0.17.1, this works on **macOS**. Linux and Windows support
> is expected in a future Ollama release. The `image` chain mode is
> implemented and ready — it will work automatically once the Ollama
> backend adds Linux support for image generation.

### Installed Image Generation Models

| Model | Params | Quant | Disk | License | Best For |
|-------|--------|-------|------|---------|----------|
| `x/flux2-klein:4b` | 8.0B | FP4 | 5.7 GB | Apache 2.0 | Fast generation, text rendering, UI mockups, product photography |
| `x/z-image-turbo` | 10.3B | FP8 | 13 GB | Apache 2.0 | Photorealistic portraits, bilingual text (EN/CN), creative composition |

### Model Details

**FLUX.2 Klein 4B** (`x/flux2-klein:4b`)
- Family: `Flux2KleinPipeline` by Black Forest Labs
- Strengths: Readable text in images, clean typography, UI/interface mockups,
  product photography with proper lighting and shadows
- Also available: `x/flux2-klein:9b` (12 GB, higher quality, non-commercial license)

**Z-Image Turbo** (`x/z-image-turbo`)
- Family: Z-Image by Alibaba Tongyi Lab (6B diffusion model)
- Strengths: Photorealistic output, bilingual text rendering (English + Chinese),
  detailed portrait photography, creative double-exposure compositions
- Also available: `x/z-image-turbo:bf16` (33 GB, full precision, highest quality)

### How the `image` Chain Mode Works

```
User prompt → [Text LLM enhances prompt] → [Image model generates] → saved PNG
```

1. The user provides a description (e.g., "a cat in a space suit")
2. A text LLM (`qwen3:8b`) rewrites the description into a detailed,
   photography-style prompt optimized for diffusion models
3. The image generation model produces the image
4. The image is saved as a PNG file in the current directory

### Prompt Tips for Best Results

- **Be specific**: "Young woman in coffee shop, natural window lighting,
  cream knit sweater, soft bokeh background" beats "woman in cafe"
- **Specify style**: "shot on 35mm film", "commercial product shot",
  "watercolor texture", "isometric illustration"
- **For text in images**: Put the exact text in quotes — `A sign reading "OPEN"`
- **Resolution**: 1024x1024 is recommended (default)
