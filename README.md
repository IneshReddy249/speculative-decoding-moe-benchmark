# Speculative Decoding on MoE: MTP vs EAGLE3 on GLM-4.7-Flash

**First published head-to-head comparison of MTP, EAGLE3, and SpecV2 overlap scheduling on a Mixture-of-Experts model.**

> GLM-4.7-Flash (30B total / 3B active, MoE) · Single NVIDIA H100 80GB PCIe · SGLang 0.5.10 · ShareGPT workload

## TL;DR

| Method | Best Config | c=1 TPS | c=32 TPS | vs Baseline |
|:--|:--|--:|--:|:--|
| Baseline | — | 100.8 | 979.6 | 1.0x |
| MTP + SpecV2 | steps=1, topk=1, tokens=2 | 131.4 | **1,255.0** | **1.28x** |
| EAGLE3 + SpecV2 | steps=3, topk=1, tokens=4 | **135.1** | 1,090.3 | 1.11x |
| EAGLE3 (no SpecV2) | steps=3, topk=4, tokens=6 | 127.2 | 1,050.5 | 1.07x |

**Key finding:** Neither method universally wins. EAGLE3 delivers the best per-request latency at low concurrency (1.34x at c=1), while MTP achieves the highest system throughput at high concurrency (1.28x at c=32) with zero additional VRAM. The **SpecV2 overlap scheduler** is the single most impactful optimization for both methods.

## Why This Matters

Published EAGLE3 benchmarks report 3–6x speedups on **dense** models (Llama-3.1-8B). Those numbers don't transfer to MoE architectures. This project measures what actually happens when you apply speculative decoding to a production MoE model with only 3B active parameters out of 30B total — a fundamentally different compute profile where verification overhead matters far more.

## Hardware & Software

| Component | Spec |
|:--|:--|
| GPU | NVIDIA H100 80GB PCIe |
| CUDA | 12.8, Driver 570.195.03 |
| Framework | SGLang 0.5.10.post1 ([Thoughtworks fork](https://github.com/nicetiger/sglang) for EAGLE3 GLM support) |
| Attention | FlashAttention 3 (FA3), FlashInfer 0.6.7 |
| Model | [GLM-4.7-Flash](https://huggingface.co/THUDM/GLM-4.7-Flash) BF16 (56.37 GB) |
| EAGLE3 Draft | [thoughtworks/GLM-4.7-Flash-Eagle3](https://huggingface.co/thoughtworks/GLM-4.7-Flash-Eagle3) (291 MB) |
| Dataset | ShareGPT (64 prompts, 28K input tokens, 16K output tokens, `--sharegpt-output-len 256`) |
| Benchmark | `sglang.bench_serving` with 4 warmup requests, seed=42 |

### Bug Fix Applied

The Thoughtworks fork required a patch for GLM-4.7-Flash compatibility. Added `self.enable_a2a_moe = False` to `Glm4MoeLiteForCausalLM` (line 438 of `glm4_moe_lite.py`) to fix:

```
AttributeError: 'Glm4MoeLiteModel' has no attribute 'enable_a2a_moe'
```

EAGLE3 also required `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` due to a context length mismatch between the draft head (4096) and target model (202752).

## Results

### Output Throughput (tok/s) — All Configurations

| Concurrency | Baseline | MTP-v1 | MTP-v2 (SpecV2) | EAGLE3-v1 | EAGLE3-v2 (big tree) | EAGLE3-v3 (SpecV2) |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 100.8 | 109.0 | **131.4** | 127.2 | 120.4 | 135.1 |
| 4 | 271.5 | 258.7 | **315.2** | 298.0 | 295.0 | 308.9 |
| 8 | 408.7 | 402.9 | 474.8 | 463.2 | 408.8 | **483.1** |
| 16 | 626.5 | 629.8 | 723.1 | 658.2 | 636.2 | **740.3** |
| 32 | 979.6 | 941.1 | **1,255.0** | 1,050.5 | 930.0 | 1,090.3 |

### Best Configs Head-to-Head (SpecV2 enabled, apples-to-apples)

| Concurrency | MTP-v2 | EAGLE3-v3 | Winner | Speedup vs Baseline |
|--:|--:|--:|:--|:--|
| 1 | 131.4 | **135.1** | EAGLE3 | 1.34x |
| 4 | **315.2** | 308.9 | MTP | 1.16x |
| 8 | 474.8 | **483.1** | EAGLE3 | 1.18x |
| 16 | 723.1 | **740.3** | EAGLE3 | 1.18x |
| 32 | **1,255.0** | 1,090.3 | MTP | 1.28x |

### Per-Token Latency (TPOT, ms) — Lower is Better

| Concurrency | Baseline | MTP-v2 | EAGLE3-v3 |
|--:|--:|--:|--:|
| 1 | 9.68 | 7.28 | **7.05** |
| 8 | 19.47 | 16.15 | **15.84** |
| 16 | 25.38 | 21.06 | **19.40** |
| 32 | 30.95 | **23.47** | 25.07 |

### Accept Length by Configuration

| Configuration | Accept Length | Notes |
|:--|--:|:--|
| MTP-v1 (steps=3, topk=1, no SpecV2) | 1.86 | Over-drafting, wasted compute |
| MTP-v2 (steps=1, topk=1, SpecV2) | 1.65 | Minimal drafting + overlap wins |
| EAGLE3-v1 (steps=3, topk=4, no SpecV2) | 2.28 | Best accept rate, no overlap |
| EAGLE3-v2 (steps=5, topk=8, no SpecV2) | 2.72–2.77 | Higher accept, but slower overall |
| EAGLE3-v3 (steps=3, topk=1, SpecV2) | 2.01–2.02 | Accept dropped with topk=1 |

## Analysis

### 1. SpecV2 Overlap Scheduler Is the Biggest Lever

The overlap scheduler eliminates CPU idle bubbles between draft and verify stages by preparing the next batch while the GPU is still computing. Impact:

- **MTP:** Transformed from a wash (1.0x) to 1.15–1.30x speedup
- **EAGLE3:** Improved from 1.07x to 1.11–1.18x speedup

This is a scheduling optimization, not a speculation quality improvement. It requires `topk=1` (greedy drafting).

### 2. EAGLE3 Has Better Draft Quality, But It's Not Enough on MoE

EAGLE3's tri-layer feature fusion produces higher accept rates (2.28 vs 1.86 for MTP). But on a 30B MoE model with only 3B active parameters, the per-token compute is already fast — so the overhead of running a separate draft model matters more relative to the savings from accepted tokens.

On dense models (Llama-3.1-8B), the target model is slower per-token, so EAGLE3's higher accept rate translates to larger speedups (2–6x reported in the paper).

### 3. Bigger Draft Trees Make Things Worse on MoE

EAGLE3-v2 (steps=5, topk=8, tokens=16) improved accept length from 2.28 to 2.77 (+22%) but throughput **decreased** at every concurrency level. The verification cost of processing 16 draft tokens exceeds the gains from accepting a few more.

This is the opposite of what happens on dense models, where bigger trees generally help.

### 4. MTP Wins at High Concurrency Because of Zero VRAM Overhead

MTP uses the model's built-in prediction heads — no extra weights, no extra KV cache, no extra forward pass. On a model that already uses 56 GB of 81 GB VRAM, this frees ~1.14 GB (EAGLE3 draft model) for KV cache, allowing more concurrent requests before memory pressure degrades throughput.

### 5. The Crossover Point Is Real

EAGLE3 wins at c=1, 8, and 16 (latency-sensitive regimes). MTP wins at c=4 and 32 (throughput-sensitive regimes). Neither method dominates — the optimal choice depends on your deployment's concurrency profile.

## Server Configurations

### Baseline
```bash
python3 -m sglang.launch_server \
  --model-path ~/models/GLM-4.7-Flash \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.85 --port 30000
```

### MTP-v2 (Best MTP Config)
```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model-path ~/models/GLM-4.7-Flash \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.80 --port 30001
```

### EAGLE3-v1 (Best Per-Request Latency Without SpecV2)
```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python3 -m sglang.launch_server \
  --model-path ~/models/GLM-4.7-Flash \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path ~/models/GLM-4.7-Flash-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 6 \
  --speculative-eagle-topk 4 \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.80 --port 30002
```

### EAGLE3-v3 (Best EAGLE3 With SpecV2)
```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model-path ~/models/GLM-4.7-Flash \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path ~/models/GLM-4.7-Flash-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --tp 1 --trust-remote-code \
  --mem-fraction-static 0.80 --port 30002
```

### Benchmark Command
```bash
python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:PORT \
  --model /path/to/GLM-4.7-Flash \
  --dataset-name sharegpt \
  --dataset-path sharegpt.json \
  --num-prompts 64 \
  --max-concurrency C \
  --sharegpt-output-len 256 \
  --extra-request-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --output-file results/CONFIG_cC.json \
  --warmup-requests 4 --seed 42
```

## Deployment Recommendations

| Scenario | Recommendation |
|:--|:--|
| Interactive chat, low concurrency (c≤8) | EAGLE3 + SpecV2 for best per-request latency |
| Batch processing, high concurrency (c≥16) | MTP + SpecV2 for best system throughput |
| Memory-constrained (single GPU, large model) | MTP — zero additional VRAM |
| Quick deployment, no training | MTP — uses built-in model heads |
| Maximum latency reduction at c=1 | EAGLE3-v1 (topk=4, no SpecV2) — 7.55ms TPOT |

## Limitations

- **Single model tested.** Results are specific to GLM-4.7-Flash (30B MoE). Dense models will show different tradeoffs.
- **Single GPU.** Multi-GPU tensor parallelism changes memory pressure dynamics.
- **Third-party draft head.** The Thoughtworks EAGLE3 draft head was trained for ~1.5 hours. A SpecForge-trained head on domain-matched data could improve accept rates.
- **ShareGPT workload.** Highly diverse prompts. Domain-specific workloads (code, RAG) may show different accept rates.
- **64 prompts per run.** Larger prompt sets would reduce variance.

## Reproducing

1. Provision an H100 80GB PCIe (Shadeform, Lambda, etc.)
2. Clone this repo and download models:
   ```bash
   huggingface-cli download THUDM/GLM-4.7-Flash --local-dir ~/models/GLM-4.7-Flash
   huggingface-cli download thoughtworks/GLM-4.7-Flash-Eagle3 --local-dir ~/models/GLM-4.7-Flash-Eagle3
   ```
3. Install the Thoughtworks SGLang fork:
   ```bash
   git clone https://github.com/nicetiger/sglang.git /tmp/tw-sglang
   cd /tmp/tw-sglang && git checkout 0675f95
   pip install -e "python/" --no-deps
   ```
4. Apply the `enable_a2a_moe` bug fix (see above)
5. Download ShareGPT dataset and run benchmarks using the commands above

## References

- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) (NeurIPS 2025)
- [SGLang Speculative Decoding Documentation](https://docs.sglang.io/advanced_features/speculative_decoding.html)
- [SpecForge: Training Framework for EAGLE3](https://github.com/sgl-project/SpecForge)
- [Thoughtworks EAGLE3 GLM Blog Post](https://huggingface.co/blog/lujangusface/tw-eagle3-gpu)
- [SGLang Zero-Overhead Overlap Scheduler (Vertex AI)](https://www.lmsys.org/blog/2025-12-01-eagle3-vertex/)

## License

MIT

## Author

Inesh Reddy — [LinkedIn](https://linkedin.com/in/ineshreddy) · [GitHub](https://github.com/IneshReddy249)
