<<<<<<< HEAD
# llama.cpp-tq3

Runtime fork for TurboQuant GGUFs in `llama.cpp`.

Public focus:
- `TQ3_1S`
- `TQ3_4S`

This repo is the runtime side only. The quantization tooling is intentionally not part of the public fork.

## Current 27B Results

`Qwen3.5-27B`, `wiki.test.raw`, full pass, `c=2048`:

| Format | PPL | Size |
|--------|-----|------|
| `TQ3_4S` | `6.8224 +/- 0.04534` | `12.9 GiB` |
| `Q3_K_S` | `6.8630 +/- 0.04583` | `11.4 GiB` |
| `TQ3_1S` | `6.9807 +/- 0.04690` | `12.9 GiB` |
| `EXL3 3.0bpw` | `7.027580` | `~13.0 GiB` |

Notes:
- `EXL3 3.0bpw` is from a local `145 x 2048` eval, not `llama-perplexity`.
- The public runtime path here is the clean base `TQ3_1S` / `TQ3_4S` support.

## Current Runtime Speed

Measured on:
=======
# buun-llama-cpp

![buunslamma](buunslamma.png)

> **This is a highly experimental fork of llama.cpp. Use at your own discretion.**

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with **Trellis-Coded Quantization (TCQ)** for KV cache compression. 2-3x more context in the same VRAM, with quality that matches or beats FP16.

**Paper**: [Closing the Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits](https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache)

## What is TCQ?

Standard KV cache quantization treats each value independently. TCQ constrains quantization indices to follow a 512-state trellis, enabling a much larger effective codebook at the same bit rate. Combined with FWHT rotation and context-adaptive norm scaling, this achieves **10-44% KL-divergence reduction** over scalar quantization at 2-3 bits per value.

At 3.25 bits per value, TCQ produces **lower perplexity than FP16** KV cache (5.802 vs 5.805).

## Build

```sh
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

## Recommended configurations

### turbo4 (4.25 bpv) -- lossless quality, great compression

The safe default. Virtually no quality loss vs FP16 with ~3.8x KV cache compression and no speed penalty.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo4 -ctv turbo4
```

### 3-bit TCQ (3.25 bpv) -- best quality at 3-bit

Beats FP16 quality at short context, stays within 2% at long context. ~5x KV cache compression.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo3_tcq
```

### 2-bit TCQ (2.25 bpv) -- maximum compression

~7x KV cache compression. Best for fitting very long contexts in limited VRAM.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo2_tcq -ctv turbo2_tcq
```

### Asymmetric 2.75 bpv -- best 2-bit quality

3-bit keys + 2-bit values. 15-17% lower KLD than the reverse, because adaptive alpha already compensates V quantization error.

```sh
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo2_tcq
```

### Scalar turbo3 / turbo2 (3.25 / 2.25 bpv) -- no trellis

Scalar quantization without TCQ. Faster encode, worse quality than TCQ equivalents.

```sh
# 3-bit scalar
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3 -ctv turbo3

# 2-bit scalar
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo2 -ctv turbo2
```

## Quality (KL-divergence, Qwen3.5-27B Q6_K, RTX 3090)

Lower is better. Measured against FP16 KV cache base logits.

| Config | bpv | KLD @2K | KLD @7K |
|--------|-----|---------|---------|
| turbo3_tcq (symmetric) | 3.25 | 0.058 | 0.074 |
| turbo3_tcq-K / turbo2_tcq-V | 2.75 | 0.078 | 0.101 |
| turbo2_tcq (symmetric) | 2.25 | 0.101 | 0.136 |

3-bit TCQ at 2K context achieves **lower perplexity than FP16** (5.802 vs 5.805) due to a mild regularizing effect from norm scaling.

## Speed (Qwen3.5-27B Q6_K, RTX 3090)

| Config | Decode (tg64) | vs q8_0 |
|--------|---------------|---------|
| q8_0 | 31.04 tok/s | -- |
| turbo3_tcq | 30.04 tok/s | 97% |
| turbo3 (scalar) | 30.04 tok/s | 97% |

Decode speed is constant across context lengths (30 tok/s at 4K, 65K, and 128K). Prefill uses tensor-core MMA path at 99%+ of q8_0 speed.

## Custom codebooks

Trained codebooks are included in `codebooks/`. The defaults are compiled into the CUDA kernels, but you can override them:

```sh
TURBO_TCQ_CB=codebooks/3bit/product_aware_iter080.bin \
TURBO_TCQ_CB2=codebooks/2bit/product_aware_iter090.bin \
./build/bin/llama-server -m model.gguf -ngl 99 -fa \
  -ctk turbo3_tcq -ctv turbo3_tcq
```

Codebook training scripts are in `scripts/tcq_train_*.py`.

## How it works

1. **FWHT rotation** with random sign flips converts correlated KV vectors into i.i.d. Gaussian entries
2. **Viterbi encoding** on a 512-state (3-bit) or 256-state (2-bit) right-shift trellis finds the globally optimal codeword assignment
3. **O(1) sliding-window decode** -- each value decodes via a bit window lookup, no trellis traversal at inference
4. **Context-adaptive alpha** -- logarithmic norm scaling formula automatically adjusts dequantization scale per context length

## Supported models

Any GGUF model with `head_dim` that is a multiple of 128 works natively. Models with other head dimensions (e.g., Phi-3 at 96, Qwen3-0.6B at 64) are supported via automatic zero-padding.

Tested on: Qwen3.5-27B, Qwen3-32B, Gemma-3-27B, Gemma-4-31B, Harmonic-Hermes-9B, Phi-3-mini, and others.

---
>>>>>>> spiritbuun/master

- GPU: `RTX 5060 Ti 16GB`
- build: public runtime branch
- command family: `llama-bench`

### Qwen3.5-27B-TQ3_4S

| Metric | Result |
|--------|--------|
| `pp2048` | `~708 tok/s` |
| `tg128` | `23.2 tok/s` |
| size | `12.99 GiB` |

Notes:

- these are the current public runtime numbers for the 27B `TQ3_4S` path
- the prompt-path headline is the important public speed claim: `~708 tok/s` at `pp2048`
- `pp` and `tg` should be compared separately
- if you benchmark with custom KV-cache settings, keep that separate from the base weight-format numbers

## Ampere Deployment Notes

### GPU Topology

All GPUs connected via PCIe PHB (no NVLink). Multi-GPU communication goes through the PCIe Host Bridge.

```
GPU0 (RTX 3090)  GPU1 (RTX 3090)  GPU2 (RTX 3060)  GPU3 (RTX 3060)  GPU4 (RTX 3090)
  └── PHB ──────────── PHB ──────────── PHB ──────────── PHB ──────────── PHB
```

### Build Flags (Ampere preset — `x64-linux-cuda-ampere-release`)

| Flag | Value | Effect |
|------|-------|--------|
| `CMAKE_CUDA_ARCHITECTURES` | `86` | Compile PTX for sm_86 only (no fallback overhead) |
| `GGML_NATIVE` | `ON` | AVX2/FMA host-side kernels |
| `GGML_CUDA_FA` | `ON` | Flash Attention compiled in |
| `GGML_CUDA_FA_ALL_QUANTS` | `ON` | FA enabled for Q4_1/Q5_0/Q5_1/Q8_0 KV caches |
| `GGML_CUDA_GRAPHS` | `ON` | CUDA graph capture (reduces kernel launch overhead) |
| `GGML_CUDA_PEER_MAX_BATCH_SIZE` | `512` | P2P batch size for multi-GPU (up from default 128) |
| RPATH | `$ORIGIN` | Prefers colocated `libllama`/`libggml` over conda env copies |

### Multi-GPU (2× RTX 3090) vs Single GPU — llama-bench results

Model: `Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf`, `-ctk q8_0 -ctv q8_0 -fa 1 -b 2048 -ub 512`, 3 reps each.

| Configuration | pp512 (t/s) | tg128 (t/s) | tg gain |
|---------------|------------:|------------:|--------:|
| Single GPU (RTX 3090, sm_86) | 713.75 ± 16 | 21.81 ± 0.1 | baseline |
| Dual GPU 0+4 (`--split-mode layer`, 50/50) | 729.48 ± 5 | 27.81 ± 0.0 | **+27.5%** |

Key observations:
- **+27.5% decode throughput** with dual GPU via memory-bandwidth halving per card (model weight loads split across both GPUs)
- Prompt processing gains are modest (+2%) because it is compute-bound; PCIe communication overhead limits further scaling
- `--split-mode row` (tensor parallel) is **not** recommended for PHB/PCIe connections — requires high-bandwidth all-reduce unsuitable for PCIe
- `--split-mode layer` is the correct choice for PCIe-connected GPUs

### Live HTTP benchmark (dual GPU, warm, port 1234)

| Prompt length | prompt tok/s | decode tok/s |
|---------------|-------------:|-------------:|
| ~6 tokens | 62.3 | 28.2 |
| ~1800 tokens | 792.9 | 27.7 |

vs prior single-GPU baseline: decode ≈ 23.1 t/s, long prompt ≈ 686.8 t/s.

### Recommended live command (dual GPU 0+4)

```bash
CUDA_VISIBLE_DEVICES=0,4 \
./build-x64-linux-cuda-ampere-release/bin/llama-server \
  -m /path/to/Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf \
  --mmproj /path/to/Qwopus3.5-27B-v3-Abliterated-mmproj.gguf \
  -ngl 99 \
  --split-mode layer \
  --tensor-split 0.5/0.5 \
  --main-gpu 0 \
  -c 112640 \
  -b 512 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --host 0.0.0.0 \
  --port 1234
```

### Single-GPU fallback

```bash
CUDA_VISIBLE_DEVICES=4 \
./build-x64-linux-cuda-ampere-release/bin/llama-server \
  -m /path/to/Qwopus3.5-27B-v3-Abliterated-TQ3_4S.gguf \
  --mmproj /path/to/Qwopus3.5-27B-v3-Abliterated-mmproj.gguf \
  -ngl 99 \
  -c 112640 \
  -b 512 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --host 0.0.0.0 \
  --port 1234
```

### VRAM layout (dual GPU, 112k context, `-ctk q8_0 -ctv q8_0`)

| Buffer | GPU 0 | GPU 4 |
|--------|------:|------:|
| Model weights | 5,994 MiB | 6,234 MiB |
| KV cache | 1,870 MiB | 1,870 MiB |
| Recurrent state | 312 MiB | 287 MiB |
| Compute buffer | 1,177 MiB | 976 MiB |
| **Total** | **~10,802 MiB** | **~9,682 MiB** |

### Historical single-GPU measurements (phase 1)

| Build / state | short prompt t/s | decode t/s | long prompt t/s |
|---------------|----------------:|----------:|----------------:|
| original live (before restart) | 48.45 | 23.10 | 686.76 |
| rebuilt RPATH preset (candidate) | 46.02 | 22.32 | 678.63 |
| restored live (warm rollback) | 50.30 | 22.80 | 680.43 |

Notes:
- `q8_0` KV was faster than `tq3_0` KV for this workload
- FA uses `BEST_FATTN_KERNEL_MMA_F16` for prompt and `BEST_FATTN_KERNEL_VEC` for decode on sm_86
- Without `GGML_CUDA_FA_ALL_QUANTS`, only f16/f32/q4_0/q8_0/tq3_0 KV support FA; enabling it also unlocks Q4_1/Q5_0/Q5_1

## Build From Source

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git
cd llama.cpp-tq3

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For a clean standalone Ampere rebuild that prefers colocated `libllama` / `libggml` instead of environment copies:

```bash
cmake --fresh --preset x64-linux-cuda-ampere-release
cmake --build build-x64-linux-cuda-ampere-release -j
```

## Download A Model

Example base `TQ3_4S` release:

```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download YTan2000/Qwen3.5-27B-TQ3_4S \
  Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  --local-dir ./models/tq3_4s
```

## Run

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 2048 \
  --port 8090
```

## KV Cache Settings

For the cleanest baseline, run with normal KV cache:

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 4096 \
  --port 8090
```

For the TurboQuant KV-cache path used in local experiments, use:

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -fa on \
  -c 8192 \
  -ctk tq3_0 \
  -ctv tq3_0 \
  --port 8090
```

Notes:
- `-ctk tq3_0 -ctv tq3_0` is the currently supported TurboQuant KV-cache setting in this fork.
- Keep weight-format and KV-cache experiments separate when benchmarking.
- If you want the exact weight-format baseline, leave KV cache at the default types.

## Benchmark

```bash
./build/bin/llama-bench \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -p 2048 \
  -n 0 \
  -r 3

./build/bin/llama-bench \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 \
  -p 0 \
  -n 128 \
  -r 3

./build/bin/llama-perplexity \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -f /path/to/wiki.test.raw \
  -c 2048 \
  -ngl 99 \
  -fa 1 \
  -t 8 \
  --no-warmup
```

## Scope

- runtime support for `TQ3_1S` and `TQ3_4S`
- CUDA path for local inference
- pre-quantized GGUF workflow

Not in scope for this public fork:
- quantizer research code
- private mixed-policy search tooling
- HF upload helpers

## Credits

- `TheTom/turboquant_plus` for public TurboQuant KV-cache engineering, benchmarking, and implementation notes:
  - https://github.com/TheTom/turboquant_plus
- `flamme-demon/llama.cpp-hip-turboquant-tq3` for HIP/ROCm compatibility port of the TQ3 native kernels:
  - https://github.com/flamme-demon/llama.cpp-hip-turboquant-tq3
- Google Research for the TurboQuant compression algorithm and technical paper:
  - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
