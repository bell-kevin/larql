# CPU Bottleneck Diagnosis

Originally recorded 2026-05-15 for ROADMAP C10.
Updated 2026-05-15 after KV-cache + direct-matvec + parallel lm_head landed.

## Baseline (pre-fix, kept for reference)

The Gemma 3 4B Q4K CPU path was 114× behind the quant-matched llama.cpp baseline:

| Engine | Model path | Quant | Decode |
|---|---|---|---:|
| LARQL | `output/gemma3-4b-q4k-v2.vindex` | Q4K | 0.36 tok/s |
| llama.cpp | `/private/tmp/larql-gemma-3-4b-it-Q4_K_M.gguf` | Q4_K_M | 40.92 tok/s |

Pre-fix per-token split:

| Stage | Time | Share |
|---|---:|---:|
| CPU full-prefix forward | 2547 ms | 91.6 % |
| lm_head + top-k          | 233 ms  | 8.4 %  |

The fallback called `predict_q4k` for every generated token over the full
growing prefix — O(N²) decode, no KV reuse, and a per-layer Q4_K → f32
dequant every step.

## Current measurement (2026-05-15, post-fix)

```
target/release/larql bench output/gemma3-4b-q4k-v2.vindex \
    --cpu --tokens 16 --warmup 1 --profile
```

| Engine | Decode | Gap vs llama.cpp |
|---|---:|---:|
| LARQL (this branch) | 5.4 tok/s | 7.5× behind |
| llama.cpp Q4_K_M    | 40.92 tok/s | — |

Per-step split:

| Stage | Time | Share |
|---|---:|---:|
| CPU fwd (attention + FFN, 33 layers, direct Q4_K/Q6_K matvec) | 146 ms | 78.4 % |
| lm_head (f32 sgemv over 262K-vocab head, row-parallel) | 40 ms | 21.6 % |

Prefill (5 tokens) is unchanged at 2674 ms — still pays the per-layer
dequant; small absolute cost so untouched.

## What changed

Three independent fixes, layered on top of each other:

1. **KV-cached decode path**  (`crates/larql-inference/src/vindex/q4k_forward/cached.rs`):
   `predict_q4k_prefill` + `predict_q4k_decode_step` split. Prefill captures
   per-layer K/V into a `CpuKvCache`; decode runs single-row attention against
   the growing cache. O(N²) → O(N) decode work on attention/FFN. Dense
   architectures only; hybrid-MoE (Gemma 4 26B A4B) and KV-shared archs
   (Gemma 4 E2B) keep the legacy loop via `supports_cached_decode`.

   In isolation: ≈ 3.8× speedup on the gemm/attention portion. Capped by the
   per-step Q4_K dequant (= ~93 % of CPU forward time on its own).

2. **Direct Q4_K / Q6_K matvec, skipping per-step dequant**
   (`predict_q4k_decode_step_direct` in the same file):
   Routes every Q/K/V/O and gate/up/down projection through
   `backend.quant_matvec` (CPU `q4k_matvec_into` / `q6k_matvec::dispatch`),
   reading the vindex's Q4_K bytes directly. No more `insert_q4k_layer_tensors`
   dequant staging during decode. The CPU `q4k_matvec` was wired to a scalar
   reference impl; switched to the sumy-precomputed `q4_common::q4k_matvec_into`
   and rayon-parallelised the row loop. Same treatment for `q6k_matvec`.

   Combined with #1: decode 378 ms → effectively bound by gemm-on-quant + lm_head.

3. **Row-parallel f32 lm_head sgemv**
   (`forward/predict/dense.rs::parallel_lm_head_logits`):
   The previous `dot_proj(last_h, lm_head)` did `h.dot(&lm_head.t())`,
   which falls off ndarray's BLAS fast path because `lm_head.t()` is a
   transpose view. Scalar fallback at 10 GB/s on a 2.7 GB head matrix
   was 247 ms/step. Hand-rolled row-parallel dot over the row-major
   `lm_head` buffer with rayon dropped this to 40 ms.

## Remaining 7.5× gap to llama.cpp

The new bottleneck is the Q4_K matvec inner loop. We're at scalar f32 Rust
with `-O3` autovec; llama.cpp uses hand-written NEON intrinsics on Apple
silicon (`Q4_K_M` is one of their hottest kernels). The realistic path to
closing the rest of the gap:

1. NEON-intrinsics rewrite of `q4k_matvec_into` (target: 4× on the inner FMA).
2. Same treatment for `q6k_matvec::dispatch` (still scalar reference).
3. Optional: fuse Q+K+V projections into a single dispatch — currently three
   separate `q4k_matvec_into` calls share an `h_norm` row each.
4. Optional: extract a vindex with `lm_head_q4.bin` so the head matmul also
   skips the f32 staging.

Items 1 and 2 together would take the kernel from ~110 GFLOPS to ~400+ GFLOPS,
landing CPU fwd in the ~30-40 ms range and the total step at ~70 ms (≈ 14 tok/s),
within ~3× of llama.cpp without any algorithmic surgery beyond what already exists.

## Verification

- Parity: `cargo test -p larql-inference --release --test test_q4k_cached_parity -- --ignored`
  exercises cached vs uncached (must bit-match) and direct-matvec vs dequant
  (must agree on the first decode token; allowed to drift on later tokens
  due to different summation order).
- Bench: `bench/baselines/cpu/gemma3-4b-cpu-after-cached-direct-2026-05-15.json`
  has the JSON envelope with prefill, ms_per_tok, and the new per-stage timings.
