# CPU Baseline Probes

Short CPU-track probe artifacts for ROADMAP C10.

These files are not full regression gates yet. They capture the exact command
surface and first measured numbers so the CPU baseline work can move from
"not started" to repeatable.

## Current measurement

- LARQL Gemma 3 4B Q4K CPU decode: 0.36 tok/s.
- llama.cpp Gemma 3 4B Q4_K_M CPU decode: 40.92 tok/s.
- Current gap: llama.cpp is about 114x faster on the short decode probe.
- LARQL CPU fallback stage split: 2547.33ms CPU full-prefix forward and
  233.10ms lm_head/top-k prediction per measured token.

## Current caveats

- `gemma3-4b-cpu-probe-2026-05-15.json` uses a Q4_K_M GGUF quantized locally
  from `output/larql-gemma-3-4b-it.gguf` with `llama-quantize`.
- `llama-bench` must be forced to `-dev BLAS` in this environment; otherwise it
  tries to initialize the Metal backend even with `-ngl 0`.
- The LARQL short runs stopped after 4 measured decode steps because EOS fired
  before the requested 5 tokens.
- CPU timing is still coarse: `CPU fwd` includes the full-prefix hidden-state
  forward, including attention, FFN, tensor insertion/removal, and prefix
  recomputation.
