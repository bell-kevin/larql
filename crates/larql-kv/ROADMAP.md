# Roadmap — larql-kv

## Current state (as of 2026-05-18)

**Performance equilibrium post W7 + W8 + W8.2 + Step 9** (Gemma 3 4B
Q4K, Metal, M3 Max):

| Engine | 50-tok tok/s | 1000-tok tok/s | Prefill (5-tok) | Gap to standard @ 1k |
|---|---:|---:|---:|---:|
| `standard` (fused) | 100.3 | 64.1 | 300 ms | — |
| `markov-rs` | 88.9 | **58.7** | 265 ms | -8.4% |
| `markov-rs-codec` | 88.8 | **57.2** | 270 ms | -10.8% |
| `unlimited-context` | 86.4 | 57.4 | 256 ms | -10.4% |
| `turbo-quant` (4-bit, 10-tok) | 37.7 | — | — | codec-bound |

All cached-state engines now cluster within ~10% of `standard`'s
fused-kernel ceiling. The 135% pre-W8.2 gap on `markov-rs` /
`markov-rs-codec` collapsed once the per-step `Array2::zeros((n+1,
kv_dim)) + slice-copy` pattern was replaced with doubling-capacity
in-place append. Prefill is no longer the wall-time dominator
(post Step 9: 10× speedup vs the 2.7 s CPU walk it used to fall back
to). See "Closed (recent)" for the milestone history.

The remaining 8-11% decode gap is fixed CPU glue (state-dump
readback into `PerLayerDecodeState`, counter bump, append-row).
Closing further requires either single-kernel prefill state-dump
(W9 — Metal kernel surgery, small wall-time win at current bench
shape) or a Metal-side path that elides the per-token CPU readback
entirely (W10 — engine-side state lives on GPU until window-close).

## Crate-shape state (2026-05-17)

- Crate extracted from `larql-inference::engines` on 2026-05-09 — see
  [`CHANGELOG.md`](CHANGELOG.md).
- **Seven engines shipped** as of 2026-05-17:
  - Original four: `standard`, `no_cache`, `markov_residual`,
    `unlimited_context`, `turbo_quant`, `apollo`.
  - Three new: `boundary_kv`, `markov_residual_codec`, `boundary_per_layer`.
    Specs in `crates/larql-inference/docs/specs/`:
    [boundary-kv-engine.md](../larql-inference/docs/specs/boundary-kv-engine.md),
    [markov-residual-codec-engine.md](../larql-inference/docs/specs/markov-residual-codec-engine.md),
    [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- Consumers wired:
  - `larql-cli bench --engine <spec>` (selector dispatch)
  - `larql-cli bench --via-executor` opts into the new `LayerExecutor`
    surface; falls through to legacy path for unmigrated engines.
  - in-crate `benches/engine_decode.rs` (criterion: dispatch helpers + Standard parity)
- Coverage policy: 90 % line coverage per source file (see
  `coverage-policy.json`); CI gate at `make larql-kv-coverage-policy`.
  Workspace `larql-kv` lib total: **95.55% lines, 95.40% regions, 94.49%
  functions** (2026-05-17 evening, up from 92.12% earlier the same day).
  **All 43 files now ≥90% lines; debt baselines cleared from policy
  file.** The post-Phase-2 push lifted `accuracy_suite/measurement.rs`
  (77→99%), `accuracy_suite/runner.rs` (75→94%), `vindex_compare.rs`
  (65→97%), and `engines/apollo/store.rs` (89→92%) by adding
  formatter + driver tests against the synthetic fixtures; the
  Phase-2 engines (`unlimited_context/engine.rs`, `turbo_quant/engine.rs`,
  `apollo/engine.rs`) all land ≥92% on the new `*_via_executor` methods.

## Architectural cuts (2026-05-17)

Substantive refactors landed; specs reflect the new boundaries.

### Naming hygiene — renamed for honesty

- **`metal_fused_prefill` / `metal_fused_decode_step`** → `fused_prefill`
  / `fused_decode_step`. The "metal" was a lie — `CpuBackend` implements
  `prefill_q4` and `decode_token` via its C Q4 kernel and also takes the
  fused path on `--cpu`. The aliases in `unlimited_context::engine`
  (`quant_prefill_metal`, `quant_decode_token`) follow.
- **`KvEngine::prefill_q4k` / `decode_step_q4k`** → `prefill_quant` /
  `decode_step_quant`. The `_q4k` suffix baked one format into the trait
  surface; the trait is quant-agnostic (dispatches on `index`'s format).
  Internals that are genuinely Q4K-specific (`prefill_q4k_moe`,
  `cpu_q4k_cache_*`, `run_ffn_decode_step_q4k_direct`) keep their names.
- **`ComputeBackend::has_q4()` → `supports_quant(format: QuantFormat)`.**
  Per-format predicate; `CpuBackend` reports support for `Q4_0`, `Q4_K`,
  `Q4_KF`, `Q6_K`; `MetalBackend` adds `Q8_0`. Backends can advertise new
  format support without trait extension.
- **Storage slots `q4k` → `kquant` for K-family fields.** `attn_q4k`,
  `interleaved_q4k`, `set_attn_q4k`, `load_attn_q4k`, etc. — these hold
  K-family quant bytes (Q4_K, Q4_KF, Q6_K — manifest tag picks). Q4_0
  (`attn_q4`) and Q8 (`attn_q8`) slots stay — genuinely format-specific.

### Engine state vs execution — new abstraction

Spec: [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).

The engines were re-coupling backend / FFN / format decisions into their
state-management code. The new shape:

- **`LayerExecutor` trait** (in `larql-inference::layer_executor`) —
  per-layer execution surface with `run_prefill_layer` /
  `run_decode_layer` returning `(hidden, SharedKV)`. Dispatch kind
  (`Fused` / `PerLayer`) is explicit.
- **`LocalWalkExecutor`** — wraps `run_attention_with_kv_backend` +
  the caller's `&dyn FfnBackend`. The critical decoupling: the executor
  does **not** construct its own `WalkFfn` — it uses whatever the engine
  was handed.
- **Engine trait extension:** `KvEngine::prefill_via_executor`,
  `decode_step_via_executor`, `prefill_quant_via_executor`,
  `decode_step_quant_via_executor`. Default impls fall through to the
  legacy methods so unmigrated engines work unchanged.

### Engines on the new surface

Every engine now runs its own state-policy code; there is no hidden
fall-through to the backend's fused kernel from per-layer engines.
`standard` (and by delegation `boundary_kv`) is the **only** engine
that exercises the fused fast path — via
`ComputeBackend::coarse_prefill` / `coarse_decode_step`, which on
Metal calls `larql_inference::vindex::fused_prefill`.

| Engine | Default dispatch | `*_via_executor` override | Honors FFN backend | Tok/s (Gemma 3 4B Q4K, Metal) | Hot state |
|---|---|---|---|---:|---:|
| `standard` | `ComputeBackend::coarse_prefill` (fused fast path) | n/a (no per-layer code to migrate) | n/a | 104 | 0 MB (backend owns K/V) |
| `boundary_kv` | Delegates to `standard` + emits boundary frames | n/a | n/a | ≈104 | 0 MB |
| `markov_residual` | Per-layer walk via `rs_prefill_walk` | ✅ | ✅ counter test | 3.6 | 6.0 MB |
| `markov_residual_codec` | Per-layer walk via `rs_prefill_codec_walk` (bf16 cold) | ✅ | ✅ counter test | 4.3 | 6.0 MB |
| `unlimited_context` | Windowed checkpoint extension via `process_q4k` | ✅ | ✅ counter test | 25.6 | 4.8 MB |
| `turbo_quant` | Per-layer WHT + Lloyd-Max compression cycle | ✅ | ✅ counter test | 3.9 | 0.6 MB |
| `boundary_per_layer` | Per-layer walk with per-layer codec policy | ✅ (dense) | ✅ counter test | — | matches markov_residual_codec |
| `apollo` | Whole-forward through `forward_layer_range` (boundary prefix + perturb) | ✅ | ✅ counter test | requires store | scales with store |
| `no_cache` | Full re-forward per step (O(N²) wall-time) | ✅ | ✅ already did on legacy `prefill` | — | token list only |

## Coverage debt

**Status (2026-05-24):** six files below the 90% per-file floor after
the Phase-2 dispatch additions and the env-var-gated diagnostic
paths in `compute.rs` drifted past the 2026-05-17 cleared state. The
post-Item-2 cross-product work added five new files (`runner/*.rs`)
all above 90%; the regression is in older engine internals that
weren't covered when the dispatch paths landed in May.

| File | Coverage | Lift path |
|---|---:|---|
| `engines/markov_residual/compute.rs` | 86.85% | env-var-gated paths (`LARQL_MARKOV_WALK_KV_DIAG`, `_FORCE_F32`, `_TOPK`) — needs `serial_test` crate or config-injection refactor |
| `engines/unlimited_context/dispatch.rs` | 59.09% | needs mock `EngineBackend` infrastructure |
| `engines/markov_residual/dispatch.rs` | 77.51% | needs mock `EngineBackend` infrastructure |
| `engines/markov_residual_codec/dispatch.rs` | 80.68% | needs mock `EngineBackend` infrastructure |
| `engines/turbo_quant/dispatch.rs` | 9.35% | needs mock `EngineBackend` infrastructure |
| `engines/boundary_per_layer/dispatch.rs` | 7.95% | needs mock `EngineBackend` infrastructure |

Progress 2026-05-24 (session-level): lifted five files above the
floor (cold_tier 88→100%, executor 85→90.6%, walk 84→95%, engine
83→90%, store 86→99.6%) and improved compute.rs from 81→86.85%
without env-var test infrastructure. 37 new tests across the five
fixed files; clippy clean; zero regressions.

### What's left

**Sub-project A — mock `EngineBackend` infrastructure (5 files).** All
five remaining dispatch files call
`backend.coarse_prefill_with_state` and
`backend.coarse_decode_step_with_state_masked` — the GPU-dispatch
surface that only `MetalBackend` implements meaningfully today. The
existing 59–80% coverage on three of them comes from
`tests/dispatch_parity.rs` (integration tests against real engines);
the 0-10% files (`boundary_per_layer/dispatch.rs`,
`turbo_quant/dispatch.rs`) aren't even reached by those.

The lift requires a shared test-only `EngineBackend` impl that
returns synthetic `PerLayerDecodeState` payloads conformant with the
trait. Once built, each dispatch file gets ~5 unit tests exercising
prefill happy-path, prefill no-state-dump fallback, decode happy-path,
decode `StateDumpMask::HOnly` / `None` cascades. Open question: where
the mock lives — `larql-inference::test_utils` (next to
`make_test_weights`) or a new `larql-kv/src/test_utils.rs` that
imports the trait. The former is more reusable across crates; the
latter is local to where the mock is consumed.

Estimated scope: ~1 hour for the mock infra + 30 min per dispatch
file = 3-4 hours total.

**Sub-project B — `compute.rs` env-var path coverage (1 file).** The
remaining ~7% gap is the diagnostic + force-f32 code paths gated on
`LARQL_MARKOV_WALK_KV_DIAG`, `LARQL_MARKOV_KV_FORCE_F32`,
`LARQL_MARKOV_WALK_KV_TOPK`, `LARQL_MARKOV_WALK_KV_SELECT_AT`. These
read process env vars at decode time — testing them in parallel
requires `serial_test` (or similar) to serialise `env::set_var` calls
without racing other tests. Alternative: refactor the helpers to
take an explicit config struct argument; the env-var read becomes a
single thin caller and the tests can pass values directly without
touching the global env.

Estimated scope: 45 min for `serial_test` integration, 1-2 hours for
the config-injection refactor.

**Acceptance criterion.** `make larql-kv-coverage-policy` passes
against a *freshly regenerated* `coverage/larql-kv/summary.json`
(via `make larql-kv-coverage-summary`, not the cached JSON). 2026-05-24
note: the gate was previously passing against a stale JSON that
predated the dispatch additions — fresh regeneration surfaced the
debt. Future commits to engine internals should run
`make larql-kv-coverage-summary` locally, not just
`make larql-kv-coverage-policy`.

### Open design questions

1. **Should the mock `EngineBackend` live in `larql-inference::test_utils`
   or in `larql-kv`?** Reusable-across-crates favours the former;
   local-to-consumers favours the latter.
2. **`serial_test` crate vs env-var-injection refactor for `compute.rs`?**
   The crate is a small dependency add; the refactor is a more
   invasive change but yields more testable code long-term.
3. **Acceptance for the GPU-only branches.** Some lines in the
   dispatch files only fire on a real Metal backend (e.g.
   `StateDumpMask::Full` paths that exercise the cached-decode
   kernel). Mock-backend tests can verify the *plumbing* but not
   the *kernel* — should those lines be excluded via
   `#[cfg(not(test_coverage))]` or similar, or accepted as
   coverable-only-via-integration-tests?

These are blocking on the implementation slice landing, not the
coverage-debt resolution itself; the design choices fall during the
sub-project work, not before.

## Open work

### P0 — engine performance (the post-bypass optimization frontier)

The fused-bypass strip (2026-05-17 night) made every engine's actual
per-step cost visible for the first time. The remaining headroom is
substantial — but the goal is to close it **without** re-introducing
bypass paths. Each per-layer engine has a state-policy contract that
defines what work cannot be skipped; the optimization budget is what
remains.

**Reference numbers** (Gemma 3 4B Q4K, Metal, M3 Max, 20-token
decode):

| Engine | tok/s | Hot state | Per-step cmd_bufs (Metal) | Per-step compute model |
|---|---:|---:|---:|---|
| `standard` (fused) | 104 | 0 MB (backend-owned) | 1 | one fused kernel, all 34 layers, append-1-row K/V |
| `unlimited_context` | 25.6 | 4.8 MB | ~103 | per-layer attn+ffn, append-1-row K/V (same compute as standard, different dispatch) |
| `markov_residual_codec` | 4.3 | 6.0 MB | ~103 | per-layer attn+ffn + **recompute K/V from `window_size` residuals every step** |
| `turbo_quant` (4-bit) | 3.9 | 0.6 MB | ~103 | per-layer attn+ffn + **decompress prior K/V + re-encode updated K/V every step** (CPU codec in inner loop) |
| `markov_residual` | 3.6 | 6.0 MB | ~103 | same as codec; no codec overhead on bench (cold tier never fired in 20-step run) |
| `apollo` | — | scales w/ store | varies | re-forward layers `crystal..N` over growing context every step (no K/V cache) |
| `no_cache` | — | token list only | varies | full re-forward every step (O(N²) by design — not an optimization target) |

#### Per-engine bottleneck diagnosis

**Post-W2 measurements — split by backend** (Gemma 3 4B Q4K, M3 Max,
10-token decode, 2026-05-17 night):

| Engine | CPU tok/s | GPU (Metal) tok/s | Where the gap lives |
|---|---:|---:|---|
| `standard` (coarse_prefill control) | 28.2 | 102.7 | GPU's fused fast path is 3.6× the CPU C kernel. |
| `unlimited_context` | 28.1 | 28.4 | **At parity** — no per-layer overhead either side. |
| `markov_residual_codec` | 26.6 | 27.5 | **At parity** (post-W2). |
| `markov_residual` | 26.5 | 26.8 | **At parity** (post-W2). |
| `turbo_quant` (4-bit) | 19.4 | 19.6 | **At parity** — codec overhead dominates on both. |

**Reading the table — the GPU/CPU split reveals an even sharper
diagnosis** (re-checked 2026-05-17 after reading the helper code):

- **On CPU**, every engine clusters at ~26-28 tok/s. The 28 tok/s
  ceiling is the M3 Max CPU compute limit for Gemma 3 4B Q4K
  rayon-parallel matvec at this prompt length.
- **On GPU**, only `standard` reaches 102.7 tok/s — the only engine
  that actually runs on the GPU. The four "per-layer Metal" engines
  all sit at 20-28 tok/s, same as CPU, **because they are running
  CPU code regardless of the `--backends metal` flag.** Tracing
  through `attention_decode_step_native` and `ffn_decode_step_native`
  (the native-quantised helpers all per-layer engines call): the
  `_backend: &dyn ComputeBackend` parameter is plumbed but never
  consulted — these helpers always dispatch to
  `matvec_q4k_or_q6k_q8k`, which is rayon-parallel CPU Q4K×Q8K
  matvec. The Metal backend isn't involved in their per-layer
  compute at all.

This changes the W1 framing. The previous diagnosis ("103 Metal
submits per token = 5-10ms of dispatch overhead") was wrong because
**there are zero Metal submits per token** for per-layer engines
today — the entire per-layer loop runs on CPU. The actual ~28 tok/s
ceiling is the CPU's rayon-parallel matvec throughput, hit equally
under both `--backends cpu` and `--backends metal`.

**The real W1**: route the per-layer Q/K/V/O and gate/up/down matvecs
through Metal kernels (per layer) so the GPU actually participates
in the per-layer engines' compute. This is a larger change than
"batch the dispatches" because today's per-layer code path doesn't
use Metal at all — there's nothing to batch yet.

W2 landed: caching the hot K/V projection across decode steps
moved both markov_residual engines from ~5 to ~27 tok/s — they now
sit on the same curve as `unlimited_context` (which already cached
K/V incrementally), within 1.5 tok/s of each other. The
`recompute_kv` stage no longer fires; FFN+attention dominate
exactly like every other cached-K/V engine. **The hot K/V state
costs ~10.8MB vs 5.3MB pre-W2** (trade memory for speed; still
~50× smaller than standard's full KV).

Reading the table: percentages are *of the engine's own per-step total*,
not vs standard. The three cached-K/V engines (markov-rs, codec,
unlimited-context) now cluster around 27-28 tok/s, all showing the
same FFN-heavy decode profile. The remaining ~4× gap to standard
is per-layer Metal dispatch overhead — W1's target.

**`unlimited_context` — 28.4 tok/s, 35 ms/tok. Per-layer attn + ffn
dominates; no recompute waste.** Compute model is identical to
standard's (append-1-row K/V per layer). 74% of the step is FFN, 25%
is attention. The 4× gap to standard is **per-layer Metal command-
buffer dispatch** — 103 cmd_bufs per token vs standard's 1. Each
submit has ~50-100µs fixed cost, so even with zero-cost compute
there'd be 5-10ms of pure scheduling per token. This is the cleanest
optimization target — the engine's contract doesn't require per-layer
submits, only per-token boundary checkpointing. **Workstream W1
(batched per-layer command buffer) should close most of the gap →
projected ~80-100 tok/s.**

**`markov_residual` / `markov_residual_codec` — 26.8 / 27.5 tok/s,
~37 ms/tok. W2 LANDED.** The hot K/V cache eliminates the 80% recompute
overhead measured pre-W2; both engines now sit on the same curve as
`unlimited_context` while preserving the residual-stream contract
(drop `hot_kv` and the next step recomputes from `stored` — the
fallback path is still there for the via_executor path that doesn't
yet capture K/V). The W2 design preserves the engine identity: K/V is
still derivable from residuals; we just don't re-derive every step.

The codec engine being marginally **faster** than the base engine
(27.5 vs 26.8) on a 10-step bench is variance — both run identical
hot-path code, and the codec's bf16 encode/decode only fires at
window-boundary evictions (rare relative to step count). At long
contexts the codec's value re-emerges as memory savings on the
cold tier.

**`turbo_quant` (4-bit) — 20.3 tok/s, 48 ms/tok. FFN dominates; codec
is ~25% of the budget, not the bottleneck.** This is a real surprise:
the pre-profile guess was "codec encode/decode is the inner-loop
killer." Measured: codec is ~25% (9.4% decode + 15.5% encode), FFN is
53%, attention is 20%. Turbo_quant is much closer to unlimited_context
(28.4 tok/s) than to markov_residual (~5 tok/s) — the engine works.
The codec is a fixed overhead per layer per step, not a quadratic
blow-up. **Workstream W3 (incremental encode of the new row only)
still applies — it would cut the 15.5% encode share roughly in half —
but the bigger lever is W1 (dispatch batching), since FFN dominates
the per-step budget and is the same per-layer-Metal bottleneck as on
unlimited_context.** W4 (SIMD WHT) is now lower-priority than originally
estimated; codec is fast enough that vectorising it shaves single-digit
percent.

**`apollo` — requires store, not benched.** Compute model is
fundamentally different: re-forward layers `crystal..num_layers` over
the growing context every decode step. Per-step cost grows linearly
with generated length. At step N: 4 layers × forward over
(N+window_tokens). This is *closer* to no_cache than to standard —
apollo never caches K/V across steps. The bottleneck isn't dispatch or
codec; it's the recomputation model. See workstream W5.

**`no_cache` — by design O(N²).** Not an optimization target;
correctness-baseline only.

#### Optimization workstreams (contract-preserving)

| ID | Workstream | Engines | Expected gain | Risk |
|---|---|---|---|---|
| W1-GPU | **Route per-layer Q/K/V/O and FFN matvecs through Metal.** Today's `attention_decode_step_native` and `ffn_decode_step_native` ignore the backend param and run rayon CPU matvec — that's why all four per-layer engines hit ~27 tok/s on both `--backends cpu` AND `--backends metal`. The GPU is not involved at all. Workstream: make these helpers actually dispatch to `MetalBackend`'s per-layer quant matvec kernels (the ones `fused_prefill` already uses internally). **GPU only.** | unlimited_context, markov_residual, markov_residual_codec, turbo_quant | Unknown — first deliverable is the measurement. Ceiling ranges from ~40 tok/s (submit overhead dominates) to ~80 tok/s (matches standard's GPU advantage). | Per-layer Metal submit cost (50-100µs each × ~6 per layer × 34 layers = ~10-20ms/token) is the open question. May need to batch within a layer (Q+K+V in one buffer, attn separately, etc.) to amortize. CPU is at parity already; no W1-CPU. |
| W2 | **Persistent hot K/V cache in markov_residual.** The engine contract says "K/V derived from residuals" — it does **not** say "recomputed every step." Cache hot K/V across steps; append-1-row on new residual; only recompute fully on cold-tier eviction (rare). Cold-tier compression remains the engine's selling point. | markov_residual, markov_residual_codec | ~20-30×; engine becomes "unlimited_context with compressed-residual cold tier" | Need to verify residual store still reflects "what we'd recompute from" — i.e., consistency check that cached K/V matches a fresh recompute under same residuals. Add a debug assertion mode. |
| W3 | **Incremental TurboQuant encode (append-only).** Only encode the new K/V row each step; keep prior compressed bytes untouched. Decompress only the new row's neighbourhood for attention scores (or the whole layer if simpler). | turbo_quant | ~10× at long context | Re-encoding for in-place updates is the slow path. Need to define when (if ever) the full layer needs re-encoding. |
| W4 | **TurboQuant SIMD WHT + Lloyd-Max.** Already on P1; promote to P0 once W3 lands so the per-row codec cost is the only remaining work. NEON on Apple Silicon, AVX2 on x86_64. | turbo_quant | 2-4× on the codec step | Mostly mechanical; landing W3 first means each step touches less data, making SIMD's batch budget go further. |
| W5 | **Apollo K/V cache across decode steps.** Cache the K/V for layers `crystal..num_layers` between steps; append-1-row per step instead of re-forwarding. Reduces per-step cost from O(N) to O(1) in generated length. | apollo | linear → constant per-step | Apollo's vec_inject perturbation fires at `injection_layer`; verify the perturbation interacts correctly with cached K/V (it should — perturbation is residual-additive, not K/V-overwriting). Needs an apollo store fixture in tree to bench. |
| W6 | **Cache attn dequant for the engine's lifetime, not per-call.** `ensure_attn_tensors_dequantised` already has an idempotency check; verify it's actually one-shot under bench. If it isn't, fix the cache. | all per-layer engines | 5-15% | Mechanical; just instrument and verify. |
| W7 | **Q4K-path engine profiler.** Today `--profile` surfaces a per-stage breakdown for markov_residual's dense path only. The Q4K decode (`rs_decode_step_walk`) doesn't populate `EngineProfiler`. Wire it, then wire the other engines so `larql bench --profile --engine markov-rs:window=512` produces an attribution. Without this, every workstream above is unfalsifiable. | all per-layer engines | 0 (instrument) | Needs to thread `&mut EngineProfiler` through `rs_decode_step_walk`, `process_q4k`, `decode_step_q4k_cpu`. |

#### Sequencing

Recommended order (revised 2026-05-17 night after W7 produced
measured numbers — replaces the earlier guess-driven sequence):

1. **W7 — DONE.** Profiler wired across markov_residual,
   markov_residual_codec, unlimited_context, turbo_quant. Each
   engine's `--profile` output produces a per-stage attribution.
   See the measured table above.
2. **W2 — DONE.** Hot K/V cache landed on `markov_residual` and
   `markov_residual_codec`. Both moved from ~5 tok/s to ~27 tok/s
   (5.5-5.7×) and now sit on the same curve as `unlimited_context`.
   Engine contract preserved: K/V still derivable from residuals,
   just not re-derived every step. Hot K/V state grew from 5.3MB
   to 10.8MB; that's the speed/memory trade. Bit-parity tests
   confirm the cached path matches the recompute path within fp
   rounding.
3. **W1-GPU — route per-layer matvecs through Metal kernels.**
   Per the corrected diagnosis above, the per-layer engines are
   *not* using Metal today — `attention_decode_step_native` and
   `ffn_decode_step_native` ignore their `_backend` parameter and
   call rayon-parallel CPU matvec. The workstream is to plumb
   per-layer Q/K/V/O and gate/up/down matvecs through Metal kernels
   (the same kernels `standard` uses internally during
   `fused_prefill`'s per-layer encode loop) so the GPU actually
   participates in per-layer engines' compute. Each layer becomes
   ~6 Metal submits (Q, K, V, attn, O, gate+up, act+down) per
   token — there's a real question whether the submit cost is
   worth it on Apple Silicon vs the CPU's 27 tok/s ceiling. **W1's
   first deliverable is the measurement, not a single decision:**
   write the per-layer Metal path, bench, and ratchet from there.
   The ceiling could be anywhere from "1.5× the CPU ceiling" (if
   submit overhead dominates) to "3× the CPU ceiling" (matching
   standard's GPU advantage, modulo per-layer dispatch). The CPU
   ceiling is already the M3 Max compute limit — no separate
   "W1-CPU" work to do; CPU is the floor.
4. **W3 — incremental TurboQuant encode.** Lower priority than
   originally thought (codec is ~25% of turbo_quant's budget, not
   80%). Still worth doing — would halve the 15.5% encode share.
5. **W4 — SIMD WHT.** Demoted; codec is fast enough that vectorising
   it shaves single-digit percent. Only worth landing if W3 already
   has and codec is the largest remaining slice.
6. **W5 — Apollo K/V caching.** Largest behavioural change; sequence
   last. Needs an apollo store fixture in tree before bench can
   surface the bottleneck.

#### What this is NOT

- **Not re-introducing fused bypass.** Standard remains the only
  fused engine. Per-layer engines stay per-layer; the goal is to
  make per-layer fast, not to skip it.
- **Not removing engine contracts.** Markov-rs's residual store
  must still be re-deriveable; turbo_quant's K/V must still be
  compressed; unlimited_context's checkpoints must still emit at
  window boundaries. Optimizations are within the contract.
- **Not optimising no_cache.** It's a correctness baseline; O(N²)
  is the design.

#### Guardrails: don't let the bypass come back

The fused-bypass pattern hid for months because nothing asserted
"the engine actually ran." Two invariants we should land before
the optimization work starts, so a future shortcut can't regress
silently:

- **State-policy assertion.** Every engine declares at least one
  invariant that holds iff its state-policy code executed. For
  example:
  - `markov_residual`: `engine.memory_bytes() > 0` after prefill on
    a non-empty prompt.
  - `markov_residual_codec`: same; plus `cold_bytes() > 0` after
    overflow.
  - `unlimited_context`: `archive.len() > 0` after at least
    `window_size` tokens.
  - `turbo_quant`: `layers.len() == num_layers` after prefill.
  - `apollo`: `context_tokens.len() > 0` after prefill.

  Add a `KvEngine::executed_state_policy() -> bool` method (or a
  test-only trait) and assert it in `larql bench` after prefill
  when `--engine` is set. The bench should print a warning if any
  engine reports `false`. This is what would have caught the
  bypass on day one.

- **Per-stage profiler coverage on the Q4K path** (W7 above). Without
  attribution we have no signal when a bypass re-emerges; the engine
  would just look mysteriously fast.

### P0 — engine performance — open after W8.2 (2026-05-18)

The W8/W8.2 alloc-churn fix collapsed the largest decode hot path
cost. The remaining levers are smaller and more scattered. Listed
in expected ROI order.

- **W9 — Single-kernel prefill state-dump.** Step 9 (2026-05-18) made
  prefill iterative (one `fused_decode_step_with_state` per prefill
  token, ~50 ms × N tokens). For N=5 this lands at ~250 ms vs
  `standard`'s ~300 ms fused — already faster on this prompt shape.
  W9 would consolidate into a single Metal kernel call that dumps
  per-position per-layer state for all prefill positions at once,
  saving the ~10 ms × N per-iter setup. Expected wall-time saving:
  ~50 ms / prefill. Small at 5-token prompts; larger at 100+ token
  prompts. **Scope: Metal-kernel surgery in
  `larql-compute-metal/src/decode/mod.rs` — likely a new
  `fused_prefill_with_state` symmetric to `fused_prefill` but with
  the W7 blit-encoder fusion baked in across positions.**
- **W10 — Engine-side state stays on GPU.** Today
  `decode_step_via_dispatch` reads per-layer K/V back into CPU
  `Array2<f32>` to update the engine's `hot_kv` store, then
  `coarse_decode_step_with_state` re-uploads the cache via its own
  K/V buffer on the next step. With engine-side state on GPU
  (`Vec<KvBufferHandle>`), the readback + re-upload pair collapses
  to zero CPU work per step on the dispatch path. The CPU-side
  `Vec<Array2<f32>>` would materialise lazily on `close_window` /
  `info()` calls. Expected: closes most of the remaining 8-11% gap
  to `standard`. **Scope: extends the `KvDispatch::PerLayerDecodeState`
  shape to carry opaque handles instead of `Vec<f32>`; needs a
  matching CPU-side shadow type for `CpuBackend` which has no
  on-GPU state.** Pre-req: stable `MetalBackend`-side KV cache
  invariants (which Step 9 already established).
- **W8.2 → `unlimited_context` CPU walk fallback.** The legacy CPU
  walk path (`process_via_executor` at engine.rs:~720) still uses
  the per-step `Array2::zeros((s_old+1, dim))` pattern. Not on the
  hot path for the bench (dispatch path is the default), but a
  consistency cleanup. Scope: ~10 lines, mirrors W8 mechanically.
- **W11 — Lift W8.2 pattern to `apollo`'s constellation cache.** Not
  measured today (apollo is bench-skipped because it needs a store);
  if/when the on-disk store loader (P1) lands, apollo's per-step
  K/V append would benefit from the same pre-allocation.

### P0 — other correctness / performance

- **`LocalFusedExecutor`.** Phase 2 of the
  [engine-state-vs-execution spec](../larql-inference/docs/specs/engine-state-vs-execution.md)
  needs a fused executor for `standard` + `boundary_kv` to migrate
  without losing Metal fast path performance. Open design question
  (spec §9): `KvHandle` opaque cache vs `SharedKV` tuple for fused
  executor's return shape. Probably needs sibling methods on the
  `LayerExecutor` trait (`run_prefill_fused` / `run_decode_step_fused`)
  with default-None for per-layer executors.
- **`BoundaryKvEngine::resume`.** Spec §6.3 describes restoring from a
  frame chain via `MarkovResidualEngine::recompute_kv`. The frame
  emission half is shipped; resume is not. Restore-parity test fixture
  needed (capture frame, verify first-5-tokens agreement under
  `D-@high`).
- **D-METAL-PLE** *(carries from larql-compute roadmap)*: Per-Layer
  Embeddings not implemented in Metal. Engines on Gemma 4 E2B fall through
  the deliberate CPU fallback in `gpu.rs:372-374`, costing ~30× decode.
  Fix is a 1-2 day Metal port of `forward/ple.rs`. Engines themselves are
  PLE-agnostic; the gain accrues through the shared `decode_token` Metal
  path.
- **Engine-level profiler coverage.** *(See W7 above — this is now
  the unblocker for the entire P0 performance workstream.)* Today
  `markov_residual`'s dense path (`rs_decode_step_profiled`)
  populates `EngineProfiler`, but the Q4K decode path
  (`rs_decode_step_walk`) does not, and the other engines never
  populate it at all. Without per-stage attribution on the Q4K
  path, the per-engine optimization workstreams (W1-W6) are
  unfalsifiable. Wire it before starting W1.

### P0 — sibling trait extraction for non-K/V engines (Apollo, Mode 5)

**Problem.** The `KvEngine` trait surface assumes per-step K/V append,
FFN dispatched through `FfnBackend`, and state reconstructible to
K/V tensors. Apollo violates all three (`engines/apollo/engine.rs`:
`_ffn` unused, `decode_step` re-forwards full `context_tokens` each
call, state is residual delta + boundary residual + token list — no
K/V). Mode 5 / Graph-Grounded will violate the same three when it
lands. The trait's `Option<T>` return type also collapses
semantically distinct outcomes — empty prompt, backend unavailable,
retrieval miss, internal error, decode-before-prefill invariant
violation — into a single `None` the harnesses route incompatibly:
`accuracy_suite/runner.rs` silently drops via `filter_map` (Apollo's
store-miss prompts disappear from the JSON, structurally shorter
result vector than other engines), while `engine_runtime.rs` aborts
with `"engine prefill failed"` on the same `None`. Same trait method,
two semantics, neither implements the spec's documented
`fallback_mode = standard` from
[`docs/state-policy.md`](docs/state-policy.md) §3.

**Resolution.** Extract a `RetrievalEngine` (or `QueryEngine`) sibling
trait that drops the per-step K/V append assumption and the
`FfnBackend` dispatch requirement. Apollo moves to it; Mode 5 lands
on it directly. Tighten both trait return types from `Option<T>` to
`Result<T, EngineError>` with a typed error enum so the two harnesses
agree on a single taxonomy and downstream consumers can route on
error kind. Harness dispatch goes through an `AnyEngine::{Kv,
Retrieval}` enum that branches once at construction.

**Scope (atomic — six touchpoints).** Partial application is worse
than no application; a half-refactored trait surface has three
disagreeing semantics instead of two.

1. New `RetrievalEngine` trait. `Apollo` impl moves from `KvEngine`
   to `RetrievalEngine`. Internal behaviour unchanged.
2. `KvEngine::prefill` / `decode_step` (and `*_quant` / `*_via_executor`
   variants) return type changes from `Option<T>` to
   `Result<T, EngineError>`. **All eight `KvEngine` impls touched** —
   `standard`, `no_cache`, `markov_residual`, `markov_residual_codec`,
   `unlimited_context`, `turbo_quant`, `boundary_kv`,
   `boundary_per_layer` — not just the one that motivated the
   refactor. The translation is mechanical: validated on three
   structurally-distinct samples (`markov_residual` for arch
   preconditions, `unlimited_context` for window boundaries,
   `boundary_per_layer` for calibration stores); every `None`-return
   in those engines maps cleanly to `InternalError(...)`. The
   remaining five are variations on already-validated patterns
   (`standard` / `no_cache` are simpler; `markov_residual_codec` /
   `boundary_kv` extend already-sampled families; `turbo_quant`'s
   destructive-codec failure modes are in-contract per
   state-policy §3 worked example and don't surface as `None`).
3. `AnyEngine::{Kv, Retrieval}` dispatch enum at harness boundary.
   Construction parses to one or the other; execution branches once.
4. Accuracy harness (`accuracy_suite/runner.rs`,
   `larql-cli/src/commands/primary/accuracy_cmd.rs`): per-error-kind
   handling replaces `filter_map`; miss-rate surfaces as a first-class
   `served_rate` column inseparable from `match_rate`.
5. Bench harness (`engine_runtime.rs`): distinguish recoverable from
   internal errors; recoverable misses log a skip note but don't
   abort the whole run.
6. `LayerEngine` / `ZoneEngine` (per
   [`layer-engine.md`](../larql-inference/docs/specs/layer-engine.md),
   [`zone-engine.md`](../larql-inference/docs/specs/zone-engine.md))
   consume `AnyEngine` rather than `Box<dyn KvEngine>`.

**Three findings from the validation pass that constrain the design.**

1. **Interim `ffn_backend` JSON limitation (until this refactor lands).**
   Item 1's schema fix (predecessor PR — see "Predecessors" below)
   reports `ffn_backend` as the value passed at engine construction,
   *not* the FFN backend actually used during the run. For engines
   where the trait method dispatches to multiple internal paths with
   different FFN usage (`markov_residual`'s CPU path uses `_ffn`;
   its `*_via_executor` path uses `ffn` — same engine, same trait,
   different ffn-honoring), the reported value may not reflect which
   backend actually executed. **Downstream consumers should not
   condition on `ffn_backend` for engines where this distinction
   matters until this refactor lands.** The fix falls out naturally
   from the typed `Result` carrying path information; deferring to
   the refactor preserves Item 1's 200-300 line scope.

2. **`InternalError` sub-taxonomy is load-bearing for production
   observability — required design decision, not discretionary.**
   "decode_step called before prefill" (`markov_residual::engine.rs:103`,
   `boundary_per_layer::engine.rs:184`, others) is structurally
   different from "the inner backend returned None for an opaque
   reason." The first indicates a harness-level dispatch bug that
   wants immediate investigation; the second indicates a runtime data
   condition that wants diagnostic logging. Collapsing both into a
   single `InternalError` makes production logs unable to distinguish
   these alerting categories. **Recommend splitting `EngineError`
   into `InvariantViolation { what: String }` and
   `BackendFailure { details: String }` as two top-level variants**
   (not a sub-tag under a single `InternalError`). This is the
   trait-extraction PR reviewer's first design call.

3. **Extensibility note — the four-variant enum is not a permanent
   ceiling.** Currently-invisible failure modes — `unlimited_context`'s
   "request crossed an uncheckpointed window boundary" (collapsed
   into generic `process()` None today), `boundary_per_layer`'s
   "calibration record missing for policy fingerprint" (a
   construction-time `.expect()` panic at `lib.rs:362` today) — are
   real conditions the typed `Result` surface *enables* surfacing
   without further trait changes. The starting enum
   (`EmptyPrompt`, `BackendUnavailable`, `RetrievalMiss { reason }`,
   `InvariantViolation`, `BackendFailure`) is the minimum-honest
   shape, not a commitment that the taxonomy is closed. New variants
   are deliberate schema changes — exhaustive enum, breaking changes
   on extension, no `#[non_exhaustive]`. Defaulting new variants
   into existing arms reproduces the silent-drop problem one layer
   down.

**Blocks.** Item 5 in the conversational priority queue (Mode 5 /
Graph-Grounded engine wiring). Mode 5 lands as a `RetrievalEngine`
impl once this refactor is in; its canonical state (retrieval graph
+ token archive) is already accommodated by
[`docs/state-policy.md`](docs/state-policy.md) §2.1's open-list of
canonical state kinds.

**Predecessors.** Item 1 schema fix (`ScoreOutcome` enum +
`served_rate` column in `accuracy_suite/runner.rs`) ships as interim
diagnosability. Its `ScoreOutcome` variants mirror the eventual
`EngineError` enum so migration is a flat projection when this
refactor lands; the field stops being interim, only its construction
path moves from harness-side `match` to engine-side `Result`.
Item 3 (Apollo into `larql accuracy` coverage) is safe to land once
Item 1 ships, but its rows will only be properly diagnosable after
this refactor.

**Closes.** [`docs/state-policy.md`](docs/state-policy.md) §8 Open
Question 1 ("Where does Apollo's fallback live? Two engines stacked
or one engine with `fallback_mode = standard`?"). The state-policy
patch declaring Q1 resolved lands in the same PR as this refactor —
patching the spec to mark Q1 resolved while the harnesses still
disagree would reproduce the same category of error the spec already
commits with `fallback_mode = standard` (documenting intent as if it
were implementation).

### P1 — capability extensions

- **Complete the FFN policy harness arc.** Item 2 v0
  (`FfnBackendKind` + `FfnLayerPolicy` parser + `ValidatedFfnLayerPolicy`
  + `BoundFfnRouter`) shipped 2026-05-24 along with the
  cross-product accuracy harness (see "Closed (recent)"). Three
  follow-ons remain, all blocked on either the sibling trait extraction
  or the `RemoteWalk` build path landing:
  - **Q4K `--ffn-policy` honoring.** `run_engine_q4k` in
    `larql-cli/.../bench/engine_runtime.rs` accepts the flag but
    logs "not yet honored" and uses the engine's internal Q4K
    routing. Honoring it requires the Q4K dispatch trait to take
    `&ModelWeights` instead of `&mut ModelWeights` so a
    `BoundFfnRouter` (which holds `&weights`) can coexist with the
    engine call. Naturally folds into the sibling trait extraction
    (P0 above) since that overhauls the trait surface anyway.
  - **`RemoteWalk` build path.** `FfnBackendKind::RemoteWalk` parses
    but errors with `RemoteWalkNotYetWired` in `build_router`. Wiring
    needs the `RemoteWalkBackend` connection pool plumbed through
    the build path. Slice estimate: ~200 lines.
  - **Bench `--ffn` URL/policy flag unification.** Bench keeps two
    flags today: `--ffn <URL>` (legacy, selects the remote-FFN
    bench scenario via `run_concurrent_ffn`) and `--ffn-policy <SPEC>`
    (new, selects engine-internal FFN backend). Once `RemoteWalk`
    builds work, `--ffn http://x:8080` can become sugar for
    `--ffn-policy remote-walk:endpoint=http://x:8080` and the two
    flags merge. Until then they stay separate. Documented in
    `engine_runtime.rs:run_engine_q4k` and the `--ffn-policy` doc
    comment in `bench/args.rs`.
- **Wire `--ffn http://...` through the executor surface.** The
  existing `--ffn` flag uses `run_concurrent_ffn` (separate path that
  routes through the `larql-metal` reference, not the engines). Once
  the four remaining engines (P0) are on `*_via_executor`, the bench
  should be able to compose `--engine markov-rs-codec:window=512
  --ffn http://shard:8080` and have the codec engine drive remote FFN
  with bounded local memory. Spec §7 calls this out as a primary use
  case.
- **Auto-rewind variant of `boundary_kv`.** Discussed mid-session as the
  only way to combine Metal's fast-path tok/s with bounded memory: emit
  boundary frame every N chunks, reset Metal's K/V cache, re-prefill
  from the last frame. Bounded memory at ~99% of fast-path tok/s with
  periodic re-prefill spikes. Would need an `evict_after_chunks` config
  on `BoundaryKvEngineConfig` plus a `backend.reset_kv_cache()` call
  after the capture. *Note (post 2026-05-17 bypass strip): this is a
  cleaner alternative to per-layer engines for "bounded memory at
  fused speed" — explicitly composes with standard rather than
  bypassing into it. Should benchmark against the W2-optimised
  markov_residual to see which model wins for long-context decode.*
- **Per-layer codec calibration sweep harness.** `BoundaryPerLayerEngine`
  ships with `BoundaryCalibrationStore` trait + `InMemoryCalibrationStore`,
  but the actual sweep tool that populates records (per-layer fragility
  measurement → policy generation → end-to-end KL validation) is not in
  tree. Per spec Phase 1 of
  [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- **Page-aligned KV slabs for `unlimited_context`.** The current
  `CheckpointStore` uses owned `Vec<f32>` per layer per checkpoint; a
  hugepage-backed slab would cut allocation churn and improve thermal
  steadiness during 370K-token replays.
- **Apollo store on disk.** `apollo` currently expects an in-memory
  `ApolloStore`. Add an mmap loader that reads the constellation map +
  boundary residuals from the same vindex-style on-disk layout as
  `down_meta.bin`, so apollo can serve ~10⁵-entry stores without RAM cost.
- **TurboQuant SIMD packing.** The Lloyd-Max codec works at scalar f32
  today; the rotation step is amenable to NEON / AVX2 vectorisation.
  *(Now also W4 in the P0 performance workstream — promote to P0 once
  W3 (incremental encode) lands so the per-row codec cost is what's
  left to vectorise.)*

### Falsified hypotheses / closed investigations (don't re-litigate)

- **`build_pipeline_layers` per-step vtable cost** — falsified
  2026-05-18 via samply flamegraph. Hypothesised as the cause of
  `standard`'s 105.9 → 99.4 regression after the kv_dispatch
  refactor; actual flamegraph showed `__bzero` +
  `zip_mut_with_same_shape` + `madvise` as 58% of CPU on per-layer
  engines (allocation churn, not dispatch overhead). The ~6 vtable
  indirections × 34 layers per step is real but ns-scale, not
  meaningful.
- **`let index = index?;` early-return branch cost** — same
  falsification. Branch is one ns-scale prediction; would not show
  as a measurable hot path.
- **`Option<&dyn KvIndex>` fat-pointer spill** — same falsification.
  Register spill is ns-scale; flamegraph showed memory operations
  not spill-related code paths.
- **`Map<I,F>::fold` 13.2% of CPU** — investigated 2026-05-18, traced
  via two-hop parent attribution to
  `larql_vindex::format::weights::load::embeddings::load_embeddings`
  → `decode_f16` of the 256K × 3072 × 2-byte embedding table. **This
  is load-time cost, not decode-time.** Visible in the profile only
  because samply records the full process lifetime; not actionable
  for the decode hot path. Don't re-investigate Map::fold as a
  decode hot-path lever.
- **`synthesize_lm_head_kquant` 19% of CPU on first profile** — same
  attribution: load-time only. The 50-tok profile had high load:decode
  ratio; at 1000 tokens it drops to 5%. Not a decode-hot-path issue.

### Investigation tooling

- **samply + `/tmp/symbolize.py` + `/tmp/symbolize_callers.py`.** The
  cargo-flamegraph-equivalent stack on this machine. Setup steps:
  1. Add `[profile.release] debug = "line-tables-only"` to root
     `Cargo.toml`. **Remember to revert before shipping** — release
     binaries bloat ~3× with line tables.
  2. `samply record --save-only --unstable-presymbolicate -o
     /tmp/profile.json --no-open -- target/release/larql bench
     gemma3-4b-q4k-v2 --tokens 1000 --engine <spec>`
  3. `python3 /tmp/symbolize.py` for top-N self-times.
  4. `python3 /tmp/symbolize_callers.py "<symbol-fragment>"` for
     two-hop call-stack attribution of generic frames.
  5. For decode-only profiles, use `--tokens 1000` so decode
     dominates over prefill / load.

### P2 — research / sequencing

- **Non-`Bf16` codecs in `markov_residual_codec`.** v0.1 ships `Bf16`
  only as the safely-defaultable cold codec. `Int8Clip3Sigma`,
  `AdaptiveBlockG32`, `PerGroupInt4G128` are present in `larql-boundary`
  but Exp 46 showed mid-layer failure for `Int8Clip3Sigma`. The
  per-architecture calibration sweep (P1) gates their promotion to
  defaults. Until then, `BoundaryPerLayerEngine` with a custom policy
  is the way to use them.
- **`MarkovResidualCodecEngine` cold tier on actual Q4K deployments.**
  Bench results confirm 50% cold tier saving on dense models and on
  Q4K Gemma with `--via-executor`. Production deployment scenario:
  long-context decode (10k+ tokens) on a 64 GB consumer Mac with a
  large model — the codec's bf16 cold tier is the difference between
  fits-in-RAM and OOM. No technical work blocking this; needs a
  recipe / docs.
- **Cross-engine comparator.** Today `larql bench --engine <spec>` runs one
  engine at a time and `benches/engine_decode.rs` exercises Standard vs the
  parity oracle. The synthesis question is: which engine wins for which
  prompt regime (long-context QA vs short-prompt multi-turn vs streaming
  generation)? A criterion harness sweeping prompt length × decode length ×
  batch size against the production `KvEngine` impls would surface this —
  the retired `kv-cache-benchmark::kv_strategies` synthetic comparator
  measured the wrong thing (encode/decode of random vectors, not real
  decode steady-state).
- **Compositional engines.** `apollo + turbo_quant` would put quantised
  K/V inside the boundary windows; `markov_residual + apollo` would let
  the residual recompute path read pre-projected boundary residuals.
  `markov_residual_codec + boundary_kv` would give bounded cold +
  cross-session resume. Neither is wired today; the trait already
  supports composition because engines hold the persistent state, not
  the dispatch — but the executor + state-policy separation (Phase 2
  spec) makes composition cleaner.

## Closed (recent)

- **2026-05-24 — Accuracy harness honesty + FFN policy cross-product
  LANDED.** Multi-PR arc that turns the accuracy suite from "silent
  drop on engine miss" into a discriminating cross-product harness:

  - **Item 1 — accuracy schema fix** (commit `07684457`).
    `ScoreOutcome` enum (exhaustive, flat-tagged serde, mirrors the
    future `EngineError` taxonomy). `PromptScore` / `ConflictScore`
    gain `outcome` field + `Option<T>` score payload with
    `served()` / `skipped()` constructors enforcing
    correlated-optionality. `StrategySplit` gains `*_served` +
    `*_served_rate` per axis as required-companion fields to
    `*_match_rate`. `compute_strategy_split` filters on served subset
    (counting skips as zero would punish honest reporting). Replaces
    `filter_map` silent-drop in all three drivers. Surfaces Apollo's
    store-miss rows as `SkippedRetrievalMiss` instead of dropping.
    `EngineKind::supported_names()` replaces hard-coded six-engine
    error string at two bench sites.

  - **Item 2 v0 — `FfnBackendKind` parser + `FfnLayerPolicy`
    (in `larql-inference::ffn_policy/`).** New crate-shape:
    `FfnBackendKind` (Dense / Walk{k} / RemoteWalk / Null),
    `RoutingPredicate` (All / Layers / Otherwise), `FfnLayerPolicy`
    with from_spec parser supporting per-layer routing
    (`{walk:k=100}@layers=14-27;{dense}@otherwise`).
    Construction-errors on overlapping ranges; exhaustive enums;
    typed error taxonomy (`PolicyParseError` /
    `PolicyValidationError`). Module lives in `larql-inference` not
    `larql-kv` — FFN policy is the FFN axis, not the KV axis.

  - **`build_router` slice — `ValidatedFfnLayerPolicy` newtype +
    `BoundFfnRouter`.** Type-system enforcement of "validate before
    build" via non-public constructor. `BoundFfnRouter<'a>` owns its
    backend instances (`Vec<Box<dyn FfnBackend + 'a>>`) so callers
    don't manage backend lifetimes alongside the router's. `impl
    FfnBackend for BoundFfnRouter` delegates per-layer via the
    trait's existing `layer: usize` parameter — drop-in for the
    `&dyn FfnBackend` surface every engine already takes. Design
    rationale: `larql-inference/docs/ffn-build-router.md`.

  - **Cross-product harness + typed axis columns.** `accuracy_cmd`
    iterates `kv_engine × ffn_backend` cross-product via
    `FfnLayerPolicy::split_specs` (comma-separated, brace-aware,
    re-parse fallback for kv-comma forms like
    `remote-walk:endpoint=X,wire=Y`). New `EvalLabels<'a>` struct
    bundles `(kv_engine, ffn_backend, strategy)` for clean signatures.
    `PromptScore` / `ConflictScore` / `StrategySplit` gain explicit
    `kv_engine: String` + `ffn_backend: String` columns alongside
    `strategy`. `format_strategy_split` grows a two-axis layout
    (`KV engine` + `FFN backend` columns) when any row has
    `ffn_backend != "dense"`; default no-`--ffn` runs keep the
    historical single-`Strategy`-column layout. Closes the
    interim-`ffn_backend`-as-user-input limitation noted in Item 1's
    ROADMAP entry.

  - **CLI wiring.** `larql accuracy --ffn dense,walk:k=100,'{walk:k=100}@layers=14-27;{dense}@otherwise'`
    now runs the cross-product in one invocation. Vindex loaded
    lazily — only when a Walk binding is present.
    `larql bench --ffn-policy <spec>` honors the policy on the
    non-Q4K (CPU) path; Q4K path accepts the flag but doesn't
    honor it yet (P1 follow-on above).

  - **Apollo into accuracy default engines.** `--engines` default
    now includes `apollo`. The schema fix above means Apollo's
    store-miss rows show `served_rate < 1.0` rather than silent
    drops — diagnostic rather than misleading.

  - **Module splits.** `accuracy_suite/runner.rs` (2050 lines) split
    into `accuracy_suite/runner/` folder (6 files: `types` /
    `scoring` / `drivers` / `aggregate` / `legacy` / `mod`). Same
    pattern that produced the `ffn_policy/` folder split in
    `larql-inference`.

  - **Coverage lift across 5 engine files.** Pre-existing engine
    internals had drifted below 90%. Lifted with synthetic-weights
    + CPU-backend tests: `boundary_per_layer/cold_tier.rs`
    (88→100%), `executor.rs` (85→90.6%), `walk.rs` (84→95%),
    `engine.rs` (83→90%), `markov_residual/store.rs` (86→99.6%).
    `markov_residual/compute.rs` partially lifted (81→86.85%);
    full lift gated on `serial_test` for env-var paths.
    Discovered the gate had been passing against a stale JSON —
    fresh `make larql-kv-coverage-summary` is now required to
    surface debt. See "Coverage debt" section above for the
    remaining 6 files.

  Test deltas across the arc: larql-kv lib 595 → 663 (+68),
  larql-inference lib 1086 → 1102 (+16). Zero regressions. Clippy
  clean. Aggregate ~3,500 lines of code + tests added across
  `larql-kv` and `larql-inference`.

  ROADMAP entry for the sibling trait extraction (P0 above)
  references "Item 1 in the conversational priority queue" — Item 1
  is the schema fix above. Mode 5 work is still gated on that P0
  refactor landing.

- **2026-05-18 — W8.2 (doubling-capacity K/V in `markov_residual` +
  `markov_residual_codec`) LANDED: 2.4× decode speedup at 1000 tokens.**
  Lifted the W8 pre-allocation pattern from `unlimited_context` to the
  two unbounded-window engines. Since `max_window=None` rules out a
  fixed pre-alloc, both stores now use a doubling-capacity strategy
  via three private helpers in each engine:
  - `window_capacity(prompt_len, window_size)` — initial cap is
    `max(window, prompt_len)` if windowed, else
    `max(prompt_len * 2, 64)`.
  - `grow_capacity_2d(src, len, cap)` — allocate `[cap, cols]` once
    at prefill, copy the prefill rows in.
  - `append_row(buf, row, len)` — in-place `slice_mut(s![len..len+1,
    ..]).assign(row)` when `len < cap`; otherwise double capacity,
    copy the live rows, then assign. Amortised O(1) per append vs the
    O(n) per step the previous `Array2::zeros((n+1, dim))` pattern
    paid.

  Store changes (both `RsStore` and `RsStoreCodec`):
  - New `pub hot_len: usize` field — logical row count, separate from
    `stored[l].shape()[0]` (which is now capacity ≥ hot_len).
  - `window_tokens()`, `memory_bytes()`, `clip_layer` /
    `clip_layer_overflow` updated to use `hot_len`.
  - New `finalise_hot_len_after_clip()` — must be called after every
    per-layer clip loop. (Subtle bug fix during impl: setting
    `hot_len = window` *inside* the per-layer loop made layers 2..N
    see `rows == window` and skip their clips, dropping half the
    cold-tier payload. Two existing tests caught this.)

  Bench (Gemma 3 4B Q4K, Metal, M3 Max):
  - **1000-tok**:
    - `markov-rs`: 24.8 → **58.7 tok/s (+137%)**
    - `markov-rs-codec`: 25.7 → **57.2 tok/s (+123%)**
    - `unlimited-context`: 49.5 → **57.4 tok/s (+16%)** (variance
      recovery from previous run + sympathy from the codepath audit)
    - `standard` unchanged at 64.1 (untouched)
  - **50-tok**:
    - `markov-rs`: 77.1 → **88.9 tok/s (+15%)**
    - `markov-rs-codec`: 77.5 → **88.8 tok/s (+15%)**

  All three cached-state engines now cluster within 11% of standard's
  64.1 tok/s ceiling at 1000 tokens. The doubling-capacity scales
  linearly with seq_len: at 50 tok the saved alloc bytes are small
  (~400 KB/step); at 1000 tok they're ~8 MB/step. The 137% win at
  long context is the alloc churn that pre-W8.2 was hiding behind
  prefill cost.

  CPU walk + executor fallback paths (`rs_decode_step_walk`,
  `rs_decode_step_codec_walk`, `process_via_executor`) still allocate
  per step — they're not on the hot path for the bench. Defensive
  consistency: every legacy RsStore/RsStoreCodec constructor sets
  `hot_len` from `stored[0].shape()[0]` so non-dispatch paths see a
  consistent invariant.

- **2026-05-18 — Step 9 (iterative Metal `coarse_prefill_with_state`)
  LANDED: ~10× prefill speedup on every state-dump engine.**
  Pre-Step 9, `MetalBackend::coarse_prefill_with_state` defaulted to
  the trait's `coarse_prefill` (no per-layer state dump); engines saw
  `state.is_complete_for() == false` and fell back to the CPU walk
  (~2.7 s on Gemma 3 4B). The new impl pre-allocates `[seq_len,
  hidden]` and `[seq_len, kv_dim]` per layer (W8-style alloc at
  source for prefill too), resets + preallocates the Metal K/V cache,
  then iterates `fused_decode_step_with_state` per prefill token,
  writing the dump into the pre-allocated row position.

  Bench (Gemma 3 4B Q4K, Metal, M3 Max, "The capital of France is",
  5 prefill tokens):
  - `markov-rs` prefill: 2757 → **254 ms** (10.9×)
  - `markov-rs-codec` prefill: 2564 → **249 ms** (10.3×)
  - `unlimited-context` prefill: 2760 → **256 ms** (10.8×)
  - `turbo-quant` prefill: 2750 → **334 ms** (8.2×)

  Predicted ~45× (5 × 12 ms decode time) didn't materialise because
  each iterative `fused_decode_step_with_state` carries per-token
  state-dump readback overhead. Remaining ~250 ms is 5 × ~50 ms
  per-iter + fixed setup. Further closure needs a single-kernel
  prefill that dumps state for all positions in one shot — separate
  Metal-kernel surgery.

  Decode steady-state also moved (W8 + Step 9 compound):
  - `unlimited-context`: 82.7 → **89.2 tok/s** (fastest cached-state
    engine; within 10% of `standard`'s 99.2 ceiling)
  - `markov-rs`: 75.3 → 77.1 tok/s
  - `markov-rs-codec`: 79.0 → 77.5 tok/s

- **2026-05-18 — W8 (pre-allocated K/V buffer in `unlimited_context`)
  LANDED: 58% of decode-CPU alloc churn removed.**
  samply flamegraph on `unlimited_context:window=1024 --tokens 1000`
  (post-W7) surfaced an unexpected hot path: 21% `__bzero` + 19%
  `ndarray::zip_mut_with_same_shape` + 18% `madvise` = **58.5% of
  main-thread CPU** spent on `Array2::<f32>::zeros((n+1, kv_dim))` +
  `slice_mut().assign(k_old)` + `slice_mut().assign(k_new_row)`
  inside `decode_step_via_dispatch` — 68 allocations per token
  (34 layers × 2), each growing linearly with `n`.

  Fix: pre-allocate `Array2::zeros((window_size, kv_dim))` per layer
  once at prefill (in `try_prefill_via_dispatch`), track a single
  `current_window_kv_len: usize` counter, and append in the hot path
  via `slot.0.slice_mut(s![pos..pos+1, ..]).assign(k_new_row)`. One
  small `kv_dim`-sized copy per layer per side, zero alloc per step.
  Readers (`close_window`, `current_kv_bytes`) updated to use the
  counter instead of `k.shape()[0]`; CPU walk fallback paths set the
  counter defensively from the returned narrow-array shape.

  Bench (Gemma 3 4B Q4K, Metal, M3 Max):
  - 50-tok: `unlimited-context:window=256` 82.7 → **86.6 tok/s
    (+4.7%)** vs `standard`'s 99.4 (gap closed ~50%)
  - 1000-tok: `unlimited-context:window=1024` 17.39 ms vs `standard`'s
    15.74 ms → 1.65 ms gap (vs pre-W8 estimated 5-10 ms slope from
    `Array2::zeros((n+1, …))` growing linearly with `n`)

  Post-W8 flamegraph: the `__bzero` / `zip_mut_with_same_shape` /
  `madvise` triple is **gone from the top-20**. Remaining main-thread
  CPU is dominated by `__psynch_cvwait` (Metal GPU wait,
  irreducible), `synthesize_lm_head_kquant` (prefill — separate
  ~2.5 s regression flagged elsewhere), and generic `Map::fold`.

  The optimisation is engine-local (`larql-kv/src/engines/unlimited_context/engine.rs`)
  with no surface change. Same pattern can be lifted to
  `markov_residual` / `markov_residual_codec` / `turbo_quant` once
  their state-policy shape is clarified — they use the same
  `Array2::zeros((n+1, kv_dim))` pattern but have unbounded windows
  by default, so the pre-allocation needs a growable strategy
  (doubling-capacity Vec-style) rather than fixed window size.
  Tracked as W8.2 candidate.

- **2026-05-18 — W7 (blit-encoder fusion) LANDED: per-layer commit
  overhead removed; +30-48% across cached-state engines.**
  Modified `decode_token_with_moe_split_fn` in
  `larql-compute-metal/src/decode/mod.rs` to pre-allocate per-layer
  staging buffers (k / v / h-in) when `state_dump` is `Some`. The
  layer loop blits `k_out` / `v_out` / `h_buf` into the staging
  buffers inside the same command buffer (`new_blit_command_encoder`
  + `copy_from_buffer`) instead of forcing per-layer commit + wait +
  CPU read. The single final commit at the bottom of the function
  flushes everything; reads happen once after that, draining staging
  into `state_dump`. Metal's command-buffer encode ordering
  guarantees blit reads see the settled compute writes.

  Measured (Gemma 3 4B Q4K, Metal, M3 Max):
  - `standard` (control, no state_dump): 105.9 → 99.4 tok/s (noise)
  - `markov-rs`: 58.0 → **75.3 tok/s (+30%)**
  - `markov-rs-codec`: 58.4 → **79.0 tok/s (+35%)**
  - `unlimited-context` (window=256): 56.0 → **82.7 tok/s (+48%)**
  - `turbo-quant` (4-bit, 10-tok bench): 33.0 → **37.7 tok/s (+14%)**

  Engine-cost decomposition post-W7: ~10 ms Metal kernel compute +
  ~3 ms CPU glue. The remaining gap to `standard`'s 99 tok/s is
  pure CPU-side state-update work (state Vec→Array2 conversion,
  appends). Closure path: in-place state updates / pre-allocated
  buffers (W8 candidate).

  Edge cases worth noting:
  - `standard` doesn't touch state_dump → blit branch is dead code
    → 0× regression confirmed.
  - `turbo_quant`'s codec inner loop is the dominant per-token cost;
    the saved 1.7 ms commit overhead is a smaller fraction.
  - The `unlimited_context` +48% win reflects its lighter post-
    kernel CPU work (just append to `current_window_kv`); engines
    with heavier post-kernel work see smaller relative gains.

- **2026-05-17 night — W1-GPU steps 4 + 6 LANDED: unlimited_context +
  turbo_quant now route through dispatch on Metal.**
  Same pattern as steps 5: each engine gains `try_prefill_via_dispatch`
  / `decode_step_via_dispatch` helpers that read per-layer captured
  state and update engine-specific state policy.
  - **turbo_quant**: state.k_new/v_new per layer feeds the
    WHT+Lloyd-Max codec via `CompressedLayer::compress` (prefill)
    and decompress→append→recompress (decode). Bench: **19.6 →
    33.0 tok/s (+68%)** on Metal. Memory stays at 0.6 MB hot
    (compression intact).
  - **unlimited_context**: state.k_new/v_new appends to
    `current_window_kv` per layer; window auto-close at
    `window_size` tokens fires the legacy `close_window` checkpoint
    emit. Bench: **28 → 56.0 tok/s on Metal (+98%)** at
    `window=256` (Gemma 3 4B, M3 Max, 50-token decode). Hot state
    15.7 MB tracks the engine-side window shadow (see KvHandle
    eviction note below).

  Engine memory note: with W1-GPU active, the backend's internal K/V
  cache grows unboundedly alongside each engine's shadow state. This
  defeats the memory benefit of `unlimited_context` /
  `markov_residual_codec` at long contexts. Follow-up: expose a
  `KvHandle::evict_oldest(n)` API on `KvDispatch` so engines can
  bound the backend cache to match their window.
- **2026-05-17 night — W1-GPU step 2 LANDED: Metal per-layer state
  dump → 2.1× decode speedup on markov-rs + codec.**
  Modified `decode_token_with_moe_split_fn` in
  `larql-compute-metal/src/decode/mod.rs` to accept an optional
  `state_dump: Option<&mut DecodeStateDump>` parameter. When active,
  the layer loop:
  1. At top of layer L: pushes `x` (for L=0) or reads `h_buf` (for
     L>0, settled by the previous layer's commit) into
     `state.h_in_per_layer`.
  2. At bottom of layer L: forces `enc.end_encoding()`, `cmd.commit()`,
     `wait_until_completed()`, reads `k_out` / `v_out` (scratch
     buffers reused across layers) into
     `state.k_new_per_layer` / `v_new_per_layer`, then restarts
     command buffer + encoder for the next layer.

  Trait wiring: new `DecodeBackend::decode_token_with_state_dump`
  method (default falls back to plain `decode_token`); MetalBackend's
  trait impl routes through the new kernel function when `state` is
  `Some`. Inference layer adds `fused_decode_step_with_state` +
  `MetalBackend::coarse_decode_step_with_state` /
  `coarse_prefill_with_state`. Engines (markov_residual, codec)
  inherit the Metal acceleration automatically — no engine-side
  changes from step 5.

  Measured (Gemma 3 4B Q4K, Metal, M3 Max, 10-token decode):
  - `markov-rs`: 27.0 → **57.7 tok/s** (+114%)
  - `markov-rs-codec`: 27.8 → **57.5 tok/s** (+107%)
  - `standard` (fused control): 100.8 tok/s (unchanged)

  Per-token cost: ~17 ms = 10 ms Metal compute + ~1.7 ms commit
  overhead (50 µs × 34 layers) + ~5 ms engine state update / CPU
  glue. The remaining gap to standard's 100 tok/s is the
  per-layer commit cost; a follow-up could use blit-encoder
  switches inside a single command buffer to eliminate the
  commit overhead and lift toward 80-100 tok/s.

  Prefill cost: ~2.8 s on Metal (CPU walk for state seeding +
  Metal `fused_prefill` for backend cache). One-shot; doesn't
  affect decode steady-state. Future optimisation: per-position
  per-layer K/V dump on the Metal prefill side to skip CPU walk.
- **2026-05-17 night — W1-GPU infrastructure (decode trait surface +
  CPU impl + engine wiring; Metal kernel modification deferred).**
  Three layered changes landed end-to-end:
  - **Trait surface (`KvDispatch`):** new `coarse_prefill_with_state` /
    `coarse_decode_step_with_state` methods take
    `Option<&mut PerLayerDecodeState>`. Default impls delegate to the
    non-state variants, so unmigrated backends keep working.
  - **`DecodeBackend` trait + `DecodeStateDump` struct** added in
    `larql-compute` for the substrate-level surface. Same default-
    delegation pattern.
  - **CPU implementation** (`predict_kquant_prefill_with_state` /
    `predict_kquant_decode_step_direct_with_state`): threads per-layer
    state capture through the existing per-layer walk at zero
    re-compute cost. Parity test in
    `kv_dispatch::cpu::coarse_decode_step_with_state_populates_and_matches_plain`
    asserts cached and non-cached outputs match within f32 rounding
    and per-layer shapes (`[1, hidden]`, `[1, kv_dim]`) are correct.
  - **Engine wiring** for `markov_residual` and
    `markov_residual_codec`: `try_prefill_via_dispatch` /
    `decode_step_via_dispatch` route through the new
    `coarse_*_with_state` API when the backend implements it. State
    capture feeds `RsStore::stored` (residuals) and `hot_kv` (W2
    cache) in a single backend call. Legacy walk path stays as the
    fallback when state isn't populated (e.g. on backends that
    haven't migrated yet — currently `MetalBackend`). Gated on
    `supports_direct_matvec_decode` so non-Q4K test fixtures skip
    the dispatch path. 113 markov tests pass.
  - **CPU bench numbers stay parity** post-W1-GPU step 5:
    markov-rs 27.4 tok/s, codec 26.6 tok/s — same as W2 (W1-GPU on
    CPU just changes the code path, not the compute; CPU was already
    at the M3 Max compute ceiling).

  **What's NOT done**: `MetalBackend::coarse_*_with_state` still uses
  the default delegation (state stays empty), so engine falls back
  to walk on Metal — no GPU speedup yet. The real Metal acceleration
  requires modifying
  `larql-compute-metal::decode::decode_token_with_moe_split_fn`
  (200+ lines) to thread per-layer dump buffers + blit-encode steps
  into the existing single command buffer. Two implementation
  shapes have been scoped:
  1. **Blit-encoder switches per layer**: cheapest in steady-state
     (~tens of µs per layer); requires careful encoder lifecycle
     management within the existing kernel function.
  2. **Per-layer commit + CPU readback**: simpler (mirror the
     existing `stage_timing_split` pattern); costs ~50µs/layer ×
     34 = ~1.7ms/token overhead. Projected ceiling: 50-80 tok/s
     (vs CPU's 27 tok/s ceiling, vs `standard`'s 102 tok/s fused).

  Choice between shapes is open. The trait surface, CPU impl, and
  engine wiring are all stable and don't change regardless of which
  Metal-side approach lands.
- **2026-05-17 night — W2: hot K/V cache for `markov_residual` and
  `markov_residual_codec`.** Added `hot_kv: Option<Vec<SharedKV>>`
  to both `RsStore` and `RsStoreCodec`; prefill captures K/V from
  the per-layer forward pass (previously discarded) and stashes it;
  decode appends one row per layer via the existing
  `run_attention_block_decode_step_backend` return tuple. On
  window-overflow `clip_layer` slices `hot_kv` consistently with
  `stored`; for `markov_residual` (lossless cold tier) the evicted
  K/V rows merge directly into `cold_kv` (no `recompute_kv` call
  needed); for `markov_residual_codec` (lossy bf16 cold tier)
  `cold_kv` is invalidated on overflow so the next step recomputes
  against the codec-decoded residual. Bench: `markov_residual`
  4.7 → 26.8 tok/s (5.7×); `markov_residual_codec` 5.0 → 27.5 tok/s
  (5.5×). Both now sit on the `unlimited_context` curve. Engine
  contract preserved — drop `hot_kv` and the next step recomputes
  from `stored` (via_executor path takes this fallback). Hot-state
  memory grew from 5.3 → 10.8 MB; still ~50× smaller than
  `standard`'s full KV cache. Parity test
  (`decode_step_quant_w2_cached_matches_recompute_from_residuals`)
  asserts the cached and recompute paths agree within fp rounding.
- **2026-05-17 night — W7: per-engine profiler wired on the quant
  path.** `EngineProfiler` now populates from `rs_decode_step_walk`
  (markov_residual), `rs_decode_step_codec_walk`
  (markov_residual_codec), `rs_extend_from_checkpoint_quant`
  (unlimited_context), and `decode_step_quant_cpu` (turbo_quant).
  Each engine's `stage_summary()` returns `Some(...)` when
  `with_profiling(true)` is set. `larql bench --profile --engine
  <name>` now produces a per-stage attribution table per engine.
  First measurement run produced the bottleneck-diagnosis table in
  the P0 section above, which inverted two of the pre-profile
  guesses: codec overhead in turbo_quant was ~25% not ~80%, and K/V
  recompute (W2 target) was the dominant cost on markov_residual
  (~80%) not dispatch (W1 target). Sequencing in P0 revised
  accordingly.
- **2026-05-17 night — `_q4k` → `_quant` on remaining internal
  function names.** The trait-surface renames earlier today
  (`prefill_q4k` → `prefill_quant`, `has_q4` →
  `supports_quant(format)`, `q4k` → `kquant` storage) missed the
  per-engine implementation wrappers:
  `unlimited_context::process_q4k`,
  `unlimited_context::extend_current_q4k`,
  `extend::rs_extend_from_checkpoint_q4k`,
  `turbo_quant::decode_step_q4k_cpu` /
  `turbo_quant::prefill_kquant_cpu`. All renamed to `_quant` since
  they dispatch on whatever format the vindex carries, not Q4_K
  specifically.
- **2026-05-17 night — Fused-bypass strip: engines are now engines.**
  Every per-layer engine (`markov_residual`, `markov_residual_codec`,
  `unlimited_context`, `turbo_quant`) had a hidden
  `if let Some(h) = fused_prefill(...) { return Some(h); }` short-
  circuit at the top of `prefill_quant` / `decode_step_quant`. The
  short-circuit meant `--engine markov-rs` on Metal silently ran
  `StandardEngine`'s fused kernel instead — five engines tied at
  ~103 tok/s with `hot=0.0MB`, masking every state-policy difference
  and making per-layer optimization invisible. Cut: removed every
  short-circuit; deleted dead `metal_prefill_done` + `force_walk`
  fields and `with_force_walk` builders; dropped the pub(crate)
  `fused_prefill`/`fused_decode_step` re-exports from
  `unlimited_context::engine` (only `StandardEngine::coarse_prefill`
  uses the underlying `larql_inference::vindex::fused_prefill` now,
  via `ComputeBackend::coarse_prefill`). `StandardEngine` remains the
  default engine and the only home of the fused fast path. Bench now
  reports honest numbers: standard 104 tok/s, markov-rs 3.6, codec
  4.3, unlimited-context 25.6, turbo-quant 3.9 — every per-layer
  engine reports non-zero `hot=` memory because their state
  structures actually materialise. The 25-30× standard-vs-per-layer
  gap is the new optimization frontier; previously it was invisible
  because every engine was running the same kernel under different
  labels.
- **2026-05-17 evening — Phase-2 migration completed for the remaining
  three engines.** `unlimited_context`, `turbo_quant`, and `apollo` all
  override `*_via_executor` methods and honor the caller-supplied
  `FfnBackend`. `CountingFfn` stub tests prove per-(token, layer)
  dispatch through the caller's backend. Same push cleared every
  `coverage-policy.json` debt baseline: all 43 files in src/ at ≥90%
  lines, workspace total 95.55%. `larql bench --ffn http://shard:8080`
  now routes through the remote shard for every per-layer engine
  instead of silently constructing a local `WalkFfn`.
- **2026-05-17 — Phase 2 engine migration to `LayerExecutor`.** Four
  engines (`markov_residual`, `markov_residual_codec`,
  `boundary_per_layer`, `no_cache`) override `*_via_executor` methods.
  They drive per-layer dispatch through `executor.run_*_layer` and
  honor the caller's `FfnBackend`. `CountingFfn` stub tests prove the
  FFN parameter is no longer silently ignored. Bench has
  `--via-executor` flag; demoed on Gemma 3 4B Q4K showing the codec
  engine's 50% cold tier saving (22.9 MB → 11.5 MB).
- **2026-05-17 — `LayerExecutor` trait + `LocalWalkExecutor`.** New
  abstraction in `larql-inference::layer_executor` separating state
  policy (engine concern) from execution strategy (executor concern).
  Spec at
  [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).
- **2026-05-17 — `q4k` → `kquant` storage rename.** K-family storage
  slots (`attn_q4k`, `interleaved_q4k`, manifests, setters, loaders)
  renamed for consistency with accessor naming (`attn_kquant_layer_data`).
  Q4_0 and Q8 slots unchanged. ~60 sites touched.
- **2026-05-17 — `has_q4()` → `supports_quant(format)`.** Per-format
  predicate on `ComputeBackend`. 79 call sites migrated to
  `supports_quant(QuantFormat::Q4_K)`. Enables future Q6_K / FP4
  fused-pipeline backends without trait extension.
- **2026-05-17 — `KvEngine::prefill_q4k` / `decode_step_q4k` →
  `prefill_quant` / `decode_step_quant`.** Trait surface naming made
  quant-agnostic. 112 sites updated. Internals that are genuinely
  Q4K-specific kept their names.
- **2026-05-17 — `metal_fused_*` → `fused_*` rename.** The "metal"
  prefix was a lie: `CpuBackend` implements `prefill_q4` and
  `decode_token` via its C Q4 kernel. Aliases in
  `unlimited_context::engine` follow.
- **2026-05-17 — `BoundaryKvEngine`, `MarkovResidualCodecEngine`,
  `BoundaryPerLayerEngine` shipped.** All three new engines have
  contracts in `crates/larql-inference/docs/specs/`. Per-file coverage
  ≥94 % lines on every new file. Bench demoed end-to-end on Gemma 3 4B,
  Gemma 4 E2B, 26B-A4B, 31B, Qwen3 0.6B (dense + Q4K).
- **2026-05-09 — Initial extraction.** `engines/` carved out of
  `larql-inference` into the new `larql-kv` crate. ~5,540 LOC moved with
  no semantic changes. All four engines + `KvEngine` + accuracy /
  profiler helpers now ship from this crate.

## Non-goals

- **Sampling.** Engines return hidden states; sampling lives in
  `larql_inference::layer_graph::generate::Sampler`. Don't add sampling
  helpers here.
- **Tokenisation / chat templates.** Out of scope; the engines operate on
  `&[u32]` token IDs already produced by `larql_inference::tokenizer` /
  `chat`.
- **Generic K/V backends for non-transformer architectures.** The
  `KvEngine` trait references `ModelWeights` directly. Generalising to
  state-space models or RNNs is not on this roadmap; rebuilds are cheap
  and that effort would belong in larql-inference's layer-graph surface.
