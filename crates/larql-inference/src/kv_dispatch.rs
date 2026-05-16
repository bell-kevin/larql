//! `KvDispatch` — engine-facing intent surface for K/V cache + attention.
//!
//! Sibling to [`crate::FfnBackend`] (FFN dispatch) and
//! [`larql_compute::ComputeBackend`] (substrate kernel primitives).
//! [`KvEngine`](crate::KvEngine) implementations call `KvDispatch`
//! methods to express *intents* (allocate K/V, append a row, attend Q
//! against K/V with optional windowing, recompute K/V from residuals,
//! upload a boundary residual). The backend decides *how* — which
//! kernel runs, which shader variant from the pipeline cache, whether
//! the K/V append fuses into the attention kernel.
//!
//! ## Why a sibling trait, not a `ComputeBackend` sub-trait
//!
//! The CPU implementation of these intents needs to call into the
//! inference-side forward-pass functions (`run_attention_*`,
//! `run_ffn`, residual ops) that live in this crate. The trait
//! therefore lives here so its CPU impl (and Metal impl, via the same
//! orphan-rule logic) can be authored in this crate. Putting the trait
//! in `larql-compute` would block the CPU impl: orphan rules forbid
//! `impl KvDispatch for CpuBackend` in `larql-inference` when both
//! trait and type are foreign, and `larql-compute` can't depend on
//! `larql-inference` (would be a cycle).
//!
//! The substrate *capability* flags
//! ([`larql_compute::Capability::FusedAttentionStep`] etc.) stay in
//! `larql-compute` — they describe what the substrate supports
//! independently of where the dispatch trait lives.
//!
//! See `docs/specs/compute-backend-redesign.md` for full design rationale.
//!
//! ## Default behaviour
//!
//! Every method has a default that either returns `None` or panics
//! with `unimplemented!()`. Backends implementing the trait override
//! what they support. Engines should check
//! [`larql_compute::ComputeBackend::supports`] with the matching
//! [`larql_compute::Capability`] flag before calling, unless the
//! method has a meaningful default decomposition documented in its
//! doc-comment.

use crate::model::ModelWeights;
use ndarray::Array2;

/// Opaque handle to a K/V cache allocation. Layout is backend-specific;
/// engines pass these around without observing structure beyond the
/// queries the trait exposes.
///
/// Backends ship their own inner type (`CpuKvHandle`, `MetalKvHandle`,
/// `VulkanKvHandle`) implementing [`KvHandleInner`]. Engines hold
/// `KvHandle` opaquely and call backend methods to manipulate it.
pub struct KvHandle {
    inner: Box<dyn KvHandleInner>,
}

impl KvHandle {
    /// Construct from a backend-specific inner. Backend implementations
    /// call this; engines never do.
    pub fn new<I: KvHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    /// Number of K/V rows currently cached.
    pub fn cached_len(&self) -> usize {
        self.inner.cached_len()
    }

    /// Hidden dim per K/V row (kv_dim, not full hidden — already
    /// accounts for GQA head count).
    pub fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    /// Which backend allocated this handle. Used for sanity checks
    /// when handles cross backend boundaries (which normally
    /// shouldn't happen — read out to host first via
    /// [`KvDispatch::read_kv_to_host`]).
    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    /// Downcast access for backend implementations. Engines never call
    /// this; only the backend that allocated the handle should.
    pub fn as_inner(&self) -> &dyn KvHandleInner {
        &*self.inner
    }

    /// Mutable downcast for backend impls.
    pub fn as_inner_mut(&mut self) -> &mut dyn KvHandleInner {
        &mut *self.inner
    }
}

/// Backend-side trait for K/V handle inner types. Backends implement
/// this on whatever GPU-side or host-side allocation they manage
/// (`MTLBuffer`, `VkBuffer`, `Vec<f32>`, or a wrapper over the
/// existing `larql_inference::attention::KvCache`).
pub trait KvHandleInner: Send + Sync + std::any::Any {
    fn cached_len(&self) -> usize;
    fn kv_dim(&self) -> usize;
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Opaque handle to a residual upload (used by `apollo` for boundary
/// residuals). Same pattern as [`KvHandle`].
pub struct ResidualHandle {
    inner: Box<dyn ResidualHandleInner>,
}

impl ResidualHandle {
    pub fn new<I: ResidualHandleInner + 'static>(inner: I) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    pub fn backend_name(&self) -> &'static str {
        self.inner.backend_name()
    }

    pub fn as_inner(&self) -> &dyn ResidualHandleInner {
        &*self.inner
    }
}

pub trait ResidualHandleInner: Send + Sync + std::any::Any {
    fn shape(&self) -> (usize, usize);
    fn backend_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Engine-facing intent surface.
///
/// All methods are synchronous (return immediately with the result;
/// any GPU work is submitted and waited on internally). Async / stream-
/// graph variants live on a future `AsyncComputeBackend` trait — not
/// part of v1. See `compute-backend-redesign.md` §11.4.
///
/// Engines hold `&dyn KvDispatch` alongside
/// `&dyn larql_compute::ComputeBackend` and [`crate::FfnBackend`].
/// The three abstractions compose orthogonally: substrate kernels +
/// engine intents + FFN routing.
pub trait KvDispatch {
    // ── Cache primitives ────────────────────────────────────────────

    /// Allocate a K/V buffer for `layer`, sized for at most `max_tokens`
    /// positions of `kv_dim`-wide K and V rows. Layout is backend-
    /// specific; engines treat the returned handle opaquely.
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        let _ = (layer, max_tokens, kv_dim);
        unimplemented!("alloc_kv_buffer not implemented for this backend")
    }

    /// Append a single K/V row at `abs_position`. The handle must have
    /// been allocated by *this* backend; cross-backend handles panic.
    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        let _ = (handle, k_row, v_row, abs_position);
        unimplemented!("append_kv not implemented for this backend")
    }

    /// Clip the handle's cached entries to at most `window_size` rows
    /// (keep the tail). Backends with bounded-ring-buffer K/V layouts
    /// may implement this as a no-op; backends with growing K/V apply
    /// a shift or drop.
    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        let _ = (handle, window_size);
        unimplemented!("clip_kv not implemented for this backend")
    }

    /// Read the full K/V back to host memory as a `(K, V)` pair.
    /// Blocking copy on GPU backends; identity on CPU. Should NOT be
    /// used in hot loops — it's the cross-backend escape hatch for
    /// fallback paths and debug inspection.
    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        let _ = handle;
        None
    }

    // ── Attention primitives ────────────────────────────────────────

    /// Run one decode-step attention: Q (one row) attends against
    /// K/V from `handle` plus the new token's K/V (which the caller
    /// must have already appended via [`Self::append_kv`]). Returns
    /// the attention output (one row, hidden_dim wide).
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::FusedAttentionStep`]. Backends
    /// that don't support fused attention return `None`; callers fall
    /// back to decomposed BLAS attention.
    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &KvHandle,
        layer: usize,
        abs_position: usize,
    ) -> Option<Array2<f32>> {
        let _ = (weights, query, kv, layer, abs_position);
        None
    }

    /// Like [`Self::attention_step`] but with a window bound baked
    /// into the dispatch — backend may use a specialised shader variant
    /// that knows the window size at compile time.
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::WindowedAttentionStep`]. Default
    /// falls back to [`Self::attention_step`] with a clipped handle
    /// (correct but not specialised).
    fn attention_step_windowed(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
    ) -> Option<Array2<f32>> {
        let _ = window;
        self.attention_step(weights, query, kv, layer, abs_position)
    }

    /// Multi-token prefill attention: tokens have been embedded into
    /// `tokens_embedded` (shape `[seq_len, hidden]`). Backend runs full
    /// attention over the sequence, populates a fresh K/V handle, and
    /// returns `(last_hidden_1xH, populated_handle)`.
    ///
    /// `window` selects the K/V cap: `None` = unbounded growth,
    /// `Some(W)` = sliding-window K/V (older positions evicted from
    /// the cache after the prefill).
    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let _ = (weights, tokens_embedded, layer, window);
        None
    }

    // ── Engine-specific primitives ──────────────────────────────────

    /// Regenerate K/V for a layer from stored pre-layer residuals.
    /// Used by `markov-rs`: residuals are the persistent state, K/V is
    /// recomputed each decode step. Backends without this intent fall
    /// back to running the Q/K/V projection through
    /// [`larql_compute::MatMul`] directly.
    fn recompute_kv_from_residuals(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        let _ = (weights, residuals, layer);
        None
    }

    /// Append compressed K/V to a handle using the given codec.
    /// Used by `turbo-quant`. Backends with native codec kernels
    /// (Metal WHT shader) implement this; others fall back to
    /// dequant → f32 append → requant via the caller.
    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        let _ = (handle, k, v, codec);
        unimplemented!("compressed_kv_append not implemented for this backend")
    }

    /// Upload a boundary residual to backend-managed memory. Returns
    /// a handle the engine can use as the starting state for
    /// [`Self::forward_from_layer`]. Used by `apollo` compressed path.
    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        let _ = residual;
        None
    }

    /// Run the forward pass starting at `start_layer` using `residuals`
    /// as the layer-`start_layer` input. Used by `apollo` to skip the
    /// pre-crystal layers when boundaries are available.
    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        let _ = (weights, start_layer, residuals, token_ids);
        None
    }

    // ── Norm + residual primitives ──────────────────────────────────

    /// Fused `residual_add + rmsnorm` for the post-attention or
    /// post-FFN residual write. Target for D-RMS-FUSE phase 2 work.
    ///
    /// Capability gate:
    /// [`larql_compute::Capability::FusedResidualNorm`]. Default
    /// decomposes into separate add + rmsnorm calls on host (correct
    /// but slow); backends with fused kernels override.
    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        // Default: decompose. add then rmsnorm.
        let added = x + residual;
        let mut out = Array2::<f32>::zeros(added.raw_dim());
        for (i, row) in added.rows().into_iter().enumerate() {
            let row_slice = row.as_slice().expect("non-contiguous row");
            let mean_sq: f32 =
                row_slice.iter().map(|v| v * v).sum::<f32>() / row_slice.len() as f32;
            let scale = (mean_sq + 1e-6).sqrt().recip();
            for (j, (val, w)) in row_slice.iter().zip(norm_weights.iter()).enumerate() {
                out[[i, j]] = val * scale * w;
            }
        }
        out
    }
}

/// Codec hook for [`KvDispatch::compressed_kv_append`]. Backends that
/// implement native compressed K/V append call back into the codec for
/// per-row encode/decode where the kernel isn't fully fused.
pub trait CompressionCodec: Send + Sync {
    fn encode(&self, vec: &[f32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8], dim: usize) -> Vec<f32>;
    fn name(&self) -> &str;
}
