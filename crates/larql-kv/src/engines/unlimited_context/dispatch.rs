//! W1-GPU dispatch path for `UnlimitedContextEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
//! surface. The state-dump payload (per-layer K_new + V_new) lands in
//! the engine's pre-allocated `current_window_kv` slabs (W8 — single
//! `slice_mut(...).assign(row)` per layer per step, no per-step
//! `Array2::zeros` allocation).
//!
//! Window auto-close fires at `current_window_tokens.len() >=
//! window_size`, archiving + checkpointing the closed window. W10
//! mask cascade: `LARQL_W10_HONLY=1` drops the engine-side K/V
//! shadow → Metal's kv cache is the truth, `close_window` reads back
//! the final row via `KvDispatch::read_kv_row_at`.

use larql_inference::attention::SharedKV;
use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use crate::engines::unlimited_context::engine::UnlimitedContextEngine;

impl UnlimitedContextEngine {
    /// W1-GPU step 4: prefill via `coarse_prefill_with_state`. The
    /// per-layer K/V dump is unpacked into pre-allocated
    /// `[window_size, kv_dim]` buffers so subsequent decode steps
    /// append a single row in-place rather than re-allocating.
    pub(super) fn try_prefill_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        if !larql_inference::vindex::supports_cached_decode(weights)
            || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
        {
            return None;
        }
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let (hidden, handle) = self.backend.as_ref().coarse_prefill_with_state(
            weights,
            token_ids,
            Some(index),
            Some(&mut state),
        )?;
        if !state.is_complete_for(num_layers) {
            return None;
        }
        let prompt_len = token_ids.len();
        let window_cap = self.window_size.max(prompt_len);
        // W10 Phase B: drop the engine-side current_window_kv shadow.
        // On by default since 2026-05-21; opt out via
        // LARQL_W10_DISABLE=1. Metal's kv cache is the truth.
        let drop_window_kv_shadow = crate::engines::w10_enabled();
        if drop_window_kv_shadow {
            drop((state.k_new_per_layer, state.v_new_per_layer));
            self.current_window_kv = None;
        } else {
            // W10 Phase A: consume each layer's K/V handle via
            // into_array() (zero-copy move on CPU happy path).
            let kv: Vec<SharedKV> = state
                .k_new_per_layer
                .into_iter()
                .zip(state.v_new_per_layer)
                .map(|(k_h, v_h)| {
                    let k_src = k_h.into_array();
                    let v_src = v_h.into_array();
                    let kv_dim = k_src.shape()[1];
                    let mut k_buf = Array2::<f32>::zeros((window_cap, kv_dim));
                    let mut v_buf = Array2::<f32>::zeros((window_cap, kv_dim));
                    if prompt_len > 0 {
                        k_buf.slice_mut(s![..prompt_len, ..]).assign(&k_src);
                        v_buf.slice_mut(s![..prompt_len, ..]).assign(&v_src);
                    }
                    (k_buf, v_buf)
                })
                .collect();
            self.current_window_kv = Some(kv);
        }
        self.current_window_kv_len = prompt_len;
        self.current_window_tokens = token_ids.to_vec();
        self.last_hidden = Some(hidden.clone());
        self.kv_handle = Some(handle);
        Some(hidden)
    }

    /// W1-GPU step 4: decode through dispatch. State capture gives us
    /// the new K/V row per layer; we append in-place to
    /// `current_window_kv` and trigger window auto-close when token
    /// count crosses `window_size`.
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let abs_position = self.abs_offset + self.current_window_tokens.len();
        let handle = self.kv_handle.as_mut()?;
        // W10 Phase B: HOnly when the window shadow was dropped at
        // prefill. close_window() reads back via KvDispatch.
        let want_h_only = self.current_window_kv.is_none() && crate::engines::w10_enabled();
        let mask = if want_h_only {
            larql_compute::StateDumpMask::HOnly
        } else {
            larql_compute::StateDumpMask::Full
        };
        let hidden = self.backend.as_ref().coarse_decode_step_with_state_masked(
            weights,
            token_id,
            Some(index),
            handle,
            abs_position,
            Some(&mut state),
            mask,
        )?;
        if !state.is_complete_under(num_layers, mask) {
            self.kv_handle = None;
            return None;
        }
        // W8: in-place row append into the pre-allocated buffers
        // (single `slice_mut().assign(row)` per layer per side).
        let pos = self.current_window_kv_len;
        if !matches!(mask, larql_compute::StateDumpMask::HOnly) {
            let window_kv = self
                .current_window_kv
                .as_mut()
                .expect("dispatch decode without prefill — kv_handle invariant violated");
            debug_assert!(
                pos < window_kv[0].0.shape()[0],
                "current_window_kv_len {pos} >= buffer capacity {} — \
                 window auto-close should have fired before this",
                window_kv[0].0.shape()[0]
            );
            let k_handles = std::mem::take(&mut state.k_new_per_layer);
            let v_handles = std::mem::take(&mut state.v_new_per_layer);
            for (slot, (k_handle, v_handle)) in window_kv
                .iter_mut()
                .zip(k_handles.into_iter().zip(v_handles))
                .take(num_layers)
            {
                let k_new_row = k_handle.into_array();
                let v_new_row = v_handle.into_array();
                slot.0.slice_mut(s![pos..pos + 1, ..]).assign(&k_new_row);
                slot.1.slice_mut(s![pos..pos + 1, ..]).assign(&v_new_row);
            }
        }
        self.current_window_kv_len = pos + 1;
        self.current_window_tokens.push(token_id);
        self.last_hidden = Some(hidden.clone());

        // Window auto-close: same trigger as the legacy process loop.
        if self.current_window_tokens.len() >= self.window_size {
            self.close_window();
        }
        Some(hidden)
    }
}
