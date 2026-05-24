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

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch path. `UnlimitedContextEngine`
    //! takes a non-optional `window_size: usize`; W10 mask cascade is
    //! gated on whether `current_window_kv` is dropped (the
    //! `LARQL_W10_DISABLE` env var). Tests are `#[serial]` because
    //! `w10_enabled()` reads a process-global env var.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    use serial_test::serial;

    use super::*;
    use crate::engines::unlimited_context::engine::UnlimitedContextEngine;

    fn fixture(window_size: usize) -> (UnlimitedContextEngine, ModelWeights, VectorIndex) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let engine = UnlimitedContextEngine::with_backend(window_size, cpu_engine_backend());
        (engine, weights, index)
    }

    fn set_w10_disable(disabled: bool) {
        if disabled {
            std::env::set_var("LARQL_W10_DISABLE", "1");
        } else {
            std::env::remove_var("LARQL_W10_DISABLE");
        }
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_returns_none_when_index_lacks_direct_matvec() {
        set_w10_disable(false);
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let mut engine = UnlimitedContextEngine::with_backend(4, cpu_engine_backend());
        let mut w = weights;
        assert!(engine
            .try_prefill_via_dispatch(&mut w, &empty_index, &[0u32, 1])
            .is_none());
        assert!(engine.kv_handle.is_none());
        assert!(engine.current_window_kv.is_none());
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_drops_window_kv_under_w10_default() {
        // W10 on by default → drop_window_kv_shadow = true. The engine
        // doesn't populate `current_window_kv`; Metal's kv cache is the
        // truth.
        set_w10_disable(false);
        let (mut engine, mut weights, index) = fixture(4);
        let h = engine
            .try_prefill_via_dispatch(&mut weights, &index, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.current_window_kv.is_none());
        assert_eq!(engine.current_window_kv_len, 3);
        assert_eq!(engine.current_window_tokens, vec![0u32, 1, 2]);
        assert!(engine.kv_handle.is_some());
        assert!(engine.last_hidden.is_some());
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_populates_window_kv_with_w10_disabled() {
        // W10 off → engine pre-allocates `[window_cap, kv_dim]` per
        // layer and copies the prefill K/V rows in.
        set_w10_disable(true);
        let (mut engine, mut weights, index) = fixture(8);
        let res = engine.try_prefill_via_dispatch(&mut weights, &index, &[0u32, 1, 2]);
        set_w10_disable(false);
        res.expect("prefill");
        let kv = engine
            .current_window_kv
            .as_ref()
            .expect("W10-disabled keeps window_kv");
        assert_eq!(kv.len(), weights.num_layers);
        // Buffer is `[window_cap, kv_dim]` where window_cap = max(window, prompt_len) = 8.
        assert_eq!(kv[0].0.shape()[0], 8);
        assert_eq!(engine.current_window_kv_len, 3);
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_without_prefill_returns_none() {
        set_w10_disable(false);
        let (mut engine, mut weights, index) = fixture(4);
        // kv_handle is None → early return.
        assert!(engine
            .decode_step_via_dispatch(&mut weights, &index, 0)
            .is_none());
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_h_only_skips_kv_append() {
        // W10 default + window_kv dropped at prefill → HOnly mask.
        // The dispatch decode runs through the `if !matches!(mask, HOnly)`
        // skip branch and still bumps the per-step counters.
        set_w10_disable(false);
        let (mut engine, mut weights, index) = fixture(4);
        engine
            .try_prefill_via_dispatch(&mut weights, &index, &[0u32, 1])
            .expect("prefill");
        let kv_len_before = engine.current_window_kv_len;
        let h = engine
            .decode_step_via_dispatch(&mut weights, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.current_window_kv_len, kv_len_before + 1);
        assert_eq!(engine.current_window_tokens.last(), Some(&2u32));
        assert!(
            engine.current_window_kv.is_none(),
            "HOnly mask keeps window_kv None"
        );
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_full_mask_appends_kv_with_w10_disabled() {
        // W10 off → Full mask. Each layer's K/V row blits into the
        // pre-allocated window_kv buffer at `current_window_kv_len`.
        set_w10_disable(true);
        let (mut engine, mut weights, index) = fixture(8);
        engine
            .try_prefill_via_dispatch(&mut weights, &index, &[0u32, 1])
            .expect("prefill");
        let kv_len_before = engine.current_window_kv_len;
        let res = engine.decode_step_via_dispatch(&mut weights, &index, 2);
        set_w10_disable(false);
        res.expect("decode");
        assert_eq!(engine.current_window_kv_len, kv_len_before + 1);
        let kv = engine.current_window_kv.as_ref().unwrap();
        // The new row sits at position `kv_len_before` (= 2) of every
        // layer's buffer. Non-zero rows imply the assign actually ran.
        for slot in kv {
            let row = slot.0.row(kv_len_before);
            let any_non_zero = row.iter().any(|v| *v != 0.0);
            assert!(
                any_non_zero,
                "K row at position {kv_len_before} should be populated"
            );
        }
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_fires_window_auto_close() {
        // window=2: prefill 2 → token count == window → close_window
        // fires inside the dispatch decode and resets the window.
        set_w10_disable(false);
        let (mut engine, mut weights, index) = fixture(2);
        engine
            .try_prefill_via_dispatch(&mut weights, &index, &[0u32])
            .expect("prefill");
        let cp_before = engine.checkpoints.len();
        engine
            .decode_step_via_dispatch(&mut weights, &index, 1)
            .expect("decode that triggers window-close");
        // close_window() emits a checkpoint and resets
        // current_window_tokens (the legacy emit path).
        assert!(
            engine.checkpoints.len() > cp_before,
            "window auto-close should emit a checkpoint"
        );
    }
}
