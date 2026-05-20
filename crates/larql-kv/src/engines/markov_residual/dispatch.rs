//! W1-GPU dispatch path for `MarkovResidualEngine`.
//!
//! Routes prefill + decode through the backend's
//! `coarse_prefill_with_state` / `coarse_decode_step_with_state_masked`
//! surface, populating the engine's residual store (and optional
//! hot_kv shadow) from per-layer state-dump payloads.
//!
//! W10 mask cascade: `LARQL_W10_HONLY=1` drops the hot_kv shadow on
//! Metal (treating K/V as derivative state); when `window_size =
//! None` it also drops the residual shadow (`stored`). The matching
//! `StateDumpMask` is flowed into the backend call so the kernel
//! skips the corresponding GPU→CPU readback.
//!
//! See `crates/larql-kv/docs/state-policy.md` and PERFORMANCE.md's
//! "W10" section.

use larql_inference::model::ModelWeights;
use larql_inference::PerLayerDecodeState;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use crate::engines::markov_residual::engine::MarkovResidualEngine;
use crate::engines::markov_residual::helpers::{append_row, grow_capacity_2d, window_capacity};
use crate::engines::markov_residual::store::RsStore;

impl MarkovResidualEngine {
    /// W1-GPU: attempt prefill through
    /// `KvDispatch::coarse_prefill_with_state`. Returns `Some(hidden)`
    /// when the backend implements the GPU/fused path; `None` when it
    /// doesn't (engine falls back to per-layer walk).
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
        // W8.2: pre-allocate `stored` and `hot_kv` to a doubling capacity
        // so subsequent decode steps append in-place.
        let prompt_len = token_ids.len();
        let initial_cap = window_capacity(prompt_len, self.window_size);
        let stored: Vec<Array2<f32>> = state
            .h_in_per_layer
            .into_iter()
            .map(|h| grow_capacity_2d(&h.into_array(), prompt_len, initial_cap))
            .collect();
        let hot_kv: Vec<larql_inference::attention::SharedKV> = state
            .k_new_per_layer
            .into_iter()
            .zip(state.v_new_per_layer)
            .map(|(k, v)| {
                (
                    grow_capacity_2d(&k.into_array(), prompt_len, initial_cap),
                    grow_capacity_2d(&v.into_array(), prompt_len, initial_cap),
                )
            })
            .collect();
        // W10 Phase B/C: drop shadows. On by default since 2026-05-21
        // (set LARQL_W10_DISABLE=1 to opt out — debug instrument).
        let drop_hot_kv_shadow = crate::engines::w10_enabled();
        let drop_stored_shadow = drop_hot_kv_shadow && self.window_size.is_none();
        let stored = if drop_stored_shadow {
            let hidden_size = weights.hidden_size;
            (0..num_layers)
                .map(|_| Array2::<f32>::zeros((0, hidden_size)))
                .collect()
        } else {
            stored
        };
        let mut rs = RsStore {
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_len: 0,
            hot_kv: if drop_hot_kv_shadow {
                None
            } else {
                Some(hot_kv)
            },
            cold_abs_start: 0,
            next_position: prompt_len,
            max_window: self.window_size,
            hot_len: if drop_stored_shadow { 0 } else { prompt_len },
        };
        // Clip window on prefill — overflow goes into cold tier via
        // the snapshot helper (already-computed K/V from dispatch).
        let pre_clip: Vec<usize> = if rs.hot_kv.is_some() {
            let window = self.window_size.unwrap_or(usize::MAX);
            let evict_count = rs.hot_len.saturating_sub(window);
            vec![evict_count; rs.stored.len()]
        } else {
            Vec::new()
        };
        let evicted_hot_kv = rs
            .hot_kv
            .as_ref()
            .filter(|_| pre_clip.iter().any(|&n| n > 0))
            .and_then(|h| RsStore::snapshot_evicted_hot_kv(h, &pre_clip));
        let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut cold);
        }
        rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: geometric-capacity cold append.
        rs.append_cold_overflow(cold, evicted_hot_kv);
        if rs.cold_len > 0 {
            rs.cold_abs_start = 0;
        }
        self.store = Some(rs);
        self.kv_handle = Some(handle);
        self.abs_position = token_ids.len();
        Some(hidden)
    }

    /// W1-GPU: decode step through
    /// `KvDispatch::coarse_decode_step_with_state_masked`. Per-layer
    /// state is appended to the engine's store/hot_kv on each step
    /// (W8.2 doubling-capacity in-place append).
    pub(super) fn decode_step_via_dispatch(
        &mut self,
        weights: &mut ModelWeights,
        index: &VectorIndex,
        token_id: u32,
    ) -> Option<Array2<f32>> {
        let t_total = std::time::Instant::now();
        let num_layers = weights.num_layers;
        let mut state = PerLayerDecodeState::with_capacity(num_layers);
        let handle = self.kv_handle.as_mut()?;
        // W10 mask cascade. On by default; opt out via
        // LARQL_W10_DISABLE=1.
        let env_on = crate::engines::w10_enabled();
        let drop_hot_kv = self
            .store
            .as_ref()
            .map(|s| s.hot_kv.is_none())
            .unwrap_or(false)
            && env_on;
        let drop_stored = self
            .store
            .as_ref()
            .map(|s| s.stored.first().map(|a| a.shape()[0] == 0).unwrap_or(false))
            .unwrap_or(false)
            && env_on;
        let mask = if drop_stored && drop_hot_kv {
            larql_compute::StateDumpMask::None
        } else if drop_hot_kv {
            larql_compute::StateDumpMask::HOnly
        } else {
            larql_compute::StateDumpMask::Full
        };
        let t_capture = std::time::Instant::now();
        let hidden = self.backend.as_ref().coarse_decode_step_with_state_masked(
            weights,
            token_id,
            Some(index),
            handle,
            self.abs_position,
            Some(&mut state),
            mask,
        )?;
        if self.profiling {
            self.profile.state_capture.record(t_capture);
        }
        if !state.is_complete_under(num_layers, mask) {
            self.kv_handle = None;
            return None;
        }
        let mut rs = self.store.take()?;
        // W8.2: append per-layer h_in / K_new / V_new in-place.
        let len = rs.hot_len;
        let h_handles = std::mem::take(&mut state.h_in_per_layer);
        let k_handles = std::mem::take(&mut state.k_new_per_layer);
        let v_handles = std::mem::take(&mut state.v_new_per_layer);
        let did_append = !matches!(mask, larql_compute::StateDumpMask::None);
        if matches!(mask, larql_compute::StateDumpMask::None) {
            drop((h_handles, k_handles, v_handles));
        } else if matches!(mask, larql_compute::StateDumpMask::HOnly) {
            drop((k_handles, v_handles));
            for (layer, h) in h_handles.into_iter().enumerate() {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        } else {
            for (layer, ((h, k), v)) in h_handles
                .into_iter()
                .zip(k_handles)
                .zip(v_handles)
                .enumerate()
            {
                let t_mat = std::time::Instant::now();
                let h_arr = h.into_array();
                let k_arr_opt = if rs.hot_kv.is_some() {
                    Some((k.into_array(), v.into_array()))
                } else {
                    None
                };
                if self.profiling {
                    self.profile.state_materialise.record(t_mat);
                }
                let t_app = std::time::Instant::now();
                append_row(&mut rs.stored[layer], &h_arr, len);
                if let Some(hot_kv) = rs.hot_kv.as_mut() {
                    if let Some((k_arr, v_arr)) = k_arr_opt {
                        append_row(&mut hot_kv[layer].0, &k_arr, len);
                        append_row(&mut hot_kv[layer].1, &v_arr, len);
                    }
                }
                if self.profiling {
                    self.profile.state_append.record(t_app);
                }
            }
        }
        if did_append {
            rs.hot_len = len + 1;
        }
        // Window clip — snapshot-evicted-into-cold flow (W2).
        let pre_clip: Vec<usize> = if rs.hot_kv.is_some() {
            let window = rs.max_window.unwrap_or(usize::MAX);
            let evict_count = rs.hot_len.saturating_sub(window);
            vec![evict_count; rs.stored.len()]
        } else {
            Vec::new()
        };
        let evicted_hot_kv = rs
            .hot_kv
            .as_ref()
            .filter(|_| pre_clip.iter().any(|&n| n > 0))
            .and_then(|h| RsStore::snapshot_evicted_hot_kv(h, &pre_clip));
        let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            rs.clip_layer(layer, &mut overflow);
        }
        rs.finalise_hot_len_after_clip();
        // 2026-05-19 audit fix: geometric-capacity cold append.
        rs.append_cold_overflow(overflow, evicted_hot_kv);
        self.store = Some(rs);
        self.abs_position += 1;
        if self.profiling {
            self.profile.decode_total.record(t_total);
        }
        Some(hidden)
    }
}
