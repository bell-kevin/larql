//! W1-GPU dispatch path for `BoundaryPerLayerEngine`.
//!
//! Mirrors `markov_residual`'s dispatch path. The two free functions
//! ([`try_prefill_via_dispatch`] and [`decode_step_via_dispatch`])
//! route through the backend's `coarse_prefill_with_state` /
//! `coarse_decode_step_with_state_masked` surface — on Metal this
//! runs the prompt through the fused per-layer kernel and dumps
//! per-layer `h_in` for the engine to pull into its residual store.
//!
//! Returns `None` (engine should fall back to the dense walk in
//! `super::walk`) when the backend / vindex doesn't support the
//! cached + direct-matvec decode path.
//!
//! **W10 mask cascade** — `boundary_per_layer` never shadows hot
//! K/V (it's recomputed at extend-cold-kv time on overflow), so
//! `LARQL_W10_HONLY=1` is always at least HOnly-safe. When
//! `window_size = None` the residual `stored` is also unused (no
//! cold-tier eviction can fire), so the engine additionally drops
//! it and requests the None mask. Bench (Gemma 3 4B Q4K, M3 Max,
//! 2026-05-21) closes the 13% gap to `standard`'s ~100 tok/s
//! ceiling.

use larql_inference::model::ModelWeights;
use larql_inference::{EngineBackend, KvHandle, PerLayerDecodeState};
use ndarray::Array2;

use crate::engines::boundary_per_layer::cold_tier::{extend_cold_kv_with_overflow, roundtrip};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

use crate::engines::w10_enabled as w10_env_on;

/// Run prefill through the W1-GPU dispatch path. Returns
/// `(last_hidden, new_store, kv_handle)` on success; `None` when the
/// backend / vindex lacks the required support (caller falls back to
/// `walk::run_prefill`).
pub(super) fn try_prefill_via_dispatch(
    weights: &mut ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer, KvHandle)> {
    if !larql_inference::vindex::supports_cached_decode(weights)
        || !larql_inference::vindex::supports_direct_matvec_decode(weights, index)
    {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let (hidden, handle) =
        backend.coarse_prefill_with_state(weights, token_ids, Some(index), Some(&mut state))?;
    if !state.is_complete_for(num_layers) {
        return None;
    }
    let prompt_len = token_ids.len();

    // W10 Phase C: when LARQL_W10_HONLY=1 + window=None, no
    // cold-tier eviction can fire and `rs.stored` is dead weight.
    // Drop it; decode steps will request the None mask, eliminating
    // both K/V and h_in readback. (HOnly without dropping stored is
    // always safe — boundary_per_layer has no hot K/V shadow — but
    // dropping stored is what enables the None-mask path.)
    let drop_stored_shadow = w10_env_on() && window_size.is_none();
    let stored: Vec<Array2<f32>> = if drop_stored_shadow {
        let hidden_size = weights.hidden_size;
        (0..num_layers)
            .map(|_| Array2::<f32>::zeros((0, hidden_size)))
            .collect()
    } else {
        state
            .h_in_per_layer
            .into_iter()
            .map(|h| h.into_array())
            .collect()
    };

    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: prompt_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };

    // Prefill-time clip only when we have a non-empty stored. With
    // drop_stored_shadow the stored is empty and clip is a no-op,
    // but we'd panic on indexing `stored[layer]` so just skip.
    if !drop_stored_shadow {
        let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            overflow_per_layer.push(rs.clip_layer_overflow(layer));
        }
        if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
            let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
            let mut cold_kv: Vec<larql_inference::attention::SharedKV> =
                Vec::with_capacity(num_layers);
            for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                let codec = policy.codec_for(layer);
                let decoded_overflow = roundtrip(overflow, codec);
                let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
                    .expect("cold K/V pre-computation failed");
                cold_kv.push((k, v));
                let mut enc = PerLayerEncodedColdLayer::empty(codec, weights.hidden_size);
                enc.append(overflow);
                encoded_layers.push(enc);
            }
            rs.cold_encoded = Some(encoded_layers);
            rs.cold_kv = Some(cold_kv);
            rs.cold_abs_start = 0;
        }
    }
    Some((hidden, rs, handle))
}

/// One decode step through the W1-GPU dispatch path. Mutates the
/// supplied `KvHandle` in place (backend appends K/V) and returns the
/// updated store. `None` signals a state-dump failure — caller should
/// clear its `kv_handle` and fall back to the dense walk.
pub(super) fn decode_step_via_dispatch(
    weights: &mut ModelWeights,
    backend: &dyn EngineBackend,
    policy: &BoundaryLayerPolicy,
    handle: &mut KvHandle,
    mut rs: RsStorePerLayer,
    index: &larql_inference::larql_vindex::VectorIndex,
    token_id: u32,
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let num_layers = weights.num_layers;
    let mut state = PerLayerDecodeState::with_capacity(num_layers);
    let abs_position = rs.next_position;

    // W10 mask cascade. boundary_per_layer never shadows hot K/V,
    // so K/V readback is always wasted overhead → drop_hot_kv is
    // unconditionally true when env_on. stored is droppable only
    // when env_on + windowless (the prefill arranged that).
    let env_on = w10_env_on();
    let drop_stored = rs
        .stored
        .first()
        .map(|a| a.shape()[0] == 0)
        .unwrap_or(false)
        && env_on;
    let mask = if drop_stored {
        larql_compute::StateDumpMask::None
    } else if env_on {
        larql_compute::StateDumpMask::HOnly
    } else {
        larql_compute::StateDumpMask::Full
    };

    let hidden = backend.coarse_decode_step_with_state_masked(
        weights,
        token_id,
        Some(index),
        handle,
        abs_position,
        Some(&mut state),
        mask,
    )?;
    if !state.is_complete_under(num_layers, mask) {
        return None;
    }

    // Append h_in to each layer's stored slab (amortised O(m) via
    // push_row). Under None mask, h_in is empty — skip the loop;
    // stored stays the empty Vec from prefill.
    if !matches!(mask, larql_compute::StateDumpMask::None) {
        for (layer, h) in state.h_in_per_layer.into_iter().enumerate() {
            let h_arr = h.into_array();
            rs.stored[layer]
                .push_row(h_arr.row(0))
                .expect("push_row shape mismatch");
        }
    }
    rs.next_position = abs_position + 1;

    // Cold-tier eviction + cold_kv extension. Under None mask there's
    // no stored to evict from; skip.
    if matches!(mask, larql_compute::StateDumpMask::None) {
        return Some((hidden, rs));
    }
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_abs_pos =
            rs.cold_abs_start + rs.cold_encoded.as_ref().map_or(0, |l| l[0].n_positions);
        match rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(overflow);
                }
            }
            None => {
                let hidden_size = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden_size);
                    enc.append(overflow);
                    layers.push(enc);
                }
                rs.cold_encoded = Some(layers);
            }
        }
        extend_cold_kv_with_overflow(
            weights,
            backend,
            policy,
            &mut rs,
            &overflow_per_layer,
            cold_abs_pos,
        );
    }
    Some((hidden, rs))
}

#[cfg(test)]
mod tests {
    //! Coverage for the W1-GPU dispatch free functions. Drives
    //! `CpuBackend` via the synthetic Q4K fixture so the per-layer
    //! `coarse_*_with_state` populates a `PerLayerDecodeState` the
    //! helpers then consume into `RsStorePerLayer`.
    //!
    //! The W10 mask cascade (`drop_stored_shadow` /
    //! `StateDumpMask::None`) is on by default; tests exercise both
    //! the windowed (`HOnly`) and windowless (`None`) shapes.

    use larql_inference::cpu_engine_backend;
    use larql_inference::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, Q4K_TEST_NUM_LAYERS,
    };
    use serial_test::serial;

    use super::*;
    use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;

    fn bf16_policy() -> BoundaryLayerPolicy {
        BoundaryLayerPolicy::bf16_uniform("test", Q4K_TEST_NUM_LAYERS)
    }

    /// Reset `LARQL_W10_DISABLE` to a known state. Every test in this
    /// module exercises code paths gated on `w10_env_on()` (which reads
    /// this var), so the tests are `#[serial]` and start clean.
    fn clear_w10_env() {
        std::env::remove_var("LARQL_W10_DISABLE");
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_returns_none_when_index_lacks_direct_matvec() {
        clear_w10_env();
        let weights = make_test_q4k_weights();
        let empty_index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            weights.hidden_size,
        );
        let backend = cpu_engine_backend();
        let mut w = weights;
        assert!(try_prefill_via_dispatch(
            &mut w,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &empty_index,
            &[0u32, 1],
        )
        .is_none());
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_windowed_populates_store_under_w10_honly() {
        clear_w10_env();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (h, rs, _handle) = try_prefill_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &index,
            &[0u32, 1, 2],
        )
        .expect("prefill via dispatch");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // W10 default: HOnly mask is selected (drop_hot_kv unconditional,
        // drop_stored_shadow only when window_size = None). Windowed
        // configuration keeps stored populated.
        assert_eq!(rs.stored.len(), weights.num_layers);
        assert_eq!(rs.stored[0].shape()[0], 3);
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    #[serial]
    fn try_prefill_via_dispatch_windowless_drops_stored_under_w10_none_mask() {
        clear_w10_env();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (_h, rs, _handle) = try_prefill_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            None,
            &index,
            &[0u32, 1, 2],
        )
        .expect("prefill via dispatch (windowless)");
        // W10 + window=None: drop_stored_shadow is true → empty stored
        // per layer.
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 0, "stored should be empty under None mask");
        }
        assert!(rs.cold_encoded.is_none());
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_appends_h_in_under_honly() {
        clear_w10_env();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (_h, rs, mut handle) = try_prefill_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(4),
            &index,
            &[0u32, 1],
        )
        .expect("prefill");
        let rows_before = rs.stored[0].shape()[0];
        let (h, rs) = decode_step_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            rs,
            &index,
            2,
        )
        .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // HOnly mask appends one h_in row per layer per step.
        assert_eq!(rs.stored[0].shape()[0], rows_before + 1);
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_windowless_takes_none_mask_path() {
        clear_w10_env();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        let (_h, rs, mut handle) = try_prefill_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            None,
            &index,
            &[0u32, 1],
        )
        .expect("prefill (windowless)");
        let (_h, rs) = decode_step_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            rs,
            &index,
            2,
        )
        .expect("decode (None mask)");
        // None mask: stored stays empty, but next_position still advances.
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 0);
        }
        assert_eq!(rs.next_position, 3);
    }

    #[test]
    #[serial]
    fn decode_step_via_dispatch_overflow_extends_cold_tier() {
        clear_w10_env();
        let mut weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = cpu_engine_backend();
        // window=2: after prefilling 2 tokens, one decode crosses the
        // window and the dispatch eviction path runs.
        let (_h, rs, mut handle) = try_prefill_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            Some(2),
            &index,
            &[0u32, 1],
        )
        .expect("prefill");
        assert!(rs.cold_encoded.is_none(), "no overflow at prefill");
        let (_h, rs) = decode_step_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            rs,
            &index,
            2,
        )
        .expect("decode");
        assert!(
            rs.cold_encoded.is_some(),
            "first decode past window should fire cold-tier append"
        );
        // Subsequent decode should extend an existing cold_encoded
        // (Some(layers) branch of the match).
        let (_h, rs) = decode_step_via_dispatch(
            &mut weights,
            backend.as_ref(),
            &bf16_policy(),
            &mut handle,
            rs,
            &index,
            3,
        )
        .expect("decode 2");
        assert!(rs.cold_encoded.is_some());
        assert!(rs.cold_encoded.as_ref().unwrap()[0].n_positions >= 2);
    }
}
