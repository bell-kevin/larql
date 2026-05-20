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
