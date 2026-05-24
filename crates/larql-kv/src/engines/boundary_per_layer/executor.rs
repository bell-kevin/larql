//! Executor-driven path for `BoundaryPerLayerEngine` (Phase 2
//! migration of the per-layer engines onto `LayerExecutor`).
//!
//! Drives the per-layer dispatch loop through a caller-supplied
//! [`LayerExecutor`] so the caller's FFN backend is honoured (e.g.
//! `--ffn http://shard:8080` routes FFN through a remote shard).
//!
//! Per-layer codec policy state is the engine's responsibility — the
//! executor handles attention + FFN compute only. On fused-kind
//! executors the engine glue falls back to the dense walk via
//! `super::walk::run_prefill` / `run_decode` since per-layer state
//! capture isn't possible.

use larql_inference::attention::SharedKV;
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::embed_tokens_pub;
use larql_inference::layer_executor::LayerExecutor;
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::cold_tier::{
    extend_cold_kv_with_overflow, last_row, roundtrip,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Executor-driven prefill. Caller MUST have already checked that
/// `executor.dispatch_kind() != Fused` (engine glue falls back to
/// `walk::run_prefill` in that case).
pub(super) fn run_prefill(
    weights: &ModelWeights,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_out, _kv) = executor.run_prefill_layer(weights, layer, &h, ffn)?;
        h = h_out;
    }

    let mut rs = RsStorePerLayer {
        stored,
        cold_encoded: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window: window_size,
        policy_codecs: policy.entries.clone(),
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
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

    Some((last_row(&h), rs))
}

/// Executor-driven decode step. Caller MUST have already checked that
/// `executor.dispatch_kind() != Fused`.
pub(super) fn run_decode(
    weights: &ModelWeights,
    executor: &dyn LayerExecutor,
    ffn: &dyn FfnBackend,
    policy: &BoundaryLayerPolicy,
    mut rs: RsStorePerLayer,
    token_id: u32,
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let backend = executor.backend();
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let prior_kv: SharedKV = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = match &rs.cold_encoded {
                Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                    let decoded = cold_layers[layer].decode();
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?
        };

        new_stored.push(h_new.clone());
        let (h_out, _new_kv) =
            executor.run_decode_layer(weights, layer, &h_new, &prior_kv, abs_position, ffn)?;
        h_new = h_out;
    }

    for (slab, new_row) in rs.stored.iter_mut().zip(new_stored.iter()) {
        slab.push_row(new_row.row(0))
            .expect("push_row shape mismatch");
    }
    rs.next_position = abs_position + 1;

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
                let hidden = weights.hidden_size;
                let mut layers: Vec<PerLayerEncodedColdLayer> = Vec::with_capacity(num_layers);
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    let codec = policy.codec_for(layer);
                    let mut enc = PerLayerEncodedColdLayer::empty(codec, hidden);
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

    Some((last_row(&h_new), rs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::ffn::NullFfn;
    use larql_inference::layer_executor::LocalWalkExecutor;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn run_prefill_no_window_returns_state_with_no_cold_tier() {
        // window_size = None → no overflow → no cold_encoded, no cold_kv.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let token_ids: Vec<u32> = vec![0, 1, 2];
        let (hidden, rs) = run_prefill(&weights, &executor, &ffn, &policy, None, &token_ids)
            .expect("prefill should succeed with synthetic weights");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 3);
        assert!(rs.cold_encoded.is_none(), "no overflow → no cold_encoded");
        assert!(rs.cold_kv.is_none(), "no overflow → no cold_kv");
        // Each layer's stored slab has all 3 tokens.
        assert_eq!(rs.stored.len(), weights.num_layers);
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 3, "each layer slab carries all 3 tokens");
        }
    }

    #[test]
    fn run_prefill_with_small_window_evicts_to_cold_tier() {
        // window_size = 2 with 3-token prefill → 1 row of overflow per
        // layer, populates cold_encoded + cold_kv.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let token_ids: Vec<u32> = vec![0, 1, 2];
        let (_hidden, rs) = run_prefill(&weights, &executor, &ffn, &policy, Some(2), &token_ids)
            .expect("prefill should succeed");
        assert!(
            rs.cold_encoded.is_some(),
            "overflow path must populate cold_encoded"
        );
        assert!(
            rs.cold_kv.is_some(),
            "overflow path must pre-compute cold_kv"
        );
        let cold_kv = rs.cold_kv.as_ref().unwrap();
        for (k, _v) in cold_kv {
            assert_eq!(k.shape()[0], 1, "1 row of overflow per layer");
        }
    }

    #[test]
    fn run_decode_extends_hot_tier_when_below_window() {
        // After a 1-token prefill with window=4, decode 1 token →
        // both fit in hot, no overflow, no cold-tier mutation.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(&weights, &executor, &ffn, &policy, Some(4), &[0]).unwrap();
        assert!(
            rs.cold_encoded.is_none(),
            "no overflow expected after prefill"
        );

        let (hidden, rs_after) =
            run_decode(&weights, &executor, &ffn, &policy, rs, 1).expect("decode should succeed");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs_after.next_position, 2);
        for slab in &rs_after.stored {
            assert_eq!(slab.shape()[0], 2, "hot slab grew to 2 rows");
        }
        // Still no overflow at this scale.
        assert!(rs_after.cold_encoded.is_none());
    }

    #[test]
    fn run_decode_promotes_to_cold_tier_on_overflow() {
        // Prefill 3 tokens with window=2 → 1 row already in cold.
        // Decode 1 more token → 2 rows in cold after eviction.
        // Exercises the Some(layers) arm of cold_encoded.as_mut().
        let weights = make_test_weights();
        let backend = CpuBackend;
        let executor = LocalWalkExecutor::new(&backend);
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(&weights, &executor, &ffn, &policy, Some(2), &[0, 1, 2]).unwrap();
        assert!(
            rs.cold_encoded.is_some(),
            "prefill should have populated cold_encoded"
        );
        let initial_cold_rows = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(initial_cold_rows, 1, "1 row in cold after prefill");

        let (_, rs_after) = run_decode(&weights, &executor, &ffn, &policy, rs, 3).unwrap();
        let after_cold_rows = rs_after
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(after_cold_rows, 2, "decode evicted 1 more row to cold");
        assert_eq!(rs_after.next_position, 4);
    }
}
