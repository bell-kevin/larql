//! CPU walk path for `BoundaryPerLayerEngine`.
//!
//! Mirrors `markov_residual_codec/walk.rs`'s shape: free functions
//! that take all inputs explicitly and return `(hidden, new_store)`
//! — the engine glue (in `engine.rs`) handles store ownership and
//! `KvHandle` lifecycle.
//!
//! The dense path is used when the W1-GPU dispatch path
//! (`dispatch::try_prefill_via_dispatch`) returns `None` —
//! typically on backends/vindexes lacking direct-matvec decode.

use larql_compute::ComputeBackend;
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use larql_inference::ffn::FfnBackend;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::boundary_per_layer::cold_tier::{
    extend_cold_kv_with_overflow, last_row, roundtrip,
};
use crate::engines::boundary_per_layer::policy::BoundaryLayerPolicy;
use crate::engines::boundary_per_layer::store::{PerLayerEncodedColdLayer, RsStorePerLayer};
use crate::engines::markov_residual::recompute_kv;

/// Run a full prefill through the dense walk. Returns
/// `(last_hidden, new_store)` — caller owns the store.
pub(super) fn run_prefill(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    window_size: Option<usize>,
    token_ids: &[u32],
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) =
            run_attention_with_kv_backend(weights, &h, layer, be).expect("attention failed");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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

/// Run one decode step through the dense walk. Consumes `rs`, returns
/// the new store alongside the hidden output.
pub(super) fn run_decode(
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    backend: &dyn ComputeBackend,
    policy: &BoundaryLayerPolicy,
    mut rs: RsStorePerLayer,
    token_id: u32,
) -> Option<(Array2<f32>, RsStorePerLayer)> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
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
            let (h_full, full_abs_start) = if let Some(cold_layers) = &rs.cold_encoded {
                let enc = &cold_layers[layer];
                if enc.n_positions > 0 {
                    let decoded = enc.decode();
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                } else {
                    (h_hot.clone(), hot_abs_start)
                }
            } else {
                (h_hot.clone(), hot_abs_start)
            };
            recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?
        };

        new_stored.push(h_new.clone());

        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_new,
            layer,
            Some(&(k_full, v_full)),
            abs_position,
            Some(backend),
        )?;

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        h_new = h_out;
    }

    // Amortised O(m) per-row append via ndarray::Array2::push_row.
    // Replaces the O(N²) per-step "Array2::zeros + .assign" rebuild
    // (bug A; see `engines/boundary_per_layer/mod.rs`).
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
        // Snapshot the absolute position at which the new overflow rows land
        // BEFORE appending to cold_encoded. Used by extend_cold_kv for RoPE.
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
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn run_prefill_no_window_returns_state_with_no_cold_tier() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (hidden, rs) = run_prefill(&weights, &ffn, &backend, &policy, None, &[0, 1, 2])
            .expect("prefill should succeed");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs.next_position, 3);
        assert!(rs.cold_encoded.is_none());
        assert!(rs.cold_kv.is_none());
        for slab in &rs.stored {
            assert_eq!(slab.shape()[0], 3);
        }
    }

    #[test]
    fn run_prefill_with_small_window_evicts_to_cold_tier() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(&weights, &ffn, &backend, &policy, Some(2), &[0, 1, 2]).unwrap();
        assert!(rs.cold_encoded.is_some(), "overflow → cold_encoded");
        assert!(rs.cold_kv.is_some(), "overflow → cold_kv pre-computed");
        let cold_kv = rs.cold_kv.as_ref().unwrap();
        for (k, _v) in cold_kv {
            assert_eq!(k.shape()[0], 1);
        }
    }

    #[test]
    fn run_decode_extends_hot_tier_when_below_window() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(&weights, &ffn, &backend, &policy, Some(4), &[0]).unwrap();
        assert!(rs.cold_encoded.is_none());

        let (hidden, rs_after) = run_decode(&weights, &ffn, &backend, &policy, rs, 1).unwrap();
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
        assert_eq!(rs_after.next_position, 2);
        for slab in &rs_after.stored {
            assert_eq!(slab.shape()[0], 2);
        }
        assert!(rs_after.cold_encoded.is_none());
    }

    #[test]
    fn run_decode_uses_cold_encoded_when_cold_kv_is_none() {
        // Defensive branch: cold_kv is None but cold_encoded is Some
        // and non-empty. In practice run_prefill always builds both
        // together, but the code carries a fallback path for the
        // case where they get desynchronised. Hand-construct that
        // state to exercise the decode path.
        use crate::engines::markov_residual_codec::codec::ColdResidualCodec;

        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);

        // First build a normal state with overflow.
        let (_, mut rs) =
            run_prefill(&weights, &ffn, &backend, &policy, Some(2), &[0, 1, 2]).unwrap();
        // Now wipe the pre-computed cold_kv. cold_encoded stays
        // populated. Decode should recompute K/V from the decoded
        // cold residuals.
        rs.cold_kv = None;
        // Sanity: the cold_encoded still carries the evicted row.
        assert!(rs.cold_encoded.as_ref().unwrap()[0].n_positions > 0);

        let _ = ColdResidualCodec::Bf16; // keep import live
        let (hidden, _) = run_decode(&weights, &ffn, &backend, &policy, rs, 3)
            .expect("decode should succeed without cold_kv");
        assert_eq!(hidden.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn run_decode_promotes_to_cold_tier_on_overflow() {
        // Prefill 3 with window=2 → 1 in cold. Decode 1 → 2 in cold.
        // Exercises Some(layers) arm of cold_encoded match.
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = NullFfn;
        let policy = BoundaryLayerPolicy::bf16_uniform("test", weights.num_layers);
        let (_, rs) = run_prefill(&weights, &ffn, &backend, &policy, Some(2), &[0, 1, 2]).unwrap();
        let initial = rs
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(initial, 1);

        let (_, rs_after) = run_decode(&weights, &ffn, &backend, &policy, rs, 3).unwrap();
        let after = rs_after
            .cold_encoded
            .as_ref()
            .map(|l| l[0].n_positions)
            .unwrap_or(0);
        assert_eq!(after, 2);
        assert_eq!(rs_after.next_position, 4);
    }
}
