//! Core residual-stream compute: prefill, decode step, K/V recomputation.

use larql_compute::{dot_proj_gpu, ComputeBackend, QuantFormat};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2, ArrayBase, ArrayView1, Data, Ix2};
use std::cell::RefCell;
use std::cmp::Ordering;

use super::store::RsStore;
use crate::profiler::EngineProfiler;
use larql_inference::attention::SharedKV;
use larql_inference::attention::{
    apply_rope_partial_at, run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::{add_bias, apply_norm, embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::residual::{rms_norm_heads, rms_norm_heads_no_weight};

#[derive(Clone, Copy)]
enum KvProjection {
    K,
    V,
}

#[derive(Clone)]
struct WalkKvSelection {
    select_layer: usize,
    top_k: usize,
    seq_len: usize,
    k_indices: Vec<Vec<usize>>,
    v_indices: Vec<Vec<usize>>,
}

thread_local! {
    static WALK_KV_SELECTION: RefCell<Option<WalkKvSelection>> = const { RefCell::new(None) };
}

pub struct RsPrefillResult {
    pub hidden: Array2<f32>,
    pub store: RsStore,
    pub memory_bytes: usize,
    pub window_tokens: usize,
}

pub fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS prefill");
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        hot_len: stored.first().map_or(0, |s| s.shape()[0]),
        stored,
        cold_residuals: None,
        cold_kv: None,
        cold_len: 0,
        hot_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    rs.finalise_hot_len_after_clip();
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = (0..num_layers)
            .map(|layer| {
                recompute_kv(weights, &cold[layer], layer, 0, backend, None)
                    .expect("cold K/V pre-computation failed")
            })
            .collect();
        // 2026-05-19 audit fix: route through the doubling-capacity
        // helper so cold_len is initialised correctly. Subsequent
        // decode-step overflows then append in amortised O(1).
        rs.append_cold_overflow(cold, Some(cold_kv));
        rs.cold_abs_start = 0;
    }

    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    rs_decode_step_inner(weights, new_token_id, rs, backend, None)
}

pub(crate) fn rs_decode_step_profiled(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    profiler: &mut EngineProfiler,
) -> Option<(Array2<f32>, RsStore)> {
    rs_decode_step_inner(weights, new_token_id, rs, backend, Some(profiler))
}

fn rs_decode_step_inner(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
) -> Option<(Array2<f32>, RsStore)> {
    use std::time::Instant;

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let t_step = if profiler.is_some() {
        Some(Instant::now())
    } else {
        None
    };
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let mut recompute_cold_us = 0.0f64;
    let mut recompute_hot_us = 0.0f64;
    let mut attention_us = 0.0f64;
    let mut ffn_us = 0.0f64;

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold_buf, v_cold_buf) = &cold_kv[layer];
            // 2026-05-19 audit fix: slice to cold_len, not shape()[0].
            // cold_kv now uses doubling-capacity (see RsStore::append_cold_overflow).
            let c = rs.cold_len;
            let k_cold = k_cold_buf.slice(s![..c, ..]);
            let v_cold = v_cold_buf.slice(s![..c, ..]);
            let t_hot = if profiler.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
            if let Some(t) = t_hot {
                recompute_hot_us += t.elapsed().as_secs_f64() * 1e6;
            }
            let kv_dim = k_cold_buf.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(&k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(&v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = if let Some(cold) = &rs.cold_residuals {
                // 2026-05-19 audit fix: slice to cold_len, not shape()[0].
                let s_cold = rs.cold_len;
                if s_cold > 0 {
                    let h_cold = cold[layer].slice(s![..s_cold, ..]);
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                    combined.slice_mut(s![..s_cold, ..]).assign(&h_cold);
                    combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                    (combined, rs.cold_abs_start)
                } else {
                    (h_hot.clone(), hot_abs_start)
                }
            } else {
                (h_hot.clone(), hot_abs_start)
            };
            let t_cold = if profiler.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let (k, v) = recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?;
            if let Some(t) = t_cold {
                recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
            }
            (k, v)
        };

        new_stored.push(h_new.clone());

        let t_attn = if profiler.is_some() {
            Some(Instant::now())
        } else {
            None
        };
        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_new,
            layer,
            Some(&(k_full, v_full)),
            abs_position,
            Some(backend),
        )?;
        if let Some(t) = t_attn {
            attention_us += t.elapsed().as_secs_f64() * 1e6;
        }

        let t_ffn = if profiler.is_some() {
            Some(Instant::now())
        } else {
            None
        };
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        if let Some(t) = t_ffn {
            ffn_us += t.elapsed().as_secs_f64() * 1e6;
        }
        h_new = h_out;
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.recompute_cold.total_us += recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.record(t_step);
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        hot_len: updated_stored.first().map_or(0, |s| s.shape()[0]),
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        cold_len: rs.cold_len,
        hot_kv: rs.hot_kv,
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    updated_rs.finalise_hot_len_after_clip();
    // 2026-05-19 audit fix: geometric-capacity cold append.
    // CPU walk path passes `evicted_kv = None` (cold_kv is rebuilt
    // from residuals on the next step), mirroring the prior behaviour
    // that invalidated cold_kv. See RsStore::append_cold_overflow.
    updated_rs.append_cold_overflow(overflow, None);

    Some((last_row(&h_new), updated_rs))
}

/// Recompute K/V from stored pre-layer residuals using `backend` for projection matmuls.
///
/// `index: Some(idx)` enables the Q4K-native fast path: per-row Q4K matvec
/// directly against the vindex's Q4K bytes, skipping the dequant-to-f32
/// step that's otherwise 8× the memory bandwidth. Quant-agnostic — the
/// backend's `quant_matvec` inspects the format byte and dispatches to
/// the right kernel (Q4K today; Q6K / future formats slot in
/// automatically). `None` keeps the f32 fallback for legacy callers.
pub fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
    index: Option<&VectorIndex>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };

    let h_norm = apply_norm(
        weights,
        h_stored,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let kv_dim = num_kv * head_dim;
    let hidden = weights.hidden_size;
    let seq_len = h_norm.shape()[0];

    let walk_kv_top_k = markov_walk_kv_top_k(layer, kv_dim);
    let walk_kv_select_at = markov_walk_kv_select_at();
    let should_cache_selection = walk_kv_select_at
        .is_some_and(|select_layer| select_layer == layer)
        && markov_walk_kv_requested_top_k(kv_dim).is_some();

    if should_cache_selection {
        if let Some((w_k, w_v)) = attn_kv_projection_weights(weights, layer) {
            let top_k = markov_walk_kv_requested_top_k(kv_dim)?;
            cache_walk_kv_selection(layer, top_k, &h_norm, w_k, w_v);
        }
    }

    // Q4K-native path: per-row matvec on the vindex's raw Q4K bytes.
    // Saves the dequant-to-f32 cost (8× memory bandwidth) when the
    // backend supports Q4K matvec and the vindex has Q4K attn data.
    //
    // Disabled when the experimental walk-KV path is active: that path
    // intentionally replaces the projection matmul with row-wise top-K
    // projection against the f32 tensor rows below.
    let q4k_path = if walk_kv_top_k.is_none() && !markov_kv_force_f32_projection() {
        index
            .and_then(|idx| idx.attn_kquant_layer_data(layer))
            .filter(|_| backend.supports_quant(::larql_compute::QuantFormat::Q4_K))
    } else {
        None
    };

    let used_q4k_projection = q4k_path.is_some();
    let (mut k, mut v) = if let Some(attn_data) = q4k_path {
        // attn_data: [(Q, fmt), (K, fmt), (V, fmt), (O, fmt)]
        let (k_bytes, k_fmt) = attn_data[1];
        let (v_bytes, v_fmt) = attn_data[2];
        let k_format = parse_quant_format(k_fmt)?;
        let v_format = parse_quant_format(v_fmt)?;

        let mut k_out = Array2::<f32>::zeros((seq_len, kv_dim));
        let mut v_out = Array2::<f32>::zeros((seq_len, kv_dim));
        for row_idx in 0..seq_len {
            let x_row = h_norm.row(row_idx);
            let x_slice = x_row.as_slice()?;
            let k_row = backend.quant_matvec(k_format, k_bytes, x_slice, kv_dim, hidden)?;
            let v_row = backend.quant_matvec(v_format, v_bytes, x_slice, kv_dim, hidden)?;
            k_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(k_row.iter())
                .for_each(|(o, &i)| *o = i);
            v_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(v_row.iter())
                .for_each(|(o, &i)| *o = i);
        }
        (k_out, v_out)
    } else {
        // f32 fallback: read dequantised weights from `weights.tensors`.
        let (w_k, w_v) = attn_kv_projection_weights(weights, layer)?;
        let (k, v) = if let Some(top_k) = walk_kv_top_k {
            let cached = walk_kv_select_at
                .filter(|&select_layer| select_layer != layer)
                .and_then(|select_layer| {
                    let k = walk_project_cached_topk(
                        &h_norm,
                        w_k,
                        top_k,
                        select_layer,
                        KvProjection::K,
                    )?;
                    let v = walk_project_cached_topk(
                        &h_norm,
                        w_v,
                        top_k,
                        select_layer,
                        KvProjection::V,
                    )?;
                    Some((k, v))
                });
            let (k, v) = if let Some(pair) = cached {
                pair
            } else {
                (
                    walk_project_topk(&h_norm, w_k, top_k)?,
                    walk_project_topk(&h_norm, w_v, top_k)?,
                )
            };
            (k, v)
        } else {
            let k = dot_proj_gpu(&h_norm, w_k, Some(backend));
            let v = dot_proj_gpu(&h_norm, w_v, Some(backend));
            (k, v)
        };
        (k, v)
    };

    if markov_walk_kv_diag_enabled() && markov_walk_kv_diag_layer(layer) {
        if let Some((w_k, w_v)) = attn_kv_projection_weights(weights, layer) {
            let dense_k = dot_proj_gpu(&h_norm, w_k, Some(backend));
            let dense_v = dot_proj_gpu(&h_norm, w_v, Some(backend));
            let walk_k = walk_project_topk(&h_norm, w_k, kv_dim)?;
            let walk_v = walk_project_topk(&h_norm, w_v, kv_dim)?;
            let path = if used_q4k_projection { "q4k" } else { "f32" };
            print_walk_kv_diag(layer, path, "K", "actual_vs_f32", &k, &dense_k);
            print_walk_kv_diag(layer, path, "V", "actual_vs_f32", &v, &dense_v);
            print_walk_kv_diag(layer, path, "K", "f32_vs_walk_full", &dense_k, &walk_k);
            print_walk_kv_diag(layer, path, "V", "f32_vs_walk_full", &dense_v, &walk_v);
            print_walk_kv_diag(layer, path, "K", "actual_vs_walk_full", &k, &walk_k);
            print_walk_kv_diag(layer, path, "V", "actual_vs_walk_full", &v, &walk_v);
        }
    }

    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v, bias);
    }
    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k, norm_w, num_kv, head_dim, qk_norm_off),
        None => k,
    };
    let k_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        arch.rope_base_for_layer(layer),
        arch.rotary_fraction_for_layer(layer),
        abs_start,
    );
    Some((k_rope, v))
}

/// Type alias for an attention K/V projection weight pair as stored in
/// `weights.tensors` (Arc-shared, `Ix2`). Used by `attn_kv_projection_weights`
/// to keep its signature readable; the clippy `type_complexity` lint
/// triggers on the inline tuple form.
type AttnKvWeightPair<'a> = (
    &'a ArrayBase<ndarray::OwnedArcRepr<f32>, Ix2>,
    &'a ArrayBase<ndarray::OwnedArcRepr<f32>, Ix2>,
);

fn attn_kv_projection_weights(
    weights: &ModelWeights,
    layer: usize,
) -> Option<AttnKvWeightPair<'_>> {
    let arch = &*weights.arch;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };
    Some((w_k, w_v))
}

/// Experimental Markov-KV walk gate.
///
/// Set `LARQL_MARKOV_WALK_KV_TOPK=N` to replace the K/V projection
/// matmul with row-wise top-K projection. By default it applies to all
/// layers; restrict it with `LARQL_MARKOV_WALK_KV_LAYERS=5-20,26`.
fn markov_walk_kv_top_k(layer: usize, kv_dim: usize) -> Option<usize> {
    let top_k = markov_walk_kv_requested_top_k(kv_dim)?;
    if let Some(select_layer) = markov_walk_kv_select_at() {
        if layer == select_layer {
            return None;
        }
    }
    if let Ok(spec) = std::env::var("LARQL_MARKOV_WALK_KV_LAYERS") {
        if !layer_in_spec(&spec, layer) {
            return None;
        }
    }
    Some(top_k)
}

fn markov_walk_kv_requested_top_k(kv_dim: usize) -> Option<usize> {
    let raw = std::env::var("LARQL_MARKOV_WALK_KV_TOPK").ok()?;
    let top_k = raw.trim().parse::<usize>().ok()?;
    if top_k == 0 {
        return None;
    }
    Some(top_k.min(kv_dim))
}

fn markov_walk_kv_select_at() -> Option<usize> {
    std::env::var("LARQL_MARKOV_WALK_KV_SELECT_AT")
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn markov_walk_kv_diag_enabled() -> bool {
    std::env::var("LARQL_MARKOV_WALK_KV_DIAG")
        .ok()
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

fn markov_kv_force_f32_projection() -> bool {
    std::env::var("LARQL_MARKOV_KV_FORCE_F32")
        .ok()
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
}

fn markov_walk_kv_diag_layer(layer: usize) -> bool {
    // `is_none_or` is MSRV 1.82; project pins MSRV 1.80. Equivalent
    // semantics: env-var absent → true (diag applies to all layers),
    // env-var present → check the comma-list.
    std::env::var("LARQL_MARKOV_WALK_KV_LAYERS")
        .ok()
        .map_or(true, |spec| layer_in_spec(&spec, layer))
}

fn layer_in_spec(spec: &str, layer: usize) -> bool {
    spec.split(',').any(|part| {
        let part = part.trim();
        if part.is_empty() {
            return false;
        }
        if let Some((start, end)) = part.split_once('-') {
            let Some(start) = start.trim().parse::<usize>().ok() else {
                return false;
            };
            let Some(end) = end.trim().parse::<usize>().ok() else {
                return false;
            };
            return start <= layer && layer <= end;
        }
        part.parse::<usize>() == Ok(layer)
    })
}

fn cache_walk_kv_selection<SK, SV>(
    select_layer: usize,
    top_k: usize,
    x: &Array2<f32>,
    w_k: &ArrayBase<SK, Ix2>,
    w_v: &ArrayBase<SV, Ix2>,
) where
    SK: Data<Elem = f32>,
    SV: Data<Elem = f32>,
{
    let k_indices = walk_select_topk_indices(x, w_k, top_k);
    let v_indices = walk_select_topk_indices(x, w_v, top_k);
    let selection = WalkKvSelection {
        select_layer,
        top_k,
        seq_len: x.shape()[0],
        k_indices,
        v_indices,
    };
    WALK_KV_SELECTION.with(|slot| {
        *slot.borrow_mut() = Some(selection);
    });
}

fn walk_select_topk_indices<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Vec<Vec<usize>>
where
    S: Data<Elem = f32>,
{
    (0..x.shape()[0])
        .map(|row_idx| {
            let pairs = walk_select_topk_scores(x.row(row_idx), weights, top_k);
            pairs.into_iter().map(|(idx, _)| idx).collect()
        })
        .collect()
}

fn walk_project_topk<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Option<Array2<f32>>
where
    S: Data<Elem = f32>,
{
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let rows = weights.shape()[0];
    if weights.shape()[1] != hidden || top_k == 0 {
        return None;
    }

    let mut out = Array2::<f32>::zeros((seq_len, rows));
    for row_idx in 0..seq_len {
        for (out_idx, score) in walk_select_topk_scores(x.row(row_idx), weights, top_k) {
            out[[row_idx, out_idx]] = score;
        }
    }
    Some(out)
}

fn walk_select_topk_scores<S>(
    x_row: ArrayView1<'_, f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
) -> Vec<(usize, f32)>
where
    S: Data<Elem = f32>,
{
    let rows = weights.shape()[0];
    let k = top_k.min(rows);
    let mut scores: Vec<(usize, f32)> = (0..rows)
        .map(|out_idx| (out_idx, dot_rows(x_row, weights.row(out_idx))))
        .collect();
    if k < scores.len() {
        scores.select_nth_unstable_by(k, compare_abs_desc);
        scores.truncate(k);
    }
    scores
}

fn walk_project_cached_topk<S>(
    x: &Array2<f32>,
    weights: &ArrayBase<S, Ix2>,
    top_k: usize,
    select_layer: usize,
    projection: KvProjection,
) -> Option<Array2<f32>>
where
    S: Data<Elem = f32>,
{
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let rows = weights.shape()[0];
    if weights.shape()[1] != hidden || top_k == 0 {
        return None;
    }

    let indices = WALK_KV_SELECTION.with(|slot| {
        let borrowed = slot.borrow();
        let selection = borrowed.as_ref()?;
        if selection.select_layer != select_layer
            || selection.top_k != top_k.min(rows)
            || selection.seq_len != seq_len
        {
            return None;
        }
        Some(match projection {
            KvProjection::K => selection.k_indices.clone(),
            KvProjection::V => selection.v_indices.clone(),
        })
    })?;

    let mut out = Array2::<f32>::zeros((seq_len, rows));
    for row_idx in 0..seq_len {
        let x_row = x.row(row_idx);
        for &out_idx in indices.get(row_idx)? {
            if out_idx >= rows {
                return None;
            }
            out[[row_idx, out_idx]] = dot_rows(x_row, weights.row(out_idx));
        }
    }
    Some(out)
}

fn compare_abs_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
}

fn dot_rows(a: ArrayView1<'_, f32>, b: ArrayView1<'_, f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, w)| x * w).sum()
}

fn print_walk_kv_diag(
    layer: usize,
    path: &str,
    projection: &str,
    label: &str,
    a: &Array2<f32>,
    b: &Array2<f32>,
) {
    let (max_abs, rms, cos) = array_diff_stats(a, b);
    eprintln!(
        "[walk-kv-diag] layer={layer:02} path={path} proj={projection} cmp={label} max_abs={max_abs:.6e} rms={rms:.6e} cos={cos:.9}"
    );
}

fn array_diff_stats(a: &Array2<f32>, b: &Array2<f32>) -> (f64, f64, f64) {
    if a.shape() != b.shape() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut max_abs = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    let mut n = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        let diff = x - y;
        max_abs = max_abs.max(diff.abs());
        sum_sq_diff += diff * diff;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
        n += 1;
    }
    let rms = if n == 0 {
        0.0
    } else {
        (sum_sq_diff / n as f64).sqrt()
    };
    let denom = norm_a.sqrt() * norm_b.sqrt();
    let cos = if denom == 0.0 { 1.0 } else { dot / denom };
    (max_abs, rms, cos)
}

fn parse_quant_format(fmt: &str) -> Option<QuantFormat> {
    match fmt {
        "Q4_K" => Some(QuantFormat::Q4_K),
        "Q4_KF" => Some(QuantFormat::Q4_KF),
        "Q6_K" => Some(QuantFormat::Q6_K),
        _ => None,
    }
}

/// Equivalent Standard KV memory in bytes for `seq_len` tokens (FP16).
pub fn kv_memory_bytes_for_seq(weights: &ModelWeights, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|l| {
            let kv_dim = arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l);
            seq_len * kv_dim * 2 * 2
        })
        .sum()
}

pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    // ── recompute_kv ──────────────────────────────────────────────────────────

    #[test]
    fn recompute_kv_returns_some_with_valid_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.5f32);
        let result = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None);
        assert!(
            result.is_some(),
            "recompute_kv should return Some with valid weights"
        );
    }

    #[test]
    fn recompute_kv_output_shape_correct() {
        let weights = make_test_weights();
        let seq_len = 4;
        let h = Array2::from_elem((seq_len, weights.hidden_size), 1.0f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[seq_len, kv_dim], "K shape mismatch");
        assert_eq!(v.shape(), &[seq_len, kv_dim], "V shape mismatch");
    }

    #[test]
    fn recompute_kv_output_is_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        assert!(
            k.iter().all(|v| v.is_finite()),
            "K contains non-finite values"
        );
        assert!(
            v.iter().all(|v| v.is_finite()),
            "V contains non-finite values"
        );
    }

    #[test]
    fn recompute_kv_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        // Different abs_start should produce different RoPE-applied K
        let (k0, _) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        let (k5, _) = recompute_kv(&weights, &h, 0, 5, &CpuBackend, None).unwrap();
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "RoPE at different positions should produce different K"
        );
    }

    #[test]
    fn walk_project_topk_full_k_matches_dense_projection() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, -2.0, 0.5, 0.25, 0.75, -1.0]).unwrap();
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.5, 1.0, -0.5, -1.0, 0.25, 0.75, 0.0, 2.0, 1.0, 1.5, -0.5, 0.25,
            ],
        )
        .unwrap();
        let walked = walk_project_topk(&x, &w, 4).unwrap();
        let dense = dot_proj_gpu(&x, &w, None);
        let max_diff = walked
            .iter()
            .zip(dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-6, "max_diff={max_diff}");
    }

    #[test]
    fn walk_project_topk_keeps_largest_absolute_outputs_per_row() {
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let w = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let walked = walk_project_topk(&x, &w, 2).unwrap();
        let non_zero: Vec<usize> = walked
            .row(0)
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v != 0.0).then_some(i))
            .collect();
        assert_eq!(non_zero, vec![1, 2]);
        assert_eq!(walked[[0, 1]], -6.0);
        assert_eq!(walked[[0, 2]], 6.0);
    }

    #[test]
    fn walk_project_cached_topk_reuses_selector_layer_indices() {
        WALK_KV_SELECTION.with(|slot| {
            *slot.borrow_mut() = None;
        });
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let selector_w_k = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 2.0, -2.0, 0.0, 0.0],
        )
        .unwrap();
        let selector_w_v = selector_w_k.clone();
        cache_walk_kv_selection(4, 2, &x, &selector_w_k, &selector_w_v);

        let later_w = Array2::from_shape_vec(
            (4, 3),
            vec![
                10.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let walked =
            walk_project_cached_topk(&x, &later_w, 2, 4, KvProjection::K).expect("cached walk");
        let non_zero: Vec<usize> = walked
            .row(0)
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v != 0.0).then_some(i))
            .collect();
        assert_eq!(non_zero, vec![1, 2]);
        assert_eq!(walked[[0, 1]], 40.0);
        assert_eq!(walked[[0, 2]], 90.0);
    }

    #[test]
    fn markov_walk_kv_layer_spec_accepts_ranges_and_singletons() {
        assert!(layer_in_spec("5-20", 5));
        assert!(layer_in_spec("5-20", 20));
        assert!(layer_in_spec(" 2, 5-7, 26 ", 6));
        assert!(layer_in_spec(" 2, 5-7, 26 ", 26));
        assert!(!layer_in_spec("5-20", 4));
        assert!(!layer_in_spec("5-20", 21));
        assert!(!layer_in_spec("x-y, 30", 29));
    }

    // ── rs_prefill ────────────────────────────────────────────────────────────

    #[test]
    fn rs_prefill_returns_correct_shape() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2], None, &CpuBackend);
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_prefill_stores_all_layers() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        assert_eq!(result.store.stored.len(), weights.num_layers);
        assert_eq!(result.store.next_position, 1);
    }

    #[test]
    fn rs_prefill_with_window_clips_hot_store() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2, 3, 4], Some(2), &CpuBackend);
        assert!(
            result.window_tokens <= 2,
            "window_tokens={} > 2",
            result.window_tokens
        );
    }

    // ── rs_decode_step ────────────────────────────────────────────────────────

    #[test]
    fn rs_decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        let (h, _) = rs_decode_step(&weights, 1, prefill.store, &CpuBackend).expect("decode step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_decode_step_advances_position() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1], None, &CpuBackend);
        assert_eq!(prefill.store.next_position, 2);
        let (_, rs2) = rs_decode_step(&weights, 2, prefill.store, &CpuBackend).unwrap();
        assert_eq!(rs2.next_position, 3);
        let (_, rs3) = rs_decode_step(&weights, 3, rs2, &CpuBackend).unwrap();
        assert_eq!(rs3.next_position, 4);
    }

    #[test]
    fn rs_decode_step_with_cold_kv_branch_produces_finite_output() {
        // Windowed prefill with prompt longer than window forces cold_kv
        // population (compute.rs lines 60-68), then decode hits the
        // `Some(cold_kv)` branch (lines 128-147) instead of the
        // cold-residual recomputation path.
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        assert!(
            prefill.store.cold_kv.is_some(),
            "expected cold_kv to be set"
        );
        let (h, rs2) = rs_decode_step(&weights, 4, prefill.store, &CpuBackend)
            .expect("decode_step over cold_kv");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        // After overflow merges into cold_residuals, cold_kv is cleared
        // (compute.rs line 260) so a second decode exercises the
        // cold_residuals-only branch (lines 149-160).
        let (h2, _) =
            rs_decode_step(&weights, 5, rs2, &CpuBackend).expect("decode_step over cold_residuals");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
        assert!(h2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kv_memory_bytes_for_seq_scales_linearly() {
        let weights = make_test_weights();
        let one = kv_memory_bytes_for_seq(&weights, 1);
        let ten = kv_memory_bytes_for_seq(&weights, 10);
        assert!(one > 0);
        assert_eq!(ten, one * 10, "kv memory must scale linearly with seq len");
    }

    // ── parse_quant_format pure helper (lines 384-391) ───────────────────

    #[test]
    fn parse_quant_format_recognises_q4k_q4kf_q6k() {
        assert!(matches!(
            parse_quant_format("Q4_K"),
            Some(QuantFormat::Q4_K)
        ));
        assert!(matches!(
            parse_quant_format("Q4_KF"),
            Some(QuantFormat::Q4_KF)
        ));
        assert!(matches!(
            parse_quant_format("Q6_K"),
            Some(QuantFormat::Q6_K)
        ));
    }

    #[test]
    fn parse_quant_format_unknown_returns_none() {
        assert!(parse_quant_format("Q8_0").is_none());
        assert!(parse_quant_format("F16").is_none());
        assert!(parse_quant_format("").is_none());
        assert!(parse_quant_format("Q4").is_none());
        assert!(parse_quant_format("nonsense").is_none());
    }

    // ── Profiler branches (lines 131, 137, 159, 164, 171, 178, 190, 195) ──
    //
    // Each timing branch fires only when `profiler.is_some()`. The existing
    // `with_profiling_enables_profiling_branch` test exercises one path;
    // these add coverage for the cold/hot/attn/ffn timing branches plus the
    // overflow-into-existing-cold-residuals merge path.

    #[test]
    fn profiled_decode_step_exercises_all_timing_branches() {
        use crate::profiler::EngineProfiler;
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        // Has cold_kv populated → exercises lines 130-147 (cold_kv branch
        // with profiler timing recompute_hot).
        assert!(prefill.store.cold_kv.is_some());
        let mut profiler = EngineProfiler::default();
        let result =
            rs_decode_step_profiled(&weights, 4, prefill.store, &CpuBackend, &mut profiler);
        assert!(result.is_some());
        // Profiler must record positive durations across all stages.
        assert!(profiler.recompute_hot.count > 0);
        assert!(profiler.attention.count > 0);
        assert!(profiler.ffn.count > 0);
        assert!(profiler.decode_total.count > 0);
    }

    #[test]
    fn profiled_decode_step_with_cold_residuals_only_path() {
        use crate::profiler::EngineProfiler;
        let weights = make_test_weights();
        // Two decodes from windowed prefill: first overflows + clears
        // cold_kv (compute.rs line 260); second hits the cold_residuals
        // branch (lines 149-160) under profiling.
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        let (_, rs2) = rs_decode_step(&weights, 4, prefill.store, &CpuBackend).unwrap();
        assert!(
            rs2.cold_kv.is_none(),
            "cold_kv should be cleared after overflow"
        );
        let mut profiler = EngineProfiler::default();
        let result = rs_decode_step_profiled(&weights, 5, rs2, &CpuBackend, &mut profiler);
        assert!(result.is_some());
        // cold_residuals branch exercises recompute_cold counter (line 171).
        assert!(profiler.recompute_cold.count > 0);
    }

    // ── Pure helpers ────────────────────────────────────────────────────────

    #[test]
    fn dot_rows_basic_arithmetic() {
        let a = ndarray::arr1(&[1.0f32, 2.0, 3.0]);
        let b = ndarray::arr1(&[4.0f32, 5.0, 6.0]);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((dot_rows(a.view(), b.view()) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn compare_abs_desc_orders_by_absolute_magnitude() {
        let a = (0usize, -5.0f32);
        let b = (1usize, 3.0f32);
        // |a| > |b| so a comes before b under descending sort.
        assert_eq!(compare_abs_desc(&a, &b), Ordering::Less);
        assert_eq!(compare_abs_desc(&b, &a), Ordering::Greater);
        // Tie: NaN/Equal fallback returns Equal.
        let c = (2usize, 5.0f32);
        let d = (3usize, -5.0f32);
        assert_eq!(compare_abs_desc(&c, &d), Ordering::Equal);
    }

    #[test]
    fn array_diff_stats_identical_arrays_returns_zero_diff_and_unit_cos() {
        // Identical arrays → max_abs=0, rms=0, cos=1.
        let a = Array2::<f32>::from_elem((2, 3), 1.5);
        let b = a.clone();
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        assert!(max_abs.abs() < 1e-12);
        assert!(rms.abs() < 1e-12);
        assert!(
            (cos - 1.0).abs() < 1e-9,
            "cos should be 1 for identical, got {cos}"
        );
    }

    #[test]
    fn array_diff_stats_reports_max_abs_and_rms() {
        let a = Array2::<f32>::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let b = Array2::<f32>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        // max_abs = 3, rms = sqrt(((-1)^2 + (-2)^2 + (-3)^2) / 3) = sqrt(14/3)
        assert!((max_abs - 3.0).abs() < 1e-9);
        assert!((rms - (14.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        // a is all zeros so cosine has denom=0 → returns 1.0 sentinel.
        assert!((cos - 1.0).abs() < 1e-9, "all-zeros a → cos sentinel = 1");
    }

    #[test]
    fn array_diff_stats_mismatched_shape_returns_nan_tuple() {
        let a = Array2::<f32>::zeros((2, 3));
        let b = Array2::<f32>::zeros((3, 2));
        let (max_abs, rms, cos) = array_diff_stats(&a, &b);
        assert!(max_abs.is_nan() && rms.is_nan() && cos.is_nan());
    }

    #[test]
    fn layer_in_spec_accepts_singleton_and_ranges() {
        // Direct test of the spec parser. Covers "5", "5-7", "1,5-7,9"
        // forms — the helper layered under markov_walk_kv_diag_layer
        // and markov_walk_kv_top_k env-var paths.
        assert!(layer_in_spec("5", 5));
        assert!(!layer_in_spec("5", 6));
        assert!(layer_in_spec("5-7", 5));
        assert!(layer_in_spec("5-7", 6));
        assert!(layer_in_spec("5-7", 7));
        assert!(!layer_in_spec("5-7", 8));
        assert!(layer_in_spec("1,5-7,9", 1));
        assert!(layer_in_spec("1,5-7,9", 6));
        assert!(layer_in_spec("1,5-7,9", 9));
        assert!(!layer_in_spec("1,5-7,9", 3));
    }

    #[test]
    fn layer_in_spec_rejects_malformed_input() {
        // Non-numeric pieces should not crash and should return false.
        assert!(!layer_in_spec("abc", 5));
        assert!(!layer_in_spec("", 5));
    }

    #[test]
    fn print_walk_kv_diag_runs_without_panicking() {
        // Pure logging helper. The body just prints diagnostic stats;
        // exercising it produces console output but no observable
        // state change. Coverage credit for the function body.
        let a = Array2::<f32>::from_elem((2, 4), 1.0f32);
        let b = Array2::<f32>::from_elem((2, 4), 0.5f32);
        print_walk_kv_diag(0, "test_path", "K", "test_label", &a, &b);
    }

    // ── Env-var-gated walk-KV paths ───────────────────────────────────────────
    //
    // These tests cover the `LARQL_MARKOV_WALK_KV_*` /
    // `LARQL_MARKOV_KV_FORCE_F32` paths in `recompute_kv` +
    // `markov_walk_kv_*` helpers. The env vars are read at decode
    // time via `std::env::var`, so each test is `#[serial]` (the
    // `serial_test` crate) and resets the relevant vars at the start
    // to prevent prior-test leakage.

    fn clear_markov_walk_env() {
        for var in [
            "LARQL_MARKOV_WALK_KV_TOPK",
            "LARQL_MARKOV_WALK_KV_LAYERS",
            "LARQL_MARKOV_WALK_KV_SELECT_AT",
            "LARQL_MARKOV_WALK_KV_DIAG",
            "LARQL_MARKOV_KV_FORCE_F32",
        ] {
            std::env::remove_var(var);
        }
    }

    #[test]
    #[serial_test::serial]
    fn markov_walk_kv_requested_top_k_parses_clamps_and_rejects_zero() {
        clear_markov_walk_env();
        // Not set → None.
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        // Set → parsed; clamp to kv_dim.
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "8");
        assert_eq!(markov_walk_kv_requested_top_k(32), Some(8));
        assert_eq!(
            markov_walk_kv_requested_top_k(4),
            Some(4),
            "clamp to kv_dim"
        );
        // Zero is rejected.
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "0");
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        // Garbage rejects.
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "abc");
        assert_eq!(markov_walk_kv_requested_top_k(32), None);
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn markov_walk_kv_select_at_parses_layer_index() {
        clear_markov_walk_env();
        assert_eq!(markov_walk_kv_select_at(), None);
        std::env::set_var("LARQL_MARKOV_WALK_KV_SELECT_AT", "7");
        assert_eq!(markov_walk_kv_select_at(), Some(7));
        std::env::set_var("LARQL_MARKOV_WALK_KV_SELECT_AT", "bad");
        assert_eq!(markov_walk_kv_select_at(), None);
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn markov_walk_kv_diag_enabled_accepts_truthy_strings() {
        clear_markov_walk_env();
        assert!(!markov_walk_kv_diag_enabled());
        for val in ["1", "true", "TRUE", "yes", "on"] {
            std::env::set_var("LARQL_MARKOV_WALK_KV_DIAG", val);
            assert!(markov_walk_kv_diag_enabled(), "should accept {val}");
        }
        for val in ["0", "false", "no"] {
            std::env::set_var("LARQL_MARKOV_WALK_KV_DIAG", val);
            assert!(!markov_walk_kv_diag_enabled(), "should reject {val}");
        }
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn markov_kv_force_f32_projection_reads_env() {
        clear_markov_walk_env();
        assert!(!markov_kv_force_f32_projection());
        std::env::set_var("LARQL_MARKOV_KV_FORCE_F32", "1");
        assert!(markov_kv_force_f32_projection());
        std::env::set_var("LARQL_MARKOV_KV_FORCE_F32", "no");
        assert!(!markov_kv_force_f32_projection());
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn markov_walk_kv_diag_layer_respects_layers_spec() {
        clear_markov_walk_env();
        // Absent → every layer counts.
        assert!(markov_walk_kv_diag_layer(0));
        assert!(markov_walk_kv_diag_layer(99));
        // Restricted to range → outside layers excluded.
        std::env::set_var("LARQL_MARKOV_WALK_KV_LAYERS", "3-5");
        assert!(markov_walk_kv_diag_layer(4));
        assert!(!markov_walk_kv_diag_layer(0));
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn markov_walk_kv_top_k_honours_layers_and_select_at_gates() {
        clear_markov_walk_env();
        // Not requested → None.
        assert_eq!(markov_walk_kv_top_k(0, 32), None);
        // Requested but layer not in spec → None.
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "4");
        std::env::set_var("LARQL_MARKOV_WALK_KV_LAYERS", "5-7");
        assert_eq!(markov_walk_kv_top_k(0, 32), None);
        assert_eq!(markov_walk_kv_top_k(6, 32), Some(4));
        // SELECT_AT == layer → None (selector layer keeps the dense path).
        std::env::remove_var("LARQL_MARKOV_WALK_KV_LAYERS");
        std::env::set_var("LARQL_MARKOV_WALK_KV_SELECT_AT", "6");
        assert_eq!(markov_walk_kv_top_k(6, 32), None);
        assert_eq!(markov_walk_kv_top_k(7, 32), Some(4));
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn recompute_kv_force_f32_disables_q4k_path() {
        clear_markov_walk_env();
        std::env::set_var("LARQL_MARKOV_KV_FORCE_F32", "1");
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[2, kv_dim]);
        assert_eq!(v.shape(), &[2, kv_dim]);
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn recompute_kv_topk_routes_through_walk_projection() {
        clear_markov_walk_env();
        // top_k forces the walk-KV f32 fallback regardless of backend.
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "2");
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
        let result = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None);
        assert!(result.is_some());
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn recompute_kv_select_at_uses_cached_indices_on_later_layers() {
        clear_markov_walk_env();
        std::env::set_var("LARQL_MARKOV_WALK_KV_TOPK", "2");
        std::env::set_var("LARQL_MARKOV_WALK_KV_SELECT_AT", "0");
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.25f32);
        // Layer 0: should_cache_selection fires, populates the
        // thread-local; layer 1: walk_project_cached_topk reads it.
        let _ = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None);
        if weights.num_layers >= 2 {
            let result = recompute_kv(&weights, &h, 1, 0, &CpuBackend, None);
            assert!(result.is_some());
        }
        clear_markov_walk_env();
    }

    #[test]
    #[serial_test::serial]
    fn recompute_kv_diag_fires_when_enabled() {
        clear_markov_walk_env();
        std::env::set_var("LARQL_MARKOV_WALK_KV_DIAG", "1");
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        // Diagnostic path runs `print_walk_kv_diag` — body is
        // observability output; we just verify no panic and a valid
        // result.
        let result = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None);
        assert!(result.is_some());
        clear_markov_walk_env();
    }

    #[test]
    fn decode_step_with_empty_cold_residuals_falls_through() {
        // Line 159: `(h_hot.clone(), hot_abs_start)` when cold tier exists
        // but s_cold == 0 (rare; happens if the engine ever clips out the
        // last cold row). Build the state by hand.
        use larql_inference::attention::SharedKV;
        use ndarray::Array2;
        let weights = make_test_weights();
        // Construct a store with cold_residuals = Some(vec![empty]) per
        // layer and cold_kv = None. The decode loop must take the "empty
        // cold" else branch (line 159).
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        let stored: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((1, hidden)))
            .collect();
        let cold_residuals: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((0, hidden)))
            .collect();
        let _ = (kv_dim, SharedKV::default()); // silence unused warnings if any
        let store = RsStore {
            hot_len: 1,
            stored,
            cold_residuals: Some(cold_residuals),
            cold_kv: None,
            hot_kv: None,
            cold_abs_start: 0,
            next_position: 1,
            max_window: None,
            cold_len: 0,
        };
        let result = rs_decode_step(&weights, 0, store, &CpuBackend);
        assert!(result.is_some());
        let (h, _) = result.unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }
}
