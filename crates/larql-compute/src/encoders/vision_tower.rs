//! SigLIP CPU forward pass.
//!
//! Reference: HuggingFace transformers `SiglipVisionTransformer` /
//! `SiglipVisionModel`. Bidirectional attention (no causal mask), GELU
//! activation, LayerNorm with both scale and bias, biases on every
//! projection.
//!
//! Phase 1b output: a `(num_patches, hidden_size)` matrix — for the
//! Gemma 3 4B-it SigLIP variant that is `(4096, 1152)`. The
//! `multi_modal_projector` connector (Phase 1c) handles the 4096 → 256
//! downsample and the projection into LM hidden size.
//!
//! `VisionEncoder` borrows `&'w VisionWeights` — no copy. The encoder
//! is intentionally stateless beyond the borrow.

use larql_models::encoders::vision_tower::{VisionLayerWeights, VisionWeights};
use larql_models::{ModalEncoder, ModalInput};
use ndarray::{s, Array2};

use super::super::forward::ops::{add_bias, softmax};
use super::super::residual::layer_norm_eps;

/// SigLIP vision encoder borrow.
pub struct VisionEncoder<'w> {
    pub weights: &'w VisionWeights,
}

impl<'w> VisionEncoder<'w> {
    pub fn new(weights: &'w VisionWeights) -> Self {
        Self { weights }
    }
}

impl ModalEncoder for VisionEncoder<'_> {
    fn family(&self) -> &str {
        if self.weights.config.is_siglip2() {
            "siglip2"
        } else {
            "siglip"
        }
    }

    fn encoder_hidden_size(&self) -> usize {
        self.weights.config.hidden_size
    }

    fn encode(&self, input: ModalInput<'_>) -> Result<Array2<f32>, String> {
        let (rgb_u8, width, height) = match input {
            ModalInput::Image { rgb, width, height } => (rgb, width, height),
            ModalInput::Audio { .. } => {
                return Err("SigLIP is an image encoder; audio input is unsupported".to_string());
            }
        };
        let cfg = &self.weights.config;
        if width != cfg.image_size || height != cfg.image_size {
            return Err(format!(
                "SigLIP expects {0}x{0} RGB input; got {width}x{height}. \
                 Resize at the host before calling encode().",
                cfg.image_size
            ));
        }
        let expected_bytes = width * height * cfg.num_channels;
        if rgb_u8.len() != expected_bytes {
            return Err(format!(
                "SigLIP input byte count mismatch: expected {expected_bytes} \
                 ({width}x{height}x{} channels), got {}",
                cfg.num_channels,
                rgb_u8.len()
            ));
        }
        Ok(forward(self.weights, rgb_u8))
    }
}

/// SigLIP forward pass on a pre-validated u8 RGB buffer.
///
/// Pipeline:
///   1. Normalize u8 [0, 255] → f32 [-1, 1]  (HF SigLIP image_mean=[0.5]·3, image_std=[0.5]·3)
///   2. Patchify (Conv2D) → `(num_patches, hidden)`
///   3. Add learned absolute position embedding
///   4. 27 transformer blocks (LayerNorm → MHA → residual → LayerNorm → MLP → residual)
///   5. post_layernorm
fn forward(weights: &VisionWeights, rgb_u8: &[u8]) -> Array2<f32> {
    let cfg = &weights.config;
    let pixels = normalize_rgb(rgb_u8, cfg.image_size, cfg.num_channels);
    let mut h = patchify(weights, &pixels);
    add_position_embed(&mut h, &weights.position_embed);

    let head_dim = cfg.head_dim();
    let num_heads = cfg.num_attention_heads;
    let eps = cfg.layer_norm_eps;
    let hidden_act = cfg.hidden_act.as_str();

    for layer in &weights.layers {
        // ── Pre-attention residual block ──
        let h_norm = layer_norm_eps(
            &h,
            Some(&layer.layer_norm1.weight),
            Some(&layer.layer_norm1.bias),
            eps,
        );
        let attn_out = bidirectional_attention(&h_norm, layer, num_heads, head_dim);
        h = h + attn_out;

        // ── Pre-MLP residual block ──
        let h_norm = layer_norm_eps(
            &h,
            Some(&layer.layer_norm2.weight),
            Some(&layer.layer_norm2.bias),
            eps,
        );
        let mlp_out = encoder_mlp(&h_norm, layer, hidden_act);
        h = h + mlp_out;
    }

    // Final post-layernorm before handing off to the connector.
    layer_norm_eps(
        &h,
        Some(&weights.post_layernorm.weight),
        Some(&weights.post_layernorm.bias),
        eps,
    )
}

/// u8 RGB row-major (H × W × C) → f32 (num_patches positions later, but
/// we first lay out as (C, H, W) for patchification).
///
/// SigLIP's image processor uses image_mean = [0.5, 0.5, 0.5] and
/// image_std = [0.5, 0.5, 0.5], so the normalization is (x/255 - 0.5)/0.5
/// = x/127.5 - 1, mapping [0, 255] → [-1, 1].
fn normalize_rgb(rgb_u8: &[u8], side: usize, channels: usize) -> Array2<f32> {
    // Output shape: (channels, side*side). Channel-first makes the
    // per-channel patch slicing in patchify a contiguous index walk.
    let mut out = Array2::<f32>::zeros((channels, side * side));
    for y in 0..side {
        for x in 0..side {
            let pixel_idx = y * side + x;
            for c in 0..channels {
                // HF processor produces channel-last bytes (H, W, C); pull
                // each channel out of the interleaved buffer.
                let byte = rgb_u8[(y * side + x) * channels + c];
                out[[c, pixel_idx]] = (byte as f32) / 127.5 - 1.0;
            }
        }
    }
    out
}

/// Apply the patch-embedding Conv2D as a matmul.
///
/// Conv2D weight shape: `(hidden, channels, patch, patch)`. Flatten the
/// trailing three axes to get `(hidden, channels * patch * patch)`,
/// then for every patch position `(i, j)` extract the
/// `(channels * patch * patch)` pixel vector and dot it.
///
/// Returns `(num_patches, hidden)`.
fn patchify(weights: &VisionWeights, pixels: &Array2<f32>) -> Array2<f32> {
    let cfg = &weights.config;
    let side = cfg.image_size;
    let p = cfg.patch_size;
    let pps = cfg.patches_per_side();
    let num_patches = cfg.num_patches();
    let channels = cfg.num_channels;
    let hidden = cfg.hidden_size;
    let patch_flat = channels * p * p;

    // (hidden, channels, patch, patch) → (hidden, channels * patch * patch)
    let kernel_flat = weights
        .patch_embed
        .view()
        .into_shape_with_order((hidden, patch_flat))
        .expect("patch_embed reshape: (hidden, C*P*P)");

    // For each patch, build a (channels * patch * patch) row in
    // (num_patches, channels * patch * patch) order.
    let mut patches = Array2::<f32>::zeros((num_patches, patch_flat));
    for pi in 0..pps {
        for pj in 0..pps {
            let patch_idx = pi * pps + pj;
            let y0 = pi * p;
            let x0 = pj * p;
            let mut col = 0usize;
            for c in 0..channels {
                for dy in 0..p {
                    for dx in 0..p {
                        let pixel_idx = (y0 + dy) * side + (x0 + dx);
                        patches[[patch_idx, col]] = pixels[[c, pixel_idx]];
                        col += 1;
                    }
                }
            }
        }
    }

    // out = patches @ kernel_flat.T  → (num_patches, hidden)
    let mut out = patches.dot(&kernel_flat.t());
    add_bias(&mut out, &weights.patch_embed_bias);
    out
}

fn add_position_embed(h: &mut Array2<f32>, pos_embed: &Array2<f32>) {
    debug_assert_eq!(h.shape(), pos_embed.shape());
    *h += pos_embed;
}

/// Bidirectional multi-head attention (no causal mask) with biased Q/K/V/O.
fn bidirectional_attention(
    x: &Array2<f32>,
    layer: &VisionLayerWeights,
    num_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
    let n = x.nrows();
    let hidden = num_heads * head_dim;

    let q = proj_with_bias(x, &layer.q_proj.weight, &layer.q_proj.bias);
    let k = proj_with_bias(x, &layer.k_proj.weight, &layer.k_proj.bias);
    let v = proj_with_bias(x, &layer.v_proj.weight, &layer.v_proj.bias);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out_flat = Array2::<f32>::zeros((n, hidden));

    for head in 0..num_heads {
        let col0 = head * head_dim;
        let col1 = col0 + head_dim;
        let q_h = q.slice(s![.., col0..col1]).to_owned();
        let k_h = k.slice(s![.., col0..col1]).to_owned();
        let v_h = v.slice(s![.., col0..col1]).to_owned();

        // scores = q_h @ k_h.T → (n, n)
        let mut scores = q_h.dot(&k_h.t());
        scores.mapv_inplace(|x| x * scale);

        // Row-wise softmax (no mask).
        let mut attn = Array2::<f32>::zeros((n, n));
        for r in 0..n {
            let row = scores.row(r);
            let s_vec = softmax(row.as_slice().expect("scores row contiguous"));
            for (c, val) in s_vec.iter().enumerate() {
                attn[[r, c]] = *val;
            }
        }

        // head_out = attn @ v_h → (n, head_dim)
        let head_out = attn.dot(&v_h);
        for r in 0..n {
            for d in 0..head_dim {
                out_flat[[r, col0 + d]] = head_out[[r, d]];
            }
        }
    }

    proj_with_bias(&out_flat, &layer.out_proj.weight, &layer.out_proj.bias)
}

/// FFN with configurable activation. SigLIP uses `gelu_pytorch_tanh`;
/// SigLIP2 may use `gelu` or `silu`.
fn encoder_mlp(x: &Array2<f32>, layer: &VisionLayerWeights, hidden_act: &str) -> Array2<f32> {
    let h1 = proj_with_bias(x, &layer.fc1.weight, &layer.fc1.bias);
    let h1 = match hidden_act {
        "silu" => h1.mapv(|v| v / (1.0 + (-v).exp())),
        "gelu" => h1.mapv(crate::ffn::gelu_tanh),
        _ => h1.mapv(crate::ffn::gelu_tanh),
    };
    proj_with_bias(&h1, &layer.fc2.weight, &layer.fc2.bias)
}

/// `x @ w.T + b` with biased projection. `w` is `(out, in)`, `x` is `(n, in)`.
fn proj_with_bias(x: &Array2<f32>, w: &Array2<f32>, b: &[f32]) -> Array2<f32> {
    let mut out = x.dot(&w.t());
    add_bias(&mut out, b);
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! ## Testing notes for vision encoders (read before touching synthetic fixtures)
    //!
    //! These traps are not SigLIP-specific — every transformer encoder
    //! with LayerNorm/RMSNorm hits them. Re-read before writing tests for
    //! Granite SigLIP2, Qwen ViT, Whisper, or the Gemma3 projector.
    //!
    //!   1. **Uniform synthetic weights collapse LayerNorm.** A constant
    //!      weight matrix produces constant rows after the first projection;
    //!      LayerNorm on a constant row → 0 (variance=0). Every subsequent
    //!      layer reads zeros. Use an asymmetric pattern (`wval(i,j)` here).
    //!
    //!   2. **Uniform pixel inputs collapse the same way.** A solid-grey
    //!      image gives constant rows after patchify; LayerNorm wipes them.
    //!      Use a gradient pattern (`gradient_rgb`).
    //!
    //!   3. **Constant-across-hidden-dims perturbations are mean-centered
    //!      out by LayerNorm.** A position-embed perturbation of
    //!      `[c; hidden]` is exactly removed by LayerNorm's mean step. Make
    //!      perturbations vary across the hidden axis.
    //!
    //!   4. **Synthetic fixtures must vary across instances, not just runs.**
    //!      A "two different images" assertion silently passes when the
    //!      two fixtures are built by the same factory with no
    //!      per-instance differentiation. Caught 2026-05-24 in the
    //!      `prepare_multimodal_input` two-image test: `write_synth_png`
    //!      generated byte-identical PNGs across calls because the pixel
    //!      pattern didn't depend on the filename. Salt the synthesised
    //!      content with the instance identifier (filename, index, seed)
    //!      so distinct fixtures actually carry distinct bytes.
    //!
    //! ## Ignored tests
    //!
    //! Real-checkpoint tests are gated with `#[ignore]` and require local
    //! fixtures. They are not CI-suitable — 14 minutes on debug CPU is fine
    //! as a one-shot correctness signal but useless as a regression net,
    //! and they need ~9 GB of weights on disk. Run locally only:
    //!
    //! ```text
    //! cargo test -p larql-compute --lib encoders::siglip::tests:: -- --ignored
    //! ```

    use super::*;
    use larql_models::encoders::vision_tower::{
        LayerNormWeights, ProjWithBias, VisionConfig, VisionLayerWeights, VisionWeights,
    };
    use ndarray::Array4;

    /// Tiny synthetic SigLIP — fast, no checkpoint required. Geometry:
    /// 4×4 image, 2×2 patches → 4 positions; hidden=8, heads=2, head_dim=4;
    /// 1 layer; intermediate=16. Just enough to exercise every code path.
    ///
    /// Weights are deterministic but **asymmetric** — uniform weights make
    /// LayerNorm collapse every row to zero (constant row → variance 0),
    /// which masks any pixel-dependent or position-dependent signal. The
    /// pseudo-random pattern below produces non-constant rows after the
    /// first projection, so the transformer blocks actually do work.
    fn synth_weights() -> VisionWeights {
        let config = VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            patch_size: 2,
            image_size: 4,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            norm_type: "layer_norm".to_string(),
        };
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let num_patches = config.num_patches();
        let channels = config.num_channels;
        let p = config.patch_size;

        // Deterministic asymmetric pattern centred near zero with small
        // magnitude — emulates a real init distribution well enough for
        // sanity tests.
        let wval = |i: usize, j: usize| {
            let h = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)).wrapping_mul(2654435761);
            ((h & 0xff) as i32 - 128) as f32 / 800.0
        };
        let pattern_2d = |rows: usize, cols: usize| {
            Array2::<f32>::from_shape_fn((rows, cols), |(i, j)| wval(i, j))
        };
        let zeros_vec = |n: usize| vec![0.0f32; n];
        let proj = |out: usize, in_: usize| ProjWithBias {
            weight: pattern_2d(out, in_),
            bias: zeros_vec(out),
        };
        let lnorm = |n: usize| LayerNormWeights {
            weight: vec![1.0f32; n],
            bias: zeros_vec(n),
        };
        let patch_embed =
            Array4::<f32>::from_shape_fn((hidden, channels, p, p), |(h, c, dy, dx)| {
                wval(h * 13, c * 7 + dy * 3 + dx)
            });

        VisionWeights {
            config,
            patch_embed,
            patch_embed_bias: zeros_vec(hidden),
            position_embed: Array2::<f32>::zeros((num_patches, hidden)),
            layers: vec![VisionLayerWeights {
                layer_norm1: lnorm(hidden),
                q_proj: proj(hidden, hidden),
                k_proj: proj(hidden, hidden),
                v_proj: proj(hidden, hidden),
                out_proj: proj(hidden, hidden),
                layer_norm2: lnorm(hidden),
                fc1: proj(inter, hidden),
                fc2: proj(hidden, inter),
            }],
            post_layernorm: lnorm(hidden),
        }
    }

    #[test]
    fn family_and_hidden_size_match_weights() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        assert_eq!(enc.family(), "siglip");
        assert_eq!(enc.encoder_hidden_size(), 8);
    }

    #[test]
    fn encode_output_shape_matches_num_patches_x_hidden() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        let rgb = vec![128u8; 4 * 4 * 3];
        let out = enc
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 4,
                height: 4,
            })
            .expect("encode");
        assert_eq!(out.shape(), &[4, 8], "(num_patches, hidden_size)");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "encoder produced non-finite values"
        );
    }

    #[test]
    fn encode_rejects_wrong_resolution() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        let rgb = vec![0u8; 6 * 6 * 3];
        let err = enc
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 6,
                height: 6,
            })
            .expect_err("wrong size should fail");
        assert!(err.contains("4x4") || err.contains("Resize"));
    }

    #[test]
    fn encode_rejects_byte_count_mismatch() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        // Right side length but missing bytes.
        let rgb = vec![0u8; 4 * 4 * 3 - 1];
        let err = enc
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 4,
                height: 4,
            })
            .expect_err("byte mismatch should fail");
        assert!(err.contains("byte count"));
    }

    #[test]
    fn encode_rejects_audio() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        let samples = [0.0f32; 16];
        let err = enc
            .encode(ModalInput::Audio { samples: &samples })
            .expect_err("audio should fail");
        assert!(err.contains("audio"));
    }

    /// Build a non-uniform RGB image so LayerNorm doesn't collapse the
    /// per-row signal to zero. A simple per-pixel gradient is enough.
    fn gradient_rgb(side: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(side * side * 3);
        for y in 0..side {
            for x in 0..side {
                // 3 distinct gradients per channel — the variance within
                // every patch row is non-zero, so LayerNorm produces signal.
                out.push(((y * 53 + x * 7) & 0xff) as u8);
                out.push(((y * 17 + x * 31) & 0xff) as u8);
                out.push(((y * 41 + x * 11) & 0xff) as u8);
            }
        }
        out
    }

    #[test]
    fn different_pixels_produce_different_outputs() {
        // Sanity check — if the forward pass collapsed to a constant,
        // shape would still match but two different RGB inputs would
        // yield the same output. Uniform-pixel inputs fool this check
        // when synthetic weights are symmetric (LayerNorm of a constant
        // row → 0), so use gradient images instead.
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        let rgb_a = gradient_rgb(4);
        let mut rgb_b = rgb_a.clone();
        // Perturb a single pixel to make the inputs differ.
        rgb_b[0] = rgb_b[0].wrapping_add(73);
        rgb_b[1] = rgb_b[1].wrapping_add(149);
        let out_a = enc
            .encode(ModalInput::Image {
                rgb: &rgb_a,
                width: 4,
                height: 4,
            })
            .unwrap();
        let out_b = enc
            .encode(ModalInput::Image {
                rgb: &rgb_b,
                width: 4,
                height: 4,
            })
            .unwrap();
        let differ = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "different RGB inputs should yield different encoder outputs"
        );
    }

    #[test]
    fn position_embed_addition_is_visible() {
        // Toggle the position embedding on and off, holding the image
        // fixed. The position-embed perturbation MUST vary across
        // hidden dims (a constant offset is removed by LayerNorm's
        // mean-centering, which would mask the signal entirely).
        let rgb = gradient_rgb(4);
        let mut w_with = synth_weights();
        let w_without = synth_weights();
        let hidden = w_with.config.hidden_size;
        for j in 0..hidden {
            // Non-constant pattern across hidden dims so LayerNorm
            // doesn't subtract it away as a mean offset.
            w_with.position_embed[[1, j]] = (j as f32) * 0.1 - 0.35;
        }
        let out_with = VisionEncoder::new(&w_with)
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 4,
                height: 4,
            })
            .unwrap();
        let out_without = VisionEncoder::new(&w_without)
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 4,
                height: 4,
            })
            .unwrap();
        let differ = out_with
            .iter()
            .zip(out_without.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            differ,
            "position embedding addition should perturb encoder output"
        );
    }

    // ── End-to-end against real Gemma 3 4B-it SigLIP weights ───────────
    //
    // Ignored by default — requires the local HF cache and is slow
    // (~30 s to load 640 MB then run 27 layers over 4096 patches on CPU).

    #[test]
    #[ignore = "requires google/gemma-3-4b-it in the local HF cache"]
    fn forward_real_gemma3_4b_it_siglip_shape_and_finite() {
        use larql_models::encoders::vision_tower::{
            load_vision_tower_from_safetensors, VisionConfig,
        };
        let snap = "/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767";
        if !std::path::Path::new(snap).exists() {
            eprintln!("snapshot not present, skipping: {snap}");
            return;
        }
        let config = VisionConfig {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            patch_size: 14,
            image_size: 896,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            norm_type: "layer_norm".to_string(),
        };
        let w = load_vision_tower_from_safetensors(snap, config).expect("load");
        let enc = VisionEncoder::new(&w);
        // Mid-grey image — every pixel = 128.
        let rgb = vec![128u8; 896 * 896 * 3];
        let out = enc
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: 896,
                height: 896,
            })
            .expect("encode");
        assert_eq!(out.shape(), &[4096, 1152]);
        assert!(out.iter().all(|v| v.is_finite()));
        // Output rows should not collapse to a single point — spread > 0.
        let col0_min = out.column(0).iter().cloned().fold(f32::INFINITY, f32::min);
        let col0_max = out
            .column(0)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            col0_max - col0_min > 1e-4,
            "first output column collapsed: min={col0_min} max={col0_max}"
        );
    }

    // ── SigLIP2 parameterization tests ───────────────────────────────────

    fn synth_siglip2_weights() -> VisionWeights {
        let mut w = synth_weights();
        w.config.hidden_act = "silu".to_string();
        w.config.norm_type = "rms_norm".to_string();
        w
    }

    #[test]
    fn siglip2_family_name() {
        let w = synth_siglip2_weights();
        let enc = VisionEncoder::new(&w);
        assert_eq!(enc.family(), "siglip2");
    }

    #[test]
    fn siglip_family_name() {
        let w = synth_weights();
        let enc = VisionEncoder::new(&w);
        assert_eq!(enc.family(), "siglip");
    }

    #[test]
    fn siglip2_encode_produces_valid_output() {
        let w = synth_siglip2_weights();
        let enc = VisionEncoder::new(&w);
        let side = w.config.image_size;
        let rgb = gradient_rgb(side);
        let out = enc
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: side,
                height: side,
            })
            .expect("SigLIP2 encode should succeed");
        assert_eq!(out.shape(), &[w.config.num_patches(), w.config.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silu_and_gelu_produce_different_outputs() {
        let side = 4;
        let rgb = gradient_rgb(side);

        let w_gelu = synth_weights();
        let enc_gelu = VisionEncoder::new(&w_gelu);
        let out_gelu = enc_gelu
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: side,
                height: side,
            })
            .unwrap();

        let w_silu = synth_siglip2_weights();
        let enc_silu = VisionEncoder::new(&w_silu);
        let out_silu = enc_silu
            .encode(ModalInput::Image {
                rgb: &rgb,
                width: side,
                height: side,
            })
            .unwrap();

        assert_eq!(out_gelu.shape(), out_silu.shape());
        let differ = out_gelu
            .iter()
            .zip(out_silu.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "SiLU and GELU-tanh should produce different outputs"
        );
    }
}
