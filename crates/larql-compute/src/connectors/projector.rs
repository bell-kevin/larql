//! Gemma 3 vision connector — CPU forward pass.
//!
//! Maps SigLIP encoder output `(num_patches, vision_hidden)` to
//! Gemma 3 LM-input `(num_soft_tokens, text_hidden)`. For the 4B-it
//! variant: `(4096, 1152) → (256, 2560)`.
//!
//! Pipeline (verified against HF transformers `Gemma3MultiModalProjector`):
//!
//! ```text
//! input: (num_patches, vision_hidden) = (4096, 1152)
//!     → reshape spatially: positions index into (64, 64) grid
//!     → AvgPool2d(kernel=4, stride=4): 64×64 → 16×16 = 256 tokens
//!     → output: (256, vision_hidden) = (256, 1152)
//!     → RMSNorm on vision_hidden axis (mm_soft_emb_norm, Gemma offset = 1.0)
//!     → matmul × mm_input_projection_weight (1152, 2560)  [x @ W, NOT x @ W.T]
//!     → output: (256, text_hidden) = (256, 2560)
//! ```
//!
//! ## Scaling ownership
//!
//! This connector owns its post-pool RMSNorm. The host does NOT apply
//! any additional `embed_scale` to the output — `Gemma3MultiModal::
//! precomputed_scaling()` returns `PrecomputedScaling::None` and that
//! pairing is intentional. If you change the protocol's scaling
//! decision, change the doc-comment here too.
//!
//! ## Patch geometry
//!
//! `patches_per_side` and `tokens_per_side` are derived from the SigLIP
//! config; `kernel_size = patches_per_side / tokens_per_side`. For
//! Gemma 3 4B: patches_per_side=64, tokens_per_side=16, kernel=4.
//! Both must divide evenly — checked at construction.

use larql_models::connectors::projector::ProjectorWeights;
use larql_models::encoders::vision_tower::VisionConfig;
use larql_models::MmConnector;
use ndarray::Array2;

use crate::residual::rms_norm_eps;

/// Gemma 3 vision projector, borrowing both the projector weights and
/// the SigLIP encoder's config (for patch geometry).
#[derive(Debug)]
pub struct VisionProjector<'w> {
    weights: &'w ProjectorWeights,
    /// Side length of the spatial patch grid (e.g. 64 for Gemma 3 4B).
    patches_per_side: usize,
    /// Side length of the soft-token grid (e.g. 16 for Gemma 3 4B's 256 tokens).
    tokens_per_side: usize,
    /// AvgPool kernel = patches_per_side / tokens_per_side (e.g. 4 for Gemma 3 4B).
    kernel: usize,
    /// RMSNorm epsilon (vision_config.layer_norm_eps; default 1e-6).
    eps: f64,
}

impl<'w> VisionProjector<'w> {
    /// Build a connector. `siglip_config` provides patch_size +
    /// image_size (which set `patches_per_side`); `mm_tokens_per_image`
    /// is the LM-side budget (256 for Gemma 3).
    ///
    /// Errors if the spatial grid doesn't divide evenly into the
    /// soft-token grid — Gemma 3's design assumes integer downsampling.
    pub fn new(
        weights: &'w ProjectorWeights,
        siglip_config: &VisionConfig,
        mm_tokens_per_image: usize,
    ) -> Result<Self, String> {
        let patches_per_side = siglip_config.patches_per_side();
        let tokens_per_side_f = (mm_tokens_per_image as f64).sqrt();
        let tokens_per_side = tokens_per_side_f as usize;
        if tokens_per_side * tokens_per_side != mm_tokens_per_image {
            return Err(format!(
                "mm_tokens_per_image must be a perfect square (got {mm_tokens_per_image})"
            ));
        }
        if !patches_per_side.is_multiple_of(tokens_per_side) {
            return Err(format!(
                "patches_per_side ({patches_per_side}) must be divisible by \
                 tokens_per_side ({tokens_per_side})"
            ));
        }
        let kernel = patches_per_side / tokens_per_side;
        // Sanity-check the weight shapes against the SigLIP config.
        if weights.vision_hidden() != siglip_config.hidden_size {
            return Err(format!(
                "projector vision_hidden ({}) does not match SigLIP hidden_size ({})",
                weights.vision_hidden(),
                siglip_config.hidden_size
            ));
        }
        if weights.soft_emb_norm.len() != siglip_config.hidden_size {
            return Err(format!(
                "soft_emb_norm length ({}) does not match SigLIP hidden_size ({})",
                weights.soft_emb_norm.len(),
                siglip_config.hidden_size
            ));
        }
        Ok(Self {
            weights,
            patches_per_side,
            tokens_per_side,
            kernel,
            eps: siglip_config.layer_norm_eps,
        })
    }
}

impl MmConnector for VisionProjector<'_> {
    fn input_dim(&self) -> usize {
        self.weights.vision_hidden()
    }

    fn output_dim(&self) -> usize {
        self.weights.text_hidden()
    }

    fn project(&self, encoder_out: &Array2<f32>) -> Array2<f32> {
        let expected_rows = self.patches_per_side * self.patches_per_side;
        assert_eq!(
            encoder_out.nrows(),
            expected_rows,
            "VisionProjector expects {} rows (= patches_per_side²); got {}",
            expected_rows,
            encoder_out.nrows()
        );
        assert_eq!(
            encoder_out.ncols(),
            self.weights.vision_hidden(),
            "encoder hidden mismatch"
        );

        let pooled = avg_pool_spatial(
            encoder_out,
            self.patches_per_side,
            self.tokens_per_side,
            self.kernel,
        );
        // Gemma RMSNorm uses (1.0 + saved_weight) at runtime — same offset
        // as Gemma3Arch::norm_weight_offset(), which is 1.0.
        let normed = rms_norm_eps(&pooled, Some(&self.weights.soft_emb_norm), 1.0, self.eps);
        // `normed` is (tokens², vision_hidden); projection is (vision_hidden,
        // text_hidden). HF matmul convention here is `x @ W` (W not transposed).
        normed.dot(&self.weights.input_projection)
    }
}

/// AvgPool2d over a flattened spatial grid.
///
/// Input rows are laid out row-major over an `(rows × rows)` spatial
/// grid — row index = `y * rows + x`. We average over `(kernel × kernel)`
/// windows with stride = kernel, preserving the hidden axis.
///
/// Returns `(out_rows², hidden)` where `out_rows = rows / kernel`.
fn avg_pool_spatial(
    input: &Array2<f32>,
    in_side: usize,
    out_side: usize,
    kernel: usize,
) -> Array2<f32> {
    let hidden = input.ncols();
    let mut out = Array2::<f32>::zeros((out_side * out_side, hidden));
    let denom = (kernel * kernel) as f32;
    for oy in 0..out_side {
        for ox in 0..out_side {
            let out_row = oy * out_side + ox;
            for dy in 0..kernel {
                for dx in 0..kernel {
                    let iy = oy * kernel + dy;
                    let ix = ox * kernel + dx;
                    let in_row = iy * in_side + ix;
                    for j in 0..hidden {
                        out[[out_row, j]] += input[[in_row, j]];
                    }
                }
            }
            for j in 0..hidden {
                out[[out_row, j]] /= denom;
            }
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! See `larql-compute::encoders::siglip::tests` for the full
    //! testing-trap rundown (uniform weights, uniform inputs,
    //! constant-across-hidden-dims perturbations). All three apply here
    //! — the RMSNorm + AvgPool combo can collapse signal the same way
    //! a transformer block can.

    use super::*;
    use ndarray::Array2;

    fn synth_config_tiny() -> VisionConfig {
        VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            // 8×8 patch grid → 2×2 soft tokens (kernel=4).
            patch_size: 1,
            image_size: 8,
            num_channels: 3,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            norm_type: "layer_norm".to_string(),
        }
    }

    fn synth_projector_weights(vision_hidden: usize, text_hidden: usize) -> ProjectorWeights {
        // Asymmetric pattern centred near zero — same rationale as the
        // SigLIP synthetic-weights helper.
        let wval = |i: usize, j: usize| {
            let h = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)).wrapping_mul(2654435761);
            ((h & 0xff) as i32 - 128) as f32 / 800.0
        };
        let input_projection =
            Array2::<f32>::from_shape_fn((vision_hidden, text_hidden), |(i, j)| wval(i, j));
        // soft_emb_norm: stored is "learned"; runtime applies (1.0 + learned).
        // Use small per-position values so the runtime weight is near 1.0.
        let soft_emb_norm = (0..vision_hidden).map(|i| wval(i, 0)).collect::<Vec<f32>>();
        ProjectorWeights {
            input_projection,
            soft_emb_norm,
        }
    }

    #[test]
    fn new_rejects_non_square_mm_tokens() {
        let cfg = synth_config_tiny();
        let w = synth_projector_weights(8, 12);
        let err = VisionProjector::new(&w, &cfg, 5).expect_err("5 is not a perfect square");
        assert!(err.contains("perfect square"));
    }

    #[test]
    fn new_rejects_non_divisible_grid() {
        let cfg = synth_config_tiny(); // patches_per_side = 8
        let w = synth_projector_weights(8, 12);
        // 9 is a perfect square (3) but 8 isn't divisible by 3.
        let err = VisionProjector::new(&w, &cfg, 9).expect_err("8 % 3 != 0");
        assert!(err.contains("divisible"));
    }

    #[test]
    fn new_rejects_hidden_mismatch() {
        let cfg = synth_config_tiny(); // hidden_size = 8
        let w = synth_projector_weights(16, 12); // projector expects 16
        let err = VisionProjector::new(&w, &cfg, 4).expect_err("hidden mismatch should fail");
        assert!(err.contains("hidden") || err.contains("match"));
    }

    #[test]
    fn project_output_shape_and_finite() {
        let cfg = synth_config_tiny(); // 8×8 = 64 patches, kernel=4 → 2×2 = 4 tokens
        let vision_hidden = cfg.hidden_size;
        let text_hidden = 12;
        let w = synth_projector_weights(vision_hidden, text_hidden);
        let conn = VisionProjector::new(&w, &cfg, 4).expect("new");
        assert_eq!(conn.input_dim(), vision_hidden);
        assert_eq!(conn.output_dim(), text_hidden);
        assert_eq!(conn.kernel, 4);
        assert_eq!(conn.tokens_per_side, 2);
        assert_eq!(conn.patches_per_side, 8);

        // Gradient encoder output — non-uniform across rows AND across hidden dims
        // so RMSNorm doesn't collapse anything.
        let encoder_out = Array2::<f32>::from_shape_fn((64, vision_hidden), |(i, j)| {
            0.1 * ((i * 7 + j * 11) % 13) as f32 - 0.5
        });
        let out = conn.project(&encoder_out);
        assert_eq!(out.shape(), &[4, text_hidden]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn avg_pool_actually_averages() {
        // Tiny direct test of the pool helper. 4×4 grid, kernel=2,
        // output 2×2. Hidden = 1 to keep it readable.
        let input = Array2::<f32>::from_shape_vec(
            (16, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, // y=0 row
                5.0, 6.0, 7.0, 8.0, // y=1
                9.0, 10.0, 11.0, 12.0, // y=2
                13.0, 14.0, 15.0, 16.0, // y=3
            ],
        )
        .unwrap();
        let out = avg_pool_spatial(&input, 4, 2, 2);
        // Top-left 2x2 window: 1,2,5,6 → mean = 3.5
        // Top-right 2x2: 3,4,7,8 → 5.5
        // Bot-left:  9,10,13,14 → 11.5
        // Bot-right: 11,12,15,16 → 13.5
        assert_eq!(out.shape(), &[4, 1]);
        assert!((out[[0, 0]] - 3.5).abs() < 1e-6);
        assert!((out[[1, 0]] - 5.5).abs() < 1e-6);
        assert!((out[[2, 0]] - 11.5).abs() < 1e-6);
        assert!((out[[3, 0]] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn project_different_inputs_produce_different_outputs() {
        let cfg = synth_config_tiny();
        let w = synth_projector_weights(cfg.hidden_size, 12);
        let conn = VisionProjector::new(&w, &cfg, 4).unwrap();
        let a = Array2::<f32>::from_shape_fn((64, cfg.hidden_size), |(i, j)| {
            0.1 * ((i + j * 3) % 13) as f32 - 0.5
        });
        let mut b = a.clone();
        // Perturb a single patch position; non-constant across hidden so
        // RMSNorm can't remove it.
        for j in 0..cfg.hidden_size {
            b[[5, j]] += (j as f32) * 0.1 - 0.35;
        }
        let out_a = conn.project(&a);
        let out_b = conn.project(&b);
        let differ = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(differ);
    }

    #[test]
    #[ignore = "requires google/gemma-3-4b-it in the local HF cache; NOT FOR CI"]
    fn project_real_gemma3_4b_it_shape_and_finite() {
        use larql_models::connectors::projector::load_projector_from_safetensors;
        let snap = "/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767";
        if !std::path::Path::new(snap).exists() {
            eprintln!("snapshot not present, skipping: {snap}");
            return;
        }
        let w = load_projector_from_safetensors(snap).expect("load");
        let cfg = VisionConfig {
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
        let conn = VisionProjector::new(&w, &cfg, 256).expect("connector");
        assert_eq!(conn.input_dim(), 1152);
        assert_eq!(conn.output_dim(), 2560);

        // Synthesize encoder output without running SigLIP — gradient pattern
        // varies across both axes so RMSNorm doesn't collapse it.
        let encoder_out = Array2::<f32>::from_shape_fn((4096, 1152), |(i, j)| {
            0.01 * ((i * 31 + j * 17) % 41) as f32 - 0.2
        });
        let out = conn.project(&encoder_out);
        assert_eq!(out.shape(), &[256, 2560]);
        assert!(out.iter().all(|v| v.is_finite()));
        // Spread check on first column — projector shouldn't collapse.
        let col0_min = out.column(0).iter().cloned().fold(f32::INFINITY, f32::min);
        let col0_max = out
            .column(0)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(col0_max - col0_min > 1e-4);
    }
}
