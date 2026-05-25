//! IBM Granite architecture.
//!
//! Llama-compatible: same tensor keys, norms, activation, RoPE.
//! Granite Vision variants additionally declare a multi-modal protocol
//! for SigLIP2 + MLP GELU connector + AnyRes tiling (Phase 2).

use crate::config::{ModelArchitecture, ModelConfig};
use crate::multimodal::{MultiModalProtocol, PlaceholderProtocol, PrecomputedScaling, TokenBudget};

/// Multi-modal contract for Granite Vision models.
pub struct GraniteVisionMultiModal;

pub const GRANITE_VISION_MULTIMODAL: GraniteVisionMultiModal = GraniteVisionMultiModal;

const GRANITE_VALID_TILE_COUNTS: &[usize] = &[1, 2, 3, 4, 5, 6];

impl MultiModalProtocol for GraniteVisionMultiModal {
    fn vision_encoder(&self) -> Option<&str> {
        Some("siglip2")
    }

    fn image_placeholder(&self) -> Option<PlaceholderProtocol> {
        Some(PlaceholderProtocol {
            start: None,
            fill: 49152,
            end: None,
        })
    }

    fn image_token_budget(&self) -> TokenBudget {
        TokenBudget::PerTile {
            tokens_per_tile: 729,
        }
    }

    fn precomputed_scaling(&self) -> PrecomputedScaling {
        PrecomputedScaling::None
    }

    fn valid_tile_counts(&self) -> &[usize] {
        GRANITE_VALID_TILE_COUNTS
    }
}

pub struct GraniteArch {
    config: ModelConfig,
}

impl GraniteArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GraniteArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn multimodal(&self) -> Option<&dyn MultiModalProtocol> {
        if self.config.has_vision_config {
            Some(&GRANITE_VISION_MULTIMODAL)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn text_only_config() -> ModelConfig {
        ModelConfig {
            model_type: "granite".into(),
            norm_eps: Some(1e-5),
            num_layers: 28,
            hidden_size: 2048,
            intermediate_size: 8192,
            head_dim: 64,
            num_q_heads: 32,
            num_kv_heads: 8,
            vocab_size: Some(49152),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: Some(12.0),
            residual_multiplier: Some(0.22),
            attention_multiplier: Some(0.015625),
            logits_scaling: Some(10.0),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            has_vision_config: false,
        }
    }

    fn vision_config() -> ModelConfig {
        let mut cfg = text_only_config();
        cfg.has_vision_config = true;
        cfg
    }

    #[test]
    fn text_only_granite_has_no_multimodal() {
        let arch = GraniteArch::from_config(text_only_config());
        assert!(arch.multimodal().is_none());
    }

    #[test]
    fn vision_granite_declares_multimodal_protocol() {
        let arch = GraniteArch::from_config(vision_config());
        let mm = arch
            .multimodal()
            .expect("Granite Vision must declare multimodal protocol");
        assert_eq!(mm.vision_encoder(), Some("siglip2"));
        assert!(mm.audio_encoder().is_none());
    }

    #[test]
    fn image_placeholder_has_fill_only() {
        let arch = GraniteArch::from_config(vision_config());
        let ph = arch
            .multimodal()
            .unwrap()
            .image_placeholder()
            .expect("Granite Vision declares image placeholder");
        assert!(ph.start.is_none());
        assert!(ph.end.is_none());
        assert_eq!(ph.fill, 49152);
    }

    #[test]
    fn token_budget_is_per_tile_729() {
        let arch = GraniteArch::from_config(vision_config());
        match arch.multimodal().unwrap().image_token_budget() {
            TokenBudget::PerTile { tokens_per_tile } => assert_eq!(tokens_per_tile, 729),
            other => panic!("expected PerTile, got {other:?}"),
        }
    }

    #[test]
    fn precomputed_scaling_is_none() {
        let arch = GraniteArch::from_config(vision_config());
        assert_eq!(
            arch.multimodal().unwrap().precomputed_scaling(),
            PrecomputedScaling::None
        );
    }

    #[test]
    fn valid_tile_counts_non_empty() {
        let arch = GraniteArch::from_config(vision_config());
        let counts = arch.multimodal().unwrap().valid_tile_counts();
        assert!(!counts.is_empty());
        assert_eq!(counts, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn no_audio_in_granite_vision() {
        let arch = GraniteArch::from_config(vision_config());
        assert!(arch.multimodal().unwrap().audio_placeholder().is_none());
    }
}
