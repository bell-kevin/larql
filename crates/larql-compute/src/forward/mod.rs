//! Substrate-level forward-pass primitives.
//!
//! Pure helpers that consume `ModelWeights` and `ndarray::Array2` but
//! carry no engine/session state and no env-var coupling. Arch-aware
//! convenience wrappers (those that consult `forward_overrides`)
//! continue to live in `larql-inference`, where the env-var registry
//! sits.
//!
//! See ADR-0022 for the layering rationale.

pub mod dump_config;
pub mod embed;
pub mod hooks;
pub mod layer;
pub mod lens;
pub mod ops;
pub mod ple;
pub mod predict;
pub mod vocab_proj;

pub use embed::embed_tokens_pub;
pub use hooks::{
    AttnZeroHook, CompositeHook, FFNZeroHook, LayerHook, NoopHook, RecordHook, SteerHook,
    ZeroAblateHook,
};
pub use layer::{
    run_attention, run_attention_inner, run_attention_public, run_attention_with_kv_cache, run_ffn,
    run_layer_with_capture, run_layer_with_capture_hooked, run_layer_with_ffn,
};
pub use ops::{add_bias, apply_norm, dot_proj, softmax};
pub use ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
pub use predict::{
    forward_from_layer, forward_raw_logits, forward_raw_logits_with_prefix, RawForward,
};
