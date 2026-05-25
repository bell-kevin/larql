//! Multi-modal projector weights and loaders.
//!
//! `ProjectorWeights` is generic — stores the projection matrix and
//! optional norm weight. Per-LM behaviour (pool type, norm offset,
//! matmul convention) dispatches at the forward-pass level in
//! `larql-compute::connectors::projector`.

pub mod mlp_connector;
pub mod projector;
