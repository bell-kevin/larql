//! CPU forward-pass implementations for multi-modal connectors.
//!
//! Per-LM connector forward passes that consume weights from
//! `larql-models::connectors::*`. Each impl satisfies the `Connector`
//! trait defined in `larql-models::multimodal` (re-exported as
//! `larql_models::MmConnector` to avoid clashing with crate-local
//! `Connector` types).

pub mod mlp_connector;
pub mod projector;
