use super::*;

impl MetalBackend {
    /// Full pipeline: attention + FFN for all layers in ONE command buffer.
    /// No CPU-GPU round-trips between layers.
    /// This is the old benchmark entry point — uses dummy norms (no residual correctness).
    pub fn full_pipeline(
        &self,
        layers: &[ops::full_pipeline::LayerWeights],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        // Convert old LayerWeights to new FullPipelineLayer with dummy norms
        let dummy_norm = vec![1.0f32; hidden];
        // Convert old LayerWeights (Q4 attention) to new FullPipelineLayer (Q8 attention)
        // For backward compat: treat Q4 data as Q8 (wrong but benchmark-only path)
        let _dummy_scales = vec![1.0f32; hidden * hidden / 32]; // oversized, reserved for Q8 path
        let full_layers: Vec<larql_compute::FullPipelineLayer> = layers
            .iter()
            .map(|l| larql_compute::FullPipelineLayer {
                wq: larql_compute::QuantWeight {
                    data: l.wq_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                wk: larql_compute::QuantWeight {
                    data: l.wk_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                wv: larql_compute::QuantWeight {
                    data: l.wv_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                wo: larql_compute::QuantWeight {
                    data: l.wo_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                gate: larql_compute::QuantWeight {
                    data: l.gate_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                up: larql_compute::QuantWeight {
                    data: l.up_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                down: larql_compute::QuantWeight {
                    data: l.down_t_q4,
                    scales: None,
                    format: larql_compute::QuantFormat::Q4_0,
                },
                input_norm: &dummy_norm,
                post_attn_norm: &dummy_norm,
                pre_ffn_norm: None,
                post_ffn_norm: None,
                norm_offset: 0.0,
                has_post_norms: false,
                activation: larql_compute::Activation::Silu,
                qk_norm_offset: 0.0,
                eps: larql_compute::RMSNORM_EPSILON_DEFAULT,
                norm_type: larql_compute::NormType::RmsNorm,
                ffn_type: larql_compute::FfnType::Gated,
                attn_scale: 0.0,
                head_dim: 0,
                num_q_heads: 0,
                num_kv_heads: 0,
                rope_base: larql_compute::ROPE_BASE_DEFAULT,
                rotary_dim: 0,
                sliding_window: 0,
                has_v_norm: false,
                layer_scalar: 0.0,
                input_norm_bias: None,
                post_attn_norm_bias: None,
                q_norm_weight: None,
                k_norm_weight: None,
                ffn_up_bias: None,
                ffn_down_bias: None,
                moe: None,
                ffn_is_remote: false,
                moe_combined_output_norm: false,
                moe_outer_post_norm: None,
                ple_input_gate: None,
                ple_projection: None,
                ple_post_norm: None,
                kv_shared_source: None,
            })
            .collect();
        ops::full_pipeline::dispatch_full_pipeline(
            &self.queue,
            &self.bufs,
            &self.q4,
            &self.ffn.geglu_pipeline,
            &self.ffn.geglu_gelu_tanh_pipeline,
            &self.ffn.silu_pipeline,
            &self.ffn.gelu_tanh_pipeline,
            &self.quant.q8_quant_pipeline,
            None,
            &self.quant.q8_matvec_pipeline.state,
            &self.attention.q8_qkv_proj_pipeline.state,
            &self.quant.q4k_matvec_pipeline,
            Some(&self.quant.q4k_matmul_pipeline),
            &self.quant.q6k_matvec_pipeline,
            &self.norms.rms_norm_pipeline,
            &self.norms.residual_add_pipeline,
            &self.norms.rms_norm_q8_pipeline,
            &self.norms.residual_norm_q8_pipeline,
            None, // no q4k_qkv_proj (legacy 148-byte)
            None,
            None, // no q4kf_qkv_proj / q4kf_proj (legacy benchmark path)
            None, // no rope_at_pos
            None, // no qk_norm
            None, // no scale_vector (no layer_scalar)
            None,
            None,
            None,
            None, // no fused activation+down (legacy benchmark path)
            None, // no KV cache
            &full_layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            1,
            0,
            0,
            0,
            0.0,
            false,
            0.0,
            None, // no MoE callback (legacy benchmark path, no MoE layers)
            None, // no intervention (legacy benchmark path)
        )
    }

    /// Multi-layer Q4 FFN in ONE command buffer.
    /// gate → up → GEGLU → down → Q8 quantize → next layer.
    /// All on GPU, no CPU return between layers.
    pub fn multi_layer_q4_ffn(
        &self,
        layers_q4: &[(&[u8], &[u8], &[u8])], // [(gate, up, down_t)]
        x: &[f32],
        inter: usize,
        hidden: usize,
    ) -> Vec<f32> {
        ops::q4_batched::multi_layer_ffn(
            &self.queue,
            &self.bufs,
            &self.q4,
            &self.ffn.geglu_pipeline,
            &self.quant.q8_quant_pipeline,
            layers_q4,
            x,
            inter,
            hidden,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    /// `multi_layer_q4_ffn` dispatches gate → up → GEGLU → down → Q8
    /// per layer in a single command buffer.
    #[test]
    fn multi_layer_q4_ffn_dispatches_two_layers() {
        let m = backend();
        let block_bytes = 18usize;
        let hidden = 32usize;
        let inter = 64usize;
        let blocks_per_row = hidden / 32;
        let gate = vec![0u8; inter * blocks_per_row * block_bytes];
        let up = vec![0u8; inter * blocks_per_row * block_bytes];
        let down = vec![0u8; hidden * (inter / 32) * block_bytes];
        let layers = vec![
            (gate.as_slice(), up.as_slice(), down.as_slice()),
            (gate.as_slice(), up.as_slice(), down.as_slice()),
        ];
        let x = vec![0.0f32; hidden];
        let out = m.multi_layer_q4_ffn(&layers, &x, inter, hidden);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
