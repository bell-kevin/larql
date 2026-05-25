//! 2-layer MLP GELU connector — CPU forward pass.
//!
//! Used by Granite Vision: takes encoder output `(seq_len, vision_hidden)`
//! and projects to `(seq_len, text_hidden)` via fc1 → GELU → fc2.
//! No spatial pooling — Granite's encoder output already has the correct
//! token count per tile.

use larql_models::connectors::mlp_connector::MlpConnectorWeights;
use larql_models::multimodal::Connector;
use ndarray::Array2;

pub struct MlpGelu<'w> {
    weights: &'w MlpConnectorWeights,
}

impl<'w> MlpGelu<'w> {
    pub fn new(weights: &'w MlpConnectorWeights) -> Self {
        Self { weights }
    }
}

impl Connector for MlpGelu<'_> {
    fn input_dim(&self) -> usize {
        self.weights.vision_hidden()
    }

    fn output_dim(&self) -> usize {
        self.weights.text_hidden()
    }

    fn project(&self, encoder_out: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            encoder_out.ncols(),
            self.weights.vision_hidden(),
            "encoder output hidden dim mismatch"
        );

        let h = proj_with_bias(
            encoder_out,
            &self.weights.fc1_weight,
            &self.weights.fc1_bias,
        );
        let h = h.mapv(crate::ffn::gelu_tanh);
        proj_with_bias(&h, &self.weights.fc2_weight, &self.weights.fc2_bias)
    }
}

fn proj_with_bias(x: &Array2<f32>, weight: &Array2<f32>, bias: &[f32]) -> Array2<f32> {
    let seq_len = x.nrows();
    let out_dim = weight.nrows();
    let mut out = Array2::<f32>::zeros((seq_len, out_dim));
    for s in 0..seq_len {
        for o in 0..out_dim {
            let mut sum = bias[o];
            for i in 0..x.ncols() {
                sum += x[[s, i]] * weight[[o, i]];
            }
            out[[s, o]] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn synth_weights(vision: usize, inter: usize, text: usize) -> MlpConnectorWeights {
        let fc1_weight = Array2::from_shape_fn((inter, vision), |(i, j)| {
            ((i * 7 + j * 3) as f32 - 50.0) / 200.0
        });
        let fc1_bias = vec![0.01; inter];
        let fc2_weight = Array2::from_shape_fn((text, inter), |(i, j)| {
            ((i * 5 + j * 11) as f32 - 40.0) / 200.0
        });
        let fc2_bias = vec![-0.01; text];
        MlpConnectorWeights {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        }
    }

    #[test]
    fn output_shape_matches() {
        let w = synth_weights(8, 16, 12);
        let connector = MlpGelu::new(&w);
        let input = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f32 * 0.1);
        let out = connector.project(&input);
        assert_eq!(out.shape(), &[4, 12]);
    }

    #[test]
    fn trait_dims_correct() {
        let w = synth_weights(8, 16, 12);
        let connector = MlpGelu::new(&w);
        assert_eq!(connector.input_dim(), 8);
        assert_eq!(connector.output_dim(), 12);
    }

    #[test]
    fn output_is_finite() {
        let w = synth_weights(8, 16, 12);
        let connector = MlpGelu::new(&w);
        let input = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f32 * 0.1);
        let out = connector.project(&input);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn different_inputs_produce_different_outputs() {
        let w = synth_weights(8, 16, 12);
        let connector = MlpGelu::new(&w);
        let a = Array2::from_shape_fn((2, 8), |(i, j)| (i + j) as f32 * 0.1);
        let b = Array2::from_shape_fn((2, 8), |(i, j)| (i + j + 5) as f32 * 0.1);
        let oa = connector.project(&a);
        let ob = connector.project(&b);
        assert_ne!(oa, ob);
    }

    #[test]
    #[should_panic(expected = "hidden dim mismatch")]
    fn panics_on_wrong_input_dim() {
        let w = synth_weights(8, 16, 12);
        let connector = MlpGelu::new(&w);
        let input = Array2::<f32>::zeros((4, 10));
        let _ = connector.project(&input);
    }
}
