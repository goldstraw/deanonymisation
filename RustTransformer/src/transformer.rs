use ndarray::{Array1, Array2, arr1};
use std::collections::HashMap;
use crate::block::Block;
use crate::dense::Dense;
use crate::encoder_block::EncoderBlock;
use crate::positional_encoder::PositionalEncoder;
use serde::{Serialize, Deserialize};

// Defines attention heads and dense layer.
#[derive(Serialize, Deserialize)]
pub struct TransformerParams {
    encoder_blocks: Array1::<EncoderBlock>,
}

// Defines multi-headed attention struct
#[derive(Serialize, Deserialize)]
pub struct Transformer {
    input: Array1::<String>,
    output: Array1::<f32>,
    num_words: usize,
    dimensionality: usize,
    pos_encoder: PositionalEncoder,
    classifier: Dense,
    embedding: HashMap<String, Vec<f32>>,
    params: TransformerParams,
}

impl Transformer {
    /// Create a new self-attention block with the given parameters
    pub fn new(num_words: usize, dimensionality: usize, num_encoders: usize, num_heads: usize, layer_sizes: Array1<usize>, embedding: HashMap<String, Vec<f32>>) -> Transformer {
        let encoder_blocks = Array1::from_shape_fn(num_encoders, |_| EncoderBlock::new(num_words, dimensionality, num_heads, layer_sizes.clone()));
        let params = TransformerParams { encoder_blocks };
        let pos_encoder = PositionalEncoder::new(num_words, dimensionality);
        let classifier = Dense::new(arr1(&[num_words*dimensionality, 2]), false, true);
        let block: Transformer = Transformer {
            input: Array1::from_shape_fn(num_words, |_| "".to_string()),
            output: Array1::<f32>::zeros(2),
            num_words,
            dimensionality,
            pos_encoder,
            classifier,
            embedding,
            params
        };

        block
    }
}

impl Block for Transformer {
    type Input = Array1<String>;
    type Output = Array1<f32>;

    fn forward_propagate(&mut self, value: Self::Input) -> Self::Output {
        self.input = value;
    
        // Convert input into embedded representation
        let embedded = Array2::<f32>::from_shape_fn((self.num_words, self.dimensionality), |(i, j)| self.embedding[&self.input[i]][j]);
    
        // Apply positional encoding to the embedded representation
        let mut enc_output = self.pos_encoder.forward_propagate(embedded);

        // Iterate through each encoder block and forward propagate the output
        for i in 0..self.params.encoder_blocks.len() {
            enc_output = self.params.encoder_blocks[i].forward_propagate(enc_output);
        }

        // Flatten the output for classification
        let flat_output = enc_output.clone().into_shape(self.num_words*self.dimensionality).unwrap();
    
        // Forward propagate the flattened output through the classifier
        self.output = self.classifier.forward_propagate(flat_output);

        // Return the output
        self.output.clone()
    }

    /// Rather than giving an error here, input a desired value.
    fn back_propagate(&mut self, error: Self::Output) -> Self::Input {
        // Calculate the error of the last layer using the given error and the output of the neural network
        let last_layer_error = self.output.clone() - error;
        
        // Back propagate the error to the classifier and get the classifier error
        let classifier_error = self.classifier.back_propagate(last_layer_error);
        
        // Reshape the classifier error to match the shape of the encoder error
        let mut encoder_error = classifier_error.into_shape((self.num_words, self.dimensionality)).unwrap();

        // Iterate over the encoder blocks in reverse order and back propagate the encoder error
        for i in (0..self.params.encoder_blocks.len()).rev() {
            encoder_error = self.params.encoder_blocks[i].back_propagate(encoder_error);
        }

        // The positional encoder doesn't have any trainable parameters
        // self.pos_encoder.back_propagate(encoder_error);

        arr1(&["".to_string()])
    }
}