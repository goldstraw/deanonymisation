// use std::io::Write;
use serde_json;
use crate::block::Block;
use ndarray::arr1;
use rand::Rng;
use crate::embedding::load_embeddings;
use crate::transformer::Transformer;
use crate::dataset::{load_chat_dataset, Message};
use log::info;

fn log_dataset_stats(dataset: &Vec<Message>) {
    info!("Loaded {} messages successfully.", dataset.len());
    let mut author_counts = [0; 2];
    for example in dataset {
        author_counts[example.author] += 1;
    }
    info!("Author counts: {:?}", author_counts);
}

pub fn run(num_words: usize, dimensionality: usize, num_encoders: usize, num_heads: usize, hidden_layer_size: usize, num_messages: usize) {
    let time = std::time::SystemTime::now();
    let str_time = time.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs().to_string();
    let model_file_name = format!("{}_chtbt_model_{}_{}_{}_{}_{}.json", str_time, num_words, dimensionality, num_encoders, num_heads, hidden_layer_size);
    let word_embeddings = load_embeddings("../chatbot_arena_embeddings.json");
    let dataset = load_chat_dataset("../train.json", num_words, word_embeddings.clone(), num_messages);
    log_dataset_stats(&dataset);
    let mut transformer = Transformer::new(num_words, dimensionality, num_encoders, num_heads, arr1(&[num_words*dimensionality,hidden_layer_size,num_words*dimensionality]), word_embeddings);
    // Load a pretrained model
    // let model_file = std::fs::File::open("1700084491_model_7_64_1_2_100.json").unwrap();
    // let mut transformer: Transformer = serde_json::from_reader(model_file).unwrap();
    let mut rng = rand::thread_rng();

    const N: usize = 5000; // Number of values to average over
    let mut avg_loss = 0.0;
    let mut index = 0;
    let test_gaps = 10; // Test runs every N * test_gaps iterations
    let mut test_count = 0; 
    const TEST_SIZE: usize = 2000; // Number of examples to test on
    let mut avg_acc = 0.0;

    let mut confusion_matrix = [[0; 2]; 2];

    
    loop {
        // Select a random example from the dataset excluding the test set
        let example = &dataset[rng.gen_range(TEST_SIZE..dataset.len())];
        
        // Forward propagate the example through the transformer model
        let val = transformer.forward_propagate(example.msg.clone());
        
        // Back propagate the author through the transformer model
        let mut desired = arr1(&[0.0; 2]);
        desired[example.author] = 1.0;
        transformer.back_propagate(desired.clone());

        // Calculate the cross entropy loss for the example
        for i in 0..2 {
            avg_loss += -desired[i] * val[i].ln();
        }
        index += 1;

        // Check if the model's prediction was correct
        let mut max = 0.0;
        let mut max_index = 0;
        for i in 0..2 {
            if val[i] > max {
                max = val[i];
                max_index = i;
            }
        }
        if max_index == example.author {
            avg_acc += 1.0;
        }

        // Update the confusion matrix
        confusion_matrix[example.author][max_index] += 1;

        if index == N {
            index = 0;
            test_count += 1;
            // Calculate and log the average loss for the current batch
            info!("{} TRAIN LOSS: {:?}", model_file_name, avg_loss / N as f32);
            info!("{}  TRAIN ACC: {:?}", model_file_name, avg_acc / N as f32);

            // Print the confusion matrix
            // info!("CONFUSION MATRIX");
            // for i in 0..8 {
            //     info!("{} is confused with {:?}", author_names[i], confusion_matrix[i]);
            // }
            confusion_matrix = [[0; 2]; 2];

            // Check if it's time to perform a test on the test set
            if test_count == test_gaps {

                // Reset the test count
                test_count = 0;
                
                let mut avg_test_loss = 0.0;
                let mut avg_test_acc = 0.0;
                let mut author_counts = [0; 2];

                // Calculate the loss for each example in the test set
                for i in 0..TEST_SIZE {
                    let example = &dataset[i];
                    author_counts[example.author] += 1;
                    desired = arr1(&[0.0; 2]);
                    desired[example.author] = 1.0;
                    let val = transformer.forward_propagate(example.msg.clone());
                    for i in 0..2 {
                        avg_test_loss += -desired[i] * val[i].ln();
                    }
                    // Check if the model's prediction was correct
                    let mut max = 0.0;
                    let mut max_index = 0;
                    for i in 0..2 {
                        if val[i] > max {
                            max = val[i];
                            max_index = i;
                        }
                    }
                    if max_index == example.author {
                        avg_test_acc += 1.0;
                    }
                }
                info!("Author counts: {:?}", author_counts);
                
                // Calculate and log the average loss for the test set
                info!("{} TEST LOSS: {:?}", model_file_name, avg_test_loss / TEST_SIZE as f32);
                info!("{}  TEST ACC: {:?}", model_file_name, avg_test_acc / TEST_SIZE as f32);
                
                let model_file = std::fs::File::create(&model_file_name).unwrap();
                serde_json::to_writer(model_file, &transformer).unwrap();
                info!("Saved model to {}", model_file_name);
            }
            
            avg_loss = 0.0;
            avg_acc = 0.0;
        }
    
    }
}