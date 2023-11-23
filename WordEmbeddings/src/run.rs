use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs::File;
use std::io::prelude::*;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Clean the chat message by removing all non-alphanumeric characters
/// and un-encoded words
fn clean_msg(msg: String) -> String {
    // "I love this chat! It's so good."
    // => "i love this chat its so good "
    let mut clean_msg: String = String::new();
    
    let chars: String = msg
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>();
    
    let words = chars
        .split_whitespace()
        .collect::<Vec<&str>>();

    for word in words {
        clean_msg.push_str(word);
        clean_msg.push(' ');
    }

    clean_msg.trim().to_string()

    
}

pub fn load_chat_dataset(json_path: &str) -> Vec<String> {
    let mut chat_dataset = Vec::new();
    let file = std::fs::File::open(json_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let json: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let messages = json.as_array().unwrap();
    for message in messages {
        let chat_msg = clean_msg(message["content"].as_str().unwrap().to_string());
        
        chat_dataset.push(chat_msg);
    }
    println!("Chat dataset size: {}", chat_dataset.len());
    chat_dataset
}

fn build_vocab(dataset: &Vec<String>) -> Vec<String> {
    let mut vocab: Vec<String> = Vec::new();
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for msg in dataset {
        for word in msg.split_whitespace() {
            if !word_counts.contains_key(&word.to_string()) {
                word_counts.insert(word.to_string(), 0);
            } else {
                word_counts.insert(word.to_string(), word_counts[&word.to_string()]+1);
            }
            // Only accepts common words to keep size manageable 
            if word_counts[&word.to_string()] == 20 {
                vocab.push(word.to_string());
            }
        }
    }

    println!("Vocab size: {}", vocab.len());
    vocab
}

fn build_co_occurrence_matrix(vocab: &Vec<String>, imdb_dataset: &Vec<String>) -> Vec<Vec<f32>> {
    let word_to_index: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, x)| (x.to_string(), i)).collect();
    let co_occurrence_window = 4;
    let mut co_occurrence_matrix = vec![vec![0.0; vocab.len()]; vocab.len()];
    for msg in imdb_dataset {
        let words = msg.split_whitespace();
        let mut filtered_words: Vec<&str> = Vec::new();
        for word in words {
            if vocab.contains(&word.to_string()) {
                filtered_words.push(word);
            }
        }

        let mut index = 0;
        for word in filtered_words.clone() {
            let mut index2 = 0;
            for word2 in filtered_words.clone() {
                let dist = (index as i32 - index2 as i32).abs();
                if dist <= co_occurrence_window {
                    let weight = 1.0 - (dist as f32 / co_occurrence_window as f32);
                    co_occurrence_matrix[word_to_index[word]][word_to_index[word2]] += weight;
                }
                index2 += 1;
            }
            index += 1;
        }
    }
    co_occurrence_matrix
}

fn power_iteration(cov: &mut Vec<Vec<f32>>, num_iterations: usize, num_eigenvectors: usize) -> Vec<Vec<f32>>{
    // Uses the power iteration algorithm to compute N eigenvectors.

    let mut eigenvectors: Vec<Vec<f32>> = Vec::new();

    for _ in 0..num_eigenvectors {
        // Generate random vector
        let mut eigenvector: Vec<f32> = vec![0.0; cov.len()];
        let mut rng = rand::thread_rng();
        for i in 0..cov.len() {
            eigenvector[i] = rng.gen();
        }

        for _ in 0..num_iterations {
            // Calculate dot product of covariance matrix and eigenvector
            let mut new_eigenvector: Vec<f32> = vec![0.0; cov.len()];
            for j in 0..cov.len() {
                for k in 0..cov.len() {
                    new_eigenvector[j] += cov[j][k] * eigenvector[k]
                }
            }

            // Calculate norm of result
            let mut sq_sum = 0.0;
            for j in 0..cov.len() {
                sq_sum += f32::powi(new_eigenvector[j], 2);
            }
            let norm = f32::powf(sq_sum, 0.5);

            // Normalise result
            for j in 0..cov.len() {
                eigenvector[j] = new_eigenvector[j] / norm;
            }
        }
        // Calculate eigenvalue from eigenvector
        let mut temp = vec![0.0; cov.len()];
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                temp[i] += cov[i][j] * eigenvector[j]
            }
        }

        let mut eigenvalue = 0.0;
        for i in 0..cov.len() {
            eigenvalue += eigenvector[i] * temp[i];
        }

        eigenvectors.push(eigenvector.clone());

        // Redirect matrix to find next eigenvector
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }

    eigenvectors
}

/// Apply principal component analysis to 'matrix'. Generate new elements with
/// a dimensionality of 'num_components'
fn pca(mut matrix: Vec<Vec<f32>>, num_components: usize, num_threads: usize) -> Vec<Vec<f32>> {
    // Normalise data
    for i in 0..matrix.len() {
        let mut total = 0.0;
        for j in 0..matrix.len() {
            total += matrix[i][j];
        }
        let mean = total / matrix.len() as f32;
        let mut dists_from_mean = 0.0;
        for j in 0..matrix[i].len() {
            dists_from_mean += f32::powi(matrix[i][j] - mean, 2);
        }
        let stdev = f32::powf(dists_from_mean / matrix[i].len() as f32, 0.5);

        for j in 0..matrix.len() {
            matrix[i][j] = (matrix[i][j] - mean) / stdev;
        }
    }

    // Find covariance matrix
    let matrix_len = matrix.len();
    let matrix = Arc::new(matrix.clone());
    let covariance_matrix = Arc::new(Mutex::new(vec![vec![0.0; matrix_len]; matrix_len]));

    // Find the covariance matrix on multiple threads
    let mut handles = vec![];
    for i in 0..num_threads {
        let matrix = Arc::clone(&matrix);
        let covariance_matrix = Arc::clone(&covariance_matrix);
        let col_per_thread = (matrix_len as f32 / num_threads as f32).ceil() as usize;
        let handle = thread::spawn(move || {
            for j in 0..col_per_thread {
                let index = i * col_per_thread + j;
                if index < matrix_len {
                    let mut inner_vector = vec![0.0; matrix_len];
                    for k in 0..matrix_len {
                        let mut sum = 0.0;
                        for l in 0..matrix_len {
                            sum += matrix[index][l] * matrix[k][l];
                        }
                        let covariance = sum / matrix_len as f32;
                        inner_vector[k] = covariance;
                    }

                    let mut cov_lock = covariance_matrix.lock().unwrap();
                    for k in 0..matrix_len {
                        cov_lock[index][k] = inner_vector[k];
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to finish
    for handle in handles {
        handle.join().unwrap();
    }

    // Compute eigenvectors and eigenvalues
    let mut cov_lock = covariance_matrix.lock().unwrap();
    let eigenvectors = power_iteration(&mut cov_lock, 10, num_components);

    let mut reduced: Vec<Vec<f32>> = Vec::new();
    for i in 0..matrix.len() {
        let mut embedding: Vec<f32> = Vec::new();
        for j in 0..num_components {
            let mut component: f32 = 0.0;
            for k in 0..matrix.len() {
                component += matrix[i][k] * eigenvectors[j][k]
            }
            embedding.push(component);
        }
        reduced.push(embedding);
    }

    reduced
}

#[derive(Serialize, Deserialize)]
struct WordEmbeddings {
    data: HashMap<String, Vec<f32>>
}

pub fn run(dimensionality: usize, output_name: &str, num_threads: usize) {
    let imdb_dataset = load_chat_dataset("../train.json");
    let vocab = build_vocab(&imdb_dataset);
    let co_occurrence_matrix = build_co_occurrence_matrix(&vocab, &imdb_dataset);
    let reduced = pca(co_occurrence_matrix, dimensionality, num_threads);
    let mut word_embeddings = WordEmbeddings {data: HashMap::new()};
    for i in 0..vocab.len() {
        word_embeddings.data.insert(vocab[i].clone(), reduced[i].clone());
    }
    let serialized = serde_json::to_string(&word_embeddings).unwrap();
    let mut file = File::create(output_name).expect("Unable to create file");
    file.write_all(serialized.as_bytes()).expect("Unable to write");
}
