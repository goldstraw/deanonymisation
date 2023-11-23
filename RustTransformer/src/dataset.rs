use ndarray::Array1;
use std::collections::HashMap;

pub struct Message {
    pub msg: Array1<String>,
    pub author: usize,
}

/// Clean the chat message by removing all non-alphanumeric characters
/// and un-encoded words
fn clean_msg(msg: String, word_embeddings: HashMap<String, Vec<f32>>) -> String {
    // "I love this chat! It's so good."
    // => "i love this chat its so good "
    let mut clean_review: String = String::new();
    
    let chars: String = msg
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>();
    
    let words = chars
        .split_whitespace()
        .collect::<Vec<&str>>();

    for word in words {
        if word_embeddings.contains_key(word) {
            clean_review.push_str(word);
            clean_review.push(' ');
        }
    }

    clean_review.trim().to_string()

    
}

/// Pads the msg with empty strings to the desired length
fn pad_msg(msg: String, msg_size: usize) -> Array1<String> {
    let words: Vec<&str> = msg.split_whitespace().collect();
    let mut padded_msg = Vec::with_capacity(msg_size);

    for i in 0..msg_size {
        let word = if i < words.len() {
            words[i].to_string()
        } else {
            "".to_string()
        };

        padded_msg.push(word);
    }

    Array1::<String>::from_vec(padded_msg)
}

pub fn load_chat_dataset(json_path: &str, msg_size: usize, word_embeddings: HashMap<String, Vec<f32>>, num_messages: usize) -> Vec<Message> {
    let mut chat_dataset = Vec::new();
    let file = std::fs::File::open(json_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let json: serde_json::Value = serde_json::from_reader(reader).unwrap();
    let messages = json.as_array().unwrap();
    let mut count = 0;
    for message in messages {
        let cleaned = clean_msg(message["content"].as_str().unwrap().to_string(), word_embeddings.clone());
        if cleaned.len() == 0 {
            continue;
        }
        let msg = pad_msg(cleaned, msg_size);
        let author_role = message["role"].as_str().unwrap();
        let author = if author_role == "user" {
            0
        } else if author_role == "assistant" {
            1
        } else {
            continue;
        };
        let chat_msg = Message {
            msg,
            author,
        };
        chat_dataset.push(chat_msg);

        count += 1;
        if count == num_messages {
            break;
        }
    }
    chat_dataset
}