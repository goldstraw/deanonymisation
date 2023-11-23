use word_embeddings::*;
use std::io;

fn main() {
    println!("Enter the desired dimensionality: ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let dimensionality = input.trim().parse().expect("Invalid input.");

    // println!("Enter the number of messages to process: ");
    // let mut input = String::new();
    // io::stdin().read_line(&mut input).expect("Failed to read input.");
    // let num_messages = input.trim().parse().expect("Invalid input.");

    println!("Output file name (e.g. data.json): ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let file_name = input.trim();

    println!("Number of threads to use: ");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input.");
    let num_threads = input.trim().parse().expect("Invalid input.");

    run::run(dimensionality, file_name, num_threads);
}
