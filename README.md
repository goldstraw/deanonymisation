# De-Anonymisation of Text Messages

This repository contains my Rust implementations of a transformer built from scratch, and a word embedding generator. The transformer is forked from [my RustTransformer repository](https://github.com/goldstraw/RustTransformer). The word embedding generator is forked from [my WordEmbeddings repository](https://github.com/goldstraw/WordEmbeddings).

The transformer is configured to be trained to de-anonymise text messages. The dataset used to train the transformer is the [Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations). The transformer is trained to predict whether the given message is from a real user or a chatbot.

My pre-trained transformer model is available [here](https://github.com/goldstraw/deanonymisation/blob/main/1700742468_chtbt_model_10_64_1_1_100.json). The embeddings used to train the transformer are available [here](https://github.com/goldstraw/deanonymisation/blob/main/chatbot_arena_embeddings.json).

## Usage

To use this transformer implementation, you must have Rust and Cargo installed on your machine. After installing Rust and Cargo, you can clone this repository to your local machine.

To run the transformer, use the following command:

```
$ cargo run --release
```

This command will train the transformer on the chatbot arena dataset and then run tests on a test set. The results of the training and testing will be printed to the console. The transformer can be easily configured to train on a different dataset by changing the `dataset.rs` file as well as some hard-coded values in the `run.rs` file.

## Further Reading

Dataset link: [Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

Original paper for dataset: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

For more information about this project, read [my blog post on transformers](https://charliegoldstraw.com/articles/transformers/).