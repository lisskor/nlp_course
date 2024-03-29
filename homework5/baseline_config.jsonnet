local embedding_dim = 128;
local hidden_dim = 128;

{
  "dataset_reader": {
    "type": "sst_tokens"
  },
  "train_data_path": "trees/train.txt",
  "validation_data_path": "trees/dev.txt",
  "model": {
    "type": "lstm_classifier",
    "word_embeddings": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embedding_dim
      }
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 20,
    "patience": 10
  }
}
