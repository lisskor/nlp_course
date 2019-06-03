{
  "dataset_reader": {
    "type": "names-reader"},
  "train_data_path": "../data/names/",
  "validation_data_path": "../data/names/",
  "model": {
    "type": "names-classifier",
        "char_embeddings": {
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 6,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": 6,
                        "hidden_size": 6
                    }
                }
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 6,
            "hidden_size": 6
        }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["name_characters", "num_token_characters"]],
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": -1,
    "optimizer": {
            "type": "sgd",
            "lr": 0.1
        },
  }
}