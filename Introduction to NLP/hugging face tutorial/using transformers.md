# What is the order of the language modeling pipeline?

The tokenizer handles text and returns IDs. The model handles these IDs and outputs a prediction. The tokenizer can then be used once again to convert these predictions back to some text.

# How many dimensions does the tensor output by the base Transformer model have, and what are they?

The sequence length, the batch size, and the hidden size

# Examples of subword tokenization

- WordPiece

  ["I", "'", "m", " hungry", " and", " I", " want", " to", " eat", " some", " desserts"]

- BPE

  {"I": 1, "'m": 2, "hungry": 3, "and": 4, "want": 5, "to": 6, "eat": 7, "some": 8, "desserts": 9}

- Unigram

  ["I'm", "hungry", "and", "I", "want", "to", "eat", "some", "desserts"]

# What is a model head?

An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output.

