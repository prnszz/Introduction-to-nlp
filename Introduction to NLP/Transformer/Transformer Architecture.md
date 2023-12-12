# Transformer
![[architecture .png]]

The Transformer is a type of neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017. The key innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing information.

In traditional sequential models, such as recurrent neural networks (RNNs) or long short-term memory networks (LSTMs), information is processed sequentially, which can lead to computational inefficiencies and difficulties in parallelization. The Transformer's self-attention mechanism enables parallelization and captures long-range dependencies in the data more effectively.

The architecture is composed of an encoder and a decoder, each consisting of multiple layers of self-attention and feedforward neural networks. Transformers have been highly successful in various natural language processing tasks, including machine translation, text summarization, and language modeling, among others. They have also been extended and adapted for use in other domains beyond NLP.
# Encoder Block
![Encoder Block](http://jalammar.github.io/images/t/transformer_resideual_layer_norm.png)
Components and construction of the encoder block:
- Input Embedding
- Positional Encoding
- Self-Attention Layer
- Add & Normalize Layer
- Feed Forward Layer
## Input(word) Embedding
Computers don’t understand words. Instead, they work on numbers, vectors or matrices. So, we need to convert our words to a vector. But how is this possible? Here’s where the concept of embedding space comes into play. It’s like an open space or dictionary where words of similar meanings are grouped together. This is called an embedding space, and here every word, according to its meaning, is mapped and assigned with a particular value. Thus, we convert our words into vectors.

## Positional Encoding
![Positional Encoding](http://jalammar.github.io/images/t/transformer_positional_encoding_example.png)
### Why we need Positional Encoding?
In different sentences, each word may take on  different meanings. So, to solve this issue, we use positional encoders. These are vectors that give context according to the position of the word in a sentence.
### Proposed method
The positional encoding vector for the $i$th position in the sequence can be represented as:
$$\begin{align*}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{{10000}^{2i/d}}\right) \\
\text{PE}_{(pos, 2i+1)} &= \cos\left(\frac{pos}{{10000}^{2i/d}}\right)
\end{align*}
$$
where:
- pos is the position of the word in the sequence,
- i is the dimension index ($2i$ means the even dimension index, $2i+i$ means the odd dimension index), and
- d is the dimensionality of the positional encoding and embeddings.
## Input Representation
The word embedding vector for each word and its corresponding positional encoding vector are added element-wise to create a new representation that includes both the semantic content of the word and information about its position in the sequence.
$$Input Representation(w,pos)=Word Embedding(w)+Positional Encoding(pos)$$
The resulting vector is then used as the input to the Transformer model.

> **Attention Mechanism in human brain**
> Attention Mechanism in transformer is originated from the mechanism of human brain.
![attention](https://shangzhih.github.io/images/2018-03-30-13-47-09.jpg)
>This image illustrates how attention works in the human brain. The red areas represent the parts of the brain that are more focused or attentive compared to other areas.

## Self-Attention Layer
Now, let's look at attention mechanism in transformer.
Say the following sentence is an input sentence we want to translate:

"`The animal didn't cross the street because it was too tired`"

What does "it" in this sentence refer to? Is it referring to the "street" or to the "animal"? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word "it", self-attention allows it to associate "it" with "animal".

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

![example sentence](http://jalammar.github.io/images/t/transformer_self-attention_visualization.png)

### How to calculate self-attention 

#### Step 1 Calculate Query, Key, and Value Vectors
The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding(**Input Representation**) by three matrices that we trained during the training process.
![step one](http://jalammar.github.io/images/t/transformer_self_attention_vectors.png)

For each input vector $x_i$, compute three vectors: a query vector $q_i$, a key vector $k_i$, and a value vector $v_i$. These can be calculated using linear projections:
$$ q_i = W_q \cdot x_i, $$
$$ k_i = W_k \cdot x_i, $$
$$v_i = W_v \cdot x_i, $$
where $W_q$, $W_k$, and $W_v$ are learnable weight matrices.

#### Step 2 Calculate Attention Scores
Compute the **attention scores** $a_{ij}$ between each pair of vectors using the dot product of query and key vectors:
$$a_{ij}=\frac{q_i⋅k_j}{\sqrt{d_k}​}$$
where $d_k$​ is the dimensionality of the key vectors. **Dividing scores by $\sqrt d_k$ leads to having more stable gradients.**
#### Step3 Apply Softmax on Attention Scores to Obtain Attention Weights
$$Attention Weights=softmax([a_{i1}​,a_{i2}​,…,a_{ij}​])$$
![](http://jalammar.github.io/images/t/self-attention_softmax.png)
Softmax normalizes the scores so Attention Weights are all positive and sum to 1. The attention weights represent the importance or relevance assigned to each position in the input sequence concerning a particular query. High attention weight means the model "attends to" or focuses more on the content of that position during the computation.

#### Step4 Multiply Each Value Vector by Attention Weights then Sum Up
$$ \text{Attention Output}_i = \sum_{j=1}^{n} \text{Attention Weights}_{ij} \cdot v_j 
$$
![](http://jalammar.github.io/images/t/self-attention-output.png)
$z_1$ on the picture refers to $Attention Output_1$. $Attention Output_1$ is the result of the attention mechanism considering the information at the first position in the sequence and combining it with the relevant information from other positions based on their attention weights.
In the actual implementation, however, this calculation is done in matrix form for faster processing. So let’s look at that now that we’ve seen the intuition of the calculation on the word level.


### Matrix Calculation of Self-Attention
The fundamental concepts and calculations of self-attention remain the same whether you're using vector or matrix notation. As a result, the main difference is "vector" with "Matrix".$${Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$![](http://jalammar.github.io/images/t/self-attention-matrix-calculation.png)
![](http://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

### What is the meaning of "Multi-Head"?
“Multi-headed” attention improves the performance of the attention layer in two ways:

1. It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the actual word itself. If we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, it would be useful to know which word “it” refers to.

2. It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.
![](http://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)
If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices
![](http://jalammar.github.io/images/t/transformer_attention_heads_z.png)
This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

How do we do that? We concat the matrices then multiply them by an additional weights matrix $W^O$.

![](http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)
To summarize:
![](http://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)
## Add & Normalize Layer
One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer in each encoder has a residual connection around it, and is followed by a layer-normalization step.
![](http://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png)

"Add"(matrix X + matrix Z) in the picture refers to "Residual connections" can be used to mitigate the vanishing gradient problem. Normalization is to Normalize the numbers according to a normal distribution. Layer-normalization contributes to the stability of the training process.

## Feed Forward Layer
If $X$ is the input to the feedforward layer, the feedforward function can be simply represented as:$$FFN(X)=ReLU(XW_1+b_1)W_2+b_2F$$​The feedforward function allows the model to capture complex, non-linear relationships in the data and helps in learning and representing abstract features. It plays a critical role in the overall effectiveness of the Transformer architecture.
# Decoder Block
![[output block.png]]

Components and construction of the decoder block:
- Output Embedding
- Positional Encoding
- Self-Attention Layer (Masked Self-Attention)
- Encoder-Decoder Attention
- Add & Normalize Layer
- Feed Forward Layer
- The Final linear and Softmax Layer
## Output Embedding
For each decoding step, the model uses the embeddings of the previously generated tokens in the target sequence as part of the input. These embeddings are often referred to as "output embeddings" because they represent the model's own previous outputs. (ignore the input from the encoder here)
![[截屏2023-12-12 下午1.45.21.png]]
## Positional Encoding 
Similar to the encoder, positional encoding is added to the output embeddings to provide information about the position of each token in the sequence.
## Self-Attention Layer (Masked Self-Attention)
### Why we should use "Mask"?
The primary reason for using Masked Self-Attention is to prevent the model from looking ahead or attending to future tokens during the generation of a particular token in the output sequence.
During autoregressive generation, the model generates tokens one at a time in a sequential manner. At each decoding step, the model predicts the next token based on the context of the previously generated tokens. It is essential to maintain a causal relationship between the generated tokens to ensure that the model only relies on information from positions that precede the current position in the sequence.
![[截屏2023-12-12 下午2.17.42.png]]
### How to achieve "Mask"?
Let's take a look of the following picture.
![[截屏2023-12-12 下午3.18.46.png]]
Grey embedding vectors are masked vectors ($x_3$, $x_4$). We can set attention score to those masked vectors to $-\infty$ (practically we just set them to for example -100000000). After applying the mask, we typically apply the $softmax$ operation to obtain the final attention weights. The $softmax$ ensures that the attention weights sum to 1 and attention weights to masked vectors should be 0.

### Intuition understanding on Matrix Calculation of Masked Self-attention
![[截屏2023-12-12 下午3.31.47.png]]
$W'$ is the matrix attention scores and after we apply $softmax$ on $W'$, we will get $W$ refers to matrix attention weights. 

## Encoder-Decoder Attention Layer
Encoder-Decoder Attention mechanism, or so called Cross Attention, is an essential component in the decoder block. The encoder-decoder attention allows the decoder to focus on different parts of the input sequence (encoder's output) while generating the output sequence.
Encoder-Decoder Attention Layer's query is derived from the output of the previous first-level decoder layer, while its key and value come from the output of the encoder. This enables each position in the decoder to attend to every position in the input sequence.
![[截屏2023-12-12 下午4.52.36.png]]

## Add & Normalize Layer
Add & Normalize Layers in a decoder block have similar roles with those layers in a encoder block.
## Feed Forward Layer
Feed Forward Layers in a decoder block have similar roles with those layers in a encoder block.
## The Final Linear and Softmax Layer
The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.
![](http://jalammar.github.io/images/t/transformer_decoder_output_softmax.png)



# Review

![](https://www.datocms-assets.com/96965/1684227303-2-transformers-explained.png)

References:
https://jalammar.github.io/illustrated-transformer/
https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php
