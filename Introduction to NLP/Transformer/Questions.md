# Why does the Transformer use multi-head attention mechanism? (Why not use a single head)
- Increased Model Representation Capacity
- Improved Stability and Generalization
- Parallel Computation
- Better Abstract Representations
Multi-head attention mechanism increases the model's representation capacity by allowing it to learn diverse relationships and patterns in different subspaces. It improves stability and generalization by enabling the model to collectively learn from multiple attentional perspectives. The use of parallel computation in each head enhances efficiency during training and inference. Additionally, the multi-head attention mechanism helps the model extract better abstract representations at various levels, capturing both local and global features in the input sequence.

# Why are different weight matrices used for Q and K in the Transformer, and why can't the same value be used for self-dot products?
- Enhancing Model Expressiveness
- Improving Generalization
If Q is set equal to K, the model is likely to generate an attention matrix resembling an identity matrix. This causes self-attention to degenerate into a point-wise linear mapping, contradicting the original design intent. Using different Q and K allows the model to learn diverse attention patterns, better capturing the complexity of the input sequence.

# Why does the Transformer choose dot product over addition when computing attention? What are the differences in terms of computational complexity and effectiveness between the two?
- Dot product has a lower computational complexity.
- Dot product is more favorable for gradient propagation.
- Dot product can better capture relationships between different positions in the sequence.

# Why is attention scaled (divided by the square root of $d_k$, the dimensionality of the query or key vector($\sqrt{d_k}$​)) before applying $softmax$, and can you explain this through formula derivation?
- Avoiding Numerical Instability and Gradient Vanishing
> When computing the softmax, the exponential function's input can become very large, leading to numerical instability or even overflow. By scaling, we ensure that the input to the softmax is numerically stable, avoiding these issues. Additionally, scaling the input helps prevent gradient vanishing, making the model easier to train.
- Controlling the Variance of Dot Products
> From a statistical perspective, assuming the query and key components are independently and identically distributed, with a mean of 0 and a variance of 1, the dot product's mean is 0, but the variance is $d_k$. This results in different dimensions having significantly different values, affecting the stability and training effectiveness of the $softmax$ function.
> By dividing by $\sqrt{d_k}$, we effectively normalize the variance of the dot product during the attention mechanism, making the $softmax$ smoother. This is crucial when dealing with a large number of attention heads or higher-dimensional attention mechanisms, where the variance of dot products can become very large.

# How to mask 'padding' when calculating attention score?
The masking is typically done by assigning a large negative value (usually -1000 is enough) to the positions corresponding to padding tokens.
```import torch

# Example input sequence and padding mask
batch_size = 3
sequence_length = 5
embedding_dim = 10

input_sequence = torch.rand((batch_size, sequence_length, embedding_dim))
padding_mask = torch.tensor([[0, 0, 1, 0, 1],
                             [0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0]])

# Apply padding mask with -10000
attention_scores = input_sequence + (1 - padding_mask.unsqueeze(-1)) * (-10000)

# Now you can apply softmax to the attention scores
attention_weights = torch.nn.functional.softmax(attention_scores, dim=-2)
```

# Why reduce the dimensionality for each head, when performing multi-head attention.
![](http://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)
In a single multi-head attention layer, the input is split into multiple heads, each with its own set of learnable parameters for key (K), query (Q), and value (V) transformations. These heads independently calculate attention scores, allowing the model to focus on different aspects of the input sequence in separate subspaces.

In summary, the core idea is to enable each attention head to independently attend to a distinct subspace of the input sequence, and by doing so, capture richer and more diverse feature information. This parallel processing across multiple heads facilitates the model's ability to learn a variety of patterns and relationships within the input data, enhancing its overall representational power and allowing it to perform better on a range of tasks.

# Can you explain why transformer use positional encoding and what's its defects and advantages?
The Transformer architecture relies on self-attention mechanisms to process input sequences in parallel. However, it lacks the inherent sequential order information present in traditional recurrent neural networks. To address this issue, positional encoding is introduced to the input embeddings in order to provide the model with information about the relative or absolute positions of the tokens in the input sequence. One common method is to use a combination of sine and cosine functions:
$$\begin{align*}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{{10000}^{2i/d}}\right) \\
\text{PE}_{(pos, 2i+1)} &= \cos\left(\frac{pos}{{10000}^{2i/d}}\right)
\end{align*}
$$
where $pos$ is the position of the token in the sequence, $i$ is the dimension of the positional encoding, and $d$ is the dimensionality of the model.

**Defects or limitations of positional encoding:**
- **Fixed Sequence Length:** The positional encoding technique is designed for a fixed sequence length. If you want to process sequences of varying lengths, you may need additional mechanisms to handle this variability.

# What's the residual structure in a transformer?
In a Transformer, the residual structure refers to adding the input directly to the output of a neural network layer, creating a residual connection. This is included in Add & Normalize layer in a transformer architecture. A residual connection solve the vanishing gradient problem.


# What's the structure of the feedforward neural network in a transformer? What's it's activation function? What's the Pros and Cons?
In the Transformer architecture, the feedforward neural network (FFN) is a component within each encoder and decoder layer. The activation function is $ReLU$. 
$$FFN(X)=ReLU(XW_1+b_1)W_2+b_2$$
# What is the interaction between the Encoder and Decode?
- **Encoder-Decoder Attention Mechanism (Cross Attention):** This attention mechanism allows each position in the Decoder to attend to all positions in the Encoder, capturing relationships between the input and output sequences. In this attention mechanism, for each position in the Decoder, a weighted sum of the Encoder's outputs is calculated based on the similarity between the current position in the Decoder and each position in the Encoder.

In summary, Cross Attention Mechanism gets $K$ and $V$ from Encoder and $Q$ from Decoder. 