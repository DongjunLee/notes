# A Structured Self-attentive Sentence Embedding

- Submitted on 2017. 3
- Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou and Yoshua Bengio

## Simple Summary

> Proposes a new model for extracting an interpretable sentence embedding by introducing self-attention. Instead of using a vector, we use a 2-D matrix to represent the embedding, with each row of the matrix attending on a different part of the sentence. We also propose a self-attention mechanism and a special regularization term for the model. As a side effect, the embedding comes with an easy way of visualizing what specific parts of the sentence are encoded into the embedding.

![images](../images/self_attn_sentence_embed_1.png)

- A self-attention mechanism for these sequential models to replace the max pooling or averaging step.
	- allows extracting different aspects of the sentence into multiple vector representations.
	- interpreting the extracted embedding becomes very easy and explicit.
- Model
	- bidirectional LSTM -> concat -> self-attention
	- the final sentence embedding to directly access previous LSTM hidden states via the attention summation.
	- `a = softmax(w_s2 tanh (W_s1 H^T))`
	- `M = AH` M: r-by-2u embedding matrix, A: annotation matrix, H: LSTM hidden states
	- Penalization Term: `P = ||(AA^T - I)||_F^2`

![images](../images/self_attn_sentence_embed_2.png)

- Experiments
	- The Author Profiling dataset
	- Sentiment Analysis: Yelp and Age Dataset
	- Textual Entailment: the SNLI corpus

- able to encode any sequence with variable length into a fixed size representation, without suffering from long-term dependency problems.
- not able to train it in an unsupervised way.

