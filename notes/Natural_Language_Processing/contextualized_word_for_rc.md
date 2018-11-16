# Contextualized Word Representations for Reading Comprehension

- Submitted on 2017. 12
- Shimi Salant and Jonathan Berant

## Simple Summary

> Evaluate the importance of context when the question and the document are each read on their own. We take a standard neural architecture for the task of reading comprehension, and show that by providing rich contextualized word representations from a large language model, and allowing the model to choose between context dependent and context independent word representations, we can dramatically improve performance and reach state-of-the-art performance on the competitive SQuAD dataset.

- hypothesize that the still-relatively-small size of current RC datasets, combined with the fact that a large part of examples can be solved with local question-document matching heuristics.

- **Contextualized Word Representations**
	- employ a re-embedding component in which a contextual representation and a noncontextual one are explicitly combined per token. ( t-th token w_t is the result of a Highway layer)
	- RNN-based token re-embedding (TR): BiLSTM
	- LM-augmented token re-embedding (TR+LM): pre-trained language model encoder + BiLSTM (a strong language model that was pre-trained on large corpora as a fixed encoder which supplies additional contextualized token representations.)
		- [Pre-trained LM](https://hb-research.github.io/notes/notes/exploring_limits_of_lm.html) with One Billion Words Benchmark dataset

- Experiments (SQuAD)
	- Base: [RaSoR Model](https://arxiv.org/abs/1611.01436)
	- RaSoR (base model) : EM 70.6, F1 78.7
	- RaSoR + TR : EM 75, F1 82.5
	- RaSoR + TR + LM(L1) : EM 77, F1 84   (* L1 : the projections of the first)
layer

- Analysis
	- the addition of our reembedding model incurs additional depth and capacity for the resultant overall model.
	- finding highlights a fundamental problem with maintaining fixed word representations:
		- albeit pre-trained on extremely large corpora, we observe that the embeddings of rare words are lacking in the sense that an encoder processing these embeddings chooses to complement them with information emanating from their context.
	- effectively showing the benefit of training a QA model in a semisupervised fashion.
	- improved robustness to two of their adversarial schemes. (Adversarial examples for RC)