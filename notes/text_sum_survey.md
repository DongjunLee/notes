# Text Summarization Techniques: A Brief Survey

- Submitted on 2017. 7
- Mehdi Allahyari, Seyedamin Pouriyeh, Mehdi Assefi, Saeid Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, Krys Kochut

## Simple Summary

> In recent years, there has been a explosion in the amount of text data from a variety of sources. This volume of text is an invaluable source of information and knowledge which needs to be effectively summarized to be useful. In this review, the main approaches to automatic text summarization are described. We review the different processes for summarization and describe the effectiveness and shortcomings of the different methods.

- `Summary`: a text that is produced from one or more texts, that conveys important information in the original text(s), and that is no longer than half of the original text(s) and usually, significantly less than that.

- Methods to determine the sentence weight: (1950s)
	1. Cue Method: based on presence or absence of words in the continuous unified electornic(cue) dictionary.
	2. Title Method: sum of all the content words appearing in the title and headings of a text.
	3. Location Method: individual paragraphs have a higher probability of being relevant.

- Approaches for automatic summarization:
	- `Extractiion`: identifying important sections of the text and generating them verbatim.
	- `Abstractive`: interpret and examine the text using advanced natural language techniques in order to generate a new shorter text.
		- cope with problems such as semantic representation, inference and natural language generation.

### Extractive Summarization

1. Construct an intermediate representation of the input text with expresses the main aspects of the text.
	- `topic representation`: transform the text into an intermediate representation and interpret the topic(s) discussed in the text.
		- eg) frequency-driven, topic word, latent semantic analysis and Bayesian topic model
	- `indicator representation`: describe every sentence as a list of features (indicators) of importance.
		- eg) sentence length, position in the document, certain phrase
2. Score the sentences based on the representation (importance score).
	- `topic representation`: how well the sentence explains some of the most important topics of the text.
	- `indicator representation`: computed by aggregating the evidence from different indicators.
3. Select a summary comprising of a number of sentences.
	- it should maximize overall importance and coherency and minimize the redundancy.

#### Topic Representation Approaches

1. topic words
	- `topic signature`: Luhn's idea, used log-likelihood ratio test to identity explanatory words which in summarization literature.
	- compute the importance of a sentence
		1. a function of the number of topic signatures it contains (higher scores to longer sentence)
		2. the propotion of the topic signatures in sentence (density of the topic words)
2. frequency-driven
	- word probability: the probability of a word `w` is determined as the number of occurrences of the word, `f(w).
		- eg) SumBasic system
	- Term Frequency Inverse Document Frequency (TFIDF)
		- Centrolid-based summarization: based on TFIDF, rank sentences by computing there salience using a set of features. 
		- metrics
			1. cluster-based relative utility (CBRU): how relevant a particular sentence is to the general topic of the entire cluster
			2. cross-sentence informational subsumption (CSIS): redundancy among sentences.
3. latent semantic analysis
	- unsupervised method for extracting a representation of text semantics based on observed words.
	- select highly ranked sentences for single and multi-document summarization in the news domain.
	- Algorithm
		1. term-sentence matrix
		2. weights of the words are computed by TFIDF
		3. SVD -> A = UΣV^T
		4. `U`: weights of a topic i, `ΣV^T`: how much a sentence represent a topic = d_ij shows the weights of the topic i and sentence j
	- Advancement (LSA-based technique)
		 1. leverage the weight of each topic to decide the relative size of the summary that should cover the topic
		 2. locate those sentences the defined the weight of the sentence (using weight fucntion `g`)
4. Bayesian Topic Models
	- limitations of multi-document summarization methos
		1. consider the sentences as independent of each other
		2. sentence score functions do not have very clear probabilistic interpretations
	- uncover and represent the topics of documents. -> can determine the similarities and differences between documents to be used summarization.
	- use KL divergence: shows the fact that good summaries are intuitively similar to the input documents.
	- Advancement
		- Latent Dirichlet allocation (LDA): unsupervised techique for extracting thematic information (topic) of a collection of documents. main idea is that documents are represented as a random mixture of latent topic, where each topic is a probability distribution over words.
		- BAYESUM: Bayesian summarization model for query-focused summarization.
		- multi-document summarization as a prediction problem based on a two-phase hybrid model.


#### Knowledge based and automatic summarization

- building more accurate summarization system is to combine summarization techiques with knowledge based (semantic-based or ontology-based summarizers).
- ontology-based extraction of sentences outperforms baseline summarizers.

#### Indicator Representation Approaches

- Graph methods
	- a common technique employed to connect two vertices is to measure the similarity of two sentences if it is greater then a threshold they are connected.
	- used for single as well as multi-document summarization.
- Machine Learning
	- Naive Bayes, decision trees, support vector machines, Hidden Markov models adn Conditional Random Fields are among the most common machine learning techniques used for summarization.
	- One of the primary issues: 1. Annotated corpora creation 2. Semi-supervised approaches

### Evaluation

- Major difficulties of automatic summary evaluation
	1. decide and specify the most important parts of the original text to preserve.
	2. have to automatically identify these pieces of important information in the candidate summary, since this information can be represented using disparate expressions.
	3. the readability of the summary in terms of grammaticality and coherence has to be evaluated.

- `ROUGH`: called Recall Oriented Understudy for Gisting Evaluation, to automatically determine the quality of a summary by comparing it to human (reference) summaries.
	- ROUGH-n: based on comparision of n-grams
	- ROUGH-L: longest common subsequence (LCS) betweeen the two sequences of text
	- ROUGH-SU: skip bi-gram and uni-gram

