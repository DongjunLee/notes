# Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models

- Submitted on 2016. 4
- Minh-Thang Luong and Christopher D. Manning

## Simple Summary

> word-character solution to achieving open vocabulary NMT. We build hybrid systems that translate mostly at the word level and consult the character components for rare words. Our character-level recurrent neural networks compute source word representations and recover unknown target words when needed.

![images](../images/nmt_hybrid_word_and_char_1.png)

- The core of the design is a word-level NMT with the advantage of being fast and easy to train.
- Source Character-based Representation: always initialized with zero states and  use Final hidden state as word representation.
- Target Character-level Generation:
	1. Hidden-state Initialization
		- target character-level generation requires the current word-level context to produce meaningful translation.
		- separate-path target generation approach works as follows. to create a counterpart vector `h˘t` that will be used to seed the character-level decoder.
		- ```h˘t = tanh(W˘[c_t;h_t])```
	2. Word-Character Generation Strategy - \<unk\> is fed to the word-level decoder “as is” using its corresponding word embedding.
		- training: choice decouples all executions over <unk> instances of the character-level decoder as soon the word-level NMT completes. 
		- test:  utilize our character-level decoder with beam search to generate actual words for these <unk>.
- demonstrated the potential of purely character-based models in producing good translations.