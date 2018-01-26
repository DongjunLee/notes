
<p align="center">
  <img src="images/logo.png" width=250>
</p>

<p align="center">

  <a href="https://github.com/hb-research/notes">
    <img src="https://img.shields.io/badge/DeepLearning-Notes-brightgreen.svg" alt="Project Introduction">
  </a>
  
  <a href="https://github.com/hb-research/notes">
    <img src="https://img.shields.io/badge/Summary-Code-brightgreen.svg" alt="Project Introduction">
  </a>

</p>

# notes: Notes of Deep Learning

## Category 

- [Background knowledge](#background-knowledge)
- [AI](#ai)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Optimization](#optimization)
- [Transfer Learning](#transfer-learning)
- [Unsupervised & Generative](#unsupervised--generative)

---



## Background knowledge

- [Gaussian Process](notes/gausian_process.md) ****`Supervised`****, ****`Regression`****
- [Importance Sampling](notes/importance_sampling.md) ****`Approximate`****

---

- Deep Learning (2015) ****`Review`****
	- [nature](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) | [note](notes/deep_learning.md)


## AI

- Building Machines That Learn and Think Like People (2016. 4)
	- ****`Cognitive`****, ****`Human-Like`****
	- [arXiv](https://arxiv.org/abs/1604.00289) | [note](notes/ml_learn_and_think_like_human.md) | [the morning paper](https://blog.acolyer.org/2016/11/25/building-machines-that-learn-and-think-like-people/)

## Computer Vision

- Network In Network (2013. 12)
	- ****`Conv 1x1`****, ****`Bottleneck`****
	- [arXiv](https://arxiv.org/abs/1312.4400) | [note](notes/network_in_network.md)
- Fractional Max-Pooling (2014. 12)
	- ****`Max-Pooling`****, ****`Data Augmentation`****, ****`Regularization`****
	- [arXiv](https://arxiv.org/abs/1412.6071) | [note](notes/fractional_max-pooling.md)
- Deep Residual Learning for Image Recognition (2015. 12)
	- ****`Residual`****, ****`ImageNet 2015`****
	- [arXiv](https://arxiv.org/abs/1512.03385) | [note](notes/residual_network.md)


## Natural Language Processing

- Distributed Representations of Words and Phrases and their Compositionality (2013. 10) 
	- ****`Word2Vec`****, ****`CBOW`****, ****`Skip-gram`****
	- [arXiv](https://arxiv.org/abs/1310.4546)
- GloVe: Global Vectors for Word Representation (2014) 
	- ****`Word2Vec`****, ****`GloVe`****, ****`Co-Occurrence`****
	- [paper](https://nlp.stanford.edu/pubs/glove.pdf)
- Convolutional Neural Networks for Sentence Classification (2014. 8) 
	- ****`CNN`****, ****`Classfication`****
	- [arXiv](https://arxiv.org/abs/1408.5882) | [code](https://github.com/DongjunLee/text-cnn-tensorflow)
- Neural Machine Translation by Jointly Learning to Align and Translate (2014. 9) 
	- ****`Seq2Seq`****, ****`Attention(Align)`****, ****`Translation`****
	- [arXiv](https://arxiv.org/abs/1409.0473) | [note](notes/bahdanau_attention.md) | [code](https://github.com/DongjunLee/conversation-tensorflow) 
- Text Understanding from Scratch (2015. 2) 
	- ****`CNN`****, ****`Character-level`****
	- [arXiv](https://arxiv.org/abs/1506.07285)
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing (2015. 6) 
	- ****`Memory`****, ****`QA`****, ****`bAbi`****
	- [arXiv](https://arxiv.org/abs/1506.07285) | [code](https://github.com/DongjunLee/dmn-tensorflow) 
- Pointer Networks (2015. 6) 
	- ****`Seq2Seq`****, ****`Attention`****, ****`Combinatorial`****
	- [arXiv](https://arxiv.org/abs/1506.03134) | [note](notes/pointer_network.md) 
- Skip-Thought Vectors (2015. 6) 
	- ****`Sentence2Vec`****, ****`Unsupervised`****
	- [arXiv](https://arxiv.org/abs/1506.06726) | [note](notes/skip_thought.md)
- A Neural Conversational Model (2015. 6) 
	- ****`Seq2Seq`****, ****`Conversation`****
	- [arXiv](https://arxiv.org/abs/1506.05869)
- Teaching Machines to Read and Comprehend (2015. 6) 
	- ****`Deepmind`****, ****`Attention`****, ****`QA`****
	- [arXiv](https://arxiv.org/abs/1506.03340) | [note](notes/teaching_machine_read_and_comprehend.md)
- Effective Approaches to Attention-based Neural Machine Translation (2015. 8) 
	- ****`Seq2Seq`****, ****`Attention`****, ****`Translation`****
	- [arXiv](https://arxiv.org/abs/1508.04025) | [note](notes/luong_attention.md) | [code](https://github.com/DongjunLee/conversation-tensorflow) 
- Character-Aware Neural Language Models (2015. 8) 
	- ****`CNN`****, ****`Character-level`****
	- [arXiv](https://arxiv.org/abs/1508.06615)
- Neural Machine Translation of Rare Words with Subword Units (2015. 8) 
	- ****`Out-Of-Vocabulary`****, ****`Translation`****
	- [arXiv](https://arxiv.org/abs/1508.07909) | [note](notes/subword_nmt.md)
- A Diversity-Promoting Objective Function for Neural Conversation Models (2015. 10) 
	- ****`Conversation`****, ****`Objective`****
	- [arXiv](https://arxiv.org/abs/1510.03055) | [note](notes/diversity_conversation.md)
- Multi-task Sequence to Sequence Learning (2015. 11) 
	- ****`Multi-Task`****, ****`Seq2Seq`****
	- [arXiv](https://arxiv.org/abs/1511.06114) | [note](notes/multi_task_seq2seq.md)
- Multilingual Language Processing From Bytes (2015. 12) 
	- ****`Byte-to-Span`****, ****`Multilingual`****, ****`Seq2Seq`****
	- [arXiv](https://arxiv.org/abs/1512.00103) | [note](notes/byte_to_span.md)
- Strategies for Training Large Vocabulary Neural Language Models (2015. 12) 
	- ****`Vocabulary`****, ****`Softmax`****, ****`NCE`****, ****`Self Normalization`****
	- [arXiv](https://arxiv.org/abs/1512.04906) | [note](notes/vocabulary_strategy.md)
- Recurrent Memory Networks for Language Modeling (2016. 1) 
	- ****`RMN`****, ****`Memory Bank`****
	- [arXiv](https://arxiv.org/abs/1601.01272)
- Exploring the Limits of Language Modeling (2016. 2) 
	- ****`Google Brain`****, ****`Language Modeling`****
	- [arXiv](https://arxiv.org/abs/1602.02410) | [note](notes/exploring_limits_of_lm.md)
- Swivel: Improving Embeddings by Noticing What's Missing (2016. 2) 
	- ****`Word2Vec`****, ****`Swivel `****, ****`Co-Occurrence`****
	- [arXiv](https://arxiv.org/abs/1602.02215)
- Incorporating Copying Mechanism in Sequence-to-Sequence Learning (2016. 3) 
	- ****`CopyNet`****, ****`Seq2Seq`****
	- [arXiv](https://arxiv.org/abs/1603.06393) | [note](notes/copynet.md)
- Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models (2016. 4) 
	- ****`Translation`****, ****`Hybrid NMT`****, ****`Word-Char`****
	- [arXiv](https://arxiv.org/abs/1604.00788) | [note](notes/nmt_hybrid_word_and_char.md)
- SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016. 6) 
	- ****`DataSet`****, ****`Reading Comprehension`****
	- [arXiv](https://arxiv.org/abs/1606.05250) | [note](notes/squad.md) | [dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- Sequence-Level Knowledge Distillation (2016. 6) 
	- ****`Distil`****, ****`Teacher-Student`****
	- [arXiv](https://arxiv.org/abs/1606.07947) | [note](notes/sequence_knowledge_distillation.md)
- Attention-over-Attention Neural Networks for Reading Comprehension (2016. 7) 
	- ****`Attention`****, ****`Cloze-style`****, ****`Reading Comprehension`****
	- [arXiv](https://arxiv.org/abs/1607.04423) | [note](notes/attn_over_attn_nn_rc.md)
- Recurrent Neural Machine Translation (2016. 7) 
	- ****`Translation`****, ****`Attention (RNN)`****
	- [arXiv](https://arxiv.org/abs/1607.08725)
- An Actor-Critic Algorithm for Sequence Prediction (2016. 7) 
	- ****`Seq2Seq`****, ****`Actor-Critic`****, ****`Objective`****
	- [arXiv](https://arxiv.org/abs/1607.07086) | [note](notes/actor_critic_for_seq.md)
- SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient (2016. 9) 
	- ****`Seq2Seq`****, ****`GAN`****, ****`RL`****
	- [arXiv](https://arxiv.org/abs/1609.05473) | [note](notes/seq_gan.md)
- Attention Is All You Need (2017. 6)
	- ****`Self-Attention`****, ****`Seq2Seq (without RNN, CNN)`****
	- [arXiv](https://arxiv.org/abs/1706.03762) | [note](notes/transformer.md) | [code](https://github.com/DongjunLee/transformer-tensorflow)  
- Neural Text Generation: A Practical Guide (2017. 11) 
	- ****`Seq2Seq`****, ****`Decoder Guide`****
	- [arXiv](https://arxiv.org/abs/1711.09534) | [note](notes/neural_text_generation.md)
- Recent Advances in Recurrent Neural Networks (2018. 1) 
	- ****`RNN`****, ****`Recent Advances`****, ****`Review`****
	- [arXiv](https://arxiv.org/abs/1801.01078)


## Optimization

- Understanding the difficulty of training deep feedforward neural networks (2010) 
	- ****`Weight Initialization (Xavier)`****
	- [paper](http://proceedings.mlr.press/v9/glorot10a.html) | [note](notes/xavier_initialization.md)
- On the difficulty of training Recurrent Neural Networks (2012. 11) 
	- ****`Gradient Clipping`****, ****`RNN`****
	- [arXiv](https://arxiv.org/abs/1211.5063)
- Dropout (2012, 2014) 
	- ****`Regulaizer`****, ****`Ensemble`****
	- [arXiv (2012)](https://arxiv.org/abs/1207.0580) | [arXiv (2014)](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) | [note](notes/dropout.md)
- Regularization of Neural Networks using DropConnect (2013) 
	- ****`Regulaizer`****, ****`Ensemble`****
	- [paper](https://cs.nyu.edu/~wanli/dropc/dropc.pdf) | [note](notes/dropconnect.md) | [wanli_summary](https://cs.nyu.edu/~wanli/dropc/)
- Batch Normalization (2015. 2) 
	- ****`Regulaizer`****, ****`Accelerate Training`****, ****`CNN`****
	- [arXiv](https://arxiv.org/abs/1502.03167) | [note](notes/batch_normalization.md)
- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015. 2) 
	- ****`PReLU`****, ****`Weight Initialization (He)`****
	- [arXiv](https://arxiv.org/abs/1502.01852) | [note](notes/he_initialization.md)
- A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015. 4) 
	- ****`Weight Initialization`****, ****`RNN`****, ****`Identity Matrix`****
	- [arXiv](https://arxiv.org/abs/1504.00941)
- Training Very Deep Networks (2015. 7) 
	- ****`Highway`****, ****`LSTM-like`****
	- [arXiv](https://arxiv.org/abs/1507.06228) | [note](notes/highway_networks.md)
- Deep Networks with Stochastic Depth (2016. 3) 
	- ****`Dropout`****, ****`Ensenble`****, ****`Beyond 1000 layers`****
	- [arXiv](https://arxiv.org/abs/1603.09382) | [note](notes/stochastic_depth.md)
- Layer Normalization (2016. 7) 
	- ****`Regulaizer`****, ****`Accelerate Training`****, ****`RNN`****
	- [arXiv](https://arxiv.org/abs/1607.06450) | [note](notes/layer_normalization.md)


## Transfer Learning

- Progressive Neural Networks (2016, 6)
	- ****`ProgNN `****, ****`Incorporate Prior Knowledge`****
	- [arXiv](https://arxiv.org/abs/1606.04671) | [the morning paper](https://blog.acolyer.org/2016/10/11/progressive-neural-networks/)
	
	
## Unsupervised & Generative

- Auto-Encoding Variational Bayes (2013, 12)
	- ****`Generative`****, ****`Approximate`****
	- [arXiv](https://arxiv.org/abs/1312.6114) | [note](notes/vae.md)
