# notes [![hb-research](https://img.shields.io/badge/hb--research-notes-green.svg?style=flat&colorA=448C57&colorB=555555)](https://github.com/hb-research)

Summary Notes, Codes and Articles of Deep Learning Research.  

If you want to download papers.

```python notes_sync.py --sync_path {path_name}```

---

## Category

- [Background Knowledge](#background-knowledge)
- [Code](#code)
	- [PyTorch](#pytorch)
	- [TensorFlow](#tensorflow)
- [Research Paper](#research-paper)
	- [Adversarial Example](#adversarial-example) 
	- [AI](#ai)
	- [Cognitive](#cognitive)
	- [Computer Vision](#computer-vision)
	- [Model](#model)
	- [Natural Language Processing](#natural-language-processing)
	- [One-Shot/Few-Shot Learing](#one-shotfew-shotmeta-learing)
	- [Optimization](#optimization)
	- [Reinforcement Learning](#reinforcement-learning)
	- [Transfer Learning](#transfer-learning)
	- [Unsupervised & Generative](#unsupervised--generative)

### Description

- **bold** : <U>important</U>
- **`tag`**: <U>keyword</U>
- paper, article, note and code

---


## Background knowledge

- Gaussian Process
	-  **`Supervised`**, **`Regression`**
	-  [note](notes/gausian_process.md)
- Importance Sampling
	- **`Approximate`** 
	- [notes](notes/importance_sampling.md)
- Information Theory: A Tutorial Introduction (2018. 2)
	- **`Shannon's Theory`**
	- [arXiv](https://arxiv.org/abs/1802.05968) 

---

## Code

The codes are implemented by TensorFlow and initiate project with [hb-base](https://github.com/hb-research/hb-base).

### PyTorch

- [gan-pytorch](https://github.com/DongjunLee/gan-pytorch) : Generative Adversarial Networks

### TensorFlow

- [transformer-tensorflow](https://github.com/DongjunLee/transformer-tensorflow) : Attention Is All You Need
- [relation-network-tensorflow](https://github.com/DongjunLee/relation-network-tensorflow) : A simple neural network module for relational reasoning for bAbi task
- [conversation-tensorflow](https://github.com/DongjunLee/conversation-tensorflow) : Conversation Models (Seq2Seq with Attentional Model)
- [dmn-tensorflow](https://github.com/DongjunLee/dmn-tensorflow) : Ask Me Anything: Dynamic Memory Networks for Natural Language Processing
- [text-cnn-tensorflow](https://github.com/DongjunLee/text-cnn-tensorflow) : Convolutional Neural Networks for Sentence Classification(TextCNN)
- [vae-tensorflow](https://github.com/DongjunLee/vae-tensorflow) : Auto-Encoding Variational Bayes
- [char-rnn-tensorflow](https://github.com/DongjunLee/char-rnn-tensorflow) : Multi-layer Recurrent Neural Networks for character-level language models

---

## Research Paper

Deep Learning (2015) **`Review`**  
	- [nature](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf), [note](notes/deep_learning.md)


### Adversarial Example 

- Explaining and Harnessing Adversarial Examples (2014. 12)
	- **`FGSM (Fast Gradient Sign Method)`**, **`Adversarial Training`**
	- [arXiv](https://arxiv.org/abs/1412.6572)
- The Limitations of Deep Learning in Adversarial Settings (2015. 11)
	- **`JSMA (Jacobian-based Saliency Map Approach)`**, **`Adversarial Training`**
	- [arXiv](https://arxiv.org/abs/1511.07528)
- Understanding Adversarial Training: Increasing Local Stability of Neural Nets through Robust Optimization (2015. 11)
	- **`Adversarial Training (generated adversarial examples)`**, **`Proactive Defense`**
	- [arXiv](https://arxiv.org/abs/1511.05432)
- Practical Black-Box Attacks against Machine Learning (2016. 2)
	- **`Black-Box (No Access to Gradient)`**, **`Generate Synthetic`**
	- [arXiv](https://arxiv.org/abs/1602.02697)
- Adversarial Patch (2017. 12)
	- **`Patch`**, **`White Box`**, **`Black Box`**
	- [arXiv](https://arxiv.org/abs/1712.09665), [the_morning_paper](https://blog.acolyer.org/2018/03/29/adversarial-patch/)


### AI

- Machine Theory of Mind (2018. 2)
	- **`ToMnet`**, **`Meta-Learning`**, **`General Model`**, **`Agent`**
	- [arXiv](https://arxiv.org/abs/1802.07740)


### Cognitive

- Building Machines That Learn and Think Like People (2016. 4)
	- **`Human-Like`**, **`Learn`**, **`Think`**
	- [arXiv](https://arxiv.org/abs/1604.00289), [note](notes/ml_learn_and_think_like_human.md), [the morning paper](https://blog.acolyer.org/2016/11/25/building-machines-that-learn-and-think-like-people/)

### Computer Vision

- **Network In Network** (2013. 12)
	- **`Conv 1x1`**, **`Bottleneck`**
	- [arXiv](https://arxiv.org/abs/1312.4400), [note](notes/network_in_network.md)
- Fractional Max-Pooling (2014. 12)
	- **`Max-Pooling`**, **`Data Augmentation`**, **`Regularization`**
	- [arXiv](https://arxiv.org/abs/1412.6071), [note](notes/fractional_max-pooling.md)
- **Deep Residual Learning for Image Recognition** (2015. 12)
	- **`Residual`**, **`ImageNet 2015`**
	- [arXiv](https://arxiv.org/abs/1512.03385), [note](notes/residual_network.md)
- Spherical CNNs (2018. 1)
	- **`Spherical Correlation`**, **`3D Model`**, **`Fast Fourier Transform (FFT)`**
	- [arXiv](https://arxiv.org/abs/1801.10130), [open_review](https://openreview.net/forum?id=Hkbd5xZRb)
- Taskonomy: Disentangling Task Transfer Learning (2018. 4)
	- **`Taskonomy `**, **`Transfer Learning`**, **`Computational modeling of task relations`**
	- [arXiv](https://arxiv.org/abs/1804.08328) 
- AutoAugment: Learning Augmentation Policies from Data (2018. 5)
	- **`Search Algorithm (RL)`**, **`Sub-Policy`**
	- [arXiv](https://arxiv.org/abs/1805.09501)



### Model

- **Dropout** (2012, 2014)
	- **`Regulaizer`**, **`Ensemble`**
	- [arXiv (2012)](https://arxiv.org/abs/1207.0580), [arXiv (2014)](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf), [note](notes/dropout.md)
- Regularization of Neural Networks using DropConnect (2013)
	- **`Regulaizer`**, **`Ensemble`**
	- [paper](https://cs.nyu.edu/~wanli/dropc/dropc.pdf), [note](notes/dropconnect.md), [wanli_summary](https://cs.nyu.edu/~wanli/dropc/)
- Recurrent Neural Network Regularization (2014. 9)
	- **`RNN`**, **`Dropout to Non-Recurrent Connections`**
	- [arXiv](https://arxiv.org/abs/1409.2329)
- **Batch Normalization** (2015. 2)
	- **`Regulaizer`**, **`Accelerate Training`**, **`CNN`**
	- [arXiv](https://arxiv.org/abs/1502.03167), [note](notes/batch_normalization.md)
- Training Very Deep Networks (2015. 7)
	- **`Highway`**, **`LSTM-like`**
	- [arXiv](https://arxiv.org/abs/1507.06228), [note](notes/highway_networks.md)
- A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (2015. 12)
	- **`Variational RNN`**, **`Dropout - RNN`**, **`Bayesian interpretation`**
	- [arXiv](https://arxiv.org/abs/1512.05287)
- Deep Networks with Stochastic Depth (2016. 3)
	- **`Dropout`**, **`Ensenble`**, **`Beyond 1000 layers`**
	- [arXiv](https://arxiv.org/abs/1603.09382), [note](notes/stochastic_depth.md)
- Adaptive Computation Time for Recurrent Neural Networks (2016. 3)
	- **`ACT`**, **`Dynamically`**, **`Logic Task`**
	- [arXiv](https://arxiv.org/abs/1603.08983)
- Layer Normalization (2016. 7)
	- **`Regulaizer`**, **`Accelerate Training`**, **`RNN`**
	- [arXiv](https://arxiv.org/abs/1607.06450), [note](notes/layer_normalization.md)
- Recurrent Highway Networks (2016. 7)
	- **`RHN`**, **`Highway`**, **`Depth`**, **`RNN`**
	- [arXiv](https://arxiv.org/abs/1607.03474), [note](notes/recurrent_highway.md)
- Using Fast Weights to Attend to the Recent Past (2016. 10)
	- **`Cognitive`**, **`Attention`**, **`Memory`**
	- [arXiv](https://arxiv.org/abs/1610.06258), [note](notes/fast_weights_attn.md)
- Professor Forcing: A New Algorithm for Training Recurrent Networks (2016. 10)
	- **`Professor Forcing`**, **`RNN`**, **`Inference Problem`**, **`Training with GAN`**
	- [arXiv](https://arxiv.org/abs/1610.09038), [note](notes/professor_forcing.md)
- Categorical Reparameterization with Gumbel-Softmax (2016. 11)
	- **`Gumbel-Softmax distribution `**, **`Reparameterization`**, **`Smooth relaxation`**
	- [arXiv](https://arxiv.org/abs/1611.01144), [open_review](https://openreview.net/forum?id=rkE3y85ee)
- Understanding deep learning requires rethinking generalization (2016. 11)
	- **`Generalization Error`**, **`Role of Regularization`**
	- [arXiv](https://arxiv.org/abs/1611.03530)
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017. 1)
	- **`MoE Layer`**, **`Sparsely-Gated`**, **`Capacity`**, **`Google Brain`**
	- [arXiv](https://arxiv.org/abs/1701.06538), [note](notes/very_large_nn_moe_layer.md)
- **A simple neural network module for relational reasoning** (2017. 6)
	- **`Relational Reasoning`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1706.01427), [note](notes/relational_network.md), [code](https://github.com/DongjunLee/relation-network-tensorflow)
- On Calibration of Modern Neural Networks (2017. 6)
	- **`Confidence calibration`**, **`Maximum Calibration Error (MCE)`**
	- [arXiv](https://arxiv.org/abs/1706.04599)
- When is a Convolutional Filter Easy To Learn? (2017. 9)
	- **`Conv + ReLU`**, **`Non-Gaussian Case`**, **`Polynomial Time`**
	- [arXiv](https://arxiv.org/abs/1709.06129), [open_review](https://openreview.net/forum?id=SkA-IE06W)
- mixup: Beyond Empirical Risk Minimization (2017. 10)
	- **`Data Augmentation`**, **`Vicinal Risk Minimization`**, **`Generalization`**
	- [arXiv](https://arxiv.org/abs/1710.09412), [open_review](https://openreview.net/forum?id=r1Ddp1-Rb)
- Measuring the tendency of CNNs to Learn Surface Statistical Regularities (2017. 11)
	- **`not learn High Level Semantics`**, **`learn Surface Statistical Regularities`**
	- [arXiv](https://arxiv.org/abs/1711.11561), [the_morning_paper](https://blog.acolyer.org/2018/05/29/measuring-the-tendency-of-cnns-to-learn-surface-statistical-regularities/)
- MentorNet: Regularizing Very Deep Neural Networks on Corrupted Labels (2017. 12)
	- **`MentorNet - StudentNet`**, **`Curriculum Learning`**, **`Output is Weight`**
	- [arXiv](https://arxiv.org/abs/1712.05055)
- Deep Learning Scaling is Predictable, Empirically (2017. 12)
	- **`Power-Law Exponents`**, **`Grow Training Sets`**
	- [arXiv](https://arxiv.org/abs/1712.00409), [the_morning_paper](https://blog.acolyer.org/2018/03/28/deep-learning-scaling-is-predictable-empirically/)
- Sensitivity and Generalization in Neural Networks: an Empirical Study (2018. 2)
	- **`Robustness`**, **`Data Perturbations`**, **`Survey`**
	- [arXiv](https://arxiv.org/abs/1802.08760), [open_review](https://openreview.net/forum?id=HJC2SzZCW)
- Can recurrent neural networks warp time? (2018. 2)
	- **`RNN`**, **`Learnable Gate`**, **`Chrono Initialization`**
	- [open_review](https://openreview.net/forum?id=SJcKhk-Ab)
- Spectral Normalization for Generative Adversarial Networks (2018. 2)
	- **`GAN`**, **`Training Discriminator`**, **`Constrain Lipschitz`**, **`Power Method`**
	- [open_review](https://openreview.net/forum?id=B1QRgziT-&noteId=BkxnM1TrM)
- On the importance of single directions for generalization (2018. 3)
	- **`Importance`**, **`Confusiing Neurons`**, **`Selective Neuron`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1803.06959), [deepmind_blog](https://deepmind.com/blog/understanding-deep-learning-through-neuron-deletion/)
- Group Normalization (2018. 3)
	- **`Group Normalization (GN)`**, **`Batch (BN)`**, **`Layer (LN)`**, **`Instance (IN)`**, **`Independent Batch Size`**
	- [arXiv](https://arxiv.org/abs/1803.08494)
- Fast Decoding in Sequence Models using Discrete Latent Variables (2018. 3)
	- **`Autoregressive`**, **`Latent Transformer`**, **`Discretization`**
	- [arXiv](https://arxiv.org/abs/1803.03382)
- How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift) (2018. 5)
	- **`Smoothing Effect`**, **`BatchNorm’s Reparametrization`**
	- [arXiv](https://arxiv.org/abs/1805.11604)
- When Recurrent Models Don't Need To Be Recurrent (2018. 5)
	- **`Approximate`**, **`Feed-Forward`**
	- [arXiv](https://arxiv.org/abs/1805.10369), [bair_blog](http://bair.berkeley.edu/blog/2018/08/06/recurrent/)
- Relational inductive biases, deep learning, and graph networks (2018, 6)
	- **`Survey`**, **`Relation`**, **`Graph`**
	- [arXiv](https://arxiv.org/abs/1806.01261)
- Universal Transformers (2018. 7)
	- **`Transformer`**, **`Weight Sharing`**, **`Adaptive Computation Time (ACT)`**
	- [arXiv](https://arxiv.org/abs/1807.03819), [google_ai_blog](https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)


### Natural Language Processing

- **Distributed Representations of Words and Phrases and their Compositionality** (2013. 10)
	- **`Word2Vec`**, **`CBOW`**, **`Skip-gram`**
	- [arXiv](https://arxiv.org/abs/1310.4546)
- GloVe: Global Vectors for Word Representation (2014)
	- **`Word2Vec`**, **`GloVe`**, **`Co-Occurrence`**
	- [paper](https://nlp.stanford.edu/pubs/glove.pdf)
- **Convolutional Neural Networks for Sentence Classification** (2014. 8)
	- **`CNN`**, **`Classfication`**
	- [arXiv](https://arxiv.org/abs/1408.5882), [code](https://github.com/DongjunLee/text-cnn-tensorflow)
- **Neural Machine Translation by Jointly Learning to Align and Translate** (2014. 9)
	- **`Seq2Seq`**, **`Attention(Align)`**, **`Translation`**
	- [arXiv](https://arxiv.org/abs/1409.0473), [note](notes/bahdanau_attention.md), [code](https://github.com/DongjunLee/conversation-tensorflow)
- Text Understanding from Scratch (2015. 2)
	- **`CNN`**, **`Character-level`**
	- [arXiv](https://arxiv.org/abs/1506.07285)
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing (2015. 6)
	- **`Memory`**, **`QA`**, **`bAbi`**
	- [arXiv](https://arxiv.org/abs/1506.07285), [code](https://github.com/DongjunLee/dmn-tensorflow)
- Pointer Networks (2015. 6)
	- **`Seq2Seq`**, **`Attention`**, **`Combinatorial`**
	- [arXiv](https://arxiv.org/abs/1506.03134), [note](notes/pointer_network.md)
- **Skip-Thought Vectors** (2015. 6)
	- **`Sentence2Vec`**, **`Unsupervised`**
	- [arXiv](https://arxiv.org/abs/1506.06726), [note](notes/skip_thought.md)
- A Neural Conversational Model (2015. 6)
	- **`Seq2Seq`**, **`Conversation`**
	- [arXiv](https://arxiv.org/abs/1506.05869)
- Teaching Machines to Read and Comprehend (2015. 6)
	- **`Deepmind`**, **`Attention`**, **`QA`**
	- [arXiv](https://arxiv.org/abs/1506.03340), [note](notes/teaching_machine_read_and_comprehend.md)
- Effective Approaches to Attention-based Neural Machine Translation (2015. 8)
	- **`Seq2Seq`**, **`Attention`**, **`Translation`**
	- [arXiv](https://arxiv.org/abs/1508.04025), [note](notes/luong_attention.md), [code](https://github.com/DongjunLee/conversation-tensorflow)
- Character-Aware Neural Language Models (2015. 8)
	- **`CNN`**, **`Character-level`**
	- [arXiv](https://arxiv.org/abs/1508.06615)
- Neural Machine Translation of Rare Words with Subword Units (2015. 8)
	- **`Out-Of-Vocabulary`**, **`Translation`**
	- [arXiv](https://arxiv.org/abs/1508.07909), [note](notes/subword_nmt.md)
- A Diversity-Promoting Objective Function for Neural Conversation Models (2015. 10)
	- **`Conversation`**, **`Objective`**
	- [arXiv](https://arxiv.org/abs/1510.03055), [note](notes/diversity_conversation.md)
- **Multi-task Sequence to Sequence Learning** (2015. 11)
	- **`Multi-Task`**, **`Seq2Seq`**
	- [arXiv](https://arxiv.org/abs/1511.06114), [note](notes/multi_task_seq2seq.md)
- Multilingual Language Processing From Bytes (2015. 12)
	- **`Byte-to-Span`**, **`Multilingual`**, **`Seq2Seq`**
	- [arXiv](https://arxiv.org/abs/1512.00103), [note](notes/byte_to_span.md)
- Strategies for Training Large Vocabulary Neural Language Models (2015. 12)
	- **`Vocabulary`**, **`Softmax`**, **`NCE`**, **`Self Normalization`**
	- [arXiv](https://arxiv.org/abs/1512.04906), [note](notes/vocabulary_strategy.md)
- Incorporating Structural Alignment Biases into an Attentional Neural Translation Model (2016. 1)
	- **`Seq2Seq`**, **`Attention with Structural Biases`**, **`Translation`**
	- [arXiv](https://arxiv.org/abs/1601.01085)
- Long Short-Term Memory-Networks for Machine Reading (2016. 1)
	- **`LSTMN`**, **`Intra-Attention`**, **`RNN`**
	- [arXiv](https://arxiv.org/abs/1601.06733)
- Recurrent Memory Networks for Language Modeling (2016. 1)
	- **`RMN`**, **`Memory Bank`**
	- [arXiv](https://arxiv.org/abs/1601.01272)
- Exploring the Limits of Language Modeling (2016. 2)
	- **`Google Brain`**, **`Language Modeling`**
	- [arXiv](https://arxiv.org/abs/1602.02410), [note](notes/exploring_limits_of_lm.md)
- Swivel: Improving Embeddings by Noticing What's Missing (2016. 2)
	- **`Word2Vec`**, **`Swivel `**, **`Co-Occurrence`**
	- [arXiv](https://arxiv.org/abs/1602.02215)
- Incorporating Copying Mechanism in Sequence-to-Sequence Learning (2016. 3)
	- **`CopyNet`**, **`Seq2Seq`**
	- [arXiv](https://arxiv.org/abs/1603.06393), [note](notes/copynet.md)
- Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models (2016. 4)
	- **`Translation`**, **`Hybrid NMT`**, **`Word-Char`**
	- [arXiv](https://arxiv.org/abs/1604.00788), [note](notes/nmt_hybrid_word_and_char.md)
- Adversarial Training Methods for Semi-Supervised Text Classification (2016. 5)
	- **`Regulaizer`**, **`Adversarial`**, **`Virtual Adversarial Training (Semi-Supervised)`**
	- [arXiv](https://arxiv.org/abs/1605.07725), [note](notes/adversarial_for_semi_sv_tc.md)
- SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016. 6)
	- **`DataSet`**, **`Reading Comprehension`**
	- [arXiv](https://arxiv.org/abs/1606.05250), [note](notes/squad.md), [dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- Sequence-Level Knowledge Distillation (2016. 6)
	- **`Distil`**, **`Teacher-Student`**
	- [arXiv](https://arxiv.org/abs/1606.07947), [note](notes/sequence_knowledge_distillation.md)
- Attention-over-Attention Neural Networks for Reading Comprehension (2016. 7)
	- **`Attention`**, **`Cloze-style`**, **`Reading Comprehension`**
	- [arXiv](https://arxiv.org/abs/1607.04423), [note](notes/attn_over_attn_nn_rc.md)
- Recurrent Neural Machine Translation (2016. 7)
	- **`Translation`**, **`Attention (RNN)`**
	- [arXiv](https://arxiv.org/abs/1607.08725)
- An Actor-Critic Algorithm for Sequence Prediction (2016. 7)
	- **`Seq2Seq`**, **`Actor-Critic`**, **`Objective`**
	- [arXiv](https://arxiv.org/abs/1607.07086), [note](notes/actor_critic_for_seq.md)
- Pointer Sentinel Mixture Models (2016. 9)
	- **`Language Modeling`**, **`Rare Word`**, **`Salesforce`**
	- [arXiv](https://arxiv.org/abs/1609.07843), [note](notes/ps-lstm.md)
- Multiplicative LSTM for sequence modelling (2016. 10)
	- **`mLSTM`**, **`Language Modeling`**,  **`Character-Level`**
	- [arXiv](https://arxiv.org/abs/1609.07959)
- Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (2016. 10)
	- **`Diverse`**, **`DBS`**
	- [arXiv](https://arxiv.org/abs/1610.02424), [note](notes/dbs.md)
- Fully Character-Level Neural Machine Translation without Explicit Segmentation (2016. 10)
	- **`Translation`**, **`CNN`**, **`Character-Level`**
	- [arXiv](https://arxiv.org/abs/1610.03017), [note](notes/fully_conv_nmt.md)
- **Neural Machine Translation in Linear Time** (2016. 10)
	- **`ByteNet`**, **`WaveNet + PixelCNN`**, **`Translation`**, **`Character-Level`**
	- [arXiv](https://arxiv.org/abs/1610.10099), [note](notes/bytenet.md)
- Bidirectional Attention Flow for Machine Comprehension (2016. 11)
	- **`QA`**, **`BIDAF`**, **`Machine Comprehension`**
	- [arXiv](https://arxiv.org/abs/1611.01603), [note](notes/bi_att_flow.md), [code](https://github.com/DongjunLee/bi-att-flow-tensorflow)
- Dynamic Coattention Networks For Question Answering (2016. 11)
	- **`QA`**, **`DCN`**, **`Coattention Encoder`**, **`Machine Comprehension`**
	- [arXiv](https://arxiv.org/abs/1611.01604)
- Dual Learning for Machine Translation (2016. 11)
	- **`Translation`**, **`RL`**, **`Dual Learning (Two-agent)`**
	- [arXiv](https://arxiv.org/abs/1611.00179), [note](notes/dual_learning_nmt.md)
- Neural Machine Translation with Reconstruction (2016. 11)
	- **`Translation`**, **`Auto-Encoder`**, **`Reconstruction`**
	- [arXiv](https://arxiv.org/abs/1611.01874), [note](notes/nmt_with_reconstruction.md)
- Quasi-Recurrent Neural Networks (2016. 11)
	- **`QRNN`**, **`Parallelism`**, **`Conv + Pool + RNN`**
	- [arXiv](https://arxiv.org/abs/1611.01576), [note](notes/dual_learning_nmt.md)
- A recurrent neural network without chaos (2016. 12)
	- **`RNN`**, **`CFN`**, **`Dynamic`**, **`Chaos`**
	- [arXiv](https://arxiv.org/abs/1612.06212)
- A Structured Self-attentive Sentence Embedding (2017. 3)
	- **`Sentence Embedding`**, **`Self-Attention`**, **`2-D Matrix`**
	- [arXiv](https://arxiv.org/abs/1703.03130), [note](notes/self_attn_sentence_embed.md)
- Dynamic Word Embeddings for Evolving Semantic Discovery (2017. 3)
	- **`Word Embedding`**, **`Temporal`**, **`Alignment`**
	- [arXiv](https://arxiv.org/abs/1703.00607), [the morning paper](https://blog.acolyer.org/2018/02/22/dynamic-word-embeddings-for-evolving-semantic-discovery/)
- Learning to Generate Reviews and Discovering Sentiment (2017. 4)
	- **`Sentiment`**, **`Unsupervised `**, **`OpenAI`**
	- [arXiv](https://arxiv.org/abs/1706.03762)
- Ask the Right Questions: Active Question Reformulation with Reinforcement Learning (2017. 5)
	- **`QA`**, **`Active Question Answering`**, **`RL`**, **`Agent (Reformulate, Aggregate)`**
	- [arXiv](https://arxiv.org/abs/1705.07830), [open_review](https://openreview.net/forum?id=S1CChZ-CZ)
- Reinforced Mnemonic Reader for Machine Reading Comprehension (2017. 5)
	- **`QA`**, **`Mnemonic (Syntatic, Lexical)`**, **`RL`**, **`Machine Comprehension`**
	- [arXiv](https://arxiv.org/abs/1705.02798)
- **Attention Is All You Need** (2017. 6)
	- **`Self-Attention`**, **`Seq2Seq (without RNN, CNN)`**
	- [arXiv](https://arxiv.org/abs/1706.03762), [note](notes/transformer.md), [code](https://github.com/DongjunLee/transformer-tensorflow)  
- Depthwise Separable Convolutions for Neural Machine Translation (2017. 6)
	- **`SliceNet`**, **`Super-Separable Conv`**, **`Depsewise + Conv 1x1`**
	- [arXiv](https://arxiv.org/abs/1706.03059), [open_review](https://openreview.net/forum?id=S1jBcueAb), [note](notes/slicenet.md)
- MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension (2017. 7)
	- **`MEMEN`**, **`QA(MC)`**, **`Embedding(skip-gram)`**, **`Full-Orientation Matching`**
	- [arXiv](https://arxiv.org/abs/1707.09098)
- On the State of the Art of Evaluation in Neural Language Models (2017. 7)
	- **`Standard LSTM`**, **`Regularisation`**, **`Hyperparemeter`**
	- [arXiv](https://arxiv.org/abs/1707.05589)
- Text Summarization Techniques: A Brief Survey (2017. 7)
	- **`Summarization`**, **`Survey`**
	- [arXiv](https://arxiv.org/abs/1707.02268), [note](notes/text_sum_survey.md)
- Adversarial Examples for Evaluating Reading Comprehension Systems (2017. 7)
	- **`Concatenative Adversaries(AddSent, AddOneSent)`**, **`SQuAD`**
	- [arXiv](https://arxiv.org/abs/1707.07328)
- Learned in Translation: Contextualized Word Vectors (2017. 8)
	- **`Word Embedding`**, **`CoVe`**, **`Context Vector`**
	- [arXiv](https://arxiv.org/abs/1708.00107)
- Simple and Effective Multi-Paragraph Reading Comprehension (2017. 10)
	- **`Document-QA`**, **`Select Paragraph-Level`**, **`Confidence Based`**, **`AllenAI`**
	- [arXiv](https://arxiv.org/abs/1710.10723), [note](notes/multi_paragraph_rc.md)
- Unsupervised Neural Machine Translation (2017. 10)
	- **`Train with both direction (tandem)`**, **`Shared Encoder`**, **`Denoising Auto-Encoder`**
	- [arXiv](https://arxiv.org/abs/1710.11041), [open_review](https://openreview.net/forum?id=Sy2ogebAW)
- Word Translation Without Parallel Data (2017. 10)
	- **`Unsupervised`**, **`Multilingual Embedding`**, **`Parallel Dictionary Induction`**
	- [arXiv](https://arxiv.org/abs/1710.04087), [open_review](https://openreview.net/forum?id=H196sainb)
- Unsupervised Machine Translation Using Monolingual Corpora Only (2017. 11)
	- **`Unsupervised`**, **`Adversarial`**, **`Monolingual Corpora`**
	- [arXiv](https://arxiv.org/abs/1711.00043), [open_review](https://openreview.net/forum?id=rkYTTf-AZ)
- Neural Text Generation: A Practical Guide (2017. 11)
	- **`Seq2Seq`**, **`Decoder Guide`**
	- [arXiv](https://arxiv.org/abs/1711.09534), [note](notes/neural_text_generation.md)
- Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (2017. 11)
	- **`MoS (Mixture of Softmaxes)`**, **`Softmax Bottleneck`**
	- [arXiv](https://arxiv.org/abs/1711.03953)
- Neural Speed Reading via Skim-RNN (2017. 11)
	- **`Skim-RNN`**, **`Speed Reading`**, **`Big(Read)-Small(Skim)`**, **`Dynamic`**
	- [arXiv](https://arxiv.org/abs/1711.09534), [open_review](https://openreview.net/forum?id=Sy-dQG-Rb)
- Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks (2017. 11)
	- **`SCAN`**, **`Compositional`**, **`Mix-and-Match`**
	- [arXiv](https://arxiv.org/abs/1711.00350)
- The NarrativeQA Reading Comprehension Challenge (2017. 12)
	- **`NarrativeQA`**, **`Dataset`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1712.07040), [dataset](https://github.com/deepmind/narrativeqa)
- Hierarchical Text Generation and Planning for Strategic Dialogue (2017. 12)
	- **`End2End Strategic Dialogue`**, **`Latent Sentence Representations`**, **`Planning + RL`**
	- [arXiv](https://arxiv.org/abs/1712.05846)
- Recent Advances in Recurrent Neural Networks (2018. 1)
	- **`RNN`**, **`Recent Advances`**, **`Review`**
	- [arXiv](https://arxiv.org/abs/1801.01078)
- Personalizing Dialogue Agents: I have a dog, do you have pets too? (2018. 1)
	- **`Chit-chat`**, **`Profile Memory`**, **`Persona-Chat Dataset`**, **`ParlAI`**
	- [arXiv](https://arxiv.org/abs/1801.07243)
- Generating Wikipedia by Summarizing Long Sequences (2018. 1)
	- **`Multi-Document Summarization`**, **`Extractive-Abstractive Stage`**, **`T-DMCA`**, **`WikiSum`**, **`Google Brain`**
	- [arXiv](https://arxiv.org/abs/1801.10198), [note](notes/generate_wiki.md), [open_review](https://openreview.net/forum?id=Hyg0vbWC-)
- MaskGAN: Better Text Generation via Filling in the______ (2018. 1)
	- **`MaskGAN`**, **`Neural Text Generation`**, **`RL Approach`**
	- [arXiv](https://arxiv.org/abs/1801.07736), [open_review](https://openreview.net/forum?id=ByOExmWAb&noteId=HJbx71pBM), [note](notes/mask_gan.md)
- Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs (2018. 1)
	- **`Contextual Decomposition (CD)`**, **`Disambiguate interactions between Gates`**
	- [arXiv](https://arxiv.org/abs/1801.05453), [open_review](https://openreview.net/forum?id=rkRwGg-0Z)
- DeepType: Multilingual Entity Linking by Neural Type System Evolution (2018. 2)
	- **`DeepType`**, **`Symbolic Information`**, **`Type System`**, **`Open AI`**
	- [arXiv](https://arxiv.org/abs/1802.01021), [openai blog](https://blog.openai.com/discovering-types-for-entity-disambiguation/)
- Deep contextualized word representations (2018. 2)
	- **`biLM`**, **`ELMo`**, **`Word Embedding`**, **`Contextualized`**, **`AllenAI`**
	- [arXiv](https://arxiv.org/abs/1802.05365), [note](notes/contextualized_word_for_rc.md)
- Ranking Sentences for Extractive Summarization with Reinforcement Learning (2018. 2)
	- **`Document-Summarization`**, **`Cross-Entropy vs RL`**, **`Extractive`**
	- [arXiv](https://arxiv.org/abs/1802.08636)
- code2vec: Learning Distributed Representations of Code (2018. 3)
	- **`code2vec`**, **`Code Embedding`**, **`Predicting method name`**
	- [arXiv](https://arxiv.org/abs/1803.09473)
- Universal Sentence Encoder (2018. 3)
	- **`Transformer`**, **`Deep Averaging Network (DAN)`**, **`Transfer`**
	- [arXiv](https://arxiv.org/abs/1803.11175)
- An efficient framework for learning sentence representations (2018. 3)
	- **`Sentence Representation`**, **`True Context`**, **`Unsupervised`**
	- [arXiv](https://arxiv.org/abs/1803.02893), [open_review](https://openreview.net/forum?id=rJvJXZb0W)
- An Analysis of Neural Language Modeling at Multiple Scales (2018. 3)
	- **`LSTM vs QRNN`**, **`Hyperparemeter`**, **`AWD-QRNN`**
	- [arXiv](https://arxiv.org/abs/1803.08240)
- Analyzing Uncertainty in Neural Machine Translation (2018. 3)
	- **`Uncertainty`**, **`Beam Search Degradation`**, **`Copy Mode`**
	- [arXiv](https://arxiv.org/abs/1803.00047)
- Training Tips for the Transformer Model (2018. 4)
	- **`Transformer`**, **`Hyperparameter`**, **`Multiple GPU`**
	- [arXiv](https://arxiv.org/abs/1804.00247)
- QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension (2018. 4)
	- **`QA`**, **`Conv - Self-Attention`**, **`Backtranslation (Data Augmentation)`**
	- [arXiv](https://arxiv.org/abs/1804.09541), [open_review](https://openreview.net/forum?id=B14TlG-RW), [note](notes/qanet.md)
-  SimpleQuestions Nearly Solved: A New Upperbound and Baseline Approach (2018. 4)
   - **`Top-K Subject Recognitio`**, **`Relation Classification`**
	- [arXiv](https://arxiv.org/abs/1804.08798)
-  Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer (2018. 4)
   - **`Sentiment Transfer`**, **`Disentangle Attribute`**, **`Unsupervised`** 
	- [arXiv](https://arxiv.org/abs/1804.06437)
- Parsing Tweets into Universal Dependencies (2018. 4)
	- **`Universal Dependencies (UD)`**, **`TWEEBANK v2`**
	- [arXiv](https://arxiv.org/abs/1804.08228)
- Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (2018. 4)
	- **`SR`**, **`Subword Sampling + Hyperparameter`**, **`Segmentation (BPE, Unigram)`**
	- [arXiv](https://arxiv.org/abs/1804.10959) 
- On the Practical Computational Power of Finite Precision RNNs for Language Recognition (2018. 5)
	- **`Unbounded counting`**, **`IBFP-LSTM`**
	- [arXiv](https://arxiv.org/abs/1805.04908)
- Paper Abstract Writing through Editing Mechanism (2018. 5)
	- **`Writing-editing Network`**, **`Attentive Revision Gate`**
	- [arXiv](https://arxiv.org/abs/1805.06064)
- A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings (2018. 5)
	- **`Unsupervised initialization scheme`**, **`Robust self-leraning`**
	- [arXiv](https://arxiv.org/abs/1805.06297)
- Efficient and Robust Question Answering from Minimal Context over Documents (2018. 5)
	- **`Sentence Selector`**, **`Oracle Sentence`**, **`Minimal Set of Sentences (SpeedUp)`**
	- [arXiv](https://arxiv.org/abs/1805.08092), [note](notes/minimal_qa.md)
- Global-Locally Self-Attentive Dialogue State Tracker (2018. 5)
	- **`GLAD`**, **`WoZ and DSTC2 Dataset`**
	- [arXiv](https://arxiv.org/abs/1805.09655) 
- Learning to Ask Good Questions: Ranking Clarification Questions using Neural Expected Value of Perfect Information (2018, 5)
	- **`Dataset`**, **`EVPI`**, **`ACL 2018 Best Paper`**
	- [arXiv](https://arxiv.org/abs/1805.04655)
- Know What You Don't Know: Unanswerable Questions for SQuAD (2018, 6)
	- **`SQuAD 2.0`**, **`Negative Example`**, **`ACL 2018 Best Paper`**
	- [arXiv](https://arxiv.org/abs/1806.03822)
- The Natural Language Decathlon: Multitask Learning as Question Answering (2018, 6)
	- **`decaNLP`**, **`Multitask Question Answering Network (MQAN)`**, **`Transfer Learning`**
	- [arXiv](https://arxiv.org/abs/1806.08730)
- GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations (2018, 6)
	- **`Transfer Learning Framework`**, **`Structured Graphical Representations`**
	- [arXiv](https://arxiv.org/abs/1806.08730)
- Improving Language Understanding by Generative Pre-Training (2018, 6)
	- **`Transformer`**, **`Generative Pre-Training`**, **`Discriminative Fine-Tuning`**
	- [paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [open\_ai\_blog](https://blog.openai.com/language-unsupervised/)
- Finding Syntax in Human Encephalography with Beam Search (2018, 6)
	- **`RNNG+beam search`**, **`ACL 2018 Best Paper`**
	- [arXiv](https://arxiv.org/abs/1806.04127)
- Let's do it "again": A First Computational Approach to Detecting Adverbial Presupposition Triggers (2018, 6)
	- **`Task`**, **`Dataset`**, **`Weighted-Pooling (WP)`** **`ACL 2018 Best Paper`**
	- [arXiv](https://arxiv.org/abs/1806.04262)

	
### One-Shot/Few-Shot/Meta Learing

- Matching Networks for One Shot Learning (2016. 6)
	- **`Matching Nets`**, **`Non-Parametric`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1606.04080), [the morning paper](https://blog.acolyer.org/2017/01/03/matching-networks-for-one-shot-learning/)
- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (2017. 3)
	- **`MAML`**, **`Meta-Learning`**, **`Few-Shot`**, **`BAIR`**
	- [arXiv](https://arxiv.org/abs/1703.03400), [bair_blog](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
- SMASH: One-Shot Model Architecture Search through HyperNetworks (2017. 8)
	- **`SMASH`**, **`HyperNet`**, **`Prior Knowledge`**
	- [arXiv](https://arxiv.org/abs/1211.5063), [open_review](https://openreview.net/forum?id=rydeCEhs-)
- Reptile: a Scalable Metalearning Algorithm (2018. 3)
	- **`Reptile`**, **`Meta-Learning`**, **`Few-Shot`**, **`OpenAI`**
	- [arXiv](https://arxiv.org/abs/1803.02999), [openai_blog](https://blog.openai.com/reptile/)


### Optimization

- Understanding the difficulty of training deep feedforward neural networks (2010)
	- **`Weight Initialization (Xavier)`**
	- [paper](http://proceedings.mlr.press/v9/glorot10a.html), [note](notes/xavier_initialization.md)
- On the difficulty of training Recurrent Neural Networks (2012. 11)
	- **`Gradient Clipping`**, **`RNN`**
	- [arXiv](https://arxiv.org/abs/1211.5063)
- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015. 2)
	- **`PReLU`**, **`Weight Initialization (He)`**
	- [arXiv](https://arxiv.org/abs/1502.01852), [note](notes/he_initialization.md)
- A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015. 4)
	- **`Weight Initialization`**, **`RNN`**, **`Identity Matrix`**
	- [arXiv](https://arxiv.org/abs/1504.00941)
- Cyclical Learning Rates for Training Neural Networks (2015. 6)
	- **`CLR`**, **`Triangular, ExpRange`**, **`Longtherm Benefit`**
	- [arXiv](https://arxiv.org/abs/1506.01186)
- On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (2016. 9)
	- **`Generalization`**, **`Sharpness of Minima`**
	- [arXiv](https://arxiv.org/abs/1609.04836)
- Neural Optimizer Search with Reinforcement Learning (2017. 9)
	- **`Neural Optimizer Search (NOS)`**, **`PowerSign`**, **`AddSign`**
	- [arXiv](https://arxiv.org/abs/1709.07417)
- On the Convergence of Adam and Beyond (2018. 2)
	- **`AMSGrad`**, **`Convex optimization`**
	- [open_review](https://openreview.net/forum?id=ryQu7f-RZ)
- Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (2018. 4)
	- **`Adafactor`**, **`Adaptive Method`**, **`Update Clipping`**
	- [arXiv](https://arxiv.org/abs/1804.04235)
- Revisiting Small Batch Training for Deep Neural Networks (2018. 4)
	- **`Generalization Performance`**, **`Training Stability`**
	- [arXiv](https://arxiv.org/abs/1804.07612)


### Reinforcement Learning

- Neural Architecture Search with Reinforcement Learning (2016. 11)
	- **`NAS`**, **`Google AutoML`**, **`Google Brain`**
	- [arXiv](https://arxiv.org/abs/1611.01578)
- Third-Person Imitation Learning (2017. 3)
	- **`Imitation Learning`**, **`Unsupervised (Third-Person)`**, **`GAN + Domain Confusion`**
	- [arXiv](https://arxiv.org/abs/1703.01703)
- Noisy Networks for Exploration (2017. 6)
	- **`NoisyNet`**, **`Exploration`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1703.1706.10295), [note](notes/noisy_network_exploration.md)
- Efficient Neural Architecture Search via Parameter Sharing (2018. 2)
	- **`ENAS`**, **`Google AutoML`**, **`Google Brain`**
	- [arXiv](https://arxiv.org/abs/1802.03268)
- Learning by Playing - Solving Sparse Reward Tasks from Scratch (2018. 2)
	- **`Scratch with minimal prior knowledge`**, **`Scheduled Auxiliary Control (SAC-X)`**, **`DeepMind`**
	- [arXiv](https://arxiv.org/abs/1802.10567), [deep_mind](https://deepmind.com/blog/learning-playing/)
- Investigating Human Priors for Playing Video Games (2018. 2)
	- **`prior knowledge`**, **`key factor`**
	- [open_review](https://openreview.net/forum?id=Hk91SGWR-)
- World Models (2018. 3)
	- **`Generative + RL`**, **`VAE (V)`**, **`MDN-RNN (M)`**, **`Controller (C)`**
	- [arXiv](https://arxiv.org/abs/1803.10122)
- Unsupervised Predictive Memory in a Goal-Directed Agent (2018. 3)
	- **`MERLIN`**, **`Memory + RL + Inference`**, **`Partial Observability`**
	- [arXiv](https://arxiv.org/abs/1803.10760), [google_ai_blog](https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)


### Transfer Learning

- Progressive Neural Networks (2016. 6)
	- **`ProgNN `**, **`Incorporate Prior Knowledge`**
	- [arXiv](https://arxiv.org/abs/1606.04671), [the morning paper](https://blog.acolyer.org/2016/10/11/progressive-neural-networks/)


### Unsupervised & Generative

- **Auto-Encoding Variational Bayes** (2013. 12)
	- **`VAE`**, **`Variational`**, **`Approximate`**
	- [arXiv](https://arxiv.org/abs/1312.6114), [note](notes/vae.md), [code](https://github.com/DongjunLee/vae-tensorflow)
- **Generative Adversarial Networks** (2014. 6)
	- **`GAN`**, **`Adversarial`**, **`Minimax`**
	- [arXiv](https://arxiv.org/abs/1406.2661), [note](notes/gan.md), [code](https://github.com/DongjunLee/gan-tensorflow)
- Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data (2016. 5)
	- **`DVBF`**, **`Variational Inference`**, **`SVGB`**
	- [arXiv](https://arxiv.org/abs/1605.06432), [open_review](https://openreview.net/forum?id=HyTqHL5xg)
- SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient (2016. 9)
	- **`Seq2Seq`**, **`GAN`**, **`RL`**
	- [arXiv](https://arxiv.org/abs/1609.05473), [note](notes/seq_gan.md)
- Structured Inference Networks for Nonlinear State Space Models (2016. 9)
	- **`Structured Variational Approximation`**, **`SVGB`**
	- [arXiv](https://arxiv.org/abs/1609.09869)
- beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework (2016. 11)
	- **`Beta-VAE`**, **`Disentangled `**
	- [open_review](https://openreview.net/forum?id=Sy2fzU9gl)
- A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning (2017. 10)
	- **`Kalman VAE`**, **`LGSSM`**
	- [arXiv](https://arxiv.org/abs/1710.05741)
- Self-Attention Generative Adversarial Networks (2018. 5)
	- **`SAGAN`**, **`Attention-Driven`**, **`Spectral Normalization`**
	- [arXiv](https://arxiv.org/abs/1805.08318) 