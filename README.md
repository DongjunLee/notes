
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
- [Optimization](#optimization)
- [Unsupervised & Generative](#unsupervised--generative)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Speech](#speech)
- [Reinforcement](reinforcement)

---

## Background knowledge

- [Gaussian Process](notes/gausian_process.md) ****`Supervised`****, ****`Regression`****
- [Importance Sampling](notes/importance_sampling.md) ****`Approximate`****

---

- Deep Learning (2015) ****`Review`****
	- [nature](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) | [note](notes/deep_learning.md)

## Unsupervised & Generative

- Auto-Encoding Variational Bayes (2013, 12) ****`Generative`****, ****`Approximate`****
	 - [arXiv](https://arxiv.org/abs/1312.6114) | [note](notes/vae.md)


## Optimization

- Dropout (2012, 2014) ****`Regulaizer`****, ****`Ensemble`****
	- [arXiv (2012)](https://arxiv.org/abs/1207.0580) | [arXiv (2014)](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) | [note](notes/dropout.md)
- Batch Normalization (2015) ****`Regulaizer`****, ****`Training`****
	- [arXiv](https://arxiv.org/abs/1502.03167) | [note](notes/batch_normalization.md)
- Layer Normalization (2016) ****`Regulaizer`****, ****`Training`****
	- [arXiv](https://arxiv.org/abs/1607.06450) | [note](notes/layer_normalization.md)


## Natural Language Processing

- Convolutional Neural Networks for Sentence Classification (2014. 8) ****`CNN`****
	- [arXiv](https://arxiv.org/abs/1408.5882) | [code](https://github.com/DongjunLee/text-cnn-tensorflow) 
- Neural Machine Translation by Jointly Learning to Align and Translate (2014. 9) ****`Seq2Seq`****, ****`Attention`****, ****`Translation`****
	- [arXiv](https://arxiv.org/abs/1409.0473) | [note](notes/bahdanau_attention.md) | [code](https://github.com/DongjunLee/conversation-tensorflow) 
- Text Understanding from Scratch (2015. 2) ****`CNN`****
	- [arXiv](https://arxiv.org/abs/1506.07285)
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing (2015. 6) ****`Memory`****, ****`QA`****
	- [arXiv](https://arxiv.org/abs/1506.07285) | [code](https://github.com/DongjunLee/dmn-tensorflow) 
- Pointer Networks (2015. 6) ****`Seq2Seq`****, ****`Attention`****
	- [arXiv](https://arxiv.org/abs/1506.03134) | [note](notes/pointer_network.md) 
- Skip-Thought Vectors (2015. 6) ****`Sent2Vec`****
	- [arXiv](https://arxiv.org/abs/1506.06726) | [note](notes/skip_thought.md) 
- Attention Is All You Need (2017. 6) ****`Attention`****
	- [arXiv](https://arxiv.org/abs/1706.03762) | [note](notes/transformer.md) | [code](https://github.com/DongjunLee/transformer-tensorflow)  
- Neural Text Generation: A Practical Guide (2017. 11) ****`Seq2Seq`****, ****`Guide`****
	- [arXiv](https://arxiv.org/abs/1711.09534) | [note](notes/neural_text_generation.md)