# SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient

- Submitted on 2016. 7
- Lantao Yu, Weinan Zhang, Jun Wang and Yong Yu

## Simple Summary

> propose a sequence generation framework, called SeqGAN. ... Modeling the data generator as a stochastic policy in reinforcement learning (RL), SeqGAN bypasses the generator differentiation problem by directly performing gradient policy update. The RL reward signal comes from the GAN discriminator judged on a complete sequence, and is passed back to the intermediate state-action steps using Monte Carlo search. Extensive experiments on synthetic data and real-world tasks demonstrate significant improvements over strong baselines.

- GAN has limitations when the goal is for generating sequences of discrete tokens. A major reason lies in that the discrete outputs from the generative model make it difficult to pass the gradient update from the discriminative model to the generative model.