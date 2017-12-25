# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

- published in 2015
- Ioffe, Sergey, and Christian Szegedy

---

## Simple Summary

> address the problem (_internal covariate shift_) by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training _mini-batch_. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout.


- Using mini-batches
	1. the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases.
	2. computation over a batch can be much more efficient than _m_ computations for individual examples, due to the parallelism afforded by the morden computing platforms.

##### internal Covariate Shift

- The change in the distribution of layers' inputs presents a problem because the layers need to continuously adapt to the new distribution. (_covariate shift_)

- The input distribution properties that makes training more efficient apply to trainig the sub-network as well. ... ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate. 

- it cause gradient vanishing/exploding problems.

##### Towards Reducing Internal Covariate Shift

- whitening : linearly transformed to have zero means and unit variances, and decorrelated.
	- issue is the gradient descenet optimization does not take into account the fact that the normalization takes place.

- for any parameters values, the network _always_ produces activations with the desired distribution. Doing so would allow the gradient of the loss with respect to the model parameters to account for the normalization, and for its dependence on the model parameters.

- each mini-batch produces estimates of the mean and variance of each activation.

##### Algorithms

- Batch Normalizing Transform, applied to activation _x_ over a mini-batch.

![image](../images/batch_normalization1.png)

- Training a Batch-Normalized Network

![image](../images/batch_normalization2.png)


##### Experiments


![image](../images/batch_normalization3.png)

![image](../images/batch_normalization4.png)