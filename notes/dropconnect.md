# Regularization of Neural Networks using DropConnect

- Submitted on 2013
- Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun and Rob Fergus

## Simple Summary

> DropConnect sets a randomly selected subset of weights within the network to zero. Each unit thus receives input from a random subset of units in the previous layer. ... We derive a bound on the generalization performance of both Dropout and DropConnect.

![images](../images/dropconnect_1.png)

- **DropConnect** is similar to Dropout as it introduces dynamic sparsity within the model, but differs in that the sparsity is on the weights W, rather than the output vectors of a layer.
- `r = a ((M * W) v)`

![images](../images/dropconnect_2.png)