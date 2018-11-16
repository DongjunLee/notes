# Pointer Networks

- published in 2015. 9
- Oriol Vinyals, Meire Fortunato and Navdeep Jaitly

## Simple Summary

- Introduce a new neural architecture to learn the conditional probability of an output sequence with elements that are discrete tokens corresponding to positions in an input sequence
- Problems such as sorting variable sized sequences, and various combinatorial optimization problems belong to this class. (eg. finding planar convex hulls, computing Delaunay triangulations, and the planar Travelling Salesman Problem)

![images](../images/pointer_network_1.png)

![images](../images/pointer_network_2.png) 

- Softmax normalizes the vector e_ij to be an output distribution over the dictionary of inputs.