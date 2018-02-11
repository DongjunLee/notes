# Recurrent Highway Networks

- Submitted on 2016. 7
- Julian Georg Zilly, Rupesh Kumar Srivastava, Jan Koutník and Jürgen Schmidhuber

## Simple Summary

>  Introduce a novel theoretical analysis of recurrent networks based on Gersgorin's circle theorem that illuminates several modeling and optimization issues and improves our understanding of the LSTM cell. Based on this analysis we propose Recurrent Highway Networks, which extend the LSTM architecture to allow step-to-step transition depths larger than one. 

![images](../images/recurrent_highway_1.png)

- Geršgorin circle theorem (GCT) (Geršgorin, 1931)
	- Using GCT we can understand the relationship between the entries of R and the possible locations of the eigenvalues of the Jacobian.
	- RNN with identity initialization: During training some eigenvalues can easily become larger than one, resulting in exploding gradients.

![images](../images/recurrent_highway_2.png)

- Recurrent Highway Networks (RHN)
	- GCT allows us to observe the behavior of the full spectrum of the temporal Jacobian, and the effect of gating units on it. (expect that for learning multiple temporal dependencies from real-world data efficiently)
	- it becomes clear that through their effect on the behavior of the Jacobian, highly non-linear gating functions can facilitate learning through rapid and precise regulation of the network dynamics.
	- the analysis of the RHN layer’s flexibility in controlling its spectrum furthers our theoretical understanding of LSTM and Highway networks and their variants.
- Each Highway layer allows increased flexibility in controlling how various components of the input are transformed or carried.