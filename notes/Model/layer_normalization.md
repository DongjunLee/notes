# Layer Normalization

- published in 2016. 7
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

----

## Simple summary

- Similar to batch normalization.
- Works well for recurrent networks with mini-batch.
- Independent with batch size.
- Robust to Input data's scale
- Robust to Weight matrix's scale and shift
- Naturally uploaded scales decreased as learning progresses
- but, Batch Norm still perform better for CNNs.