# Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models

- Submitted on 2016. 10
- Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R. Selvaraju, Qing Sun, Stefan Lee, David Crandall and Dhruv Batra

## Simple Summary

>  propose Diverse Beam Search(DBS), an alternative to BS that decodes a list of diverse outputs by optimizing for a diversity-augmented objective. We observe that our method finds better top-1 solutions by controlling for the exploration and exploitation of the search space -- implying that DBS is a better search algorithm. Moreover, these gains are achieved with minimal computational or memory overhead as compared to beam search.

![images](../images/dbs_1.png)

-  optimize an objective that consists of two terms – the sequence likelihood under the model and a dissimilarity term that encourages beams across groups to differ.
-  This diversity-augmented model score is optimized in a doubly greedy manner – greedily optimizing along both time (like BS) and groups (like DivMBest).

![images](../images/dbs_2.png)