# Building Machines That Learn and Think Like People

- Submitted on 2016. 4
- Brenden M. Lake, Tomer D. Ullman, Joshua B. Tenenbaum and Samuel J. Gershman

## Simple Summary

> review progress in cognitive science suggesting that truly human-like learning and thinking machines will have to reach beyond current engineering trends in both what they learn, and how they learn it.
>
> (a) build causal models of the world that support explanation and understanding, rather than merely solving pattern recognition problems;  
> (b) ground learning in intuitive theories of physics and psychology, to support and enrich the knowledge that is learned;  
> (c) harness compositionality and learning-to-learn to rapidly acquire and generalize knowledge to new tasks and situations.

**Human** 

- Ingredient `start-up software` (Infants):
	1. intuitive physics (primitive object concepts, Causality)
	2. intuitive psychology (goals and beliefs)
- Learning 
	1. model building : explaining observed data through the construction of causal models of the world
	2. compositionality : capable of learning these richly structured models
from very limited amounts of experience
	3. learning to learn : make this type of rapid model learning possible
- Action
	1. rich models our minds build are put into action, in real time.
	2. `learn to do inference` : model-free methods can accelerate slow model-based inferences in perception and cognition

- Integration of **model-based** and **model-free** methods in reinforcement learning.

---

- Explore Challenges
	1. Recognizing new characters and objects 
	2. Learning to play the game Frostbite.
- The ingredients outlined in this article will prove useful for working towards this goal: seeing objects and agents rather than features, building causal models and not just recognizing patterns, recombining representations without needing to retrain, and learning-to-learn rather than starting from scratch.