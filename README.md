# Dynamics of Moral Behavior in Heterogeneous Populations of Learning Agents

This repository contains implementation and analysis code for the following paper: 
Dynamics of Moral Behavior in Heterogeneous Populations of Learning Agents, AIES'24. 

[(arXiv version with Appendix)](https://arxiv.org/abs/2403.04202) 


## Cite us
***

If you use this code, please cite the following paper:

```bibtex
@INPROCEEDINGS{Tennant-AIES2024,
  title     = {Dynamics of Moral Behavior in Heterogeneous Populations of Learning Agents},
  author    = {Tennant, Elizaveta and Hailes, Stephen and Musolesi, Mirco},
  booktitle = {Proceedings of the 7th AAAI/ACM Conference on AI, Ethics & Society (AIES'24)},
  publisher = {AAAI / ACM},
  editor    = {},
  pages     = {},
  year      = {2024},
  month     = {},
  note      = {Main Track},
  doi       = {},
  url       = {[https://doi.org/10.24963/ijcai.2023/36](https://arxiv.org/abs/2403.04202)},
}

```

You can contact the authors at: `l.karmannaya.16@ucl.ac.uk`

## Setup

Intall packages listed in requirements.txt into a Python environment. 
```
pip install -r requirements.txt
```

## The environment 

This code can be used to run a simulation of social dilemma games within populations agents - at every step, an agent M selects an opponent O, and then M and O play a one-shot Prisoner's Dilemma game. We use a Reinforcement Learning paradigm where each agent learns accoridng to a reward signal:

![Reinformcenet Learning by a Moral learning agent M and a learning opponent O](pics/diagram_V2.png "Reinformcenet Learning by a Moral learning agent M and a learning opponent O")

the reward is defined by the agent's payoff in a game.

<p align="center">
  <img src="https://github.com/Liza-Karmannaya/modeling_moral_choice_dyadic/blob/main/pics/payoffs.png" />
</p>

We use a apradig where each gent learns how to select a partner AND learns how to play the game using the same reward signal. 

## The agents 

These experiments conduct a systematic comparison of interactions between pairs of various moral learning agents in each of the dilemma games. The moral agents are defined using the following intrnisic rewards: 

![Rewards](pics/moralrewards.png "Rewards")



## Run the experiments

...


## Parameters

...

## Plotting 

...


