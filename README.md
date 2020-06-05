# dilemmaRL



Code for our paper: 

**"Online Learning in Iterated Prisoner's Dilemma to Mimic Human Behavior"** 

by [Baihan Lin](http://www.columbia.edu/~bl2681/) (Columbia), [Djallel Bouneffouf](https://scholar.google.com/citations?user=i2a1LUMAAAAJ&hl=en) (IBM Research), [Guillermo Cecchi](https://researcher.watson.ibm.com/researcher/view.php?person=us-gcecchi) (IBM Research).



For the latest full paper: https://arxiv.org/abs/



All the experimental results can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.


**Abstract**


Prisoner's Dilemma mainly treat the choice to cooperate or defect as an atomic action. We propose to study online learning algorithm behavior in the Iterated Prisoner's Dilemma (IPD) game, where we explored the full spectrum of reinforcement learning agents: multi-armed bandits, contextual bandits and reinforcement learning. We have evaluate them based on a tournament of iterated prisoner's dilemma where multiple agents can compete in a sequential fashion. This allows us to analyze the dynamics of policies learned by multiple self-interested independent reward-driven agents, and also allows us study the capacity of these algorithms to fit the human behaviors. Results suggest that considering the current situation to make decision is the worst in this kind of social dilemma game. Multiples discoveries on online learning behaviors and clinical validations are stated.


## Info

Language: Python3, Python2, bash


Platform: MacOS, Linux, Windows

by Baihan Lin, April 2020


## Citation

If you find this work helpful, please try the models out and cite our works. Thanks!

    @article{lin2020online,
      title={Online Learning in Iterated Prisoner's Dilemma to Mimic Human Behavior},
      author={Lin, Baihan and Bouneffouf, Djallel and Cecchi, Guillermo},
      journal={arXiv preprint arXiv:},
      year={2020}
    }



## Tasks

* Iterated Prisoner's Dilemma (IPD) with two players

* Iterated Prisoner's Dilemma (IPD) with N players

  

## Algorithms:

* **Bandits:** UCB1, Thompson Sampling, epsilon Greedy, EXP3, Human Behavior Thompson Sampling

* **Contextual bandits:** LinUCB, Contextual Thompson Sampling, EXP4, Split Contextual Thompson Sampling

* **Reinforcement learning:** Q Learning, Double Q Learning, SARSA, Split Q Learning

* **Handcrafted:** Always cooperate, Always defect, Tit for tat

  

## Requirements

* numpy and scikit-learn



## Mental variants

* For the specifics about the mental variants used in this work, check out: https://github.com/doerlbh/mentalRL



