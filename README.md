# Intrinsically Motivated Goal-Conditioned Reinforcement Learning in Multi-Agent Environments

![python](https://img.shields.io/badge/python-3.9-blue)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This repo contains the code base of the project Intrinsically Motivated Goal-Conditioned Reinforcement Learning in Multi-Agent Environments.

## Context
Building autonomous machines that can explore open-ended environments, discover possible interactions and autonomously build repertoires of skills is a general objective of artificial intelligence. We’ll consider the framework of Intrinsically Motivated Goal Exploration Process (IMGEPs, see (Colas et al., 2021) for a recent review). In an IMGEP, the agent is considered as autotelic, in the sense that it self-generates its own goals.

We would like to generalize the IMGEP framework to MARL. In few words, the setting is multi-task, multi-agent, and agents select their own goals. For this aim, we will design a complex multi-agent environment filled with various types of objects offering a large goal space to agents, some of these goals requiring certain levels of cooperation to be achieved. Goal and rewards distributions will be fixed and externally defined, however, agents will need to learn to sample from them and choose which goal to follow on each episode.

The project will start with two baselines, with the intention of estimating the worst and best possible performances a set of agents might get:

1.    Independent IM-GC-RL agents (lower bound on the performance)
        Each agent sample its goal independently on each episode. We don’t expect them to learn how to collaborate and thus we expect them to fail in cooperative tasks.

2.    GC-RL agents with an external goal supervisor (higher bound on the performance)
        Goals are the same for all agents at each episode. With full coordination, we expect them to learn how to deal with both individual and cooperative tasks.

The challenge then is to enable some kind of communication between agents so they can align their goals before starting an episode. This will be done by adding a negotiation step before the start of the episode. Each agent will negotiate and decide which goal to follow. We’ll also consider an after-episode communication step, so agents can also perform hindsight learning ( instance where agents can reinterpret a past trajectory collected while pursuing a given goal in the light of a new goal, see (Colas et al., 2021))

The final objective is to study how agents can sample their own goals and align them at the population level, learning how to collectively achieve them in a fully autonomous fashion for a given time budget, without any external supervision.

## How to use
First you need to clone the repository. For that, you can use the following command line:
```Bash
git clone https://anonymous.4open.science/r/Dec-IMSAP.git
```
Then we recommend using a virtual environment, this can be done by the following:
```Bash
python3 -m venv env
source env/bin/activate
```
Finally, in order to install the package, you can simply run:
```Bash
pip install -e .
```
