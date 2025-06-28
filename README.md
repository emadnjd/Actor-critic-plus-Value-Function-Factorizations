🧠 Project Title
Exploring Value Function Factorization Methods in Deep Multi-Agent COMA Algorithm

📄 Overview
This repository contains the implementation and evaluation of a novel exploration into combining value function factorization methods with the Counterfactual Multi-Agent (COMA) algorithm within deep multi-agent reinforcement learning (MARL).
The project investigates how integrating VDN, QMIX, and a non-monotonic factorization method into the COMA framework influences performance, coordination, and learning efficiency in cooperative multi-agent environments.

🎯 Objectives
Integrate VDN, QMIX, and a non-monotonic method with COMA to enhance credit assignment and coordination.

Evaluate and compare these hybrid approaches against standard COMA in two distinct environments.

Analyze the impact of factorization on policy learning and value estimation within actor-critic MARL settings.

🧪 Environments
The code supports two distinct evaluation environments:

Matrix Game (a discrete, static coordination game)

Predator-Prey Game (a dynamic grid-based cooperative environment)

Each algorithm (COMA and its hybrid variants) was tested in both environments under consistent training conditions for fair and reproducible comparisons.

⚙️ Implemented Algorithms:
Baseline: Standard COMA

Hybrid Variants:

VDN-COMA

QMIX-COMA

Non-monotonic-COMA

These variants maintain COMA’s actor-critic architecture while altering the centralized critic’s value decomposition strategy.

📈 Evaluation Strategy
To ensure fair comparisons:

All models were trained under identical conditions across both environments.

Key performance metrics (e.g., average episode reward, policy stability) were collected and compared.

The standard COMA implementation serves as the baseline reference.

🧮 Hyperparameters
The hyperparameters used for training are consistent across algorithms to isolate the impact of the factorization method.

Actor Learning rate (lr_a)

Critic Learning rate (lr_c)

Discount factor (gamma) 

Batch size 

Exploration: epsilon-greedy with decay 

Replay buffer size

Target network update rate

total_timesteps
