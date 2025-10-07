# Research on the Efficiency of Modern DQN Modifications under Limited Computational Resources

This repository provides a modular and extensible framework for training and evaluating multiple **Deep Q-Network (DQN)** variants under **limited computational resources**.  
The project investigates how different improvements to DQN affect training efficiency, stability, and performance when running on constrained hardware (e.g., single GPU or CPU).

## Implemented Algorithms

| Algorithm | Reference | Key Idea |
|------------|------------|-----------|
| **Vanilla DQN** | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) | Deep Q-learning from raw pixels |
| **Double DQN** | [van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461) | Reduces overestimation bias |
| **Dueling DQN** | [Wang et al., 2016](https://arxiv.org/abs/1511.06581) | Separates value and advantage estimations |
| **Prioritized Experience Replay (PER)** | [Schaul et al., 2016](https://arxiv.org/abs/1511.05952) | Samples important transitions more frequently |
| **Noisy Networks** | [Fortunato et al., 2018](https://arxiv.org/abs/1706.10295) | Uses noise for exploration instead of ε-greedy |
| **Categorical DQN (C51)** | [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887) | Learns a distribution of returns |
| **Quantile Regression DQN (QR-DQN)** | [Dabney et al., 2018](https://arxiv.org/abs/1710.10044) | Models quantiles of the return distribution |
| **Munchausen DQN** | [Vieillard et al., 2020](https://arxiv.org/abs/2007.14430) | Adds a log-policy term for improved stability |
| **DrQ (Data-Regularized Q-Learning)** | [Yarats et al., 2021](https://arxiv.org/abs/2004.13649) | Regularizes Q-learning with data augmentations |
| **Dropout Q-functions** | [Osband et al., 2021](https://arxiv.org/abs/2110.02034) | Uses dropout for uncertainty and exploration |
| **CURL (Contrastive Unsupervised Representation Learning)** | [Laskin et al., 2020](https://arxiv.org/abs/2004.04136) | Improves representation learning via contrastive loss |
| **RND (Random Network Distillation)** | [Burda et al., 2019](https://arxiv.org/abs/1810.12894) | Intrinsic motivation through prediction error |

## Configuration

All hyperparameters and architecture options are defined in [`configs/config.yaml`](configs/config.yaml).  
Each DQN modification can be enabled or disabled using the `agent_params` block:

```yaml
agent_params:
  use_dueling: True
  use_double: True
  use_per: True
  use_c51: True
  use_qr: False
  use_munchausen: False
  use_noisy: True
  use_dropout: False
  use_drq: False
  use_rnd: False
  use_curl: False
  n_step: 5
```

No code changes are required — you can freely toggle modules via the YAML configuration.


## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/smiling-demon/efficient-dqn-research.git
cd efficient-dqn-research
```


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run training
```bash
python ./main.py
```

By default, it will load parameters from `configs/config.yaml`.  
You can modify the YAML file to change algorithms, hyperparameters, and environment settings.


## Logging & Evaluation

- Training logs (if enabled) are written to `runs/`
- Model checkpoints are saved to `models/`
- You can visualize metrics using TensorBoard:
```bash
tensorboard --logdir runs/
```