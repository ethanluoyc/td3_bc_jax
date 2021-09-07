# TD3_BC in JAX
Direct port of https://github.com/sfujim/TD3_BC to JAX using Haiku and optax.

# Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```

# Usage
Refer to the original README for usage.

Below is the original README.
----

# A Minimalist Approach to Offline Reinforcement Learning

TD3+BC is a simple approach to offline RL where only two changes are made to TD3: (1) a weighted behavior cloning loss is added to the policy update and (2) the states are normalized. Unlike competing methods there are no changes to architecture or underlying hyperparameters. The paper can be found [here](https://arxiv.org/abs/2106.06860).

### Usage
Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The paper results can be reproduced by running:
```
./run_experiments.sh
```

---
*This is not an official Google product. 