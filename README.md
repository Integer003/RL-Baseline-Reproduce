Forked from:
```
https://github.com/facebookresearch/drqv2
https://github.com/XuGW-Kevin/DrM
https://github.com/suninghuang19/mentor
https://github.com/Aladoro/Stabilizing-Off-Policy-RL
https://github.com/FrankZheng2022/TACO
https://github.com/jzndd/CP3ER
https://github.com/lilucse/Normalization-Enhances-Generalization-in-Visual-Reinforcement-Learning
```

Installation

```bash
sudo apt update
sudo apt install libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3 
conda env create -f conda_env.yml 
conda activate rl_baselines
pip install torch torchvision torchaudio
pip install swanlab
cd envs
cd metaworld
pip install -e .
cd ..
cd rrl-dependencies
pip install -e .
cd mj_envs
pip install -e .
cd ..
cd mjrl
pip install -e .
```

Environment varianbles for gcc compile:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
```

Run DMC task:

```bash
python train_dmc.py task=dog_stand agent=drm
```

Run metaworld task:

```bash
python train_mw.py task=coffee-push agent=drm_mw
```