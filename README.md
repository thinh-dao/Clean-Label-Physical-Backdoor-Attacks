# Clean-Label Physical Backdoor Attacks With Dataset Distillation 

This code is the official PyTroch implementation of our paper: [Clean-Label Physical Backdoor Attacks With Dataset Distillation](https://arxiv.org/abs/2407.19203). 

The code is built based on the framework from [Industrial Scale Data Poisoning via Gradient Matching](https://github.com/JonasGeiping/poisoning-gradient-matching) and [Sleeper Agent](https://github.com/hsouri/Sleeper-Agent.git)

To use this codebase, you have to first download the dataset of Animal classification in this link: https://drive.google.com/file/d/1E97dY5bm3xgAfwIydVxfCuHYR7AGXKw1/view?usp=drive_link and unzip.

To use with gdown:
```bash
    gdown 1E97dY5bm3xgAfwIydVxfCuHYR7AGXKw1
    tar -xzf Animal_classification.tar.gz
```

All of our experiment scripts are given in the experiment folder. The main table results (Table 1) can be reproduced by running the scripts in  ```experiment/paper_exp.sh```. The estimated time to run an experiment with the GM attack under full-finetuning in that script is ~7 hours on A5000 GPUs. You can set smaller retraining factor and smaller optimization steps reduce experimentation time. The scripts in the ```experiment/defense_exp.sh``` folder are for reproducing the defense experiments.

Example running script on the Animal Classification dataset with GM attack and CW loss:

```bash
python main.py --recipe=gradient-matching --dataset=Animal_classification --eps=16 --alpha=0.1 --source_criterion=cw --trigger=tennis --net=resnet18_imagenet --poisonkey=11-19 --trigger=tennis --devices=0,1 --save_poison=poison_only --model_seed=123456 --poison_seed=123456 --exp_name=Animal_classification_gm 
```
