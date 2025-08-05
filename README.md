# Towards Clean-Label Physical Backdoor Attacks 

This code is the official PyTroch implementation of our paper: Towards Clean-Label Backdoor Attacks in the Physical World. The code is built based on the framework from [Industrial Scale Data Poisoning via Gradient Matching](https://github.com/JonasGeiping/poisoning-gradient-matching) and [Sleeper Agent](https://github.com/hsouri/Sleeper-Agent.git)

All of our experiment scripts are given in the experiment folder. The main table results (Table 1) can be reproduced by running the scripts in  ```experiment/paper_exp.sh```. The scripts in the ```experiment/defense_exp.sh``` folder are for reproducing the defense experiments.

Example running script on the Animal Classification dataset:

```bash
python main.py --recipe=gradient-matching --dataset=Animal_classification --eps=16 --alpha=0.05 --trigger=tennis --net=resnet18_imagenet --poisonkey=11-35 --devices=0,1 --save_poison=poison_only --model_seed=123456 --poison_seed=123456 --exp_name=Animal_classification_gm 
```
For details on the arguments, please see `options.py`.
The dataset will be updated after the paper is accepted.

