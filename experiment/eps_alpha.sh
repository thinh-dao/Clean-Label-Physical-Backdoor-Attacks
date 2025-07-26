#!/bin/bash

alphas=(0.1 0.2 0.3)
epsilons=(16 32 48)
devices=(0 1 2 3 4 5 6 7)

idx=0
for alpha in "${alphas[@]}"; do
  for eps in "${epsilons[@]}"; do
    dev1=${devices[$(( (idx*2)   % 8 ))]}
    dev2=${devices[$(( (idx*2+1) % 8 ))]}
    python main.py --devices=${dev1},${dev2} --recipe=gradient-matching --source_criterion=cw \
      --attackiter=4000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=1000 \
      --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
      --alpha=${alpha} --eps=${eps} --model_seed=123456 --poison_seed=123456 \
      --save_poison=poison_only --exp_name=eps_alpha_gm &
    idx=$((idx+1))
  done
done