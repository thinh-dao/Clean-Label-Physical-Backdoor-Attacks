python main.py --devices=2,3 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=3,2 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=2,3 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=3,2 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500  --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &


python main.py --devices=0,1 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --net=deit_tiny --optimization=transformer-adamw --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=1,0 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --net=deit_tiny --optimization=transformer-adamw --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=0,1 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --net=deit_tiny --optimization=transformer-adamw --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &
python main.py --devices=1,0 --recipe=gradient-matching --source_criterion=cw --attackiter=2000 --net=deit_tiny --optimization=transformer-adamw --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm &



######## DEIT_TINY AUGMENTATION ########
python main.py --devices=0,1 --recipe=gradient-matching --source_criterion=cw --attackiter=2500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=1,0 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=transfer --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=0,1 --recipe=gradient-matching --source_criterion=cw --attackiter=2500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=1,0 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=transfer --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &



######## GM no CW ########
python main.py --devices=2,3 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_normal &
python main.py --devices=3,2 --recipe=gradient-matching --attackiter=1500 --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_normal &
python main.py --devices=5,6 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_normal &
python main.py --devices=6,5 --recipe=gradient-matching --attackiter=1500 --scenario=transfer --retrain_scenario=from-scratch --retrain_iter=500  --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=20 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_normal &




########## Batch size ##########
python main.py --devices=4,5 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --pbatch=8 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm
python main.py --devices=5,4 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --pbatch=16 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm

########## Full Data ##########
python main.py --devices=5,4 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --full_data --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm

########## Step ##########
python main.py --devices=1,2 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --step --step_every=100 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_step
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw --attackiter=500 --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --step --step_on_poison --step_every=100 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_steppoison


python main.py --devices=1,2 --recipe=gradient-matching --source_criterion=cw --attackiter=1000 --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --step --step_every=200 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_step


python main.py --devices=1,2 --recipe=gradient-matching --source_criterion=cw --attackiter=1000 --sample_from_trajectory --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --alpha=0.1 --eps=16 --step --step_every=200 --model_seed=123456 --poison_seed=123456 --save_poison=poison_only --exp_name=final_gm_traj



########################## Ablation study ##########################
python main.py --devices=0,1 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_traj_different_idx
python main.py --devices=1,0 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_traj_same_idx
python main.py --devices=2,3 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_traj
python main.py --devices=3,2 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm
python main.py --devices=4,5 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --step --step_every=250 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_step
python main.py --devices=5,4 --recipe=gradient-matching --attackiter=750 --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_no_retrain



python main.py --devices=3,2,1 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_ensemble
python main.py --devices=6,5,4 --recipe=gradient-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --step --step_every=250 --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_gm_step_retrain