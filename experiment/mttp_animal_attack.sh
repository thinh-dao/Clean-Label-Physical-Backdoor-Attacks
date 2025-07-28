python main.py --devices=1,0 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &
python main.py --devices=0,1 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-28 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &

python main.py --devices=2,3 --num_experts=5 --expert_epochs=2 --backdoor_training_epoch=2 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_2_2 &
python main.py --devices=3,2 --num_experts=5 --expert_epochs=2 --backdoor_training_epoch=2 --poisonkey=11-28 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_2_2 &



python main.py --devices=4,7 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-19 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_6_6 &
python main.py --devices=7,4 --num_experts=3 --expert_epochs=3 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_3_3_6 &


python main.py --devices=7,4 --num_experts=3 --expert_epochs=3 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=finetuning --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_3_3_6

##### DeiT Tiny #####
python main.py --devices=0,1 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=deit_tiny --optimization=transformer-adamw --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &
python main.py --devices=1,0 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=3000 --retrain_scenario=from-scratch --trigger=tennis --net=deit_tiny --optimization=transformer-adamw --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &




##### DeiT Tiny Augmentation #####
python main.py --devices=0,1 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=2500 --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=1,0 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=500 --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=transfer --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &








python main.py --devices=7,4 --num_experts=1 --expert_epochs=5 --backdoor_training_epoch=5 --backdoor_training_mode=poison_only --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=finetuning --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_poisononly
python main.py --devices=1,2 --num_experts=1 --expert_epochs=5 --backdoor_training_epoch=5 --backdoor_training_mode=full-data --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=finetuning --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_full_data





python main.py --devices=0,3 --recipe=mttp --attackiter=500 --skip_clean_training --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --retrain_max_epoch=30 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --tau=0.1 --eps=16 --exp_name=final_fm_iterative_test 

python main.py --devices=0,3 --recipe=mttp --attackiter=500 --skip_clean_training --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --retrain_max_epoch=30 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=24 --exp_name=final_fm_iterative_test --dryrun


python main.py --devices=0,1 --recipe=mttp --attackiter=300 --mtt_validate_every=50 --scenario=finetuning --net=resnet18_imagenet --source_criterion=cw --bkd_iter=3 --bkd_lr=0.001 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --exp_name=final_mttp_validate
python main.py --devices=0,1 --recipe=mttp --attackiter=300 --mtt_validate_every=50 --scenario=finetuning --net=resnet18_imagenet --sample_from_trajectory --retrain_max_epoch=30 --source_criterion=cw --bkd_iter=3 --bkd_lr=0.001 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --exp_name=final_mttp_validate_traj