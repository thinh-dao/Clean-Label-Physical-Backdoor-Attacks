python main.py --devices=1,0 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &
python main.py --devices=0,1 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-28 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &

python main.py --devices=2,3 --num_experts=5 --expert_epochs=2 --backdoor_training_epoch=2 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_2_2 &
python main.py --devices=3,2 --num_experts=5 --expert_epochs=2 --backdoor_training_epoch=2 --poisonkey=11-28 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_2_2 &



python main.py --devices=4,7 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-19 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_6_6 &
python main.py --devices=7,4 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_6_6 &




##### DeiT Tiny #####
python main.py --devices=0,1 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=deit_tiny --optimization=transformer-adamw --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &
python main.py --devices=1,0 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=3000 --retrain_scenario=from-scratch --trigger=tennis --net=deit_tiny --optimization=transformer-adamw --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &




##### DeiT Tiny Augmentation #####
python main.py --devices=0,1 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=2500 --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=1,0 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=500 --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=transfer --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --save_poison=poison_only --exp_name=deit_tiny_animal &