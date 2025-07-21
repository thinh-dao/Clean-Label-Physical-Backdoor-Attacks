python main.py --devices=1,0 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-19 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &
python main.py --devices=0,1 --num_experts=5 --expert_epochs=5 --backdoor_training_epoch=10 --poisonkey=11-28 --recipe=mttp --attackiter=6000 --retrain_iter=1500 --scenario=finetuning --retrain_scenario=from-scratch --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --save_poison=poison_only --exp_name=final_mttp_5_3_6 &





python main.py --devices=4,7 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-19 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=tennis --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_6_6 &
python main.py --devices=7,4 --num_experts=5 --expert_epochs=6 --backdoor_training_epoch=6 --backdoor_training_mode=poison_only --poisonkey=11-28 --recipe=mttp --attackiter=500 --scenario=transfer --trigger=phone --net=resnet18_imagenet --model_seed=123456 --poison_seed=123456 --dataset=Animal_classification --tau=0.01 --save_poison=poison_only --exp_name=final_mttp_5_6_6 &

