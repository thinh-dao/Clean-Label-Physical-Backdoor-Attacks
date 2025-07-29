python main.py --devices=3,0 --attackiter=1000 --scenario=transfer --net=resnet18_imagenet --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_retrain_1000_0.01 &
python main.py --devices=3,0 --attackiter=1000 --scenario=transfer --net=resnet18_imagenet --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_retrain_1000_0.01 &

python main.py --devices=1,0 --attackiter=2000 --retrain_iter=500 --retrain_scenario=from-scratch --scenario=finetuning --net=resnet18_imagenet --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_retrain_500_0.01 &
python main.py --devices=0,1 --attackiter=2000 --retrain_iter=500 --retrain_scenario=from-scratch --scenario=finetuning --net=resnet18_imagenet --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_retrain_500_0.01 &

python main.py --devices=2,3 --attackiter=500 --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_deit_0.01
python main.py --devices=3,2 --attackiter=4000 --scenario=finetuning --retrain_iter=1000 --retrain_scenario=from-scratch --net=deit_tiny --optimization=transformer-adamw --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_deit_1000_0.01




python main.py --devices=5,6 --attackiter=2500 --retrain_iter=500 --retrain_scenario=from-scratch --scenario=finetuning --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --recipe=feature-matching --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_retrain_500_0.01 


python main.py --devices=4,5 --recipe=feature-matching --attackiter=2500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=default --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --tau=0.01 --save_poison=poison_only --exp_name=deit_tiny_animal
python main.py --devices=6,7 --recipe=feature-matching --attackiter=2500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=default --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --sample_gradient --poison_seed=123456 --tau=0.01 --save_poison=poison_only --exp_name=deit_tiny_animal_sample_grad


python main.py --devices=7,6 --recipe=feature-matching --attackiter=2500 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --model_seed=123456 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --sample_gradient --poison_seed=123456 --tau=0.01 --save_poison=poison_only --exp_name=deit_tiny_animal_sample_grad_no_augment


python main.py --devices=4,5 --recipe=feature-matching --attackiter=5000 --net=deit_tiny,deit_tiny,deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=1000 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --tau=0.01 --save_poison=poison_only --exp_name=deit_tiny_animal &
python main.py --devices=5,4 --recipe=feature-matching --attackiter=500 --net=deit_tiny --ensemble=3 --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --alpha=0.1 --eps=20 --poison_seed=123456 --tau=0.01 --save_poison=poison_only --exp_name=deit_tiny_animal &



python main.py --devices=5,6,7  --recipe=feature-matching --skip_clean_training --attackiter=2000 --scenario=finetuning --retrain_scenario=finetuning --retrain_iter=250 --retrain_max_epoch=5 --validate_every=5 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_iterative



# 
python main.py --devices=7,6,5,4 --recipe=feature-matching --attackiter=2000 --warm_start --warm_start_epochs=5 --skip_clean_training --scenario=finetuning --retrain_scenario=finetuning --retrain_iter=250 --retrain_max_epoch=5 --sample_from_trajectory --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-28 --trigger=phone --model_seed=123456 --poison_seed=123456 --tau=0.01 --exp_name=final_fm_iterative



########################## Ablation study ##########################
python main.py --devices=4,5 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_traj_different_idx
python main.py --devices=4,5 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_traj_same_idx
python main.py --devices=5,4 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --sample_from_trajectory --sample_every=5 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_traj
python main.py --devices=5,4 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative
python main.py --devices=5,4 --recipe=feature-matching --attackiter=1500 --step --step_every=250 --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_step
python main.py --devices=5,4 --recipe=feature-matching --attackiter=750 --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_no_retrain



python main.py --devices=4,5 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet,resnet18_imagenet,resnet18_imagenet --ensemble=3 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_ensemble
python main.py --devices=5,4 --recipe=feature-matching --attackiter=1500 --scenario=finetuning --retrain_scenario=from-scratch --step --step_every=250 --retrain_iter=750 --retrain_max_epoch=25 --net=resnet18_imagenet --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis --model_seed=123456 --poison_seed=123456 --tau=0.01 --eps=16 --exp_name=final_fm_iterative_step_retrain