################### White-box transferability experiments ###################
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=resnet18_imagenet --vnet=resnet18_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models && \

python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=vgg11_imagenet --vnet=vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models && \

python main.py --devices=4,3 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=mobilenetv2_imagenet --vnet=mobilenetv2_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models && \

python main.py --devices=4,3 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=deit_tiny --vnet=deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \

################### Black-box transferability experiments ###################
# Resnet34
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=resnet34_imagenet --ensemble=4 --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \

# MobileNetV3
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=mobilenetv3_imagenet --ensemble=4 --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \

# VGG13
python main.py --devices=1,2,3 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=vgg13_imagenet --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \

# Deit-Small
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=deit_small --optimization=transformer-adamw --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \

# Ensemble 0
python main.py --devices=4,5 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --ensemble=3 --vnet=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models

# Ensemble 1
python main.py --devices=4,5 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=resnet34_imagenet,vgg13,mobilenetv3_imagenet,deit_small --vnet=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models

# Ensemble 2
python main.py --devices=3,4 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=3000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=750 \
    --net=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --ensemble=4 --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet,deit_tiny --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models & \


###### CLEAN TRAINING
python main.py --devices=3,2 --net=vgg11_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --clean_training_only --save_clean_model --model_seed=123456 --exp_name=clean_training
python main.py --devices=3,2 --net=vgg13_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --clean_training_only --save_clean_model --model_seed=123456 --exp_name=clean_training
python main.py --devices=3,2 --net=deit_small --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --clean_training_only --save_clean_model --model_seed=123456 --exp_name=clean_training  
python main.py --devices=3,2 --net=mobilenetv2_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --clean_training_only --save_clean_model --model_seed=123456 --exp_name=clean_training 
python main.py --devices=4,5 --net=mobilenetv3_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --clean_training_only --save_clean_model --model_seed=123456 --exp_name=clean_training 

