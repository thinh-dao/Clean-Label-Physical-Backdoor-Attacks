python main.py --devices=1,0 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=resnet18_imagenet --vnet=resnet18_imagenet,vgg11_imagenet,mobilenetv2_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models

python main.py --devices=2,3 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=vgg11_imagenet --vnet=vgg11_imagenet,resnet18_imagenet,mobilenetv2_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models

python main.py --devices=3,2 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=mobilenetv2_imagenet --vnet=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models

python main.py --devices=4,5 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=2500 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --vnet=mobilenetv2_imagenet,resnet18_imagenet,vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --exp_name=transfer_models


###### CLEAN TRAINING
python main.py --devices=3,2 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=1000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=vgg11_imagenet --vnet=resnet18_imagenet,mobilenetv2_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --clean_training_only --exp_name=clean_training 

python main.py --devices=2,3 --recipe=gradient-matching --source_criterion=cw \
    --attackiter=1000 --scenario=finetuning --retrain_scenario=from-scratch --retrain_iter=500 \
    --net=mobilenetv2_imagenet --vnet=resnet18_imagenet,vgg11_imagenet --vruns=1 --dataset=Animal_classification --poisonkey=11-19 --trigger=tennis \
    --alpha=0.2 --eps=16 --alpha=0.2 --model_seed=123456 --poison_seed=123456 \
    --save_poison=poison_only --clean_training_only --exp_name=clean_training 