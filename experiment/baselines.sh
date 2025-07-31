######### dirty-label-physical #########
python main.py --recipe=dirty-label-physical --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &

python main.py --recipe=dirty-label-physical --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &\
python main.py --recipe=dirty-label-physical --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical & 


python main.py --recipe=dirty-label-physical --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &
python main.py --recipe=dirty-label-physical --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &
python main.py --recipe=dirty-label-physical --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &
python main.py --recipe=dirty-label-physical --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-physical &

# Augment
python main.py --recipe=dirty-label-physical --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-physical & \
python main.py --recipe=dirty-label-physical --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-physical &


######### dirty-label-digital #########
python main.py --recipe=dirty-label-digital --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &

python main.py --recipe=dirty-label-digital --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &\
python main.py --recipe=dirty-label-digital --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital & 

python main.py --recipe=dirty-label-digital --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &
python main.py --recipe=dirty-label-digital --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &
python main.py --recipe=dirty-label-digital --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &
python main.py --recipe=dirty-label-digital --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty-label-digital &

# Augment
python main.py --recipe=dirty-label-digital --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-digital & \
python main.py --recipe=dirty-label-digital --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=dirty-label-digital &

######### Naive Attack #########
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive &

python main.py --recipe=naive --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=1,2 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=1,2 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=2,1 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=2,1 --model_seed=123456 --poison_seed=123456 --exp_name=naive & 

python main.py --recipe=naive --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive &

# Augment
python main.py --recipe=naive --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-28 --alpha=0.1 --devices=4,3 --poison_seed=123456 --exp_name=naive &




######### LC #########
python main.py --recipe=label-consistent --scenario=finetuning --dataset=Facial_recognition --trigger=sunglasses --poisonkey=9-5 --attackiter=250 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --dataset=Facial_recognition --trigger=sunglasses --poisonkey=9-5 --attackiter=250 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=finetuning --dataset=Facial_recognition --trigger=real_beard --poisonkey=6-1 --attackiter=250 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --dataset=Facial_recognition --trigger=real_beard --poisonkey=6-1 --attackiter=250 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent &

python main.py --recipe=label-consistent --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --attackiter=250 --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --attackiter=250 --poisonkey=11-19 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --attackiter=250 --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --attackiter=250 --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & 

python main.py --recipe=label-consistent --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=tennis --attackiter=250 --poisonkey=11-19 --alpha=0.1 --devices=2,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & 
python main.py --recipe=label-consistent --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --dataset=Animal_classification --trigger=phone --attackiter=250 --poisonkey=11-28 --alpha=0.1 --devices=3,2 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & 

# Augment
python main.py --recipe=label-consistent --scenario=finetuning --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=0,1 --poison_seed=123456 --exp_name=label-consistent & \
python main.py --recipe=label-consistent --scenario=transfer --net=deit_tiny --optimization=transformer-adamw --augment --paugment --data_aug=mixed --model_seed=1234567 --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=1,0 --poison_seed=123456 --exp_name=label-consistent & 