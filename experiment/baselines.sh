######### Dirty-Label #########
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &

python main.py --recipe=dirty-label --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=3,4,7 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \
python main.py --recipe=dirty-label --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=4,3,7 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \
python main.py --recipe=dirty-label --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &\
python main.py --recipe=dirty-label --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & 

######### Naive Attack #########
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive & \
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=naive &

######### LC #########
python main.py --recipe=label-consistent --scenario=finetuning --dataset=Facial_recognition --trigger=sunglasses --poisonkey=9-5 --attackiter=250 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --dataset=Facial_recognition --trigger=sunglasses --poisonkey=9-5 --attackiter=250 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=finetuning --dataset=Facial_recognition --trigger=real_beard --poisonkey=6-1 --attackiter=250 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --dataset=Facial_recognition --trigger=real_beard --poisonkey=6-1 --attackiter=250 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent &

python main.py --recipe=label-consistent --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --attackiter=250 --poisonkey=11-19 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --attackiter=250 --poisonkey=11-19 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --attackiter=250 --poisonkey=11-28 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & \
python main.py --recipe=label-consistent --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=phone --attackiter=250 --poisonkey=11-28 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=label_consistent & 
