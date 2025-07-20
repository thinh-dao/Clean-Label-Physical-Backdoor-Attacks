######### Dirty-Label #########
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=0,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=1,0 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=3,4 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label & \
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,3 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &

python main.py --recipe=dirty-label --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=7,4,3,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \
python main.py --recipe=dirty-label --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=tennis --poisonkey=11-19 --alpha=0.1 --devices=7,4,3,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \
python main.py --recipe=dirty-label --scenario=finetuning --net=resnet18_imagenet --dataset=Animal_classification --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=7,4,3,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \
python main.py --recipe=dirty-label --scenario=transfer --net=resnet18_imagenet --dataset=Animal_classification --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=7,4,3,1 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label && \

######### Naive Attack #########
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=naive --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=naive --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &