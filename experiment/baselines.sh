######### Dirty-Label #########
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=sunglasses --poisonkey=9-5 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=dirty-label --scenario=finetuning --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
python main.py --recipe=dirty-label --scenario=transfer --dataset=Facial_recognition_extended --trigger=real_beard --poisonkey=6-1 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=dirty_label &
