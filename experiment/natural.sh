python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.0 --devices=0,1,2,3 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.05 --devices=2,3,0,1 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.15 --devices=5,6,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=natural &

wait

python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.0 --devices=0,1,2,3 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.05 --devices=2,3,0,1 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.15 --devices=5,6,1,2 --model_seed=123456 --poison_seed=123456 --exp_name=natural &

########### TRANSFER-LEARNING ###########

python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.0 --devices=0,1,2,3 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.05 --devices=2,3,0,1 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.15 --devices=5,6,1,2 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &

wait

python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.0 --devices=0,1,2,3 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.05 --devices=2,3,0,1 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.1 --devices=4,5,1,2 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &
python main.py --recipe=naive --net=vit_face --optimization=transformer-adamw --dataset=Facial_recognition_extended --poisonkey=0-6 --alpha=0.15 --devices=5,6,1,2 --model_seed=123456 --poison_seed=123456 --scenario=transfer --exp_name=natural &