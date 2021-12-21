python pretrain.py --name pretrain 
python main.py --load --name train --ckpt pretrain/epoch_100
python main.py --load --name test --ckpt train/best
