export CUDA_VISIBLE_DEVICES=1
seed=(0 2 75 1000 63710)
for sd in ${seed[@]}
    do
    echo ------------$sd--------------
    python ../../normal_train.py --truly_finetune_epochs 20 --seed $sd --baseline scratch --finetune_lr 0.002
    python ../../normal_train.py --truly_finetune_epochs 20 --seed $sd --baseline pretrained
    python ../../normal_train.py --resume_ckpt ../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_15_10_46_44/ckpts/best@loop_1299@tarloss0.072.pt --baseline our --truly_finetune_epochs 20 --seed $sd
    echo -------------------------------------
    done