#!/usr/bin/env python

git pull
git add .
git commit -m run
git push
commit_id=$(git rev-parse HEAD)
notes="unlearn for porn"
ORIGINAL_DATASET=imagenet
TARGET_DATASET=porn
OUTPUT=results_unlearn/${ORIGINAL_DATASET}_${TARGET_DATASET}/
mkdir -p ${OUTPUT}
python3 ../../final.py \
    --root ${OUTPUT} \
    --commit "$commit_id" \
    --original_train_list "list/${ORIGINAL_DATASET}_train_list.txt" \
    --original_test_list "list/${ORIGINAL_DATASET}_test_list.txt" \
    --target_defense_list "./list/${TARGET_DATASET}_defense1_list.txt" \
    --target_test_list "./list/${TARGET_DATASET}_test_list.txt" \
    --target_finetune_list "./list/${TARGET_DATASET}_finetune_list.txt" \
    --target porn \
    --workers 40 \
    --epochs 2000 \
    --start-epoch 0 --syn 0 \
    --batch-size 20 --batches 1 --shots 55 --gpu 2 --seed 40 --only_decoder 1 \
    --alpha 0.01 --beta 1 --cigma 0.5 --gamma 0.1 --weighted 0 --mimic 1 \
    --coefficients '1,0,0' \
    --lr 5e-5 --fast_lr 0.001 --resolution 256 --partial 0 --finetune_epochs 0 --truly_finetune_epochs 10 \
    --momentum 0.9 --x 0 --pretest 0 --unlearn 1 \
    --weight-decay 2e-4 \
    --same_path 0 \
    --arch '32x4' --maml m \
    --test_iterval 100 --truly_finetune_freq 1000 \
    --notes="${notes}" --name='unlearn' \
    --resume ../../code/imagenet-autoencoder-main/assets/pretrained32x32x4.pt
   
