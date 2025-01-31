#!/usr/bin/env python

git pull
git add .
git commit -m run
git push
commit_id=$(git rev-parse HEAD)
name="cancel nash adaptive weights"
ORIGINAL_DATASET=imagenet
TARGET_DATASET=porn
OUTPUT=results_mixed/${ORIGINAL_DATASET}_${TARGET_DATASET}/
mkdir -p ${OUTPUT}
# bash run/run.sh > output.log 2>&1
python3 ../../final.py \
    --root ${OUTPUT} \
    --commit "$commit_id" \
    --original_train_list "list/${ORIGINAL_DATASET}_train_list.txt" \
    --original_test_list "list/${ORIGINAL_DATASET}_test_list.txt" \
    --target_defense_list "./list/${TARGET_DATASET}_defense_list.txt" \
    --target_test_list "./list/${TARGET_DATASET}_test_list.txt" \
    --target_finetune_list "./list/${TARGET_DATASET}_finetune_list.txt" \
    --target ${TARGET_DATASET} \
    --workers 40 \
    --epochs 6000 \
    --start-epoch 0 --syn 0 \
    --batch-size 15 --batches 10 --gpu 2 --shots 18 --seed 40 --only_decoder 1 \
    --alpha 0.1 --beta 1 --cigma 0.1 --gamma 1 --mimic 1 \
    --coefficients '1,0,0' --disalbe_wandb 1 \
    --lr 1e-5 --fast_lr 0.01 --resolution 256 --partial 0 --finetune_epochs 1 --truly_finetune_epochs 5 \
    --momentum 0.9 --pretest 0 --unlearn 0 --threshold 4 \
    --weight-decay 1e-4 --target_feature_dir '../../code/imagenet-autoencoder-main/target_features/2000pornfeatures/feature,../../code/imagenet-autoencoder-main/target_features/pornfeatures/feature'\
    --same_path 0 \
    --arch '32x4' --maml x --mixed 1 \
    --test_iterval 50 --truly_finetune_freq 400 --finetune_number 500 \
    --notes="${name}" \
    --resume ../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_15_10_46_44/ckpts/best@loop_1299@tarloss0.072.pt
    
    #best unlearn: ../../code/imagenet-autoencoder-main/results_unlearn/imagenet_porn/5_12_22_32_38/finetuned_ckpts/1999.pt