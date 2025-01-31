export CUDA_VISIBLE_DEVICES=2

# pretrained
python ../../normal_train_clip_score.py --truly_finetune_epochs 20

# scratch
python ../../normal_train_clip_score.py --baseline scratch --truly_finetune_epochs 20 --finetune_lr 0.002

# ours 
python ../../normal_train_clip_score.py --resume_ckpt ../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_15_10_46_44/ckpts/best@loop_1299@tarloss0.072.pt --baseline our --truly_finetune_epochs 20