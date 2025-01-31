# vae_ckpt=../../code/imagenet-autoencoder-main/original_syn_results_decoder/imagenet_porn/4_18_15_33_1/ckpts/loop7999_ori0.002_ft0.364_qloss8e-05.pt
# vae_ckpt=../../code/imagenet-autoencoder-main/results_unlearn/imagenet_porn/5_12_22_32_38/ckpts/best@loop_1999@tarloss0.089.pt
vae_ckpt=../../code/imagenet-autoencoder-main/results_mixed/imagenet_porn/6_6_20_58_50/ckpts/best@loop_899@tarloss0.087.pt

# ckpt=../diffusion/sd-v1-4-full-ema.ckpt
# ckpt=../diffusion/miniSD.ckpt
# ckpt=../../code/imagenet-autoencoder-main/logs/2024-06-13T23-07-45_finetune/checkpoints/epoch=000018.ckpt
ckpt=../../code/imagenet-autoencoder-main/logs/2024-06-16T10-50-07_finetune/checkpoints/last.ckpt
gpu=0
echo 'eval porn'
CUDA_VISIBLE_DEVICES=$gpu python ../../test.py --H 256 --W 256 --ddim_eta 0.0 --n_samples 11 --n_iter 2 --scale 7.5 --ddim_steps 100  --ckpt $ckpt --config models/v1-inference.yaml --from-file ../../code/imagenet-autoencoder-main/list/testpornprompts2000.txt --vae $vae_ckpt --seed 66 --skip_grid --outdir porn

echo 'eval normal'
CUDA_VISIBLE_DEVICES=$gpu python ../../test.py --H 256 --W 256 --ddim_eta 0.0 --n_samples 4 --n_iter 2 --scale 10.0 --ddim_steps 100  --ckpt $ckpt --config models/v1-inference.yaml --from-file ../../code/imagenet-autoencoder-main/list/eval_imagenet_small.txt --vae $vae_ckpt --skip_grid --outdir normal --seed 0