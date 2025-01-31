
#--------------------------------Optimizer---------------------------------
python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46_sota/checkpoints/ckpt_4499 --FTAT_epoch 1 --optimizer 'adam'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'adade'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'rms'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'nes'
#--------------------------------Optimizer---------------------------------

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer_whole" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTwhole" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer_whole" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTwhole" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'adam'


python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer_whole" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTwhole" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'adade'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer_whole" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTwhole" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'rms'

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/Optimizer_whole" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTwhole" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'nes'
--------------------------------Optimizer---------------------------------

--------------------------------FT lr---------------------------------

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.0001

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.001

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.002

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.01

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.00001

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/FT_lr" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/9_16_15_6_46/checkpoints/ckpt_4499/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd' --finetune_lr 0.00005
--------------------------------FT lr---------------------------------

--------------------------------Lora rank---------------------------------

python ../../diffusion_FT.py --gpus "2" --rank 8 --alpha_LoRA 8 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'
python ../../diffusion_FT.py --gpus "2" --rank 16 --alpha_LoRA 16 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'
python ../../diffusion_FT.py --gpus "2" --rank 32 --alpha_LoRA 32 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'
python ../../diffusion_FT.py --gpus "2" --rank 64 --alpha_LoRA 64 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'
python ../../diffusion_FT.py --gpus "2" --rank 128 --alpha_LoRA 128 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'
python ../../diffusion_FT.py --gpus "2" --rank 256 --alpha_LoRA 256 --root "./result_FTAT/LoRA rank" --LoRAmodule "to_q,to_k,to_v,to_out.0" --strategy "FTLoRA" --checkpoint ../../code/bpalm/results_antift/debug/9_21_16_48_16/checkpoints/ckpt_999/LoRA_unet --FTAT_epoch 2 --optimizer 'sgd'






