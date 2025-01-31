python ../../diffusion_unlearn.py --alpha 1 --beta 0.01 --gpus "0,1" --epochs 5000 --rank 8 --alpha_LoRA 8 --batches 1 --root "./result_unlearn" --sample "random" --module lora --lr 0.0001 --LoRAmodule "to_q,to_k,to_v,to_out.0"

