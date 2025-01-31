python plot/plot.py -n "lr exp" -v "finetune_lr" -p exp_nodetach/lr_whole -e "Diffusion Lsoss" 
python plot/plot.py -n "optimizer exp" -v "optimizer" -p exp_nodetach/optimizer_whole -e "Diffusion Lsoss" 
python plot/plot.py -n "LoraRank exp" -v "rank" -p result_FTAT/LoRA_rank -e "Diffusion Lsoss" 