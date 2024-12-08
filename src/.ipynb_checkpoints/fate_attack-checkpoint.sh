#!/bin/bash

ptb_modes=("flip" "delete" "add")
ptb_rates=("0.05" "0.1" "0.15" "0.2" "0.25")
datasets=("pokec_n" "pokec_z" "bail")
 
for ptb_rate in "${ptb_rates[@]}"; do
    for ptb_mode in "${ptb_modes[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Running script with $ptb_rate, $ptb_mode, $dataset"
            python fate_attack.py --dataset "$dataset" --fairness statistical_parity --ptb_mode "$ptb_mode" --ptb_rate "$ptb_rate" --attack_steps 3 --attack_seed 25 --cuda_device 0
        done
    done
done
    
