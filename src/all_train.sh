#!/bin/bash

training_methods=("gcn" "fairgnn" "fairgnnExtractor")
attack_methods=("dice_sensitive" "fagnn" "fate" "random")
ptb_rates=("0.05" "0.1" "0.15" "0.2" "0.25")
ptb_modes=("flip" "delete" "add")
datasets=("pokec_n" "pokec_z" "bail")


for train_method in "${training_methods[@]}"; do
    for attack_method in "${attack_methods[@]}"; do
        for ptb_rate in "${ptb_rates[@]}"; do
            for ptb_mode in "${ptb_modes[@]}"; do
                for dataset in "${datasets[@]}"; do
                    echo "Running script with $train_method, $attack_method, $ptb_rate, $ptb_mode, $dataset"
                    python train.py --dataset "$dataset" --fairness statistical_parity --ptb_mode "$ptb_mode" --ptb_rate "$ptb_rate" --attack_steps 3 --attack_seed 25 --attack_method "$attack_method" --victim_model "$train_method" --hidden_dimension 128 --num_epochs 400 --cuda_device 0
                done
            done
        done
    done
done