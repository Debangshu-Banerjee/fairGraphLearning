#!/bin/bash

models=("gcn" "fairgnn")
attack_methods=("fagnn" "dice" "random")
# ptb_rates=("0.05" "0.1" "0.15" "0.2" "0.25")
datasets=("pokec_n" "pokec_z" "bail")

for model in "${models[@]}"; do
    for attack_method in "${attack_methods[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "Running script with $model, $attack_method, $ptb_rate, $dataset"
            python baseline_attack.py --dataset "$dataset" --model "$model" --attack_type "$attack_method" --ptb_rate 0.05 0.1 0.15 0.2 0.25 --device 0
        done
    done
done



