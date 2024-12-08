Implementations of FATE attack was borrowed from the [github repo](https://github.com/jiank2/FATE) of ICLR 2024 submission "Deceptive Fairness Attacks on Graphs via Meta Learning"

## Requirements
Sttored in `requirements.txt` for pip and `fate_new.yaml` for conda.

## Running attacks
To generate the poisoned graph with FATE and baselines, go to `src/` folder and run `fate_attack.sh` and `baselines_attack.sh` bash scripts respectively.

## Running training for the victim models
To train the victim model, go to `src/` folder and run the `all_train.sh` bash script to train using vanilla GCN, fair GNN, and our fair training method.
