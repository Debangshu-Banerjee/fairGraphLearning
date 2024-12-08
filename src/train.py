import argparse
import os

import torch

from utils.configs import init_attack_configs, init_dataset_configs, init_train_configs, init_get_extract_configs
from utils.trainer_fairgnn import FairGNNTrainer
from utils.extractor_fairgnn import FairGNNExtractor
from utils.trainer_gcn import GCNTrainer
import random
import string
import json

def generate_random_string(length):
    # Combine letters and digits to create a pool of characters
    characters = string.ascii_letters + string.digits
    # Use random.choices to generate a random string
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

def append_to_file(filename, content, folder="../output"):
    """
    Appends content to a file in the specified folder. Creates the file and folder if they do not exist.

    Args:
        filename (str): Name of the file to append to.
        content (str): Content to append to the file.
        folder (str): The folder where the file is located or should be created. Default is 'output'.
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Full path to the file
    filepath = os.path.join(folder, filename)
    
    # Open the file in append mode and write content
    with open(filepath, 'a') as file:
        file.write(content + '\n')

def train(
    args,
    dataset_configs,
    attack_configs,
    train_configs,
    random_seed_list,
    extract_configs={}
):
    folder = f"../output/training={train_configs['model']}__attackMethod={args.attack_method}/"
    while True:
        folder_subdir = generate_random_string(10)
        x = os.path.join(folder,folder_subdir)
        if not os.path.exists(x):
            folder = x
            os.makedirs(folder,exist_ok=True)
            break
    for (configs_name, configs) in [
        ("attack_configs", attack_configs),
        ("dataset_configs", dataset_configs),
        ("train_configs", train_configs)
    ]:
        with open(os.path.join(folder,f"{configs_name}.json"), "w") as file:
            json.dump(configs, file, indent=4)
    
    if train_configs["model"] in (
        "gcn",
        "gat",
        "inform_gcn",
    ):
        trainer = GCNTrainer(
            dataset_configs=dataset_configs,
            train_configs=train_configs,
            attack_configs=attack_configs,
            no_cuda=(not args.enable_cuda),
            device=(
                torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu"
            ),
            random_seed_list=random_seed_list,
            attack_method=args.attack_method,
        )
        best_val = trainer.train()
        dataset = attack_configs["dataset"]
        budget = attack_configs["perturbation_rate"]
        attack_type = attack_configs["perturbation_mode"]
        output = f"{dataset} {budget} {attack_type} {best_val}"
        append_to_file("results_standard_baseline.txt", output, folder)
    elif train_configs["model"] in ("fairgnn",):
        trainer = FairGNNTrainer(
            dataset_configs=dataset_configs,
            train_configs=train_configs,
            attack_configs=attack_configs,
            no_cuda=(not args.enable_cuda),
            device=(
                torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu"
            ),
            random_seed_list=random_seed_list,
            attack_method=args.attack_method,
        )
        best_val = trainer.train()
        dataset = attack_configs["dataset"]
        budget = attack_configs["perturbation_rate"]
        attack_type = attack_configs["perturbation_mode"]
        output = f"{dataset} {budget} {attack_type} {best_val}"
        append_to_file("results_fair_baseline.txt", output, folder)
    elif train_configs["model"] in ("fairgnnExtractor",):
        trainer = FairGNNExtractor(
            dataset_configs=dataset_configs,
            train_configs=train_configs,
            attack_configs=attack_configs,
            extract_configs=extract_configs,
            no_cuda=(not args.enable_cuda),
            device=(
                torch.device(f"cuda:{args.cuda_device}") if args.enable_cuda else "cpu"
            ),
            random_seed_list=random_seed_list,
            attack_method=args.attack_method,
        )
        best_val = trainer.extract()
        dataset = attack_configs["dataset"]
        budget = attack_configs["perturbation_rate"]
        attack_type = attack_configs["perturbation_mode"]
        output = f"{dataset} {budget} {attack_type} {best_val}"
        append_to_file("results.txt", output, folder)
    else:
        raise ValueError(
            "Model in train_configs should be one of (gcn, gat, fairgnn, inform_gcn)!"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="pokec_n",
        help="choose the dataset to attack.",
        choices=["pokec_n", "pokec_z", "bail"],
    )
    parser.add_argument(
        "--fairness",
        type=str,
        default="statistical_parity",
        help="choose the fairness definition to attack.",
        choices=["statistical_parity", "individual_fairness"],
    )
    parser.add_argument(
        "--ptb_mode",
        type=str,
        default="flip",
        help="flip or add edges.",
        choices=["flip", "add", "delete"],
    )
    parser.add_argument(
        "--ptb_rate", type=float, default=0.05, help="perturbation rate."
    )
    parser.add_argument(
        "--attack_steps", type=int, default=3, help="number of attacking steps."
    )
    parser.add_argument(
        "--attack_seed", type=int, default=25, help="random seed to set in attacker."
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        default="fate",
        help="which attacking method's poisoned dataset.",
        choices=["fate", "random", "dice_sensitive", "fagnn"],
    )
    parser.add_argument(
        "--victim_model",
        type=str,
        default="gcn",
        help="victim model to train.",
        choices=[
            "gcn",
            "fairgnn",
            "inform_gcn",
            "gat",
            "fairgnnExtractor",
        ],
    )
    parser.add_argument(
        "--hidden_dimension",
        type=int,
        default=128,
        help="hidden dimension of the victim model.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=400,
        help="number of epochs to train the victim model.",
    )
    parser.add_argument(
        "--enable_cuda", action="store_true", default=True, help="enable CUDA."
    )
    parser.add_argument(
        "--cuda_device", type=int, default=0, choices=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--random_update", action="store_true", default=True, help="enable CUDA."
    )

    args = parser.parse_args()

    dataset_configs = init_dataset_configs()
    attack_configs = init_attack_configs()
    train_configs = init_train_configs()
    extract_configs = init_get_extract_configs(random_update=args.random_update)

    attack_configs["inform_similarity_measure"] = "cosine"
    train_configs["weight_decay"] = 1e-5
    train_configs["lr"] = 1e-3

    dataset_configs["name"] = attack_configs["dataset"] = args.dataset

    attack_configs["fairness_definition"] = args.fairness
    attack_configs["perturbation_mode"] = args.ptb_mode
    attack_configs["perturbation_rate"] = args.ptb_rate
    attack_configs["attack_steps"] = args.attack_steps
    attack_configs["seed"] = args.attack_seed

    train_configs["model"] = args.victim_model
    train_configs["hidden_dimension"] = args.hidden_dimension
    train_configs["num_epochs"] = args.num_epochs

    print(f"exp config {extract_configs}")

    train(
        args=args,
        dataset_configs=dataset_configs,
        attack_configs=attack_configs,
        train_configs=train_configs,
        random_seed_list=[
            0,
            1,
            2,
            42,
            100,
        ],
        extract_configs=extract_configs
    )
