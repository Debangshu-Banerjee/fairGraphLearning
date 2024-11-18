import copy
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.fairgnn import FairGNN
from utils.evaluator import Evaluator
from utils.helper_functions import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairGNNExtractor:
    def __init__(
        self,
        dataset_configs,
        train_configs,
        attack_configs,
        extract_configs,
        no_cuda,
        device,
        random_seed_list,
        attack_method=None,
    ):
        # get configs
        self.dataset_configs = dataset_configs

        self.train_configs = train_configs
        self.extract_configs = extract_configs

        self.attack_configs = attack_configs
        self.attack_folds = self.attack_configs["num_cross_validation_folds"]

        self.attack_method = attack_method if attack_method else None

        # get cuda-related info
        self.no_cuda = no_cuda
        self.device = device

        # get random seed list
        self.random_seed_list = random_seed_list

    def _init_params(self, random_seed):
        # set random seed
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if not self.no_cuda:
                torch.cuda.manual_seed(self.random_seed)

        # load data
        vanilla_data, attacked_data = self._load_data()
        self.original_graph = sparse_matrix_to_sparse_tensor(
            symmetric_normalize(
                vanilla_data["adjacency_matrix"]
                + sp.eye(vanilla_data["adjacency_matrix"].shape[0])
            )
        ).to_dense()
        self.attacked_graph = sparse_matrix_to_sparse_tensor(
            symmetric_normalize(
                attacked_data["adjacency_matrix"]
                + sp.eye(attacked_data["adjacency_matrix"].shape[0])
            )
        ).to_dense()
        self.num_nodes = attacked_data["num_nodes"]
        self.num_edges = attacked_data["num_edges"]
        self.features = torch.FloatTensor(attacked_data["node_features"])
        self.labels = torch.LongTensor(attacked_data["labels"])
        self.sensitive_labels = torch.LongTensor(attacked_data["sensitive_labels"])
        self.train_idx = torch.LongTensor(attacked_data["train_idx"])
        self.val_idx = torch.LongTensor(attacked_data["val_idx"])
        self.test_idx = torch.LongTensor(
            np.setdiff1d(
                np.arange(attacked_data["adjacency_matrix"].shape[0]),
                np.union1d(attacked_data["train_idx"], attacked_data["val_idx"]),
            )
        )

        # get models
        self.attacked_model = FairGNN(
            nfeat=self.features.shape[1],
            nhid=self.train_configs["hidden_dimension"],
            dropout=self.train_configs["dropout"],
            lr=self.train_configs["lr"],
            weight_decay=self.train_configs["weight_decay"],
        )
        # self.attacked_model = copy.deepcopy(self.vanilla_model)

        # move to corresponding device
        if not self.no_cuda:
            self.original_graph = self.original_graph.to(self.device)
            self.attacked_graph = self.attacked_graph.to(self.device)
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
            self.sensitive_labels = self.sensitive_labels.to(self.device)
            self.train_idx = self.train_idx.to(self.device)
            self.val_idx = self.val_idx.to(self.device)
            self.test_idx = self.test_idx.to(self.device)
            # self.vanilla_model.to(self.device)
            self.attacked_model.to(self.device)

        # init optimizers
        # self.vanilla_model.init_optimizers()
        self.attacked_model.init_optimizers()

        # init loss
        self.utility_criterion = nn.BCEWithLogitsLoss()

        # init evaluator
        self.evaluator = Evaluator()

    def extract(self, total_epochs=20):
        for epoch in range(total_epochs):
            self._train_attacked()
            topology_budget = (
                1
                if epoch == 0
                else int(
                    (int(self.attack_configs["perturbation_rate"] * self.num_edges) - 1)
                    // 2
                    * 2
                    // (self.attack_configs["attack_steps"] - 1)
                )
            )
            self.graph_update(topology_budget=topology_budget)
        # Final fair model training
        self._train_attacked()

    def _train_vanilla(self):
        
        
        best_val = {
            "micro_f1": -1.0,
            "macro_f1": -1.0,
            "binary_f1": -1.0,
            "roc_auc": -1.0,
            "bias": 100,
        }

        ## warmup training on corrupted graph
        best_test = copy.deepcopy(best_val)
        for _ in range(self.extract_configs["warmup_num_epochs"]):
            # train
            self.vanilla_model.train()
            self.vanilla_model.optimize(
                adj=self.original_graph,
                x=self.features,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx_train=self.train_idx,
                alpha=self.train_configs["fairgnn_regularization"]["alpha"],
                beta=self.train_configs["fairgnn_regularization"]["beta"],
                retain_graph=False,
                enable_update=True,
            )
            vanilla_loss_train = self.vanilla_model.loss_classifiers.detach()
            vanilla_output = self.vanilla_model(self.original_graph, self.features)
            _ = self.evaluator.eval(
                loss=vanilla_loss_train.detach().item(),
                output=vanilla_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.train_idx,
                stage="train",
            )

            # val
            self.vanilla_model.eval()
            vanilla_output = self.vanilla_model(self.original_graph, self.features)
            vanilla_loss_val = 0
            vanilla_eval_val_result = self.evaluator.eval(
                loss=vanilla_loss_val,
                output=vanilla_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.val_idx,
                stage="validation",
            )

            # test
            if vanilla_eval_val_result["micro_f1"] > best_val["micro_f1"]:
                best_val = vanilla_eval_val_result
                vanilla_loss_test = 0
                best_test = self.evaluator.eval(
                    loss=vanilla_loss_test,
                    output=vanilla_output,
                    labels=self.labels,
                    sensitive_labels=self.sensitive_labels,
                    idx=self.test_idx,
                    stage="test",
                )
                self._save_model_ckpts(self.vanilla_model)

        
        # return best_val, best_test

    def _train_attacked(self):
        best_val = {
            "micro_f1": -1.0,
            "macro_f1": -1.0,
            "binary_f1": -1.0,
            "roc_auc": -1.0,
            "bias": 100,
        }
        best_test = copy.deepcopy(best_val)
        # warmup training
        for i in range(self.extract_configs["warmup_num_epochs"]):
            # train
            rand_int = random.randint(0, len(self.random_seed_list) -1)
            rand_seed = self.random_seed_list[rand_int]
            self._init_params(random_seed=rand_seed)
            self.attacked_model.train()
            self.attacked_model.optimize(
                adj=self.attacked_graph,
                x=self.features,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx_train=self.train_idx,
                alpha=self.train_configs["fairgnn_regularization"]["alpha"],
                beta=self.train_configs["fairgnn_regularization"]["beta"],
                retain_graph=False,
                enable_update=True,
            )
            attacked_loss_train = self.attacked_model.loss_classifiers.detach()
            print(f"iteration {i} loss: {attacked_loss_train}")
            attacked_output = self.attacked_model(self.attacked_graph, self.features)
            _ = self.evaluator.eval(
                loss=attacked_loss_train.detach().item(),
                output=attacked_output,
                labels=self.labels,
                sensitive_labels=self.sensitive_labels,
                idx=self.train_idx,
                stage="train",
            )

            # val
            # self.attacked_model.eval()
            # attacked_output = self.attacked_model(self.attacked_graph, self.features)
            # attacked_loss_val = 0

            # attacked_eval_val_result = self.evaluator.eval(
            #     loss=attacked_loss_val,
            #     output=attacked_output,
            #     labels=self.labels,
            #     sensitive_labels=self.sensitive_labels,
            #     idx=self.val_idx,
            #     stage="validation",
            # )

            # # test
            # if attacked_eval_val_result["micro_f1"] > best_val["micro_f1"]:
            #     best_val = attacked_eval_val_result
            #     attacked_loss_test = 0
            #     best_test = self.evaluator.eval(
            #         loss=attacked_loss_test,
            #         output=attacked_output,
            #         labels=self.labels,
            #         sensitive_labels=self.sensitive_labels,
            #         idx=self.test_idx,
            #         stage="test",
            #     )
                # self._save_model_ckpts(self.attacked_model)
        
        ## updating the corrupted graph iteratively

    def _hypergradient_computation(
        self,
        perturbed_graph,
        train_idx,
        val_idx,
    ):
        # initialize params
        graph_diff = torch.zeros_like(perturbed_graph)

        perturbed_graph_with_grad = perturbed_graph.detach()
        perturbed_graph_with_grad.requires_grad_(True)
        perturbed_graph_with_grad_normalized = symmetric_normalize_tensor(
            perturbed_graph_with_grad
            + torch.eye(perturbed_graph_with_grad.shape[0]).to(self.device)
        )

        perturbed_graph_normalized = symmetric_normalize_tensor(
            perturbed_graph + torch.eye(perturbed_graph.shape[0]).to(self.device)
        )
        self.features.requires_grad_()
        self.attacked_model.optimize(
            adj=perturbed_graph_with_grad,
            x=self.features,
            labels=self.labels,
            sensitive_labels=self.sensitive_labels,
            idx_train=self.train_idx,
            alpha=self.train_configs["fairgnn_regularization"]["alpha"],
            beta=self.train_configs["fairgnn_regularization"]["beta"],
            retain_graph=False,
            enable_update=False,
        )

        loss = self.attacked_model.loss_classifiers 

        # import pdb; pdb.set_trace()
        # loss = torch.tensor(0.0).to(self.device)

        # ### pre-train model ###
        # backbone = GCN(
        #     nfeat=self.num_node_features,
        #     nhid=self.attack_configs["hidden_dimension"],
        #     nclass=1,
        #     dropout=self.attack_configs["pre_train_dropout"],
        # )
        # if not self.no_cuda:
        #     backbone.to(self.device)
        # opt = torch.optim.Adam(
        #     backbone.parameters(),
        #     lr=self.attack_configs["pre_train_lr"],
        #     weight_decay=self.attack_configs["pre_train_weight_decay"],
        # )
        # backbone = self._gcn_pre_train(
        #     model=backbone,
        #     opt=opt,
        #     graph=perturbed_graph_normalized,
        #     num_epochs=self.attack_configs["pre_train_num_epochs"],
        #     train_idx=train_idx,
        # )

        # ### calculate loss ###
        # # for high-order hypergradients, change it to some number larger than 1
        # for _ in range(1):
        #     backbone.train()
        #     opt.zero_grad()
        #     output = backbone(
        #         perturbed_graph_normalized,
        #         self.features,
        #         with_nonlinearity=self.with_nonlinearity,
        #     )
        #     loss_train = self.utility_criterion(
        #         output[train_idx],
        #         self.labels[train_idx].unsqueeze(1).float(),
        #     )
        #     loss_train.backward(retain_graph=False)
        #     opt.step()

        #     # obtain bias grad on validation set
        #     output = backbone(
        #         perturbed_graph_with_grad_normalized,
        #         self.features,
        #         with_nonlinearity=self.with_nonlinearity,
        #     )
        #     if self.attack_configs["fairness_definition"] == "individual_fairness":
        #         loss = -self.fairness_criterion(output)
        #     elif self.attack_configs["fairness_definition"] == "statistical_parity":
        #         output = torch.sigmoid(output)
        #         loss = -self.fairness_criterion(
        #             output[val_idx],
        #             # labels=self.labels[val_idx],
        #             self.sensitive_labels[val_idx],
        #             bandwidth=self.attack_configs["bandwidth"],
        #             tau=self.attack_configs["tau"],
        #             # is_statistical_parity=True,
        #         )
        grad_graph = torch.autograd.grad(loss, perturbed_graph_with_grad, create_graph=True)
        grad_graph = grad_graph[0].data
        graph_diff = (
            grad_graph
            + grad_graph.permute(1, 0)
            - torch.diag(torch.diagonal(grad_graph, 0))
        )

        return graph_diff

    
    def graph_update(self, topology_budget):
        perturbed_adj = copy.deepcopy(self.attacked_graph)
        ones_graph = torch.ones(self.num_nodes, self.num_nodes)
        if not self.no_cuda:
            ones_graph = ones_graph.to(self.device)
        # perturbed graph to tensor

        perturbed_graph = torch.FloatTensor(perturbed_adj.cpu()).to(self.device)

        # get idx for unlabeled nodes
        unlabeled_idx = torch.LongTensor(
            self.train_idx.tolist() + self.val_idx.tolist() + self.test_idx.tolist()
        )

        # prepare graph
        # original_graph_normalized = sparse_matrix_to_sparse_tensor(
        #     symmetric_normalize(
        #         self.original_graph + sp.eye(self.original_graph.shape[0])
        #     )
        # ).to_dense()

        # Convert the original graph to a SciPy sparse matrix format, if it’s not already
        original_graph_sparse = sp.coo_matrix(self.original_graph.cpu())

        # Add identity matrix (eye) to the sparse matrix
        normalized_graph = symmetric_normalize(original_graph_sparse + sp.eye(original_graph_sparse.shape[0]))

        # Convert the normalized graph to a PyTorch sparse tensor
        original_graph_normalized = sparse_matrix_to_sparse_tensor(normalized_graph).to_dense()
        # put into cuda
        if not self.no_cuda:
            unlabeled_idx = unlabeled_idx.to(self.device)
            original_graph_normalized = original_graph_normalized.to(self.device)

        # get hypergradient
        graph_delta = self._hypergradient_computation(
            perturbed_graph=perturbed_graph,
            train_idx=self.train_idx,
            val_idx=unlabeled_idx,
        )

        if self.attack_configs["perturbation_mode"] == "flip":
            pass
        elif self.attack_configs["perturbation_mode"] == "delete":
            graph_delta[graph_delta < 0] = 0  # only keep the positive grad terms
        elif self.attack_configs["perturbation_mode"] == "add":
            graph_delta[graph_delta > 0] = 0  # only keep the negative grad terms


        s_adj = -graph_delta * (ones_graph - 2 * perturbed_graph.data)
        _, idx = torch.topk(s_adj.flatten(), topology_budget)
        idx_row, idx_column = np.unravel_index(
            idx.cpu().numpy(), perturbed_adj.shape
        )
        for i in range(len(idx_row)):
            perturbed_adj[idx_row[i], idx_column[i]] = (
                1 - perturbed_adj[idx_row[i], idx_column[i]]
            )
        self.attacked_graph = perturbed_adj


    def train_against_data_poisoning(self):
        b = max(0,self.extract_configs["b"])
        steps = self.extract_configs["steps"]
        for s in range(steps):
            # Train the model

            self._train_attacked()



        return

    @staticmethod
    def _get_accuracy(output, labels):
        if output.shape[1] == 1:
            preds = (output.squeeze() > 0).type_as(labels)
        else:
            preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def _load_data(self):
        # load vanilla data
        file_path = os.path.join(
            "..",
            "dataset",
            "clean",
            self.dataset_configs["name"],
            "{name}_{prefix_sensitive_attr}_sensitive_attr.pt".format(
                name=self.dataset_configs["name"],
                prefix_sensitive_attr="no"
                if self.dataset_configs["no_sensitive_attribute"]
                else "with",
            ),
        )

        vanilla_data = torch.load(file_path)

        # load attacked data
        attack_setting = "rate={rate}_mode={mode}_steps={steps}_lr={lr}_nepochs={nepochs}_seed={seed}".format(
            rate=self.attack_configs["perturbation_rate"],
            mode=self.attack_configs["perturbation_mode"],
            steps=self.attack_configs["attack_steps"],
            lr=self.attack_configs["pre_train_lr"],
            nepochs=self.attack_configs["pre_train_num_epochs"],
            seed=self.attack_configs["seed"],
        )

        folder_path = os.path.join(
            "..",
            "data",
            f"{self.attack_method}" if self.attack_method else "perturbed",
            self.attack_configs["dataset"],
            self.attack_configs["fairness_definition"],
        )

        # if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass

        attacked_data = torch.load(os.path.join(folder_path, f"{attack_setting}.pt"))

        return vanilla_data, attacked_data

    def _get_path(self, result_type="ckpts"):
        attack_setting = "rate={rate}_mode={mode}_steps={steps}_lr={lr}_nepochs={nepochs}_seed={seed}".format(
            rate=self.attack_configs["perturbation_rate"],
            mode=self.attack_configs["perturbation_mode"],
            steps=self.attack_configs["attack_steps"],
            lr=self.attack_configs["pre_train_lr"],
            nepochs=self.attack_configs["pre_train_num_epochs"],
            seed=self.attack_configs["seed"],
        )

        train_setting = "model={model}_lr={lr}_weight-decay={weight_decay}_hidden-dim={hidden_dim}_informreg={reg}".format(
            model=self.train_configs["model"],
            lr=self.train_configs["lr"],
            weight_decay=self.train_configs["weight_decay"],
            hidden_dim=self.train_configs["hidden_dimension"],
            reg=self.train_configs["inform_regularization"],
        )

        folder_path = os.path.join(
            "..",
            f"{result_type}_{self.attack_method}"
            if self.attack_method
            else result_type,
            self.attack_configs["dataset"],
            self.attack_configs["fairness_definition"],
            f"{attack_setting}",
        )

        # if not os.path.isdir(folder_path):
        try:
            os.makedirs(folder_path)
        except:
            pass

        return folder_path, train_setting

    def _save_result(self, result):
        folder, train_setting = self._get_path(result_type="result")
        file_path = os.path.join(folder, f"{train_setting}.txt")
        with open(file_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

    def _save_model_ckpts(self, model):
        folder, train_setting = self._get_path(result_type="ckpts")
        file_path = os.path.join(folder, f"{train_setting}.pt")
        torch.save(model.state_dict(), file_path)

    def _get_mean_and_std(
        self,
        result,
    ):
        if len(result) == 0:
            return result

        metrics_list = [
            "micro_f1",
            "macro_f1",
            "binary_f1",
            "roc_auc",
            "bias",
        ]

        stats = dict()
        for result_type in ["vanilla", "attacked"]:
            stats[result_type] = dict()
            stats[result_type]["eval"] = dict()
            stats[result_type]["test"] = dict()
            for metric in metrics_list:
                if result[self.random_seed_list[0]][result_type]["eval"] == {
                    "micro_f1": -1.0
                }:
                    eval_res = [-1.0]
                else:
                    eval_res = [
                        v[result_type]["eval"][metric] for k, v in result.items()
                    ]
                test_res = [v[result_type]["test"][metric] for k, v in result.items()]
                stats[result_type]["eval"][metric] = {
                    "mean": np.mean(eval_res),
                    "std": np.std(eval_res),
                }
                stats[result_type]["test"][metric] = {
                    "mean": np.mean(test_res),
                    "std": np.std(test_res),
                }
        result["stats"] = stats
        return result
