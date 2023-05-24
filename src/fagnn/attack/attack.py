import os

import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, Random
from deeprobust.graph.utils import (
    normalize_adj,
    preprocess,
    sparse_mx_to_torch_sparse_tensor,
    to_scipy,
)

import fagnn.structack.node_connection as nc
import fagnn.structack.node_selection as ns
from fagnn.attack.fair_attack import Fair_Attack
from fagnn.attack.fast_dice import DICE
from fagnn.attack.sacide import SACIDE
from fagnn.attack.sp_increase import (
    MetaSPI,
    RandomMetropolisHastingSPI,
    RewireMetropolisHastingSPI,
    RewireSPI,
    SPI_heuristic,
)

# from attack.metattackSA import MetattackSA
# from attack.targeted_spi import RandomSPI, NettackSPI, TargetRewireSPI
from fagnn.structack.structack import build_custom


def build_random(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return Random()


def build_dice(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return DICE()


def build_sacide(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return SACIDE()


def build_SPI_heuristic(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return SPI_heuristic(device=device)


def build_rewirespi(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return RewireSPI(device=device)


def build_iter2(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return RewireMetropolisHastingSPI(device=device)


def build_iter3(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    return RandomMetropolisHastingSPI()


def build_fair_attack(adj, features, labels, idx_train, idx_test, device):
    return Fair_Attack()


# def build_target_randomspi(adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None):
#     return RandomSPI(device=device)
#
#
# def build_target_nettackspi(adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None):
#     return NettackSPI(device=device)
#
# def build_target_rewirespi(adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None):
#     return TargetRewireSPI(device=device)


def attack_random(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_dice(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(adj, labels, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_sacide(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(adj, sens, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_SPI_heuristic(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(adj, labels, sens, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_rewirespi(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(adj, features, labels, sens, idx_train, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def attack_fair_attack(
    model,
    adj,
    features,
    labels,
    sens,
    idx_train,
    n_perturbations,
    direction,
    strategy,
    deg,
    deg_direction,
    dataset,
    idx_sens_train,
):
    model.attack(
        adj,
        features,
        labels,
        sens,
        idx_train,
        n_perturbations,
        direction,
        strategy,
        deg,
        deg_direction,
        dataset,
        idx_sens_train,
    )
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def postprocess_adj(adj):
    # adj = normalize_adj(adj)
    #     adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj.tocoo()


def attack_structack(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled
):
    model.attack(adj, n_perturbations)
    modified_adj = model.modified_adj
    return postprocess_adj(modified_adj)


def apply_perturbation(
    model_builder,
    attack,
    args,
    adj,
    features,
    labels,
    sens,
    idx_train,
    idx_val,
    idx_test,
    ptb_rate=0.05,
    cuda=False,
    seed=42,
    idx_sens_train=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:1" if cuda else "cpu")
    print(f"Device {device}")

    idx_unlabeled = np.union1d(idx_val, idx_test)

    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    print(f"n_perturbations = {n_perturbations}")
    sens = sens.long()

    if (
        model_builder == build_metattack
        or model_builder == build_metaspi
        or model_builder == build_prbcd
    ):
        adj, features, labels = preprocess(
            adj,
            sp.coo_matrix(features.cpu().numpy()),
            labels.cpu().numpy(),
            preprocess_adj=False,
        )
    elif model_builder == build_MetaDiscriminator:
        sens = torch.FloatTensor(sens).type(torch.LongTensor).to(device)
    # print(sens)
    if model_builder == build_MetaDiscriminator:
        # build the model
        model = model_builder(adj, features, sens, idx_train, idx_test, device)
    elif model_builder == build_MetaSA:
        model = model_builder(adj, features, labels, sens, idx_train, idx_test, device)
    else:
        # build the model
        model = model_builder(adj, features, labels, idx_train, idx_test, device)

    # perform the attack
    # modified_adj = attack(model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens)

    if model_builder == build_fair_attack:
        modified_adj = attack(
            model,
            adj,
            features,
            labels.to(device),
            sens.to(device),
            idx_train,
            n_perturbations,
            args.direction,
            args.strategy,
            args.deg,
            args.deg_direction,
            args.dataset,
            idx_sens_train,
        )
    else:
        modified_adj = attack(
            model,
            adj,
            features,
            labels.to(device),
            n_perturbations,
            idx_train,
            idx_unlabeled,
            sens.to(device),
        )
    return modified_adj


def build_metattack(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    lambda_ = 0

    # Setup Surrogate Model
    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    )
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    print(f"{torch.cuda.device_count()} GPUs available")
    print("built surrogate")
    model = Metattack(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=lambda_,
        lr=0.005,
    )
    print("built model")
    model = model.to(device)
    print("to device")
    return model


def build_metaspi(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    lambda_ = 1

    # Setup Surrogate Model
    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    )
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    print(f"{torch.cuda.device_count()} GPUs available")
    print("built surrogate")
    model = MetaSPI(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=lambda_,
        lr=0.01,
    )
    print("built model")
    model = model.to(device)
    print("to device")
    return model


def build_MetaDiscriminator(
    adj=None, features=None, sens=None, idx_train=None, idx_test=None, device=None
):
    lambda_ = 0.5

    # Setup Surrogate Model
    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=sens.max().item() + 1,
        nhid=16,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    )
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, sens, idx_train)
    print(f"{torch.cuda.device_count()} GPUs available")
    print("built surrogate")
    model = Metattack(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=lambda_,
        lr=0.005,
    )
    print("built model")
    model = model.to(device)
    print("to device")
    return model


def build_MetaSA(
    adj=None,
    features=None,
    labels=None,
    sens=None,
    idx_train=None,
    idx_test=None,
    device=None,
):
    lambda_ = 0.5

    # Setup Surrogate Model
    surrogate = GCN(
        nfeat=features.shape[1],
        nclass=labels.max().item() + 1,
        nhid=16,
        dropout=0.5,
        with_relu=False,
        with_bias=True,
        weight_decay=5e-4,
        device=device,
    )
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    print(f"{torch.cuda.device_count()} GPUs available")
    print("built surrogate")
    model = MetattackSA(
        model=surrogate,
        nnodes=adj.shape[0],
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device=device,
        lambda_=lambda_,
        lr=0.005,
    )
    print("built model")
    model = model.to(device)
    print("to device")
    return model


def attack_metattack(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    labels = labels.cpu().numpy()
    model.attack(
        features,
        adj,
        labels,
        idx_train,
        idx_unlabeled,
        n_perturbations,
        ll_constraint=False,
    )
    return to_scipy(model.modified_adj)


def attack_metaspi(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(
        features,
        adj,
        labels,
        sens,
        idx_train,
        idx_unlabeled,
        n_perturbations,
        ll_constraint=False,
    )
    return to_scipy(model.modified_adj)


def attack_MetaDiscriminator(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(
        features,
        adj,
        sens,
        idx_train,
        idx_unlabeled,
        n_perturbations,
        ll_constraint=False,
    )
    return to_scipy(model.modified_adj)


def attack_MetaSA(
    model, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    model.attack(
        features,
        adj,
        labels,
        sens,
        idx_train,
        idx_unlabeled,
        n_perturbations,
        ll_constraint=False,
    )
    return to_scipy(model.modified_adj)


# from rgnn_at_scale.attacks import create_attack
# from rgnn_at_scale.models.gcn import GCN as prGCN


def build_prbcd(
    adj=None, features=None, labels=None, idx_train=None, idx_test=None, device=None
):
    # model = prGCN(n_features=features.shape[1], n_classes=labels.max().item() + 1)
    # attack_params = {}  # TODO: can set attack params
    # attack_model = create_attack("PRBCD", attr=features, adj=adj, labels=labels, model=model, idx_attack=idx_test.detach().cpu().numpy(),
    #                              device=device, data_device=device, binary_attr=True,
    #                              make_undirected=True, **attack_params)
    # return attack_model
    pass


def attack_prbcd(
    adversary, adj, features, labels, n_perturbations, idx_train, idx_unlabeled, sens
):
    adversary.attack(n_perturbations)
    (
        pert_adj,
        pert_attr,
    ) = adversary.get_pertubations()  # TODO still the pert_attr should be used somehow
    return pert_adj.to_scipy()


def attack(
    args,
    ptb_rate,
    adj,
    features,
    labels,
    sens,
    idx_train,
    idx_val,
    idx_test,
    seed,
    dataset_name,
    sens_attr=None,
    idx_sens_train=None,
):
    """
    builds the attack, applies the perturbation
    # :param attack_name: random, dice, metattack, fair_attack
    :param ptb_rate: [0,1]
    :param adj: scipy_sparse
    :param features:
    :param labels:
    :param sens:
    :param idx_train:
    :param idx_val:
    :param idx_test:
    :param seed:
    :param dataset_name: required only for structack
    :return: perturbed graph (scipy_sparse)
    """

    if not os.path.exists(f"../dataset/cached_attacks/"):
        os.makedirs("../dataset/cached_attacks/")
    attack_name = args.attack_type
    print(dataset_name)
    print(attack_name)
    print(ptb_rate)
    print(seed)
    print(args.train_percent_atk)
    # if sens_attr == 'gender' and 'region_job' in dataset_name and attack_name == 'sacide':
    #     cached_filename = f'../dataset/cached_attacks/{dataset_name}_{sens_attr}_{attack_name}_{ptb_rate:.2f}_{seed}.npz'
    # elif 'fair_attack' in attack_name:
    #     cached_filename = f'../dataset/cached_attacks/{dataset_name}_{attack_name}_{args.direction}_{args.strategy}_deg{str(args.deg)}_{args.deg_direction}_{ptb_rate:.2f}_train{str(args.train_percent_atk)}_{seed}.npz'
    # else:
    #     cached_filename = f'../dataset/cached_attacks/{dataset_name}_{attack_name}_{ptb_rate:.2f}_{seed}.npz'
    # check if modified_adj of (dataset_name, attack_name, ptb_rate, seed) are stored
    # if os.path.exists(cached_filename):
    #     print(f'Perturbed adjacency matrix already exists at {cached_filename}. Loading...')
    #     modified_adj = sp.load_npz(cached_filename)
    #     print('Perturbed adjacency matrix loaded successfully!')
    #     return modified_adj
    print(f"Applying {attack_name} attack to input graph")
    builds = {
        "random": build_random,
        "dice": build_dice,
        "fagnn": build_fair_attack,
    }
    # 'y1s1-DD': build_SPI_heuristic, 'metaspi': build_metaspi,
    # 'MetaDiscriminator': build_MetaDiscriminator, 'rspis': build_rewirespi,
    # 'iter3': build_iter3, 'iter2': build_iter2,
    # 'MetaDiscriminator': build_MetaDiscriminator, 'metasa': build_MetaSA,
    # 'iter3': build_iter3, 'iter2': build_iter2, 'target_randomspi': build_target_randomspi,
    # 'target_nettackspi': build_target_nettackspi}
    attacks = {
        "random": attack_random,
        "dice": attack_dice,
        "metattack": attack_metattack,
        "sacide": attack_sacide,
        "prbcd": attack_prbcd,
        "fair_attack": attack_fair_attack,
    }
    # 'y1s1-DD': attack_rewirespi, 'metaspi': attack_metaspi,
    # 'MetaDiscriminator': attack_MetaDiscriminator, 'rspis': attack_rewirespi,
    # 'iter3': attack_rewirespi, 'iter2': attack_rewirespi,
    # 'prbcd': attack_prbcd, 'y1s1-DD-no-surrogate': attack_rewirespi,
    # 'MetaDiscriminator': attack_MetaDiscriminator, 'metasa': attack_MetaSA,
    # 'iter3': attack_rewirespi, 'iter2': attack_rewirespi, 'target_randomspi': attack_rewirespi,
    # 'target_nettackspi': attack_rewirespi}
    baseline_attacks = list(builds.keys())

    if attack_name in baseline_attacks:
        modified_adj = apply_perturbation(
            builds[attack_name],
            attacks[attack_name],
            args,
            adj,
            features,
            labels,
            sens,
            idx_train,
            idx_val,
            idx_test,
            ptb_rate=ptb_rate,
            seed=seed,
            cuda=torch.cuda.is_available(),
            idx_sens_train=idx_sens_train,
        )
        print(f"Attack finished, returning perturbed graph")
        # print(f'Storing perturbed adjacency matrix at {cached_filename}.')
        # sp.save_npz(cached_filename, modified_adj)
        print(
            "Num edges after attack: {}".format(
                int((modified_adj.nnz + modified_adj.diagonal().sum()) // 2)
            )
        )
        return modified_adj
    elif "structack" in attack_name:
        if not os.path.exists(f"../dataset/cached_structack/"):
            os.mkdir("../dataset/cached_structack/")
        selections = {
            "dg": ns.get_nodes_with_lowest_degree,
            "pr": ns.get_nodes_with_lowest_pagerank,
            "ev": ns.get_nodes_with_lowest_eigenvector_centrality,
            "bt": ns.get_nodes_with_lowest_betweenness_centrality,
            "cl": ns.get_nodes_with_lowest_closeness_centrality,
        }
        connections = {
            "katz": nc.katz_hungarian_connection,
            "comm": nc.community_hungarian_connection,
            "dist": nc.distance_hungarian_connection,
        }
        _, selection, connection = attack_name.split("_")
        modified_adj = attack_structack(
            build_custom(selections[selection], connections[connection], dataset_name),
            adj,
            features,
            labels,
            int(ptb_rate * (adj.sum() // 2)),
            idx_train,
            np.union1d(idx_val, idx_test),
        )
        print(f"Attack finished, returning perturbed graph")
        # print(f'Storing perturbed adjacency matrix at {cached_filename}.')
        # sp.save_npz(cached_filename, modified_adj)
        return modified_adj
    else:
        print("attack not recognized")
        return None