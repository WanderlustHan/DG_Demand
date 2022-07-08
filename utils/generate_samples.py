from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pickle

import yaml
import torch
import numpy as np
from tqdm import tqdm

from utils.data_container import get_data_loader, get_data_loader_samples
from utils.evaluate import masked_rmse_np
from utils.graphutils import laplaEigen
from sklearn.cluster import KMeans
import random

TF_ENABLE_ONEDNN_OPTS = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='taxi', type=str,
                    help='Type of dataset for training the model.')
parser.add_argument('--save_dir', default='../data/nogrid/taxi_samples.npz', type=str,
                    help='samples save dir')
parser.add_argument('--graph', default='../data/adj_data/adj_mx_taxi.pkl', type=str,
                    help='node nums')
parser.add_argument('--time_graph', default='../data/nogrid/taxi_graph_T.npz', type=str,
                    help='Configuration filename for restoring the time graph')
parser.add_argument('--num_nodes', default=266, type=int, help='node nums')
parser.add_argument('--eigen_k', default=30, type=int, help='k for knn')
parser.add_argument('--eigen_dim', default=16, type=int, help='dim of eigenmap')
parser.add_argument('--eigen_t', default=5.0, type=float, help='t for generate W')
parser.add_argument('--k_means_k', default=8, type=int, help='k for k-means')
args = parser.parse_args()


def _init_seed(SEED=10):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def main(conf, data_category, args):
    f = open(args.graph, 'rb')
    support = pickle.load(f)[2]
    _, eigen_adjmx_f = laplaEigen(support, args.eigen_k, args.eigen_t, args.eigen_dim)
    adj_Tmx = np.load(args.time_graph)['data']
    _, eigen_adjTmx_f = laplaEigen(adj_Tmx, args.eigen_k, args.eigen_t, args.eigen_dim)
    eigen_mx = torch.cat((eigen_adjmx_f, eigen_adjTmx_f), 1)  # 将eigenmap后得到的两个矩阵合并

    km_pred = KMeans(n_clusters=args.k_means_k).fit_predict(eigen_mx)  # k-means聚类
    cate_dict = {i: [] for i in range(args.k_means_k)}
    for i in range(km_pred.size):
        cate_dict[km_pred[i]].append(i)

    data_loader, normal = get_data_loader_samples(**conf['data'], data_category=data_category, device=device)
    tqdm_loader = tqdm(enumerate(data_loader['train']))
    positive = []
    negative = []
    for _, (x, y) in tqdm_loader:
        x, y = prepare_data(x, y)
        batch_posi_list = []
        batch_nega_list = []
        # 从同时刻同类中选取top1正样本
        for node in range(args.num_nodes):
            node_category = km_pred[node]
            min_rmse = np.inf
            posi_top = node
            for neib in cate_dict[node_category]:
                if neib != node:
                    rmse = compute_rmseloss(y[:, node], y[:, neib])
                    if rmse < min_rmse:
                        min_rmse = rmse
                        posi_top = neib
            batch_posi_list.append(posi_top)
            # 随机选择一个其他类的节点作为负样本
            nega_category = (node_category + random.randint(1, args.k_means_k - 1)) % args.k_means_k
            batch_nega_list.append(cate_dict[nega_category][random.randint(0, len(cate_dict[nega_category]) - 1)])
        positive.append(batch_posi_list)
        negative.append(batch_nega_list)
    np.savez(args.save_dir, embedding=np.array(eigen_mx), positive=np.array(positive), negative=np.array(negative))
    print("saved in ", args.save_dir)
    return positive, negative


def prepare_data(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x: shape (batch_size, num_sensor, seq_len, input_dim)
             y: shape (batch_size, num_sensor, horizon, input_dim)
    """
    x = x.permute(0, 2, 1, 3)
    y = y.permute(0, 2, 1, 3)
    return x.to(device), y.to(device)


def compute_rmseloss(y_true, y_predicted):
    return masked_rmse(y_predicted, y_true, 0.0)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


if __name__ == '__main__':
    _init_seed(64)
    con = "config-" + args.type
    data = [args.type]
    with open(os.path.join('../config', f'{con}.yaml')) as f:
        conf = yaml.safe_load(f)
    main(conf, data, args)
