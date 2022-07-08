import pickle
import argparse

import h5py
import numpy as np
import pandas as pd
import os


def get_weighted_adjacency_matrix(distance_df, sensor_ids):  # 用W_s表示
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        dist_mx[sensor_id_to_ind[row[1]], sensor_id_to_ind[row[0]]] = row[2]
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    adj_mx = np.exp(-np.square(dist_mx / std))

    return sensor_ids, sensor_id_to_ind, adj_mx


def get_time_volume_matrix(data_category, save_filename, period=12 * 24 * 7):  # 用W_v表示
    # data = np.load(data_filename)['data'][:, :, 0]
    with h5py.File(f"../data/nogrid/{data_category}_data.h5", 'r') as hf:
        data_pick = hf[f'{data_category}_pick'][:]
    with h5py.File(f"../data/nogrid/{data_category}_data.h5", 'r') as hf:
        data_drop = hf[f'{data_category}_drop'][:]
    data = np.dstack((data_pick, data_drop))
    num_samples, num_nodes, _ = data.shape
    num_train = int(3001)
    num_ave = int(num_train / period) * period

    time_volume_mx = np.zeros((num_nodes, 7, 288), dtype=np.float32)
    for node in range(num_nodes):
        for i in range(7):  # 星期一~星期天
            for t in range(288):  # 一天有288个时间段  将所有星期一的288个时间段的流量求均值。同理, 所有星期二, 星期三
                time_volume = []  # i*288+t表示星期XXX的0点时数据所对应的行数
                for j in range(i * 288 + t, num_ave, period):
                    time_volume.append(data[j][node])

                time_volume_mx[node][i][t] = np.array(time_volume).mean()

    time_volume_mx = time_volume_mx.reshape(num_nodes, -1)  # (num_nodes, 7*288)

    # 计算l2-norm
    similarity_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    similarity_mx[:] = np.inf
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity_mx[i][j] = similarity_mx[j][i] = np.sqrt(np.sum((time_volume_mx[i] - time_volume_mx[j]) ** 2))

    distances = similarity_mx[~np.isinf(similarity_mx)].flatten()
    std = distances.std()
    similarity_mx = np.exp(-np.square(similarity_mx / std))  # 主对角线为0
    np.savez(save_filename, data=similarity_mx)
    return time_volume_mx, similarity_mx


def construct_T(sim_mx, threshold, filename, direct):  # 用W_V构造T,用knn原理选择每行前threshold为True
    num_nodes = sim_mx.shape[0]
    temporal_graph = np.zeros((num_nodes, num_nodes), dtype=bool)
    for row in range(num_nodes):  # 主对角线为0
        indices = np.argsort(sim_mx[row])[::-1][:threshold]  # 取top k个为True,sim_mx主对角线为0,因此top k不会出现在主对角线上
        temporal_graph[row, indices] = True

    if not direct:  # 构造对称矩阵
        temporal_graph = np.maximum.reduce([temporal_graph, temporal_graph.T])
        print('构造的时间相似性矩阵是对称的')

    # np.savez(filename, data=temporal_graph)
    return temporal_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_category', type=str, default='bike',
                        help='category of data')
    parser.add_argument('--filename_T', type=str, default='../data/nogrid/bike_graph_T.npz',
                        help='save directory of temporal graph')
    args = parser.parse_args()

    print('Constructing temporal matrix......')
    time_volume_mx, sim_mx = get_time_volume_matrix(args.data_category, args.filename_T)  # 构造时间相似性矩阵
