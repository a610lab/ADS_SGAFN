# -*- coding = utf-8 -*-
# @Time : 2024/3/22 10:13
# @Author : cb
# @File :utils.py
# @Software : PyCharm

import random
from sklearn.metrics.pairwise import pairwise_kernels
import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GAT
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_and_split_data(data_folder, test_size=0.1, random_state=1111):
    ad_folder = os.path.join(data_folder, 'AD')
    nc_folder = os.path.join(data_folder, 'NC')
    # mci_folder = os.path.join(data_folder, 'MCI')

    ad_subjects = [os.path.join(ad_folder, file) for file in os.listdir(ad_folder)]
    nc_subjects = [os.path.join(nc_folder, file) for file in os.listdir(nc_folder)]
    # mci_subjects = [os.path.join(mci_folder, file) for file in os.listdir(mci_folder)]
    #
    subjects = ad_subjects + nc_subjects
    labels = [1] * len(ad_subjects) + [0] * len(nc_subjects)
    # subjects = ad_subjects + nc_subjects + mci_subjects
    # labels = [2] * len(ad_subjects) + [0] * len(nc_subjects) + [1] * len(mci_subjects)

    # 记录样本ID信息
    ids = [os.path.splitext(os.path.basename(sub))[0] for sub in subjects]

    # 打乱subjects和对应的labels
    combined = list(zip(subjects, ids, labels))  # 增加了ids信息
    random.seed(random_state)
    random.shuffle(combined)
    subjects[:], ids[:], labels[:] = zip(*combined)

    # 加载所有数据
    dti, fmri, all_labels = load_data(subjects, labels)

    # 划分索引
    train_idx, test_idx, _, _ = train_test_split(
        range(len(dti)), all_labels, test_size=test_size,
        random_state=random_state, stratify=all_labels
    )

    return dti, fmri, all_labels, train_idx, test_idx, ids


# 数据加载函数
def load_data(subjects, labels):
    dti_features = []
    dti_ts = []
    fmri_features = []
    fmri_ts = []

    for subject, label in zip(subjects, labels):
        data = scipy.io.loadmat(subject)

        dti_matrix = data['DTI']
        fmri_matrix = data['fMRI']

        dti_features.append(dti_matrix)
        fmri_features.append(fmri_matrix)

    dti = np.array(dti_features)
    fmri = np.array(fmri_features)

    dti_features = torch.from_numpy(dti).float()
    fmri_features = torch.from_numpy(fmri).float()
    labels = torch.tensor(labels).long()

    return dti_features, fmri_features, labels


# def build_adjacency_matrix(features, ids):
#     num_samples, num_features = features.shape
#
#     # 从Excel文件读取对应的sex和age信息
#     df = pd.read_excel("D:\\cb/DATA/s.xlsx")
#     df.set_index('ids', inplace=True)
#
#     sex_values = df.loc[ids, 'Sex'].values
#     age_values = df.loc[ids, 'Age'].values
#
#     # 构建sex矩阵和age矩阵
#     sex_matrix = (sex_values[:, np.newaxis] == sex_values[np.newaxis, :]).astype(int)
#     age_diff_matrix = np.abs(age_values[:, np.newaxis] - age_values[np.newaxis, :]) <= 2
#     age_matrix = age_diff_matrix.astype(int)
#
#     # 合并两个矩阵并取平均
#     combined_matrix = (sex_matrix + age_matrix) / 2.0
#     # combined_matrix = np.multiply(sex_matrix, age_matrix)
#
#     # similarity_matrix = cosine_similarity(features)
#     sigma = 1.0  # 高斯核的带宽参数
#     kernel = 'rbf'  # 径向基函数，即高斯核
#     similarity_matrix = pairwise_kernels(features, metric=kernel, gamma=1 / (2 * sigma ** 2))
#
#     feature_similarity_matrix = np.multiply(similarity_matrix, combined_matrix)
#     threshold = 0.7
#     feature_similarity_matrix[np.abs(feature_similarity_matrix) < threshold] = 0
#     feature_similarity_matrix[np.abs(feature_similarity_matrix) >= threshold] = 1
#     # 手动构建 edge_index 和 edge_attr
#     edge_index = []
#     edge_attr = []
#
#     for i in range(num_samples):
#         for j in range(num_samples):
#             weight = feature_similarity_matrix[i, j]
#             if weight != 0:
#                 edge_index.append([i, j])
#                 # edge_attr.append(weight.item())
#     # edge_weight = feature_similarity_matrix[feature_similarity_matrix != 0]
#     # edge_index = feature_similarity_matrix.nonzero(as_tuple=False)
#
#     data = Data(x=torch.tensor(features, dtype=torch.float),  # 特征信息
#                 edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),  # 边的索引  # 边的权重（如果有）
#                 )
#
#     # return torch.tensor(features).cuda(), torch.tensor(feature_similarity_matrix, dtype=torch.float).cuda()
#     return data


def spectral_feature_extraction(adj_matrices, threshold=0):
    num_samples, num_rows, _ = adj_matrices.size()
    low_freq_eigenvalues = []
    low_freq_eigenvectors = []
    wavelet = []
    m = []

    for i in range(num_samples):
        # 计算每个样本的度矩阵和邻接矩阵
        I = torch.eye(adj_matrices[i].size(0)).cuda()
        similarity_matrix = F.cosine_similarity(adj_matrices[i].unsqueeze(1), adj_matrices[i].unsqueeze(0), dim=2) - I
        # matrices = similarity_matrix
        # matrices = adj_matrices[i] * 0.1 + similarity_matrix * 0.9
        # matrices = adj_matrices[i] * 0.2 + similarity_matrix * 0.8
        # matrices = adj_matrices[i] * 0.3 + similarity_matrix * 0.7
        # matrices = adj_matrices[i] * 0.4 + similarity_matrix * 0.6
        # matrices = adj_matrices[i] * 0.5 + similarity_matrix * 0.5
        # matrices = adj_matrices[i] * 0.6 + similarity_matrix * 0.4
        # matrices = adj_matrices[i] * 0.7 + similarity_matrix * 0.3
        # matrices = adj_matrices[i] * 0.8 + similarity_matrix * 0.2
        matrices = adj_matrices[i] * 0.9 + similarity_matrix * 0.1
        # matrices = adj_matrices[i]

        matrices = torch.where(matrices > threshold, 1, 0)

        degree_matrix = torch.diag(matrices.sum(dim=1))
        laplacian_matrix = degree_matrix - matrices

        # 转换数据类型为浮点数
        laplacian_matrix = laplacian_matrix.float()

        # 对拉普拉斯矩阵进行谱分解
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix, UPLO='L')
        # eigenvalues, eigenvectors = torch.linalg.eigh(adj_matrices[i], UPLO='L')
        # U, S, V = torch.svd(laplacian_matrix)

        # 选择特征值和对应的特征向量
        low_eigenvalues = eigenvalues[-90:]
        low_eigenvectors = eigenvectors[:, -90:]

        low_freq_eigenvalues.append(low_eigenvalues)

        low_freq_eigenvectors.append(low_eigenvectors)
        wavelet.append(laplacian_matrix)
        m.append(matrices)

    eigenvalues1 = torch.stack(low_freq_eigenvalues)
    eigenvectors1 = torch.stack(low_freq_eigenvectors)
    L = torch.stack(wavelet)
    ma = torch.stack(m)

    return eigenvectors1, eigenvalues1

