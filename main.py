# -*- coding = utf-8 -*-
# @Time : 2024/3/22 15:24
# @Author : cb
# @File :main.py
# @Software : PyCharm
from idea.model import Model
from idea.utils import load_and_split_data, spectral_feature_extraction
import math
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"



data_folder = "D:\\cb/DATA"
dti, fmri, labels, train_idx, test_idx, ids = load_and_split_data(data_folder)
dti = dti.to(device)
fmri = fmri.to(device)
labels = labels.to(device)

num_epochs = 500
w = 0.00005
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002,weight_decay=0.0001) # AD 0.0002

def modality_alignment_loss(x1, x2):
    # kl
    kl_div1 = torch.sum(x1 * torch.log(x1 / x2), dim=1)
    kl_div2 = torch.sum(x2 * torch.log(x2 / x1), dim=1)
    L = kl_div1 + kl_div2
    return L.mean()
#
# def compute(output, labels):
#     # 将预测的概率转换为类别（0或1）
#     preds = output.argmax(dim=1)
#     # 计算分类正确的样本数
#     correct = (preds == labels).sum().item()
#     # 计算总样本数
#     total = len(labels)
#     # 计算准确率
#     accuracy = correct / total
#
#     predicted = preds.cpu().detach().numpy()
#     label = labels.cpu().detach().numpy()
#
#     # sensitivity, specificity = sen_spe(label, predicted)
#     tn, fp, fn, tp = confusion_matrix(label, predicted).ravel()
#     # Sensitivity (Recall)
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     # Specificity
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#
#     try:
#         # probs = np.max(output.cpu().numpy(), axis=1)  # Get probability for positive class
#         auc = roc_auc_score(label, predicted)
#     except ValueError:
#         auc = 0.0
#
#     return accuracy, sensitivity, specificity, auc
# from sklearn.metrics import confusion_matrix, roc_auc_score
# import numpy as np
#
def compute(output, labels):
    # 将预测的概率转换为类别
    preds = output.argmax(dim=1)
    # 计算分类正确的样本数
    correct = (preds == labels).sum().item()
    # 计算总样本数
    total = len(labels)
    # 计算准确率
    accuracy = correct / total

    predicted = preds.cpu().detach().numpy()
    label = labels.cpu().detach().numpy()

    # 计算混淆矩阵
    cm = confusion_matrix(label, predicted)

    # 初始化每类的灵敏度和特异度列表
    sensitivity_list = []
    specificity_list = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity_list.append(sensitivity)

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)

    # 计算平均灵敏度和特异度
    mean_sensitivity = np.mean(sensitivity_list)
    mean_specificity = np.mean(specificity_list)

    # 计算多类AUC
    try:
        auc = roc_auc_score(label, predicted, multi_class='ovr')
    except ValueError:
        auc = 0.0

    # 返回总体的准确率，平均灵敏度，平均特异度，以及AUC
    return accuracy, mean_sensitivity, mean_specificity, auc


# 计算准确率
#
# import scipy.io as sio
# file_path1 = "dti1.txt"
# file_path2 = "dti2.txt"
# max_test_acc = 0.0  # 用于存储最大测试准确率
# train_losses = []
# train_accuracies = []
# test_accuracies = []
# AUC = []
# F1 = []
# F2 = []
# fu = []
# filter_matrices1 = []
# filter_matrices2 = []
#
# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     optimizer.zero_grad()
#     # output, H, f, t = model(dti, fmri, ids)
#     output, x1, x2 = model(dti, fmri, ids)
#     # output = model(dti, fmri, ids)
#     # L = attention_regularization_loss(f1, f2) + filter_smoothness_loss(H, f1, f2)
#     loss_train = criterion(output[train_idx], labels[train_idx]) + w * modality_alignment_loss(x1, x2)
#     loss_train.backward()
#     optimizer.step()
#
#     acc_train, *_ = compute(output[train_idx], labels[train_idx])
#     acc_test, sen, spe, auc = compute(output[test_idx], labels[test_idx])
#
#     # if epoch == 500:
#     #     predicted = torch.argmax(output[train_idx], dim=1)
#     #     l = labels[train_idx]
#     #     correct_indices = (predicted == l).nonzero(as_tuple=True)[0]
#     #     # 将正确预测的样本及其对应的滤波器矩阵存储起来
#     #     f_dti = f[0][correct_indices].cpu().detach().numpy()
#     #     f_fMRI = f[1][correct_indices].cpu().detach().numpy()
#     #     # for idx in correct_indices:
#     #     #     F1.append(H[0][idx].cpu().detach().numpy())
#     #     #     F2.append(H[1][idx].cpu().detach().numpy())
#     #     #     fu.append(t[idx].cpu().detach().numpy())
#     #     # mat1 = {"MCIAD": F1}
#     #     # mat2 = {"MCIAD": F2}AD
#     #     mat_dict1 = {"NCAD": f_dti}
#     #     mat_dict2 = {"NCAD": f_fMRI}
#     #     # mat_fu = {"MCIAD": fu}
#     #     sio.savemat("filter_dti.mat", mat_dict1)
#     #     sio.savemat("filter_fMRI.mat", mat_dict2)
#     #     # sio.savemat("F_dti.mat", mat1)
#     #     # sio.savemat("F_fMRI.mat", mat2)
#     #     # sio.savemat("Fu.mat", mat2)
#     # 打印训练和测试信息
#     print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train}, Train Acc: {acc_train}, Val Acc: {acc_test}, '
#           f'sen: {sen}, spe: {spe}, auc: {auc}')
#     # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train}, Train Acc: {acc_train}')
#     train_losses.append(loss_train.item())
#     test_accuracies.append(acc_test)
#     train_accuracies.append(acc_train)
#     AUC.append(auc)
#
#
#     # 更新最大测试准确率
#     if acc_test > max_test_acc:
#         max_test_acc = acc_test
#
#
# # 画出损失和准确率变化图
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 4, 1)
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 4, 2)
# plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.subplot(1, 4, 3)
# plt.plot(range(1, num_epochs + 1), test_accuracies, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.subplot(1, 4, 4)
# plt.plot(range(1, num_epochs + 1), test_accuracies, label='Val AUC')
# plt.xlabel('Epoch')
# plt.ylabel('AUC')
# plt.legend()
#
# plt.show()
# print(f'New max test accuracy: {max_test_acc}')
#

# import numpy as np

# 设置K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化一个列表，用来存储每个fold的测试准确率
test_accuracies = []
SEN = []
SPE = []
AUC = []

# 开始K折交叉验证
for fold, (train_index, test_index) in enumerate(kf.split(dti)):
    labels_train, labels_test = labels[train_index], labels[test_index]

    # 在当前fold上进行训练和测试
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        output, x1, x2 = model(dti, fmri, ids)
        m_loss = modality_alignment_loss(x1[train_index], x2[train_index])
        loss_train = criterion(output[train_index], labels_train) + w * m_loss
        loss_train.backward()
        optimizer.step()

    # 在测试集上进行预测
    model.eval()
    output, *_ = model(dti, fmri, ids)
    acc_test, sen, spe, auc = compute(output[test_index], labels_test)

    test_accuracies.append(acc_test)
    SEN.append(sen)
    SPE.append(spe)
    AUC.append(auc)

    # 打印当前fold的训练和测试信息
    print(f'Fold {fold + 1}/{kf.n_splits}, Test Acc: {acc_test}, sen: {sen}, spe: {spe}, auc: {auc}')

# 输出每个fold的测试准确率
for fold, acc in enumerate(test_accuracies, 1):
    print(f"Fold {fold} Test Accuracy: {acc}")

# 输出平均测试准确率
mean_test_accuracy = np.mean(test_accuracies)
mean_sen = np.mean(SEN)
mean_spe = np.mean(SPE)
mean_auc = np.mean(AUC)
print(f"Mean Test Accuracy: {mean_test_accuracy}")
print(f"Mean Test SPE: {mean_spe}")
print(f"Mean Test SEN: {mean_sen}")
print(f"Mean Test AUC: {mean_auc}")
