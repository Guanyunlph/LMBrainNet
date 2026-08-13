import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, roc_auc_score,accuracy_score,mean_absolute_error


def COR(pred, true):
    return pearsonr(true, pred)[0]


def MAE(pred, true):
    return mean_absolute_error(true, pred)


def ACC(pred, true):
    return accuracy_score(true, pred)

def AUC(pred, true):
    return roc_auc_score(true, pred)

def regre_metric(pred, true):
    corr = COR(pred, true)
    mae = MAE(pred, true)
    return corr, mae

def class_metric(pred, true):
    acc = ACC(pred, true)
    auc = AUC(pred, true)
    return acc, auc

    # def _select_metric(self,pred, true):
    
    #     true_array = np.array(true)
    #     pred_array = np.array(pred)

    #     # Determine predicted classes based on maximum value index
    #     predicted_classes = np.argmax(pred_array, axis=1)
    #     true_classes = np.argmax(true_array, axis=1)

    #     # Compute Accuracy
    #     acc = accuracy_score(true_classes, predicted_classes)

    #     # Compute AUC for the positive class (class 1)
    #     # probabilities = np.exp(pred_array) / np.sum(np.exp(pred_array), axis=1, keepdims=True)  # softmax to get probabilities
    #     # auc = roc_auc_score(true_classes, probabilities[:, 1])

    #     # Compute Sensitivity and Specificity
    #     tn, fp, fn, tp = confusion_matrix(true_classes, predicted_classes).ravel()
    #     sensitivity = tp / (tp + fn)
    #     specificity = tn / (tn + fp)

    #     return acc,sensitivity


# def RSE(pred, true):
#     return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
#     return (u / d).mean()


# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))


# def MSE(pred, true):
#     return np.mean((pred - true) ** 2)


# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))


# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))


# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))


# # Ƥ��ѷ���ϵ��
# def COR(pred, true):
#     return pearsonr(true, pred)[0]


# def ACC(true, pred):
#     sum = 0
#     for i in range(pred.shape[0]):
#         if pred[i] >= 0.5:
#             pred[i] = 1.0
#         else:
#             pred[i] = 0.0
#         if (pred[i] == true[i]):
#             sum += 1

#     return sum / true.shape[0]


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#     cor = COR(pred, true)
#     # auc = roc_auc_score(pred, true)
#     auc =0
#     acc = ACC(pred, true)

#     return mae, mse, rmse, mape, mspe, cor, acc, auc
