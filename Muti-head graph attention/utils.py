import numpy as np
from scipy.sparse import coo_matrix
import random
import matplotlib.pyplot as plt
# from torch_geometric.utils import negative_sampling
import torch
import pandas as pd
import os
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False





def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)


# turn dense matrix into a sparse foramt
def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data

def construct_het_mat(rna_dis_mat, dis_mat):
    drug_mat=np.zeros((218, 218))
    mat1 = np.hstack((drug_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def construct_adj_mat(training_mask):
    adj_tmp = training_mask.copy()
    drug_mat = np.zeros((training_mask.shape[0], training_mask.shape[0]))
    circ_mat = np.zeros((training_mask.shape[1], training_mask.shape[1]))
    mat1 = np.hstack((drug_mat, adj_tmp))
    mat2 = np.hstack((adj_tmp.T, circ_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def calculate_loss(pred, pos_edge_idx, neg_edge_idx):
    pos_pred_socres = pred[pos_edge_idx[1], pos_edge_idx[0]]
    neg_pred_socres = pred[neg_edge_idx[1], neg_edge_idx[0]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = torch.hstack((torch.Tensor(pos_edge_idx[2]), torch.Tensor(neg_edge_idx[2])))
    loss_fun=torch.nn.BCELoss(reduction='mean')
    return loss_fun(pred_scores, true_labels)