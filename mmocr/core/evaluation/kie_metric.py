import torch
import numpy as np


def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score of prediction.

    Args:
        preds (Tensor): The predicted probability NxC map
            with N and C being the sample number and class
            number respectively.
        gts (Tensor): The ground truth vector of size N.
        ignores (list): The index set of classes that are ignored when
            reporting results.
            Note: all samples are participated in computing.

     Returns:
        The numpy list of f1-scores of valid classes.
    """
    C = preds.size(1)
    classes = torch.LongTensor(sorted(set(range(C)) - set(ignores)))
    hist = torch.bincount(
        gts * C + preds.argmax(1), minlength=C**2).view(C, C).float()
    diag = torch.diag(hist)
    recalls = diag / hist.sum(1).clamp(min=1)
    precisions = diag / hist.sum(0).clamp(min=1)
    f1 = 2 * recalls * precisions / (recalls + precisions).clamp(min=1e-8)
    return f1[classes].cpu().numpy()

def search(i, A, L, class_num=1):
    if i in L:
        return L

    L.append(i)
    for j in range(A.shape[0]):
        if i == j:
            A[i, j] = 0
            continue

        if A[i, j] == class_num:
            A[i, j] = 0
            A[j, i] = 0
            L = search(j, A, L, class_num=class_num)
    return list(set(L))

def get_same_class_index(A, class_num, node_nums):
    same_class_indexs = []
    while True:
        no_none_node = np.where(np.sum(A == class_num, axis=1) > 0)[0]
        if len(no_none_node) == 0:
            break

        next_i = no_none_node[0]
        L = []
        same_class_index = search(next_i, A, L, class_num=class_num)
        same_class_indexs.append(sorted(same_class_index))

    for i in range(node_nums):
        if i not in same_class_indexs:
            node_nums.append([i])
    return same_class_indexs

def cal_f1(gt, pred):
    TP_FP = len(pred)
    TP_FN = len(gt)
    TP = 0
    eps = 1e-5

    for i in pred:
        if i in gt:
            TP += 1

    precition = TP / float(TP_FP + eps)
    recall = TP / float(TP_FN + eps)
    F1 = 2 * recall * precition / (recall + precition + eps)
    return (precition, recall, F1)

def cal_row_col_f1(node_nums, batch_label, pred, src, dst, n_classes=2):
    precition_row, recall_row, right_row = 0, 0, 0
    precition_col, recall_col, right_col = 0, 0, 0
    precition_more_row, recall_more_row, right_more_row = 0, 0, 0
    precition_more_col, recall_more_col, right_more_col = 0, 0, 0

    gt_A = np.zeros((node_nums, node_nums))
    labels = [i for i in batch_label.numpy().astype(int)]
    pred_A = np.zeros((node_nums, node_nums))
    pred_labels = [i for i in pred.numpy().astype(int)]

    for i in range(len(labels)):
        if labels[i] != 0:
            gt_A[src[i], dst[i]] = labels[i]
            # gt_A[dst[i], src[i]] = labels[i]

    for i in range(len(pred_labels)):
        if pred_labels[i] != 0:
            pred_A[src[i], dst[i]] = pred_labels[i]
            # pred_A[dst[i], src[i]] = pred_labels[i]

    #row calculate
    class_num = 1
    gt_rows = get_same_class_index(gt_A, class_num, node_nums)
    pred_rows = get_same_class_index(pred_A, class_num, node_nums)

    # col calculate
    if n_classes == 3 or n_classes == 5:
        class_num = 2
        gt_cols = get_same_class_index(gt_A, class_num, node_nums)
        pred_cols = get_same_class_index(pred_A, class_num, node_nums)

    # cross rows,cols
    if n_classes == 5:
        gt_cross_rows = [[i] for i in np.where(np.sum(gt_A == 3, axis=1) > 0)[0]]
        gt_cross_cols = [[i] for i in np.where(np.sum(gt_A == 4, axis=1) > 0)[0]]
        pred_cross_rows = [[i] for i in np.where(np.sum(pred_A == 3, axis=1) > 0)[0]]
        pred_cross_cols = [[i] for i in np.where(np.sum(pred_A == 4, axis=1) > 0)[0]]

    #cal acc
    row_result = cal_f1(gt_rows, pred_rows)
    if n_classes == 2:
        return [row_result[0]], [row_result[1]], [row_result[2]]
    col_result = cal_f1(gt_cols, pred_cols)
    if n_classes == 3:
        return [row_result[0], col_result[0]], [row_result[1], col_result[1]], [row_result[2],col_result[2]]
    cross_row_result = cal_f1(gt_cross_rows, pred_cross_rows)
    cross_col_result = cal_f1(gt_cross_cols, pred_cross_cols)
    return [row_result[0], col_result[0], cross_row_result[0], cross_col_result[0]], [row_result[1], col_result[1], cross_row_result[1], cross_col_result[1]], [row_result[2],col_result[2], cross_row_result[2], cross_col_result[2]]
