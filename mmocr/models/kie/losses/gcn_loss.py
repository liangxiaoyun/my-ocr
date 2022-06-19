import torch
from mmdet.models.builder import LOSSES
from mmdet.models.losses import accuracy
from torch import nn
import numpy as np

@LOSSES.register_module()
class GCNLoss(nn.Module):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """

    def __init__(self, ignore=-100, used_ohem=False, ohem=3, neg_class=[0,25]):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.ignore = ignore
        self.ohem = ohem
        self.used_ohem = used_ohem
        self.neg_class= neg_class

    def _ohem(self, pred, label):
        pred = pred.data.cpu().numpy()
        label = label.data.cpu().numpy()

        for neg_c in self.neg_class:
            if neg_c != 0:
                label[label == neg_c] = 0

        pos_num = sum(label != 0)
        neg_num = pos_num * self.ohem

        pred_value = pred[:, 1:].max(1)
        neg_score_sorted = np.sort(-pred_value[label==0])

        if neg_score_sorted.shape[0] > neg_num:
            threshold = -neg_score_sorted[neg_num - 1]
            mask = ((pred_value >= threshold) | label != 0)
        else:
            mask = label != -1

        return  torch.from_numpy(mask)


    def forward(self, node_preds, gts):
        if self.used_ohem :
            mask_label = gts.clone()
            mask = self._ohem(node_preds, gts)
            mask = mask.to(node_preds.device)
            mask_label[mask==False] = self.ignore
            loss = self.loss_node(node_preds, mask_label)
        else:
            loss = self.loss_node(node_preds, gts)

        return dict(
            loss_node=loss,
            acc_node=accuracy(node_preds, gts))
