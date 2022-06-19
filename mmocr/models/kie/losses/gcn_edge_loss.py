import torch
from mmdet.models.builder import LOSSES
from mmdet.models.losses import accuracy
from torch import nn

from .gcn_loss import GCNLoss

@LOSSES.register_module()
class GCNEdgeLoss(GCNLoss):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """
    def forward(self, edge_preds, gts):
        if self.used_ohem :
            mask_label = gts.clone()
            mask = self._ohem(edge_preds, gts)
            mask = mask.to(edge_preds.device)
            mask_label[mask==False] = self.ignore
            loss = self.loss_node(edge_preds, mask_label)
        else:
            loss = self.loss_node(edge_preds, gts)

        return dict(
            loss_edge=loss,
            acc_edge=accuracy(edge_preds, gts))
