import torch
import math
from mmcv.cnn import normal_init
from mmdet.models.builder import HEADS, build_loss
from torch import nn
from torch.nn import functional as F
import dgl
from torch.nn.utils.rnn import pack_padded_sequence
from .gcn_head import GCNHead

@HEADS.register_module()
class GCNEdgeHead(GCNHead):
    def __init__(self,
                 num_chars=92,
                 hidden_dim=512,
                 MLP_hidden_dim=512,
                 node_input=10,
                 edge_input=2,
                 num_gnn=8,
                 num_classes=26,
                 dropout=0.,
                 bidirectional=True,
                 graph_norm=True,
                 batch_norm=True,
                 residual=True,
                 loss=dict(type='GCNLoss'),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(num_chars,hidden_dim,MLP_hidden_dim,node_input,edge_input,num_gnn,num_classes,dropout,bidirectional,graph_norm,batch_norm,residual,loss,train_cfg,test_cfg)

    def forward(self, g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size):
        '''
        :param g: dgl
        :param h:  (box_num, 10)
        :param e: (edge_num, 2)
        :param text: (box_num, max_length)
        :param text_length: (box_num)
        :param snorm_n: (box_num,1)
        :param snorm_e: (box_num,1)
        :param graph_node_size: (box_num)
        :param graph_edge_size: (box_num)
        :param x:
        :return:
        '''
        h = self.embedding_h(h)  # all_N, hidden_dim
        e = self.embedding_e(e)

        text = text.long()
        text_embedding = self.embedding_text(text)
        text_embedding = self.lstm_text_embedding(text_embedding, text_length)  # all_N, max_length, hidden_dim
        text_embedding = F.normalize(text_embedding)

        pe = self.position_embeding(h)

        h = h + text_embedding + pe
        for gnn_layer in self.gnn_layers:
            h, e = gnn_layer(g, h, e, snorm_n, snorm_e, graph_node_size, graph_edge_size)

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            e = self.MLP_layer(e)
            return {'e':e}

        g.apply_edges(_edge_feat)
        return g.edata['e']

