import torch
import math
from mmcv.cnn import normal_init
from mmdet.models.builder import HEADS, build_loss
from torch import nn
from torch.nn import functional as F
import dgl
from torch.nn.utils.rnn import pack_padded_sequence


@HEADS.register_module()
class GCNHead(nn.Module):

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
        super().__init__()

        self.embedding_text = nn.Embedding(num_chars, hidden_dim)
        self.embedding_h = nn.Linear(node_input, hidden_dim)
        self.embedding_e = nn.Linear(edge_input, hidden_dim)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional)
        self.gnn_layers = nn.ModuleList(
            [GatedGCNLayer(hidden_dim, hidden_dim, dropout, graph_norm, batch_norm, residual) for _ in range(num_gnn)])

        self.MLP_layer = MLPReadout(MLP_hidden_dim, num_classes)
        self.loss = build_loss(loss)

    def init_weights(self, pretrained=False):
        normal_init(self.embedding_e, mean=0, std=0.01)

    def lstm_text_embedding(self, text, text_length):
        packed_sequence = pack_padded_sequence(text, text_length, batch_first=True, enforce_sorted=False)#将三维的输入去掉padding搞成二维的
        outputs_packed, (h_last, c_last) = self.rnn(packed_sequence)
        return h_last.mean(0)

    def position_embeding(self, h):
        node_num, hidden_dim = h.size()[:2]
        pe = torch.zeros(node_num, hidden_dim)
        position = torch.arange(0, node_num).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(h.device)
        return pe

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

        h_out = self.MLP_layer(h)
        return h_out

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim//2**l, input_dim//2**(l+1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim//2**L, output_dim, bias=True))
        self.FC_layer = nn.ModuleList(list_FC_layers)
        self.L = L
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layer[l](y)
            y=F.relu(y)
        y = self.FC_layer[self.L](y)
        return y

class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(self.num_features))
        self.beta = nn.Parameter(torch.ones(self.num_features))

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, x, graph_size):
        x_list = torch.split(x, graph_size)
        bn_list = []
        for x in x_list:
            bn_list.append(self.norm(x))

        x = torch.cat(bn_list, 0)
        return self.gamma * x + self.beta

class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = GraphNorm(output_dim)
        self.bn_node_e = GraphNorm(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + edges.src['Dh'] + edges.dst['Eh']
        edges.data['e'] = e_ij
        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)
        return {'h': h}

    def forward(self, g, h, e, snorm_n, snorm_e, graph_node_size, graph_edge_size):
        h_in = h
        e_in = e

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)

        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)

        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']
        e = g.edata['e']

        if self.graph_norm:
            h = h * snorm_n
            e = e * snorm_e

        if self.batch_norm:
            h = self.bn_node_h(h, graph_node_size)
            e = self.bn_node_e(e, graph_edge_size)

        h = F.relu(h)
        e = F.relu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        return h, e


