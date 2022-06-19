import torch
import torch.nn as nn
from torch.nn import Dropout

from mmcv.cnn import uniform_init, xavier_init

import mmocr.utils as utils
from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder
from mmocr.models.textrecog.layers.transformer_layer import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward

@ENCODERS.register_module()
class MASTEREncoder(BaseEncoder):
    """Implementation of encoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_

    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_do_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self,
                 d_model=512,
                 _multi_heads_count=8,
                 _dropout=0.2,
                 _MultiHeadAttention_dropout=0.1,
                 _feed_forward_size=2048,
                 _with_encoder=False,
                 **kwargs):
        super().__init__()
        assert isinstance(_dropout, float)
        assert isinstance(d_model, int)

        self.position = PositionalEncoding(d_hid=d_model, n_position=5000)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = Dropout(_dropout)
        self.attention = MultiHeadAttention(n_head=_multi_heads_count, d_model=d_model, dropout=_dropout)
        self.position_feed_forward = PositionwiseFeedForward(d_model, _feed_forward_size, _dropout, act_layer=nn.ReLU)
        self.with_encoder = _with_encoder

    def init_weights(self):
        # initialize weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

    def _generate_mask(self, _position_encode_tensor):
        target_length = _position_encode_tensor.size(1)
        return torch.ones((target_length, target_length), device=_position_encode_tensor.device)

    def forward(self, feat, img_metas=None):
        b, c, h, w = feat.shape  # （B， C， H/8, W/4）
        feat = feat.view(b, c, h * w)
        feat = feat.permute((0, 2, 1))
        output = self.position(feat)
        if self.with_encoder:
            source_mask = self._generate_mask(output)
            for i in range(self.stacks):
                normed_output = self.layer_norm(output)
                output = output + self.dropout(
                    self.attention(normed_output, normed_output, normed_output, source_mask)
                )
                normed_output = self.layer_norm(output)
                output = output + self.dropout(self.position_feed_forward(normed_output))
            output = self.layer_norm(output)

        return output
