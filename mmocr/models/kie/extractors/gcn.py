import warnings
import dgl
import torch
import mmcv
from mmdet.core import bbox2roi
from mmdet.models.builder import DETECTORS, build_roi_extractor
from mmdet.models.detectors import SingleStageDetector
from torch import nn
from torch.nn import functional as F

from mmocr.core import imshow_edge_node
from mmocr.utils import list_from_file


@DETECTORS.register_module()
class GCN(SingleStageDetector):
    """The implementation of the paper: Spatial Dual-Modality Graph Reasoning
    for Key Information Extraction. https://arxiv.org/abs/2103.14470.

    Args:
        visual_modality (bool): Whether use the visual modality.
        class_list (None | str): Mapping file of class index to
            class name. If None, class index will be shown in
            `show_results`, else class name.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7),
                     featmap_strides=[1]),
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 class_list=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.class_list = class_list

    def pipline_process(self, text_inds, text_length, src, dst, edge_data, norm_boxes, labels=None):
        graphs = []
        for one_src, one_dst, one_edge_data, one_norm_boxes in zip(src, dst, edge_data, norm_boxes):
            g = dgl.DGLGraph()
            g.add_nodes(one_norm_boxes.size(0))
            g.add_edges(one_src, one_dst)
            g.edata['feat'] = one_edge_data
            g.ndata['feat'] = one_norm_boxes
            graphs.append(g)

        tab_size_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_size_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt().to(text_inds[0].device)
        tab_size_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_size_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt().to(text_inds[0].device)
        batched_graph = dgl.batch(graphs)

        graph_node_size = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        graph_edge_size = [graphs[i].number_of_edges() for i in range(len(graphs))]

        h = batched_graph.ndata['feat']
        e = batched_graph.edata['feat']

        max_length = max([i.shape[1] for i in text_inds])
        texts = torch.cat([torch.cat([text, text.new_zeros(text.size(0), max_length - text.size(1))], -1) for text in text_inds])
        text_length = torch.cat(text_length)

        if labels is not None:
            labels = torch.cat(labels)
            return labels, batched_graph, h, e, texts, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size
        else:
            return batched_graph, h, e, texts, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size

    #'labels', 'text_inds', 'text_length', 'src', 'dst', 'edge_data', 'norm_boxes'
    def forward_train(self, img, img_metas, labels, text_inds, text_length, src, dst, edge_data, norm_boxes):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details of the values of these keys,
                please see :class:`mmdet.datasets.pipelines.Collect`.
            relations (list[tensor]): Relations between bboxes.
            texts (list[tensor]): Texts in bboxes.
            gt_bboxes (list[tensor]): Each item is the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[tensor]): Class indices corresponding to each box.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        labels, g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size = self.pipline_process(text_inds, text_length, src, dst, edge_data, norm_boxes, labels)
        node_preds = self.bbox_head.forward(g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size)
        return self.bbox_head.loss(node_preds, labels)

    def forward_test(self,
                     img,
                     img_metas,
                     text_inds, text_length, src, dst, edge_data, norm_boxes, rescale=False):
        g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size = self.pipline_process(
            text_inds, text_length, src, dst, edge_data, norm_boxes)

        node_preds = self.bbox_head.forward(g, h, e, text, text_length, snorm_n, snorm_e, graph_node_size, graph_edge_size)
        return [
            dict(
                img_metas=img_metas,
                nodes=F.softmax(node_preds, -1))
        ]
