import copy
from os import path as osp

import numpy as np
import torch
import dgl
import cv2

from mmdet.datasets.builder import DATASETS

from mmocr.core import compute_f1_score
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.pipelines import sort_vertex8
from mmocr.utils import is_type_list, list_from_file

@DATASETS.register_module()
class GCNDataset(BaseDataset):
    """
    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        loader (dict): Dictionary to construct loader
            to load annotation infos.
        img_prefix (str, optional): Image prefix to generate full
            image path.
        test_mode (bool, optional): If True, try...except will
            be turned off in __getitem__.
        dict_file (str): Character dict file path.
        norm (float): Norm to map value from one range to another.
    """
    def __init__(self,
                 ann_file,
                 loader,
                 dict_file,
                 img_prefix='',
                 pipeline=None,
                 norm=10.,
                 directed=False,
                 test_mode=True,
                 **kwargs):
        super().__init__(
            ann_file,
            loader,
            pipeline,
            img_prefix=img_prefix,
            test_mode=test_mode)
        assert osp.exists(dict_file)

        self.norm = norm
        self.directed = directed
        self.dict = {
            '': 0,
            **{
                line.rstrip('\r\n'): ind
                for ind, line in enumerate(list_from_file(dict_file), 1)
            }
        }
        self.dict['UNKNOW'] = len(self.dict)
        self.dict['NULL'] = len(self.dict)
        self.key_labels_map = {1:1,3:2,5:3,7:4,9:5,11:6,13:7,15:8,17:9,19:10,21:11,23:12,
                               0:0,2:0,4:0,6:0,8:0,10:0,12:0,14:0,16:0,18:0,20:0,22:0,24:0,25:0}

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
        results['ori_texts'] = results['ann_info']['text_inds']
        results['filename'] = osp.join(self.img_prefix,
                                       results['img_info']['filename'])
        results['ori_filename'] = results['img_info']['filename']
        # a dummy img data
        results['img'] = np.zeros((0, 0, 0), dtype=np.uint8)

    def _parse_anno_info(self, annotations):
        """Parse annotations of boxes, texts and labels for one image.
        Args:
            annotations (list[dict]): Annotations of one image, where
                each dict is for one character.

        Returns:
            dict: A dict containing the following keys:

                - bboxes (np.ndarray): Bbox in one image with shape:
                    box_num * 4. They are sorted clockwise when loading.
                - relations (np.ndarray): Relations between bbox with shape:
                    box_num * box_num * D.
                - texts (np.ndarray): Text index with shape:
                    box_num * text_max_len.
                - labels (np.ndarray): Box Labels with shape:
                    box_num * (box_num + 1).
        """

        assert is_type_list(annotations, dict)
        assert len(annotations) > 0, 'Please remove data with empty annotation'
        assert 'box' in annotations[0]
        assert 'text' in annotations[0]

        boxes, texts, text_inds, text_length, labels, edges = [], [], [], [], [], []
        for ann in annotations:
            text = ann['text']
            if len(text) == 0:
                text = 'NULL'
                texts.append(text)
                text_length.append(1)
                text_ind = [self.dict[text]]
            else:
                text_length.append(len(text))
                text_ind = []
                for c in text:
                    if c not in self.dict:
                        text = text.replace(c, 'UNKNOW')
                        c = 'UNKNOW'
                    text_ind.append(self.dict[c])
                texts.append(text)

            box = ann['box']
            sorted_box = sort_vertex8(box[:8])
            sorted_box.extend([np.max(sorted_box[0::2]) - np.min(sorted_box[0::2]), np.max(sorted_box[1::2]) - np.min(sorted_box[1::2])]) # w, h
            boxes.append(sorted_box)

            text_inds.append(text_ind)
            label = ann.get('label', 0)
            labels.append(self.key_labels_map[label])
            edges.append(ann.get('edge', 0))

        ann_infos = dict(
            boxes=np.array(boxes),
            texts=np.array(texts),
            text_inds=np.array(text_inds),
            text_length=np.array(text_length),
            edges=np.array(edges),
            labels=np.array(labels))

        return self.list_to_numpy(ann_infos)

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_ann_info = self.data_infos[index]
        img_info = {
            'filename': img_ann_info['file_name'],
            'height': img_ann_info['height'],
            'width': img_ann_info['width']
        }
        ann_info = self._parse_anno_info(img_ann_info['annotations'])
        results = dict(img_info=img_info, ann_info=ann_info)

        self.pre_pipeline(results)

        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='macro_f1',
                 metric_options=dict(macro_f1=dict(ignores=[])),
                 **kwargs):
        # allow some kwargs to pass through
        assert set(kwargs).issubset(['logger'])

        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['macro_f1']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        return self.compute_macro_f1(results, **metric_options['macro_f1'])

    def compute_macro_f1(self, results, ignores=[]):
        node_preds = []
        node_gts = []
        for idx, result in enumerate(results):
            node_preds.append(result['nodes'].cpu())
            box_ann_infos = self.data_infos[idx]['annotations']
            node_gt = [self.key_labels_map[box_ann_info['label']] for box_ann_info in box_ann_infos]
            node_gts.append(torch.Tensor(node_gt))

        node_preds = torch.cat(node_preds)
        node_gts = torch.cat(node_gts).int()

        node_f1s = compute_f1_score(node_preds, node_gts, ignores)

        res = {}
        for i, s in enumerate(node_f1s):
            res[str(i)] = s
        res['macro_f1'] = node_f1s.mean()
        # return {
        #     'macro_f1': node_f1s.mean(),
        # }
        return res

    def compare_key(self, x):
        points = x[1]
        box = np.array(points)[:8].reshape(4,2)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        return center[1], center[0]

    def sorted_boxes(self, boxes, text_inds, text_length, labels, edges):
        sorted_indexs = sorted(enumerate(boxes), key=self.compare_key)
        idx = [i[0] for i in sorted_indexs]
        return boxes[idx], text_inds[idx], text_length[idx], labels[idx], edges[idx]

    def normlize(self, boxes, edge_data):
        box_min = boxes.min(0)
        box_max = boxes.max(0)
        boxes = (boxes - box_min) / (box_max - box_min)
        boxes = (boxes - 0.5) / 0.5

        edge_min = edge_data.min(0)
        edge_max = edge_data.max(0)
        edge_data = (edge_data - edge_min) / (edge_max - edge_min)
        edge_data = (edge_data - 0.5) / 0.5
        return boxes, edge_data

    def list_to_numpy(self, ann_infos):
        """Convert bboxes, relations, texts and labels to ndarray."""

        boxes, text_inds, text_length, labels, edges = ann_infos['boxes'], ann_infos['text_inds'], ann_infos['text_length'], ann_infos.get('labels', None), ann_infos.get('edges', None)
        # boxes, text_inds, text_length, labels, edges = self.sorted_boxes(boxes, text_inds, text_length, labels, edges)

        node_nums = labels.shape[0]
        edge_nums = 0
        src = []
        dst = []
        edge_data = []

        for i in range(node_nums):
            for j in range(node_nums):
                if i == j:
                    continue

                y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                w = boxes[i, 8]
                h = boxes[i, 9]

                if np.abs(y_distance) > 3 * h:
                    continue

                edge_data.append([y_distance, x_distance])
                src.append(i)
                dst.append(j)
                edge_nums += 1

        edge_data = np.array(edge_data)

        norm_boxes, edge_data = self.normlize(boxes, edge_data)
        padded_text_inds = self.pad_text_indices(text_inds)

        return dict(
            src=src,
            dst=dst,
            norm_boxes=norm_boxes,
            edge_data=edge_data,
            text_length=text_length,
            text_inds=padded_text_inds,
            labels=labels
            )

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = np.zeros((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds
