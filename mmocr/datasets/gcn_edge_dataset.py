import copy
from os import path as osp

import numpy as np
import torch
import dgl
import cv2
import math

from mmdet.datasets.builder import DATASETS

from mmocr.core import compute_f1_score, cal_row_col_f1
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.datasets.pipelines import sort_vertex8
from mmocr.utils import is_type_list, list_from_file

@DATASETS.register_module()
class GCNEdgeDataset(BaseDataset):
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
                 edge_type='row',
                 classes=2,
                 w_thres = 2,
                 h_thres=2,
                 fix_max_edge=False,
                 max_edge_num=20,
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
        self.edge_type = edge_type
        self.w_thres = w_thres
        self.h_thres = h_thres
        self.fix_max_edge = fix_max_edge
        self.classes = classes
        self.max_edge_num = max_edge_num

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []

        results['ori_texts'] = results['ann_info']['text_inds']
        results['edge_labels'] = results['ann_info']['labels']
        results['node_num'] = len(results['ann_info']['norm_boxes'])
        results['ori_src'] = results['ann_info']['src']
        results['ori_dst'] = results['ann_info']['dst']

        results['filename'] = osp.join(self.img_prefix,
                                       results['img_info']['filename'])
        results['ori_filename'] = results['img_info']['filename']
        # a dummy img data
        results['img'] = np.zeros((0, 0, 0), dtype=np.uint8)

    def _parse_anno_info(self, annotations, file_name):
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

        boxes, texts, text_inds, text_length, row_labels, col_labels = [], [], [], [], [], []
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
            sorted_box.extend([sorted_box[4]-sorted_box[0], sorted_box[5]-sorted_box[1]])
            boxes.append(sorted_box)

            text_inds.append(text_ind)

            row_label = ann.get('row_label', None)
            if row_label is not None:
                row_label = list(map(int, row_label.split(',')))
            else:
                row_label = [None]

            col_label = ann.get('col_label', None)
            if col_label is not None:
                col_label = list(map(int, col_label.split(',')))
            else:
                col_label = [None]

            row_labels.append(row_label)
            col_labels.append(col_label)

        ann_infos = dict(
            boxes=np.array(boxes),
            texts=np.array(texts),
            text_inds=np.array(text_inds),
            text_length=np.array(text_length),
            row_labels=np.array(row_labels),
            col_labels = np.array(col_labels))

        return self.list_to_numpy(ann_infos, file_name)

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
        ann_info = self._parse_anno_info(img_ann_info['annotations'], img_ann_info['file_name'])
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
        edge_preds = [result['nodes'].cpu() for result in results]
        edge_gts = [torch.Tensor(result['img_metas'][0]['edge_labels']) for result in results]
        node_nums = [result['img_metas'][0]['node_num'] for result in results]
        srcs = [result['img_metas'][0]['ori_src'] for result in results]
        dsts = [result['img_metas'][0]['ori_dst'] for result in results]

        edge_f1s = compute_f1_score(torch.cat(edge_preds), torch.cat(edge_gts).int(), ignores)

        Ps, Rs, F1s = [], [], []
        for node_num, src, dst, gt, pred in zip(node_nums, srcs, dsts, edge_gts, edge_preds):
            scores = pred.softmax(1)
            values, pred = scores.max(1)
            P, R, F1 = cal_row_col_f1(node_num, gt, pred, src, dst, self.classes)
            Ps.apeend(P)
            Rs.append(R)
            F1s.append(F1)


        if self.classes == 2:
            return {
            'macro_f1': edge_f1s.mean(),
            'precition': np.mean([i[0] for i in Ps]),
            'recall': np.mean([i[0] for i in Rs]),
            'f1_score': np.mean([i[0] for i in F1s])
        }
        if self.classes == 3:
            return {
            'macro_f1': edge_f1s.mean(),
            'row_precition': np.mean([i[0] for i in Ps]),
            'row_recall': np.mean([i[0] for i in Rs]),
            'row_f1_score': np.mean([i[0] for i in F1s]),
            'col_precition': np.mean([i[1] for i in Ps]),
            'col_recall': np.mean([i[1] for i in Rs]),
            'col_f1_score': np.mean([i[1] for i in F1s])
        }
        elif self.classes == 5:
            return {
                'macro_f1': edge_f1s.mean(),
                'row_precition': np.mean([i[0] for i in Ps]),
                'row_recall': np.mean([i[0] for i in Rs]),
                'row_f1_score': np.mean([i[0] for i in F1s]),
                'col_precition': np.mean([i[1] for i in Ps]),
                'col_recall': np.mean([i[1] for i in Rs]),
                'col_f1_score': np.mean([i[1] for i in F1s]),
                'cross_row_precition': np.mean([i[2] for i in Ps]),
                'cross_row_recall': np.mean([i[2] for i in Rs]),
                'cross_row_f1_score': np.mean([i[2] for i in F1s]),
                'cross_col_precition': np.mean([i[3] for i in Ps]),
                'cross_col_recall': np.mean([i[3] for i in Rs]),
                'cross_col_f1_score': np.mean([i[3] for i in F1s])
            }
        else:
            print('self.classes not in [2,3,5]!!!!!!!!!!!!!!')

    def compare_key(self, x):
        points = x[1]
        box = np.array(points)[:8].reshape(4,2)
        rect = cv2.minAreaRect(box)
        center = rect[0]
        return center[1], center[0]

    def sorted_boxes(self, boxes, text_inds, text_length, row_labels, col_labels):
        sorted_indexs = sorted(enumerate(boxes), key=self.compare_key)
        idx = [i[0] for i in sorted_indexs]
        return boxes[idx], text_inds[idx], text_length[idx], row_labels[idx], col_labels[idx]

    def normlize(self, boxes, edge_data):
        box_min = boxes.min(0)
        box_max = boxes.max(0)
        eps = 1e-5
        boxes = (boxes - box_min) / (box_max - box_min + eps)
        boxes = (boxes - 0.5) / 0.5

        edge_min = edge_data.min(0)
        edge_max = edge_data.max(0)
        edge_data = (edge_data - edge_min) / (edge_max - edge_min + eps)
        edge_data = (edge_data - 0.5) / 0.5
        return boxes, edge_data

    def list_to_numpy(self, ann_infos, file_name):
        """Convert bboxes, relations, texts and labels to ndarray."""

        boxes, text_inds, text_length, row_labels, col_labels = ann_infos['boxes'], ann_infos['text_inds'], ann_infos['text_length'], ann_infos.get('row_labels', None), ann_infos.get('col_labels', None)
        boxes, text_inds, text_length, row_labels, col_labels = self.sorted_boxes(boxes, text_inds, text_length, row_labels, col_labels)

        node_nums = col_labels.shape[0]
        edge_nums = 0
        src = []
        dst = []
        edge_data = []
        edge_labels = []

        for i in range(node_nums):
            i_edge_collect = {}
            for j in range(node_nums):
                if i == j:
                    continue

                y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                w = boxes[i, 8]
                h = boxes[i, 9]

                if self.edge_type == 'row':
                    if np.abs(y_distance) > self.h_thres * h:
                        continue
                    edge_i, edge_j = row_labels[i], row_labels[j]
                    commonEle = [val for val in edge_i if (val in edge_j and val is not None)]
                    if len(commonEle) > 0:
                        edge_labels.append(1)
                    else:
                        edge_labels.append(0)

                elif self.edge_type == 'col':
                    if np.abs(x_distance) > self.w_thres * w:
                        continue
                    edge_i, edge_j = col_labels[i], col_labels[j]
                    commonEle = [val for val in edge_i if (val in edge_j and val is not None)]
                    if len(commonEle) > 0:
                        edge_labels.append(1)
                    else:
                        edge_labels.append(0)

                else:
                    if not(np.abs(y_distance) < self.h_thres * h or np.abs(x_distance) < self.w_thres * w):
                        continue

                    #同行判断
                    row_edge_i, row_edge_j = row_labels[i], row_labels[j]
                    row_commonEle = [val for val in row_edge_i if (val in row_edge_j and val is not None)]
                    if len(row_commonEle) > 0:
                        if self.classes == 5:
                            if len(row_edge_i) > 1:
                                if self.fix_max_edge:
                                    i_edge_collect[j] = {'y_distance':y_distance, 'x_distance':y_distance, 'edge_label': 3}
                                else:
                                    edge_labels.append(3)
                            else:
                                if self.fix_max_edge:
                                    i_edge_collect[j] = {'y_distance':y_distance, 'x_distance':y_distance, 'edge_label': 1}
                                else:
                                    edge_labels.append(1)
                        elif self.classes == 3:
                            if self.fix_max_edge:
                                i_edge_collect[j] = {'y_distance': y_distance, 'x_distance': y_distance,
                                                     'edge_label': 1}
                            else:
                                edge_labels.append(1)
                    else:
                        col_edge_i, col_edge_j = col_labels[i], col_labels[j]
                        col_commonEle = [val for val in col_edge_i if (val in col_edge_j and val is not None)]
                        if len(col_commonEle) > 0:
                            if self.classes == 5:
                                if len(col_edge_i) > 1:
                                    if self.fix_max_edge:
                                        i_edge_collect[j] = {'y_distance': y_distance, 'x_distance': y_distance,
                                                             'edge_label': 4}
                                    else:
                                        edge_labels.append(4)
                                else:
                                    if self.fix_max_edge:
                                        i_edge_collect[j] = {'y_distance': y_distance, 'x_distance': y_distance,
                                                             'edge_label': 2}
                                    else:
                                        edge_labels.append(2)
                            elif self.classes == 3:
                                if self.fix_max_edge:
                                    i_edge_collect[j] = {'y_distance': y_distance, 'x_distance': y_distance,
                                                         'edge_label': 2}
                                else:
                                    edge_labels.append(2)
                        else:
                            if self.fix_max_edge:
                                i_edge_collect[j] = {'y_distance': y_distance, 'x_distance': y_distance,
                                                     'edge_label': 0}
                            else:
                                edge_labels.append(0)

                if not self.fix_max_edge:
                    edge_data.append([y_distance, x_distance])
                    src.append(i)
                    dst.append(j)

            if self.fix_max_edge:
                retain_i_edge_j = []
                if len(i_edge_collect) > self.max_edge_num:
                    sorted_i_edge_collect = sorted(i_edge_collect.items(), key=lambda x: math.sqrt(x[1]['y_distance']**2+x[1]['x_distance']**2))[:self.max_edge_num]
                    retain_i_edge_j = [t[0] for t in sorted_i_edge_collect]
                for k_j, v_j in i_edge_collect.items():
                    if len(retain_i_edge_j) > 0 and k_j not in retain_i_edge_j: continue
                    edge_labels.append(v_j['edge_label'])
                    edge_data.append([v_j['y_distance'], v_j['x_distance']])
                    src.append(i)
                    dst.append(k_j)

        #TODO
        if len(edge_data) == 0:
            print(file_name, 'len(edge_data) == 0')
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
            labels=edge_labels
            )

    def pad_text_indices(self, text_inds):
        """Pad text index to same length."""
        max_len = max([len(text_ind) for text_ind in text_inds])
        padded_text_inds = np.zeros((len(text_inds), max_len), np.int32)
        for idx, text_ind in enumerate(text_inds):
            padded_text_inds[idx, :len(text_ind)] = np.array(text_ind)
        return padded_text_inds
