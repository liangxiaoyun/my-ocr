import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmocr.core.evaluation.ocr_metric import eval_ocr_metric

@DATASETS.register_module()
class OnlineGenerationDataset(Dataset):
    def __init__(self,
                 pipeline,
                 dataset_len,
                 test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.dataset_len = dataset_len
        self.pipeline = Compose(pipeline)
        self._set_group_flag()
        self.CLASSES = 0

    def __len__(self):
        return self.dataset_len

    def _set_group_flag(self):
        """Set flag."""
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        # results['img_prefix'] = self.img_prefix
        pass

    def prepare_train_img(self, index):
        """Get training data and annotations from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        # img_info = self.data_infos[index]
        # results = dict(img_info=img_info)
        # self.pre_pipeline(results)
        # return self.pipeline(results)
        return self.pipeline({})

    def prepare_test_img(self, img_info):
        """Get testing data from pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        return self.prepare_train_img(img_info)

    def _log_error_index(self, index):
        """Logging data info of bad index."""
        # try:
        #     data_info = self.data_infos[index]
        #     img_prefix = self.img_prefix
        #     print_log(f'Warning: skip broken file {data_info} '
        #               f'with img_prefix {img_prefix}')
        # except Exception as e:
        #     print_log(f'load index {index} with error {e}')
        print_log(f'load index {index} with error')

    def _get_next_index(self, index):
        """Get next index from dataset."""
        self._log_error_index(index)
        index = (index + 1) % len(self)
        return index

    def __getitem__(self, index):
        """Get training/test data from pipeline.

        Args:
            index (int): Index of data.

        Returns:
            dict: Training/test data.
        """
        if self.test_mode:
            return self.prepare_test_img(index)

        while True:
            try:
                data = self.prepare_train_img(index)
                if data is None:
                    raise Exception('prepared train data empty')
                break
            except Exception as e:
                print_log(f'prepare index {index} with error {e}')
                index = self._get_next_index(index)
        return data

    def format_results(self, results, **kwargs):
        """Placeholder to format result to dataset-specific output."""
        pass

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        gt_texts = []
        pred_texts = []
        for i in range(len(self)):
            item_info = self.data_infos[i]
            text = item_info['text']
            gt_texts.append(text)
            pred_texts.append(results[i]['text'])

        eval_results = eval_ocr_metric(pred_texts, gt_texts)

        return eval_results
