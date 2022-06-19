import os
import torch
import six
import lmdb
import random
import glob
import math
from PIL import Image, ImageFont, ImageFilter
import torchvision
import numpy as np
from torch.utils.data import Dataset

import mmcv
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile

from mmocr.datasets.utils.data_generation_tools import *

@PIPELINES.register_module()
class LoadTextAnnotations(LoadAnnotations):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p).astype(np.float32) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        ann_info = results['ann_info']
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = ann_info['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        gt_masks_ignore = ann_info.get('masks_ignore', None)
        if gt_masks_ignore is not None:
            if self.poly2mask:
                gt_masks_ignore = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks_ignore],
                    h, w)
            else:
                gt_masks_ignore = PolygonMasks([
                    self.process_polygons(polygons)
                    for polygons in gt_masks_ignore
                ], h, w)
            results['gt_masks_ignore'] = gt_masks_ignore
            results['mask_fields'].append('gt_masks_ignore')

        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


@PIPELINES.register_module()
class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['img'].dtype == 'uint8'

        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            img = mmcv.bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            img = mmcv.gray2bgr(img)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

@PIPELINES.register_module()
class LoadImageFromLMDB(object):
    def __init__(self, color_type='color'):
        self.color_type = color_type
        self.env = None
        self.txn = None

        orig_func = torch.utils.data._utils.worker._worker_loop

        def wl(*args, **kwargs):
            print("Running modified workers in dataloader.")
            ret = orig_func(*args, **kwargs)
            if self.env is not None:
                self.env.close()
                self.env = None
                print("Lmdb loader closed.")

        torch.utils.data._utils.worker._worker_loop = wl

    def __call__(self, results):
        lmdb_index = results['img_info']['filename']
        data_root = results['img_info']['ann_file']
        img_key = b'image-%09d' % int(lmdb_index)

        if self.env is None:
            env = lmdb.open(data_root, readonly=True)
            self.env = env
        else:
            env = self.env

        if self.txn is None:
            txn = env.begin(write=False)
            self.txn = txn
        else:
            txn = self.txn

        #read img
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            if self.color_type == 'grayscale':
                img = Image.open(buf).convert('L')
            else:
                img = Image.open(buf).convert('RGB')

            img = np.asarray(img)
        except IOError:
            raise IOError("Corrupted image for {}".format())

        results['filename'] = lmdb_index
        results['ori_filename'] = lmdb_index
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']

        return results

    def __repr__(self):
        return '{} (color_type={})'.format(self.__class__.__name__, self.color_type)

    def __del__(self):
        print('DEL!!!')
        if self.env is not None:
            self.env.close()

@PIPELINES.register_module()
class RandomLoadString(object):
    def __init__(self, file_path=None, string_list=None, string_gen=None):
        if file_path is not None:
            with open(file_path, 'r') as f:
                string_list = f.readlines()
                string_list = [s.strip() for s in string_list]
        if string_list is not None:
            self.string_list = string_list
            self.iter = self._gen()
        else:
            self.iter = string_gen()

    def _gen(self):
        while True:
            random.shuffle(self.string_list)
            if len(self.string_list) == 0:
                raise RuntimeError
            for n in  self.string_list:
                yield n

    def __call__(self, results):
        string = next(self.iter)
        results['generate_text'] = string
        return results

@PIPELINES.register_module()
class RandomInsert(object):
    def __init__(self, alpha_path=None, alphabet=None, prob=0.1):
        self.prob = prob
        self.alpha_path = alpha_path
        if alpha_path is not None:
            alphabet = read_txt(alpha_path)
        self.character = txt_generator((list(alphabet)), shufle=True, length=(float('inf')))

    def __call__(self, results):
        text = results['generate_text']
        rand = [random.random() for _ in text]
        out = []
        for s, r in zip(text, rand):
            if r < self.prob:
                out.append(next(self.character))
            out.append(s)
        out = ''.join(out)
        results['generate_text'] = out
        return results

@PIPELINES.register_module()
class RandomReplace(object):
    def __init__(self, alpha_path=None, alphabet=None, prob=0.1):
        self.prob = prob
        self.alpha_path = alpha_path
        if alpha_path is not None:
            alphabet = read_txt(alpha_path)
        self.character = txt_generator((list(alphabet)), shufle=True, length=(float('inf')))

    def __call__(self, results):
        text = results['generate_text']
        rand = [random.random() for _ in text]
        out = []
        for s, r in zip(text, rand):
            if r < self.prob:
                out.append(next(self.character))
            else:
                out.append(s)
        out = ''.join(out)
        results['generate_text'] = out
        return results

@PIPELINES.register_module()
class OnlinePartimgGenerate(Dataset):
    def __init__(self, material_path='', max_len=30, max_fonts_per_img=1, font_size_range=[10,20], color_type='grayscale'):
        self._setup(material_path)
        self.material_path = material_path
        self.max_len = max_len
        self.max_fonts_per_img = max_fonts_per_img
        self.font_size_range = font_size_range
        self.color_type = color_type
        assert color_type in ('rgb', 'grayscale')

    def _setup(self, material_path):
        self.bg_img_names = glob.glob(os.path.join(material_path, 'bgs/*.*'))
        font_ch_names = glob.glob(os.path.join(material_path, 'fonts/*.ttf'))
        font_en_names = glob.glob(os.path.join(material_path, 'fonts/*.ttf'))
        self.font_ch_names = txt_generator(font_ch_names, length=(float('inf')))
        self.font_en_names = txt_generator(font_en_names, length=(float('inf')))
        self.support_chars = {}
        for name in font_ch_names + font_en_names:
            self.support_chars[name] = ''.join([chr(pair[0]) for pair in get_supported_chars_iter(os.path.join(material_path, name))])
        self.bg_generator = bg_generator((self.bg_img_names), cache=True)

    def __call__(self, results):
        state = self.generate_random_state(results)
        results['generation_state'] = state
        image, text = self.generate_example(state)

        if self.color_type == 'rgb':
            image = image.convert('RGB')
        elif self.color_type == 'grayscale':
            image = image.convert('L')

        img = np.asarray(image)
        results['img_info'] = {'filename':'', 'ann_file':self.material_path, 'text':text}
        results['img_prefix'] = ''
        results['text'] = text
        results['filename'] = ''
        results['ori_filename'] = ''
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']

        return results

    def generate_random_state(self, pipline_results):
        results = {}
        results['font_ch_name'] = [next(self.font_ch_names) for _ in range(random.randint(1, self.max_fonts_per_img))]
        results['font_en_name'] = [next(self.font_en_names) for _ in range(random.randint(1, self.max_fonts_per_img))]
        results['font_size'] = [random.randint(*self.font_size_range) for _ in results['font_ch_name'] + results['font_en_name']]
        results['bgs'] = next(self.bg_generator)
        results['inv_scale'] = random.random() * 5 + 1
        results['bg_transpose'] = random.random() < 0.0
        results['bg_rotate_angle'] = random.random() * 10 - 5
        results['text_color'] = random.randint(0, 150)
        results['text_rotate_angle'] = random.random() * 5 - 2.5
        results['do_gaussion_filter'] = random.random() < 0.5
        results['gaussion_blur'] = 0 if random.random() < 0.9 else random.random()  * 1
        results['resize_factor'] = random.random() * 0.6 - 0.5 + 1
        results['padding'] = tuple([int(random.random() * 8 - 1),
                                    int(random.random() * 8 - 1),
                                    int(random.random() * 8),
                                    int(random.random() * 8 - 2)])
        results['do_resize'] = random.random() < 0.5
        methods = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS]
        random.shuffle(methods)
        results['scale_down_method'] = methods[0]
        random.shuffle(methods)
        results['scale_up_method'] = methods[0]
        results['generate_text'] = pipline_results['generate_text'][:self.max_len-1]
        results['spacing'] = [random.randint(0, 4) for _ in results['generate_text']]

        return results

    def generate_example(self, results):
        text = results['generate_text']
        font_names = results['font_en_name'] + results['font_ch_name']
        char_to_font = []
        for c in text:
            if is_chinese(c):
                char_to_font.append(random.randint(0, len(results['font_ch_name']) - 1) + len(results['font_en_name']))
            else:
                char_to_font.append(random.randint(0, len(results['font_en_name']) - 1))
        font_size = results['font_size']
        fonts = [ImageFont.truetype(os.path.join(self.material_path, fn), fs) for fn, fs in zip(font_names, font_size)]

        for c, font_id in zip(text, char_to_font):
            if c not in self.support_chars[font_names[font_id]]:
                text = text.replace(c, '')
        if len(text) == 0 and len(results['generate_text']) != 0:
            pass
        elif len(text) == 0 and len(results['generate_text']) == 0:
            pass

        image, path = results['bgs']
        size = image.size[::-1]
        inv_scale = results['inv_scale']
        image = torchvision.transforms.Resize([math.ceil(size[0] / inv_scale), math.ceil(size[1] / inv_scale)])(image)
        if results['bg_transpose']:
            image = image.transpose(Image.ROTATE_90)
        rotate_angle = results['bg_rotate_angle']
        image = image.rotate(rotate_angle)
        color = results['text_color']
        rotation_angle = results['text_rotate_angle']
        resize_factor = results['resize_factor']
        padding = results['padding']
        spacing = results['spacing']
        image = get_txt_patch(image, text if len(text) != 0 else ' ', fonts=fonts, char_to_font=char_to_font, x_spacing=spacing,
                              rotation_angle=rotation_angle, padding=padding, fill=(color, color, color))

        if results['do_gaussion_filter']:
            image = image.filter(ImageFilter.GaussianBlur(radius=(results['gaussion_blur'])))
        if results['do_resize']:
            image = image.resize((int(resize_factor * image.size[0]),
                                  int(resize_factor * image.size[1])), results['scale_down_method'])
            image = image.resize((image.size[0], image.size[1]), results['scale_up_method'])
        return (image, text)


@PIPELINES.register_module()
class LoadGCNAnnotations(object):
    def __init__(self):
        pass

    def __call__(self, results, keys=['labels','text_inds','text_length','norm_boxes','src','dst','edge_data']):
        ann_info = results['ann_info']
        for k in keys:
            results[k] = ann_info[k]
        return results