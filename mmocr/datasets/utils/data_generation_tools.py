import os
import math
import copy
import random
import numpy as np
from collections import Iterable
from PIL import Image, ImageDraw

from itertools import chain
from fontTools.ttLib import TTCollection, TTFont
from shapely.affinity import rotate
from shapely.geometry import MultiPoint
from torchvision.transforms import RandomCrop, Pad
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool()
map = pool.map

def read_txt(path, encode='utf-8'):
    with open(path, 'rb') as f:
        return map(lambda byte: byte.decode(encode).strip(), f.readlines())

def txt_generator(library, length=None, shufle=False, transforms=None):
    '''
    生成器，从字符表中
    :param library: list，需要从中选择元素的列表
    :param length: int，生成器长度
    :param shufle: bool， 是否打乱library
    :param transforms: list， 包含若干函数，这些函数均只接受一个字符串输入，返回一个字符串
    :return:
    '''
    library = copy.deepcopy(library)
    if length is None:
        length = len(library)
    count = 0
    while True:
        if shufle:
            random.shuffle(library)
        for n in library:
            if transforms is not None:
                for t in transforms:
                    n = t(n)
            count += 1
            yield n
            if count >= length:
                return

def get_chars_iter(x):
    return [y for y in x.cmap.items()]

def load_font(font_path):
    if font_path.endswith('ttc'):
        ttc = TTCollection(font_path)
        return ttc.fonts[0]

    if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('tof'):
        ttf = TTFont(font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
        return ttf

def get_supported_chars_iter(ttf_path, muti_processing=False):
    '''
    得到每个字体能支持和不能支持的字符
    :param ttf_path:
    :param muti_processing:
    :return:
    '''
    with load_font(ttf_path) as ttf:
        if muti_processing:
            supported_chars_iter = map(lambda t: get_chars_iter(t), ttf['cmap'].tables)
        else:
            iteration = chain.from_iterable([y for y in x.cmap.items()] for x in ttf['cmap'].tables)
            supported_chars_iter = list(iteration)
    return supported_chars_iter

def bg_generator(bg_paths, cache=False):
    '''
    生成器，随机读取背景图片
    :param bg_paths: list， 每个元素为背景图片的完整路径
    :param cache: 是否需要缓存图片
    :return: PIL Image
    '''
    horizontal_pad_rate = 0.2
    if cache:
        cache = {}
    while True:
        random.shuffle(bg_paths)
        for path in bg_paths:
            if cache and path in cache:
                image = cache[path]
            else:
                image = Image.open(path)

            if cache:
                cache[path] = image

            size = image.size
            image = image.crop([int(size[0] * horizontal_pad_rate), 0, int(size[0] * (1- horizontal_pad_rate)), size[1]])
            if np.array(image).mean() < 100:#去掉过于黑暗的背景
                continue
            yield image, path

def wrap_as_generator(inp):
    #为了用生成器来统一支持list、生成器、int、float等输入
    if isinstance(inp, Iterable):
        for i in inp:
            yield i
    else:
        while True:
            yield inp

def is_chinese(c):
    uchar = bytes(c, encoding='unicode_escape')
    if (uchar >= b'\\u4e00' and uchar <= b'\\u9fa5') or uchar == b'\\u3007':
        return True
    else:
        return False

_txt_size_cache = {}

def _get_size(font, t):
    if t in _txt_size_cache:
        font_size = _txt_size_cache[font.path]
    else:
        font_size = font.getsize(t)
        _txt_size_cache[font.path] = font_size
    return font_size

def get_txt_size(text, fonts, char_to_font, x_spacing=0):
    '''
    计算字符串按照一定字体在涂上画出来的大小
    :param text:
    :param fonts:
    :param char_to_font:
    :param x_spacing:
    :return: 字符串的宽高
    '''
    w, h = 0, 0
    for t, x_spacing, font_id in zip(text, wrap_as_generator(x_spacing), char_to_font):
        font_size = _get_size(fonts[font_id], t)
        w += font_size[0]
        h = max(h, font_size[1])
        w += x_spacing
    return w, h

def get_rotated_txt_size(text, fonts, char_to_font, x_spacing=0, rotation_angle=0, padding=(0,0,0,0)):
    size_text_line = list(get_txt_size(text, fonts, char_to_font, x_spacing))
    size_orig_text = list(size_text_line)
    size_text_line[0] += (padding[0] + padding[2])
    size_text_line[1] += (padding[1] + padding[3])
    size_padded_text = list(size_text_line)
    text_boundary = MultiPoint([[0,0], [0, size_text_line[1]], size_text_line, [size_text_line[0], 0]])
    rotated_box = rotate(text_boundary, rotation_angle).bounds
    size_rotated_padded_once = [math.ceil(rotated_box[2] - rotated_box[0]), math.ceil(rotated_box[3] - rotated_box[1])]

    rotated_text_boundary = MultiPoint([[0,0], [0, size_rotated_padded_once[1]], size_rotated_padded_once, [size_rotated_padded_once[0], 0]])
    rotated_twice_text_boundary = rotate(rotated_text_boundary, -rotation_angle).bounds
    size_rotated_twice_padded_once = [math.ceil(rotated_twice_text_boundary[2] - rotated_twice_text_boundary[0]), math.ceil(rotated_twice_text_boundary[3] - rotated_twice_text_boundary[1])]

    orig_text_boundary = MultiPoint([[0,0], [0, size_orig_text[1]], size_orig_text, [size_orig_text[0], 0]])
    rotated_box = rotate(orig_text_boundary, rotation_angle).bounds
    size_rotated_text_line = [math.ceil(rotated_box[2] - rotated_box[0]), math.ceil(rotated_box[3] - rotated_box[1])]
    return size_rotated_twice_padded_once, size_rotated_padded_once, size_rotated_text_line, size_padded_text, size_orig_text

def get_txt_patch(image, text, fonts, char_to_font, fill=0, x_spacing=0, rotation_angle=0, padding=(0,0,0,0)):
    '''
    对PIL的draw.text进行封装， 按一定间隔写字
    :param image:
    :param text:
    :param fonts:
    :param char_to_font: 文本到字体的映射
    :param fill:
    :param x_spacing: 字体间隔
    :param rotation_angle:
    :param padding:
    :return:
    '''
    pos = [0, 0]
    size_rotated_twice_padded_once, size_rotated_padded_once, size_rotated_text_line, size_padded_text, size_orig_text = \
    get_rotated_txt_size(text, fonts, char_to_font, x_spacing, rotation_angle, padding=padding)

    if image.size[0] < size_rotated_twice_padded_once[0] * 1.2:
        scale = size_rotated_twice_padded_once[0] * 1.2 / image.size[0]
        image = image.resize((int(image.size[0]*scale), int(image.size[1]*scale)))

    if image.size[1] < size_rotated_twice_padded_once[1] * 1.2:
        scale = size_rotated_twice_padded_once[1] * 1.2 / image.size[1]
        image = image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))

    pos = [size_rotated_twice_padded_once[0] // 2 - size_orig_text[0] // 2,
           size_rotated_twice_padded_once[1] // 2 - size_orig_text[1] // 2]

    txt = image
    txt = RandomCrop(size_rotated_twice_padded_once[::-1])(txt)
    txt_draw = ImageDraw.Draw(txt)
    for t, x_spacing, font_id in zip(text, wrap_as_generator(x_spacing), char_to_font):
        font_size = _get_size(fonts[font_id], t)
        h_offset = (size_orig_text[1] - font_size[1])
        txt_draw.text([pos[0], pos[1] + h_offset], t, fill=fill, font=fonts[font_id])
        pos[0] += _get_size(fonts[font_id], t)[0]
        pos[0] += x_spacing
    txt = txt.rotate(rotation_angle, expand=1)
    txt = txt.crop([txt.size[0] // 2 - size_rotated_padded_once[0] // 2,
                    txt.size[1] // 2 - size_rotated_padded_once[1] // 2,
                    txt.size[0] // 2 + size_rotated_padded_once[0] // 2,
                    txt.size[1] // 2 + size_rotated_padded_once[1] // 2,
                    ])
    # txt = Pad(padding, fill=int(0))(txt)
    return txt