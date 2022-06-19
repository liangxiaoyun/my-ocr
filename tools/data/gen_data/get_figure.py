import random
import cv2
from glob import glob
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans
from collections import defaultdict

import generate_chars
from fontcolor import FontColor, get_fonts, chose_font, get_bestcolor
import generate_chars

class GetFigure(object):
    def __init__(self, configs):
        # 字体库
        font_path = configs['font_folder_path']
        # 背景图库
        bg_root_path = configs['bg_folder_path']
        bg_imgs_paths = glob(bg_root_path + "/" + r"*.jpg")

        self.chars_process = configs['chars_process']
        self.img_process = configs['img_process']

        font_sizes = list(
            range(self.chars_process['font_size']['min_size'], configs['chars_process']['font_size']['max_size']))

        # 通过读取txt文件来获取总的chars列表，来生成数据
        chars_sequence = []
        if self.chars_process['chars_from_txt_file'] is not None:
            chars_sequence = generate_chars.get_all_chars_in_txt(configs['chars_from_txt_file'])

        # all alphabets  用于后续得字符删除/添加/替换得增强
        with open(configs['front_unsupported_chars_json'], 'r') as f:
            self.front_unsupported_chars = json.load(f)
        self.front_supported_chars = defaultdict(list)
        alphabets = None
        if configs['alphabets_path'] != 'None':
            alphabets = generate_chars.get_all_keys(configs['alphabets_path'])
            for k, v in self.front_unsupported_chars.items():
                for i in alphabets:
                    if i not in v and i >= '\u4e00' and i < '\u9fa5':
                        self.front_supported_chars[k].append(i)
        # 读入字体色彩库
        self.color_lib = FontColor('colors_new.cp')
        self.fonts = get_fonts(font_path, font_sizes)
        self.font_sizes = font_sizes
        self.bg_imgs_paths = bg_imgs_paths
        self.chars_sequence = chars_sequence
        self.alphabets = alphabets


    def rotate_img_box(self, image, box, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        rotate_angle = angle
        M = cv2.getRotationMatrix2D((cX, cY), rotate_angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotate_image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rotate_box = []

        for ponit in box:
            rot_point = np.dot(M, np.array([ponit[0], ponit[1], 1]))
            rotate_box.append([np.int(rot_point[0]), np.int(rot_point[1])])

        new_box = [min(rotate_box[0][0], rotate_box[1][0], rotate_box[2][0], rotate_box[3][0]),
                   min(rotate_box[0][1], rotate_box[1][1], rotate_box[2][1], rotate_box[3][1]),
                   max(rotate_box[0][0], rotate_box[1][0], rotate_box[2][0], rotate_box[3][0]),
                   max(rotate_box[0][1], rotate_box[1][1], rotate_box[2][1], rotate_box[3][1])]

        return Image.fromarray(rotate_image.astype('uint8')).convert('RGB'), new_box

    def bg_img_refine(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if w < 400:
            img = img.resize((400, h), Image.ANTIALIAS)
            w = 400
        if h < 30:
            img = img.resize((w, 30), Image.ANTIALIAS)
            h = 30
        return img, w, h

    def judge_font_can_support_chars(self, font_name, chars):
        for i in chars:
            if i in self.front_unsupported_chars[font_name]:
                return False
        return True

    def replace_char_for_font(self, font_name, chars):
        for i in chars:
            if i in self.front_unsupported_chars[font_name]:
                if len(self.front_supported_chars) and font_name in self.front_supported_chars.keys():
                    chars = chars.replace(i, random.choice(self.front_supported_chars[font_name]))
                elif self.alphabets is not None:
                    chars = chars.replace(i, random.choice(self.alphabets))
        return chars


    def get_horizontal_text_picture(self, index):
        bg_img_path = random.choice(self.bg_imgs_paths)
        retry = 0
        img = Image.open(bg_img_path)
        img, w, h = self.bg_img_refine(img)
        ori_img = Image.fromarray(np.array(img).copy())
        x1 = 0  # text的开始位置
        y1 = 0

        #找到合适的字符串、crop位置和文字颜色
        while True:
            if self.chars_process['chars_from_txt_file']:
                chars = self.chars_sequence[index]
            else:
                chars = generate_chars.gen_vin()

            if self.alphabets is not None and random.random() < self.chars_process['character_process_ratio']:#字符串处理
                chars = generate_chars.character_process(chars, self.alphabets)

            if (chars is None or chars == ''):
                continue

            font_name, font = chose_font(self.fonts, self.font_sizes)
            # if not self.judge_font_can_support_chars(font_name, chars): continue

            #replace char for font
            chars = self.replace_char_for_font(font_name, chars)

            print('font_name: ', font_name)

            f_w, f_h = font.getsize(chars)
            if f_w < w:
                if (w - f_w) < 1:
                    print("if (w - f_w)<1:")
                    continue
                if (h - f_h) < 1:
                    print("if (h - f_h)<1:")
                    continue
                x1 = random.randint(0, w - f_w - 1)
                y1 = random.randint(0, h - f_h - 1)
                x2 = x1 + f_w
                y2 = y1 + f_h

                # 随机加一点偏移
                if random.random() < self.img_process['img_crop_shift_rario']:  # 设定偏移的概率
                    crop_x1 = int(max(0, x1 - random.uniform(0, f_h / 3)))
                    crop_x2 = int(min(w - 1, x2 + random.uniform(0, f_h / 3)))
                    crop_y1 = int(max(0, y1 - random.uniform(0, f_h / 6)))
                    crop_y2 = int(min(h - 1, y2 + random.uniform(0, f_h / 6)))
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2
                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                if self.img_process['use_color_lib']:
                    crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
                    if np.linalg.norm(
                            np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 35 and retry < 30:  # 颜色标准差阈值，颜色太丰富就不要了
                        retry = retry + 1
                        print("retry = retry+1")
                        print('bg_image_file:   ', self.bg_imgs_paths)
                        continue
                    best_color = get_bestcolor(self.color_lib, crop_lab)
                else:
                    best_color = random.choice(self.img_process['font_colors'])

                break
            else:
                print("pass:")
                pass

        draw = ImageDraw.Draw(img)

        #单字符写，扩大字符间距离
        if self.img_process['one_char_write']:
            c_w = 0
            interval = random.randint(self.img_process['one_char_write_min_interval'], self.img_process['one_char_write_max_interval'])
            # print(interval)
            for nn, c in enumerate(chars):
                draw.text((x1 + c_w, y1), c, fill=best_color, font=font)
                c_w += font.getsize(c)[0] + interval

            crop_x2 = int(x1 + c_w)
            if crop_x2 > w - 1:
                img = ori_img.resize((crop_x2, h), Image.ANTIALIAS)
                draw = ImageDraw.Draw(img)
                c_w = 0
                for nn, c in enumerate(chars):
                    draw.text((x1 + c_w, y1), c, fill=best_color, font=font)
                    c_w += font.getsize(c)[0] + interval

        #使用正常的font间距
        else:
            draw.text((x1, y1), chars, fill=best_color, font=font)

        # 画水平直线
        if random.random() < self.img_process['add_line_ratio']:
            start_x = random.randint(crop_x1, crop_x2 - 1)
            end_x = random.randint(min(start_x + 15, crop_x2 - 1), crop_x2)
            start_y = random.choice([random.randint(crop_y1, min(crop_y1 + 20, crop_y2)), random.randint(min(crop_y1, crop_y2 - 20), crop_y2)])
            end_y = min(start_y + random.randint(1,3), crop_y2)
            draw.rectangle((start_x, start_y, end_x, end_y), best_color)

        # 旋转
        if random.random() < self.img_process['rotate']['ratio']:
            angle = random.randint(-self.img_process['rotate']['angle'], self.img_process['rotate']['angle'])
            # crop_img = crop_img.rotate(random.randint(-5,5))
            # img.resize((crop_x2-crop_x1, crop_y2- crop_y1),Image.ANTIALIAS)
            rotate_img, rotate_box = self.rotate_img_box(np.array(img),
                                                    [[crop_x1, crop_y1], [crop_x1, crop_y2], [crop_x2, crop_y2],
                                                     [crop_x2, crop_y1]], angle)
            crop_img = rotate_img.crop((rotate_box[0], rotate_box[1], rotate_box[2], rotate_box[3]))
        else:
            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 加模糊
        if random.random() < self.img_process['blur']['ratio']:
            mohu = random.randint(1, 3)
            if self.img_process['blur']['mode'] == 'mean':
                crop_img = Image.fromarray(cv2.blur(np.array(crop_img), (mohu, mohu)).astype('uint8')).convert('RGB')
            elif self.img_process['blur']['mode'] == 'Gaussian':
                crop_img = crop_img.filter(ImageFilter.GaussianBlur)

        return crop_img, chars

    def get_vertical_text_picture(self, index):
        bg_img_path = random.choice(self.bg_imgs_paths)
        img = Image.open(bg_img_path)
        img, w, h = self.bg_img_refine(img)
        ori_img = Image.fromarray(np.array(img).copy())
        retry = 0
        x1 = 0  # text的开始位置
        y1 = 0

        while True:
            if self.chars_process['chars_from_txt_file']:
                chars = self.chars_sequence[index]
            else:
                chars = generate_chars.gen_vin()

            if self.alphabets is not None and random.random() < self.chars_process['character_process_ratio']:  # 字符串处理
                chars = generate_chars.character_process(chars, self.alphabets)

            if (chars is None or chars == ''):
                continue

            font_name, font = chose_font(self.fonts, self.font_sizes)
            print('font_name: ', font_name)

            ch_w = []
            ch_h = []
            for ch in chars:
                wt, ht = font.getsize(ch)
                ch_w.append(wt)
                ch_h.append(ht)
            f_w = max(ch_w)
            f_h = sum(ch_h)

            # 完美分割时应该取的,也即文本位置
            if h > f_h:
                x1 = random.randint(0, w - f_w - 1)
                y1 = random.randint(0, h - f_h - 1)
                x2 = x1 + f_w
                y2 = y1 + f_h
                # 随机加一点偏移
                rd = random.random()
                if rd < self.img_process['img_crop_shift_rario']:  # 设定偏移的概率  0.2
                    crop_x1 = x1 - random.random() / 4 * f_w
                    crop_y1 = y1 - random.random() / 2 * f_w
                    crop_x2 = x2 + random.random() / 4 * f_w
                    crop_y2 = y2 + random.random() / 2 * f_w
                    crop_y1 = int(max(0, crop_y1))
                    crop_x1 = int(max(0, crop_x1))
                    crop_y2 = int(min(h, crop_y2))
                    crop_x2 = int(min(w, crop_x2))
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2

                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                if self.img_process['use_color_lib']:
                    crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
                    if np.linalg.norm(
                            np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 35 and retry < 30:  # 颜色标准差阈值，颜色太丰富就不要了
                        retry = retry + 1
                        continue
                    best_color = get_bestcolor(self.color_lib, crop_lab)
                else:
                    best_color = random.choice(self.img_process['font_colors'])
                break

            else:
                pass
        draw = ImageDraw.Draw(img)
        i = 0

        if self.img_process['one_char_write']:
            interval = random.randint(self.img_process['one_char_write_min_interval'],
                                      self.img_process['one_char_write_max_interval'])
            for ch in chars:
                draw.text((x1, y1), ch, best_color, font=font)
                y1 = y1 + ch_h[i] + interval
                i = i + 1
            crop_y2 = int(y1) - interval
            if crop_y2 > h-1:
                img = ori_img.resize((w, crop_y2), Image.ANTIALIAS)
                draw = ImageDraw.Draw(img)
                i = 0
                for ch in chars:
                    draw.text((x1, y1), ch, best_color, font=font)
                    y1 = y1 + ch_h[i] + interval
                    i = i + 1
        else:
            for ch in chars:
                draw.text((x1, y1), ch, best_color, font=font)
                y1 = y1 + ch_h[i]
                i = i + 1

        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_img = crop_img.transpose(Image.ROTATE_270)
        # 加模糊
        if random.random() < self.img_process['blur']['ratio']:
            mohu = random.randint(1, 3)
            if self.img_process['blur']['mode'] == 'mean':
                crop_img = Image.fromarray(cv2.blur(np.array(crop_img), (mohu, mohu)).astype('uint8')).convert('RGB')
            elif self.img_process['blur']['mode'] == 'Gaussian':
                crop_img = crop_img.filter(ImageFilter.GaussianBlur)
        return crop_img, chars

