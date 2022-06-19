import os
import cv2
import math
import random
import numpy as np
import argparse

# from faker import Faker
from shapely.geometry import Polygon, MultiPoint
from PIL import ImageFont, ImageDraw, Image, ImageFilter

def judge_intersect(poly, all_polies):
    for p in all_polies:
        if float(poly.intersection(p).area) > 0:
            return True
    return False

def get_point(angle, d, base):
    angle = angle / 180.0 * math.pi
    _x, _y = math.cos(angle) * d, math.sin(angle) * d
    return (int(base[0] + _x), int(base[1] - _y))

def star(img, x, y, star_angle, center, color=(0,0,255)):
    '''
    画五角星
    :param img: np.int8, w*h*3
    :param x: 起始x坐标
    :param y: 起始y坐标
    :param star_angle: 画笔方向 逆时针转动度数
    :param center: 五角星中心
    :param color: 颜色
    :return:
    '''
    points = []
    y = x / (math.cos(0.2 * math.pi) + math.sin(0.2 * math.pi) / math.tan(0.1 * math.pi))
    for i in range(5):
        points.append(get_point(star_angle, x, center))
        star_angle -= 36
        points.append(get_point(star_angle, y, center))
        star_angle -= 36
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=color)
    return img


def add_signet2img(img, texts, signet_color, signet_size, signet_type_texts=[]):
    font_path = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/公司印章.ttf'
    excircle_radius = signet_size // 2
    innercircle_radius = excircle_radius // 2
    R = (excircle_radius + innercircle_radius) // 2
    text_size = int((excircle_radius - innercircle_radius) / 2)#9 * 5)
    all_polies = []
    gen_boxes = []


    width, height = img.size

    for signet_num, text in enumerate(texts):
        char_num = len(text)
        middle_index = char_num // 2 - 1
        mid_angle = random.randint(0, 360)
        star_angle = mid_angle + 90
        font = ImageFont.truetype(font_path, text_size)

        if len(signet_type_texts) > 0:
            if (text_size * char_num) > 260:
                each_text_angle = 260 // char_num
            else:
                each_text_angle = text_size
        else:
            if (text_size * char_num) > 280:
                each_text_angle = 280 // char_num
            elif (text_size * char_num) < 160:
                each_text_angle = 160 // char_num
            else:
                each_text_angle = text_size

        center = None
        while True:
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            poly = np.array([[start_x, start_y], [start_x+signet_size, start_y], [start_x+signet_size, start_y+signet_size], [start_x, start_y+signet_size]])
            poly = Polygon(poly).convex_hull
            if not judge_intersect(poly, all_polies) and (start_x + signet_size + 7) < width and (start_y + signet_size + 7) < height:
                center = ((start_x + signet_size) // 2, (start_y + signet_size) // 2)
                all_polies.append(poly)
                break
        gen_boxes.append([start_x, start_y, start_x + signet_size + 7, start_y + signet_size + 7])

        #画圆
        draw = ImageDraw.Draw(img)
        draw.ellipse((center[0] - excircle_radius, center[1] - excircle_radius, center[0] + excircle_radius, center[1] + excircle_radius), outline=signet_color, width=5)

        #写字
        #获取每个文字的角度
        angle_list= []
        for i in range(char_num):
            if i == middle_index:
                angle_list.append(mid_angle)
            else:
                angle_list.append(mid_angle + (middle_index - i) * each_text_angle)

        #写印章公司名字
        start_point, end_point, start_second_point, start_w, start_h = None, None, None, None, None
        for i in range(char_num):
            angle = angle_list[i]
            char_img = Image.new('RGBA', (text_size, text_size), (0,0,0,0))
            d = ImageDraw.Draw(char_img)
            d.text((0,0), text[i], font=font, fill=signet_color[:3])

            #图片拉长
            char_img = char_img.resize((text_size, int(text_size / 2 * 3)))
            rotate_char_img = char_img.rotate(angle, expand=1)
            tmp_char_img = Image.new('RGBA', rotate_char_img.size, (0,0,0,0))
            rotate_char_img = Image.composite(rotate_char_img, tmp_char_img, rotate_char_img)
            out = rotate_char_img
            if angle > 0:
                x = center[0] - int(R * np.sin(angle / 180 * np.pi)) - int(rotate_char_img.size[0]/2)
                y = center[1] - int(R * np.cos(angle / 180 * np.pi)) - int(rotate_char_img.size[1] / 2)
            else:
                x = center[0] + int(R * np.sin(-angle / 180 * np.pi)) - int(rotate_char_img.size[0] / 2)
                y = center[1] - int(R * np.cos(-angle / 180 * np.pi)) - int(rotate_char_img.size[1] / 2)

            img.paste(out, (x,y), out)

            if i == 0:
                start_point = (x,y)
                start_w, start_h = rotate_char_img.size
            elif i == char_num - 1:
                end_point = (x, y)

        #添加五角星
        star_size = int(excircle_radius * 4 / 5)
        star_img = Image.new('RGBA', (star_size, star_size), (0,0,0,0))
        star_img = star(star_img, star_size // 2, 0, star_angle, (star_size // 2, star_size // 2), color=signet_color)
        img.paste(star_img, (center[0]-star_size//2, center[1]-star_size//2), star_img)

        #写上印章类型
        if len(signet_type_texts) > 0:
            signet_type_text = signet_type_texts[signet_num]
            #第一个字和最后一个字的中间角度
            text_angle = - (180.0 - angle_list[0]) #-math.atan((end_point[1] - start_point[1]) / (end_point[0] - start_point[0]))
            #中间文本最大长度
            text_length = math.sqrt(2 * (1 + math.cos((angle_list[-1] - angle_list[0]) / 180 * math.pi))) * excircle_radius
            #字符最大长度
            signet_type_text_size = text_length // (len(signet_type_text)-1)
            signet_type_text_size = (signet_type_text_size + len(signet_type_text) // 3) if len(signet_type_text) > 5 else signet_type_text_size
            signet_type_text_size = int(signet_type_text_size)
            font = ImageFont.truetype(font_path, signet_type_text_size)
            #中间文本图片大小
            w,h = font.getsize(signet_type_text)
            signet_type_img = Image.new('RGBA', (w,h), (0,0,0,0))
            d = ImageDraw.Draw(signet_type_img)
            d.text((0,0), signet_type_text, font=font, fill=signet_color[:3])
            #拉长文字的h
            signet_type_img = signet_type_img.resize((w, h // 2 * 3))
            #旋转中间文字
            signet_type_img = signet_type_img.rotate(text_angle/math.pi*180, expand=1)
            signet_type_img_w = signet_type_img.size[0]
            # signet_type_text_x, signet_type_text_y = int(start_point[0] + start_w/2), int(start_point[1] + start_h / 2)
            # signet_type_text_y = signet_type_text_y - int(signet_type_img_w * math.tan((text_angle-180)/math.pi*180))
            signet_type_text_x, signet_type_text_y = int(start_point[0] - w / 2), int(
                (start_point[1] + end_point[1]) / 2 - h / 2)
            signet_type_text_y = signet_type_text_y - int(
                signet_type_img_w * math.tan((text_angle - 180) / math.pi * 180))

            img.paste(signet_type_img, (signet_type_text_x, signet_type_text_y), signet_type_img)

    #模糊
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.convert('RGB')
    return img, gen_boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('--img_path', type=str, default='/Users/duoduo/Desktop/截图/截屏2021-04-24 下午2.07.36.png', help='The path of img which need to be added signet')
    parser.add_argument('--signet_num', type=int, default=1, help='The num of signet added to img')
    parser.add_argument('--signet_texts_txt', type=str, default=None, help='The txt file contains texts of signet')
    parser.add_argument('--signet_texts', type=list, default=['中国平安财产保险股份有限公司'], help='The texts of signet')
    parser.add_argument('--signet_color', type=tuple, default=(239, 7, 5, 255), help='The texts of signet')
    parser.add_argument('--signet_size', type=int, default=300, help='The size of signet')#外圆直径
    parser.add_argument('--save_img_folder', type=str, default='/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/gen_signets', help='The path of saving img folder')
    parser.add_argument('--signet_type_texts', type=list, default=['报销专用章'], help='The texts of signet')

    args = parser.parse_args()

    img = Image.open(args.img_path).convert('RGBA')

    texts = []
    signet_type_texts = []
    if args.signet_texts_txt is not None:
        with open(args.signet_texts_txt, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if 5 < len(line) < 17:
                    texts.append(line)
        texts = texts[:args.signet_num]

    elif len(args.signet_texts) > 0:
        texts = args.signet_texts[:args.signet_num]

    else: #随机生成
        pass
        # fake = Faker("zh_CN")
        # for i in range(args.signet_num):
        #     texts.append(fake.company())

    signet_type_texts = args.signet_type_texts[:args.signet_num]

    gen_img, gen_boxes = add_signet2img(img, texts, args.signet_color, args.signet_size, signet_type_texts)
    gen_img.save(os.path.join(args.save_img_folder, args.img_path.split('/')[-1]))


