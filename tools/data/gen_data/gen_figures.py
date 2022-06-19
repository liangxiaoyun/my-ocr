'''
_*_ coding: utf-8 _*_
@author: LiangXiaoyun
@time: 20210415
'''

import os
import yaml

from get_figure import GetFigure

def main():
    f = open('gen_figure_config.yaml', 'r', encoding='utf-8')
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(configs)
    save_img_pth = configs['save_img_folder']
    suffix = configs['save_img_suffix']
    if not os.path.exists(save_img_pth):
        os.mkdir(save_img_pth)

    GF = GetFigure(configs)

    labels_path = save_img_pth + '/labels.txt'
    gs = 0
    if os.path.exists(labels_path):  # 支持中断程序后，在生成的图片基础上继续
        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = list(f.readlines())
            new_lines = [l for l in lines if suffix in l]
            # print("lines:", new_lines)

        if len(new_lines) > 0:
            gs = int(new_lines[-1].strip().split(configs['img_pth_char_split_in_labels'])[0].split('/')[-1].split('_')[0]) + 1
            print('Resume generating from step %d' % gs)

    f = open(labels_path, 'a', encoding='utf-8')
    print('start generating...')

    for i in range(gs, int(configs['save_num'])):
        try:
            if configs['img_process']['img_direction'] == 'horizontal':
                gen_img, chars = GF.get_horizontal_text_picture(i)
            else:
                gen_img, chars = GF.get_vertical_text_picture(i)
            if gen_img.mode != 'RGB':
                gen_img = gen_img.convert('RGB')

            save_img_name = str(i).zfill(7) + suffix

            save_img_name = os.path.join(save_img_pth, save_img_name)
            gen_img.save(save_img_name)

            f.write(save_img_name + configs['img_pth_char_split_in_labels'] + chars + '\n')
            print('gennerating:-------' + save_img_name + 'chars:' + chars)
        except Exception as e:
            print(e)
    f.close()

if __name__ == '__main__':
    main()