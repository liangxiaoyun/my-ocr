import os
import glob
import json

from itertools import chain
from fontTools.ttLib import TTCollection, TTFont
from fontTools.unicode import Unicode

from generate_chars import load_alphabet

def load_font(font_pth):
    if font_pth.endswith('ttc'):
        ttc = TTCollection(font_pth)
        return ttc.fonts[0]
    if font_pth.endswith('ttf') or font_pth.endswith('TTF') or font_pth.endswith('otf'):
        ttf = TTFont(font_pth, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
        return ttf

def check_font_chars(font, alphabet):
    chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in font["cmap"].tables)
    chars_int = []
    for c in chars:
        chars_int.append(c[0])

    unsupported_chars = []
    supported_chars = ''
    for c in alphabet:
        if ord(c) not in chars_int:
            unsupported_chars.append(c)
        else:
            supported_chars = supported_chars + c

    font.close()
    return unsupported_chars, supported_chars

def get_unsupported_chars(font_pth, alphabet):
    font = load_font(font_pth)
    unsupported, _ = check_font_chars(font, alphabet)
    return unsupported


def find_front_unsupported_chars(font_file, alphabet, save_json_file=None):
    font_list = []
    if os.path.isfile(font_file):
        font_list = [font_file]
    elif os.path.isdir(font_file):
        font_list = glob.glob(font_file+'/*.*')
    else:
        print('{} is a special file'.format(font_file))

    # if save_txt_file is not None:
    #     sf = open(save_txt_file, 'w', encoding='utf-8')

    unsupported_chars_dict = {}
    for font_pth in font_list:
        try:
            unsupported_chars = get_unsupported_chars(font_pth, alphabet)
            if len(unsupported_chars):
                unsupported_chars_dict[font_pth.split('/')[-1]] = unsupported_chars
            # if save_txt_file is not None and len(unsupported_chars):
            #     line = font_pth + ';'
            #     for char in unsupported_chars:
            #         line += char + ','
            #     line = line[:-1] + '\n'
            #     sf.write(line)
        except Exception as e:
            print(font_pth, e)

    if save_json_file is not None:
        with open(save_json_file, 'w', encoding='utf-8') as f:
            json.dump(unsupported_chars_dict, f)#, ensure_ascii=False)

    return unsupported_chars_dict

if __name__ == '__main__':
    font_file = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/fonts_song'
    alphabet = load_alphabet('/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/chinese_alphabet.txt')
    save_json_file = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/front_unsupported_chars.json'
    unsupported_chars_dict = find_front_unsupported_chars(font_file, alphabet, save_json_file=save_json_file)



