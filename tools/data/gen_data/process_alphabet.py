import os

all_alphabets = []
with open('/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/chinese_alphabet.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        all_alphabets.extend(list(line))
all_alphabets.append(' ')
all_alphabets = set(all_alphabets)
print(len(all_alphabets))
all_alphabets = ''.join(all_alphabets)
with open('/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/chinese_alphabet2.txt', 'w', encoding='utf-8') as sf:
    sf.write(all_alphabets)