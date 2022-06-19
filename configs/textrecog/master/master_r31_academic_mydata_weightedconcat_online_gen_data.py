_base_ = ['../../_base_/default_runtime.py']

find_unused_parameters = True
start_end_same = True
with_unknown = True

alphabet_pth = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/alphabet.txt'

max_length = 40

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', dict_file=alphabet_pth, with_unknown=with_unknown, start_end_same=start_end_same) #alphabet在这里指定

model = dict(
    type='MASTERNet',
    backbone=dict(type='ConvEmbeddingGC', in_channels=1),
    encoder=dict(
        type='MASTEREncoder',
        d_model=512,
        _multi_heads_count=8,
        _dropout=0.2,
        _MultiHeadAttention_dropout=0.1,
        _feed_forward_size=2048,
        _with_encoder=False,
    ),
    decoder=dict(
        type='MasterDecoder',
        n_layers=3,
        d_embedding=512,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        n_position=5000,
        dropout=0.2,
        max_seq_len=max_length,
    ),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=max_length)

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='Prober', alphabet_pth=alphabet_pth),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=256,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=256,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio']),
    ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiRotateAugOCR',
#         rotate_degrees=[0, 90, 270],
#         transforms=[
#             dict(
#                 type='ResizeOCR',
#                 height=32,
#                 min_width=32,
#                 max_width=256,
#                 keep_aspect_ratio=True,
#                 width_downsample_ratio=0.25),
#             dict(type='ToTensorOCR'),
#             dict(type='NormalizeOCR', **img_norm_cfg),
#             dict(
#                 type='Collect',
#                 keys=['img'],
#                 meta_keys=[
#                     'filename', 'ori_shape', 'img_shape', 'valid_ratio'
#                 ]),
#         ])
# ]

online_gen_train_pipline = [
    # dict(type="balanceSample", max_len=max_length),#根据数据字符频率来采样频率较低的字符
    dict(type="RandomLoadString", file_path='/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/chars_sequences.txt'), #从txt中读取字符串来生成数据
    # dict(type="RandomInsert", alpha_path=alphabet_pth, prob=0.05), #语义增强：随机插入字符
    # dict(type="RandomReplace", alpha_path=alphabet_pth, prob=0.1), #语义增强：随机替换字符
    dict(type='OnlinePartimgGenerate', material_path='/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data',
         max_len=max_length, max_fonts_per_img=1, color_type='grayscale'), #需要有背景图目录：bgs和字体目录：fonts   max_fonts_per_img:一张图片最多可能有多少种字体
    dict(type='Prober', alphabet_pth=alphabet_pth),#使用balanceSample时，需要所有train_pipline都加上这个来记录字符频率
    dict(
            type='ResizeOCR',
            height=32,
            min_width=32,
            max_width=256,
            keep_aspect_ratio=True,
            width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]


dataset_type = 'OCRDataset'

train_ann_file1 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'
train_ann_file2 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'

train_ann_file3 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'
train_ann_file4 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'


train1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file1,
    loader=dict(
        type='LoaderParsertxt',
        repeat=1,
        keys=['filename', 'text'],
        separator='\t',
        data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=train_pipeline,
    test_mode=False)

train2 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file2,
    loader=dict(
        type='LoaderParsertxt',
        repeat=1,
        keys=['filename', 'text'],
        separator='\t',
        data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=train_pipeline,
    test_mode=False)

train3 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file3,
    loader=dict(
        type='LoaderParsertxt',
        repeat=1,
        keys=['filename', 'text'],
        separator='\t',
        data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=train_pipeline,
    test_mode=False)

train4 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file4,
    loader=dict(
        type='LoaderParsertxt',
        repeat=1,
        keys=['filename', 'text'],
        separator='\t',
        data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=train_pipeline,
    test_mode=False)

balance_online_gen = dict(
    type='OnlineGenerationDataset',
    pipeline=online_gen_train_pipline,
    dataset_len=1000000,
    test_mode=False,
)

#gen data
gen_data = dict(
    type='WeightedConcatDataset',
    datasets=[train1, train2, balance_online_gen],
    weights=[1, 1, 1],
    len_epoch=10000,
)
#real data
real_data = dict(
    type='WeightedConcatDataset',
    datasets=[train3, train4],
    weights=[1,1],
    len_epoch=10000,
)

test_ann_file1 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'

test1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=test_ann_file1,
    loader=dict(
        type='LoaderParsertxt',
        repeat=1,
        keys=['filename', 'text'],
        separator='\t',
        data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=test_pipeline,
    test_mode=True)

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=2,  #debug时设置为0，即可pdb进去
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='WeightedConcatDataset',
        datasets=[
            gen_data, real_data
        ],
        weights=[10, 1], #跟数据量无关，设置的是每个batch采样的比例，一般生成数据和真实数据设置为10：1
        len_epoch=10000),
    val=test1,
    test=test1)

evaluation = dict(interval=1, metric='acc')
# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=100,
                 warmup_ratio=1.0 / 3,
                 step=[8, 13])
total_epochs = 15