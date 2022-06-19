_base_ = ['../../_base_/default_runtime.py']

find_unused_parameters = True
start_end_same = True
with_unknown = True

alphabet_pth = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/alphabet.txt'
num_classes = 93
# alphabet = []
# from mmocr.utils import list_from_file
# for line in list_from_file(alphabet_pth):
#     line = line.strip()
#     if line != '':
#         alphabet.extend(list(line))
#
# num_classes = len(alphabet) + 1 #padding
# if start_end_same:
#     num_classes += 1
# else:
#     num_classes += 2
# if with_unknown:
#     num_classes += 1

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', dict_file=alphabet_pth, with_unknown=with_unknown, start_end_same=start_end_same) #alphabet在这里指定

model = dict(
    type='MASTERNet',
    backbone=dict(type='ConvEmbeddingGC', in_channels=3),
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
        num_classes=num_classes, #DICT90
        max_seq_len=30,
    ),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 5

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
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
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=256,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ])
]

dataset_type = 'OCRDataset'

train_ann_file1 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'
train_ann_file2 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'


train1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t'),
            data_root=None),  #data_root: 图片的真实所在目录的路径，None的话，默认读取的txt文件所在路径为图片路径。例：'/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/'
    pipeline=None,
    test_mode=False)

#注意：要改字典元素里的元素的值（例：parser），同时会改变train1，所以得重新定义，而不是用下方的复制方法
train2 = {key: value for key, value in train1.items()}
train2['ann_file'] = train_ann_file2

test_ann_file1 = '/Users/duoduo/PycharmProjects/pingan_projects/mmocr-main/data/recog_data/labels.txt'


test1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='\t')),
    pipeline=None,
    test_mode=True)

data = dict(
    samples_per_gpu=96,
    workers_per_gpu=2,  #debug时设置为0，即可pdb进去
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[
            train1, train2
        ],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test1],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test1],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
