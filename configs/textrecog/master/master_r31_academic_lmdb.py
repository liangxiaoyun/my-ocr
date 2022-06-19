_base_ = ['../../_base_/default_runtime.py']

#fp16   使用fp16，加这句话即可
fp16 = dict(loss_scale='dynamic')# 512.

find_unused_parameters = True

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True) #alphabet在这里指定

model = dict(
    type='MASTERNet',
    backbone=dict(type='ConvEmbeddingGC', in_channels=3),
    encoder=dict(
        type='MASTEREncoder',
        d_model=512,
        _multi_heads_count=8,
        _dropout=0.2,
        _MultiHeadAttention_dropout=0.1,
        _feed_forward_size=2024,
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
        d_inner=2024,
        n_position=5000,
        dropout=0.0,
        num_classes=93, #DICT90
        max_seq_len=30,
    ),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)

# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333,
    step=[8, 13])
total_epochs = 15

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromLMDB'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,),
        # width_downsample_ratio=0.25),
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
    dict(type='LoadImageFromLMDB'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=160,
                keep_aspect_ratio=True,),
                # width_downsample_ratio=0.25),
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

train_prefix = 'data/mixture/'

train_ann_file1 = train_prefix + 'icdar_2011'
train_ann_file2 = train_prefix + 'icdar_2013'
train_ann_file3 = train_prefix + 'icdar_2015'
train_ann_file4 = train_prefix + 'coco_text'
train_ann_file5 = train_prefix + 'III5K'
train_ann_file6 = train_prefix + 'SynthText_Add'
train_ann_file7 = train_prefix + 'SynthText'
train_ann_file8 = train_prefix + 'Syn90k'

train1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file1,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=20,
        parser=dict(
            type='LineStrParser2',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train2 = {key: value for key, value in train1.items()}
train2['ann_file'] = train_ann_file2

train3 = {key: value for key, value in train1.items()}
train3['ann_file'] = train_ann_file3

train4 = {key: value for key, value in train1.items()}
train4['ann_file'] = train_ann_file4

train5 = {key: value for key, value in train1.items()}
train5['ann_file'] = train_ann_file5

train6 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=train_ann_file6,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser2',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train7 = {key: value for key, value in train6.items()}
train7['ann_file'] = train_ann_file7

train8 = {key: value for key, value in train6.items()}
train8['ann_file'] = train_ann_file8

test_prefix = 'data/mixture/'
test_ann_file1 = test_prefix + 'IIIT5K/'
test_ann_file2 = test_prefix + 'svt/'
test_ann_file3 = test_prefix + 'icdar_2013/'
test_ann_file4 = test_prefix + 'icdar_2015/'
test_ann_file5 = test_prefix + 'svtp/'
test_ann_file6 = test_prefix + 'ct80/'


test1 = dict(
    type=dataset_type,
    img_prefix='',
    ann_file=test_ann_file1,
    loader=dict(
        type='MJSTLmdbLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser2',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test2 = {key: value for key, value in test1.items()}
test2['ann_file'] = test_ann_file2

test3 = {key: value for key, value in test1.items()}
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['ann_file'] = test_ann_file4

test5 = {key: value for key, value in test1.items()}
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['ann_file'] = test_ann_file6

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=2,
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
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3, test4, test5, test6],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')


