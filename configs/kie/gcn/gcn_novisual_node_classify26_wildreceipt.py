img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1024, 512

train_pipeline = [
    dict(type='LoadGCNAnnotations', ),
    dict(type='GCNFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'labels', 'text_inds', 'text_length', 'src', 'dst', 'edge_data', 'norm_boxes'],
        meta_keys=('filename', 'ori_texts'))
]
test_pipeline = [
    dict(type='LoadGCNAnnotations'),
    dict(type='GCNFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'text_inds', 'text_length', 'src', 'dst', 'edge_data', 'norm_boxes'],
        meta_keys=('filename', 'ori_texts'))
]

dataset_type = 'GCNDataset'
data_root = '/liangxiaoyun583/data/wildreceipt'

loader = dict(
    type='HardDiskLoader',
    repeat=1,
    parser=dict(
        type='LineJsonParser',
        keys=['file_name', 'height', 'width', 'annotations']))

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/train.txt',
    pipeline=train_pipeline,
    img_prefix=data_root,
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=False)
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/test.txt',
    pipeline=test_pipeline,
    img_prefix=data_root,
    loader=loader,
    dict_file=f'{data_root}/dict.txt',
    test_mode=True)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=train,
    val=test,
    test=test)

evaluation = dict(
    interval=1,
    metric='macro_f1',
    metric_options=dict(
        macro_f1=dict(
            ignores=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])))

model = dict(
    type='GCN',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='GCNHead', num_chars=94, num_classes=26, hidden_dim=512, MLP_hidden_dim=512, num_gnn=8,
        loss=dict(type='GCNLoss', used_ohem=False, ohem=3, neg_class=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25])),
    train_cfg=None,
    test_cfg=None,
    class_list=f'{data_root}/class_list.txt')

optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[200, 300])
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1,
#     warmup_ratio=1,
#     step=[40, 50])
total_epochs = 100

checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True

#Macro F1-Score: 0.8738 (epoch 68)
