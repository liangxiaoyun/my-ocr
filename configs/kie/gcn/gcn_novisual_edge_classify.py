img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_scale, min_scale = 1024, 512

train_pipeline = [
    dict(type='LoadGCNAnnotations'),
    dict(type='GCNFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'labels', 'text_inds', 'text_length', 'src', 'dst', 'edge_data', 'norm_boxes'],
        meta_keys=('filename', 'ori_texts', 'edge_labels', 'node_num', 'ori_src', 'ori_dst'))
]
test_pipeline = [
    dict(type='LoadGCNAnnotations'),
    dict(type='GCNFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'text_inds', 'text_length', 'src', 'dst', 'edge_data', 'norm_boxes'],
        meta_keys=('filename', 'ori_texts', 'edge_labels', 'node_num', 'ori_src', 'ori_dst'))
]

dataset_type = 'GCNEdgeDataset'
data_root = '/liangxiaoyun583/data/wildreceipt'
CLASSES = 5

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
    dict_file=f'/liangxiaoyun583/data/wildreceipt/dict.txt',
    edge_type='row_col', #'row':CLASSES 2, 'col':CLASSES 2, 'row_col':CLASSES 3/5
    classes=CLASSES,
    w_thres=2,
    h_thres=2,
    fix_max_edge=True, #'row_col' 下才设置True
    max_edge_num=20,
    test_mode=False)
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/test.txt',
    pipeline=test_pipeline,
    img_prefix=data_root,
    loader=loader,
    dict_file=f'/liangxiaoyun583/data/wildreceipt/dict.txt',
    edge_type='row_col',
    classes=CLASSES,
    w_thres=2,
    h_thres=2,
    fix_max_edge=True,
    max_edge_num=20,
    test_mode=True)

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=0,
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
            ignores=[0])))

model = dict(
    type='GCN',
    backbone=dict(type='UNet', base_channels=16),
    bbox_head=dict(
        type='GCNEdgeHead', num_chars=94, num_classes=CLASSES, hidden_dim=512, MLP_hidden_dim=512*2, num_gnn=4,
        loss=dict(type='GCNEdgeLoss', used_ohem=False, ohem=3, neg_class=[0])),
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
total_epochs = 200

checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True

'''
SciTSR 从中抽2000训练，120验证
classes3 max_edges:20  8卡 batchsize：12
epoch(44) macro_f1:0.9965, row_recall:0.9880, row_precision:0.9799, row_F1_score:0.9835
col_recall:0.9338, col_precision:0.9369, col_F1_score:0.9333

classes5: max_edges:20  8卡 batchsize：12
epoch(94) macro_f1:0.77646, row_recall:0.9867, row_precision:0.9848, row_F1_score:0.9855
col_recall:0.9713, col_precision:0.9721, col_F1_score:0.9716
more_row_recall:0, more_row_precision:0, more_row_F1_score:0  (本无多行)
more_col_recall:0.1708, more_col_precision:0.1708, more_col_F1_score:0.1694
'''


