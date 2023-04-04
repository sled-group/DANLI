dataset_type = 'CocoPanopticDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
CLASSES = [
    'AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair',
    'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds',
    'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced',
    'ButterKnife', 'CD', 'Cabinet', 'Candle', 'CellPhone', 'Chair', 'Cloth',
    'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk',
    'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge', 'DogBed', 'Drawer',
    'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet', 'FloorLamp',
    'Footstool', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel',
    'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
    'Laptop', 'LaundryHamper', 'Lettuce', 'LettuceSliced', 'LightSwitch',
    'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Pan',
    'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate',
    'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'RoomDecor',
    'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShelvingUnit',
    'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable',
    'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon',
    'SprayBottle', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'TVStand',
    'TableTopDecor', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox',
    'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato',
    'TomatoSliced', 'Towel', 'TowelHolder', 'VacuumCleaner', 'Vase', 'Watch',
    'WateringCan', 'Window', 'WineBottle', 'AirConditioner', 'Bag', 'Bookcase',
    'CabinetBody', 'Carpet', 'Ceiling', 'CounterSide', 'Cube', 'Curtains',
    'Cylinder', 'Dishwasher', 'DomeLight', 'Door', 'Floor', 'LightFixture',
    'Painting', 'Poster', 'StoveBase', 'TargetCircle', 'Wall', 'OTHERS'
]
THING_CLASSES = ['AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced',
    'ButterKnife', 'CD', 'Cabinet', 'Candle', 'CellPhone', 'Chair', 'Cloth',
    'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk',
    'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge', 'DogBed', 'Drawer',
    'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet', 'FloorLamp',
    'Footstool', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel',
    'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
    'Laptop', 'LaundryHamper', 'Lettuce', 'LettuceSliced', 'LightSwitch',
    'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Pan',
    'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate',
    'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'RoomDecor',
    'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShelvingUnit',
    'ShowerCurtain', 'ShowerDoor', 'ShowerGlass', 'ShowerHead', 'SideTable',
    'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon',
    'SprayBottle', 'Statue', 'Stool', 'StoveBurner', 'StoveKnob', 'TVStand',
    'TableTopDecor', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox',
    'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato',
    'TomatoSliced', 'Towel', 'TowelHolder', 'VacuumCleaner', 'Vase', 'Watch',
    'WateringCan', 'Window', 'WineBottle']

STUFF_CLASSES = ['AirConditioner', 'Bag', 'Bookcase',
    'CabinetBody', 'Carpet', 'Ceiling', 'CounterSide', 'Cube', 'Curtains',
    'Cylinder', 'Dishwasher', 'DomeLight', 'Door', 'Floor', 'LightFixture',
    'Painting', 'Poster', 'StoveBase', 'TargetCircle', 'Wall', 'OTHERS']

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(1024, 1024),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
Dataset_base = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Perception_Dataset/"
# Train_base = Dataset_base +  'train/'
# Val_seen_base = Dataset_base + 'val_seen/'
Train_base = Dataset_base +  'train/'
Val_unseen_base = Dataset_base + 'val_unseen/'

# Train_base = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Test_Dataset/"
# Val_seen_base = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Test_Dataset/"
# Val_unseen_base = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Test_Dataset/"

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=1,
    train=dict(
        type='CocoPanopticDataset',
        ann_file= Train_base + 'coco_annotation/panoptic.json',
        img_prefix=Train_base + 'replay_images',
        seg_prefix=Train_base + 'replay_segmentations',
        # ins_ann_file= Train_base + 'coco_annotation/detection.json'
        ins_ann_file= None,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='LoadPanopticAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                ratio_range=(0.1, 2.0),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(1024, 1024),
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='DefaultFormatBundle', img_to_float=True),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                    'gt_semantic_seg'
                ])
        ],
    ),
    val=dict(
        type='CocoPanopticDataset',
        ann_file= Val_unseen_base + 'coco_annotation/panoptic.json',
        img_prefix= Val_unseen_base + 'replay_images',
        seg_prefix= Val_unseen_base + 'replay_segmentations',
        classes  = CLASSES,
        # ann_file=
        # '/home/ubuntu/code/mask2former/Dataset/coco_annotation/panoptic.json',
        # img_prefix='/home/ubuntu/code/mask2former/Dataset/replay_images',
        # seg_prefix='/home/ubuntu/code/mask2former/Dataset/replay_segmentations',
        # ins_ann_file= Val_unseen_base + 'coco_annotation/detection.json'
        ins_ann_file= None,
        # '/home/ubuntu/code/mask2former/Dataset/coco_annotation/detection.json'
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
    ),
    test=dict(
        type='CocoPanopticDataset',
        ann_file= Val_unseen_base + 'coco_annotation/panoptic.json',
        img_prefix= Val_unseen_base + 'replay_images',
        seg_prefix= Val_unseen_base + 'replay_segmentations',
        classes = CLASSES,
        # ins_ann_file= Val_unseen_base + 'coco_annotation/detection.json'
        ins_ann_file= None,
        # ann_file=
        # '/home/ubuntu/code/mask2former/Dataset/coco_annotation/panoptic.json',
        # img_prefix='/home/ubuntu/code/mask2former/Dataset/replay_images',
        # seg_prefix='/home/ubuntu/code/mask2former/Dataset/replay_segmentations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        # '/home/ubuntu/code/mask2former/Dataset/coco_annotation/detection.json'
    ),
)
evaluation = dict(
    # interval=5000, metric=['PQ', 'bbox', 'segm'], dynamic_intervals=[(1, 10)])
    interval=100000, metric=['PQ'])
checkpoint_config = dict(
    interval=1000, by_epoch=False, save_last=True, max_keep_ckpts=3)
log_config = dict(
    interval= 10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
# resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
num_things_classes = 120
num_stuff_classes = 21
num_classes = 141
model = dict(
    type='Mask2Former',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
        ),
        pretrain_img_size=384),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[128, 256, 512, 1024],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=120,
        num_stuff_classes=21,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=120,
        num_stuff_classes=21,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True),
    init_cfg=None)
image_size = (1024, 1024)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    # NOTE mofified from 1e-4 as we are finetuning
    lr=0.00003,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat':
            dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.patch_embed.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed':
            dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.2.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.3.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.4.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.5.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.6.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.7.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.8.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.9.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.10.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.11.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.12.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.13.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.14.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.15.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.16.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.17.norm':
            dict(lr_mult=0.1, decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[327778, 355092],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10)
max_iters = 3000000
runner = dict(type='IterBasedRunner', max_iters=max_iters)
interval = 1
dynamic_intervals = [(1, 10)]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
depths = [2, 2, 18, 2]
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'absolute_pos_embed':
    dict(lr_mult=0.1, decay_mult=0.0),
    'relative_position_bias_table':
    dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'query_feat':
    dict(lr_mult=1.0, decay_mult=0.0),
    'level_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'backbone.stages.0.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.2.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.3.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.4.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.5.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.6.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.7.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.8.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.9.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.10.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.11.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.12.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.13.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.14.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.15.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.16.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.17.norm':
    dict(lr_mult=0.1, decay_mult=0.0)
})
samples_per_gpu = 3
resume_from = None
load_from = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Mask2Former/work_dirs/m2former_swin_b/latest.pth"
#load_from = "/nfs/turbo/coe-chaijy/jiayipan/simbot/Summer/Mask2Former/pretrained_weights/mask2former_swin-b-p4-w12-384-in21k_lsj_8x2_50e_coco-panoptic_20220329_230021-3bb8b482.pth"
