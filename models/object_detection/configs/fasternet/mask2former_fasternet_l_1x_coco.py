_base_ = [
    '../__base__/models/mask2former_r50_lsj_8x2_50e_coco-panoptic.py',
    '../common/mstrain_3x_coco_panoptic.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_l',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../model_ckpt/fasternet_l-epoch=299-val_acc1=83.5060.pth',
            ),
        # init_cfg=None,
        ),
)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)