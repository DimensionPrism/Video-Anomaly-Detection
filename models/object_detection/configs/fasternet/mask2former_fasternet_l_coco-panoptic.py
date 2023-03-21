_base_ = [
    '../_base_/models/mask2former_r50_lsj_8x2_50e_coco-panoptic.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='fasternet_l',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./model_ckpt/fasternet_l-epoch.299-val_acc1.83.5060.pth',
            ),
        # init_cfg=None,
        ),
)