_base_ = './retinanet_r50_fpn_1x_cityscapes.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='checkpoints/resnet101-63fe2227.pth')))
# load_from="checkpoints/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth"