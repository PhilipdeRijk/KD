## Results & Models

## Pre-trained Models

|    Backbone     |   Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: |  :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |     1x    |   3.8    |      19.0      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|    R-101-FPN       |     3x    |   5.4   |      15.0 |  41    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_mstrain_640-800_3x_coco.py)      | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.log.json)


## Knowledge Distillation - Single Shot Object Detection

|    Backbone     | Backbone (teacher) |   Lr schd | Method | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: |  :-----: |  :-----: | :------: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | - |     1x    | GT | 3.8    |      19.0      |  36.5  | . | . |
|    R-50-FPN     | R-101-FPN |     1x    | GT + All Features (L2) | 3.8    |      19.0      |  <b>37.0<b> | . | . |
|  | | |  |  | | | | |
|    R-50-FPN     | R-101-FPN |     1x    | Output (Hard) | 3.8    |      19.0      |  34.6 | . | . |
|    R-50-FPN     | R-101-FPN |     1x    | Output (Hard) + All Features (L2) | 3.8    |      19.0      |  35.7 | . | . |
|    R-50-FPN     | R-101-FPN |     1x    | Output (Hard) + All Features (SSIM) | 3.8    |      19.0      |  <b>36.5<b> | . | . |


