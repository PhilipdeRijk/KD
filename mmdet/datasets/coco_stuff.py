import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from prettytable import PrettyTable

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from terminaltables import AsciiTable
# from mmdet.core import resize

from mmdet.core import eval_recalls
from mmdet.core.evaluation.semantic_metrics import eval_metrics
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class COCOStuffDataset(CustomDataset):
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    CLASSES_STUFF = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood')

    PALETTE = [
        [0, 192, 64],
        [0, 192, 64],
        [0, 64, 96],
        [128, 192, 192],
        [0, 64, 64],
        [0, 192, 224],
        [0, 192, 192],
        [128, 192, 64],
        [0, 192, 96],
        [128, 192, 64],
        [128, 32, 192],
        [0, 0, 224],
        [0, 0, 64],
        [0, 160, 192],
        [128, 0, 96],
        [128, 0, 192],
        [0, 32, 192],
        [128, 128, 224],
        [0, 0, 192],
        [128, 160, 192],
        [128, 128, 0],
        [128, 0, 32],
        [128, 32, 0],
        [128, 0, 128],
        [64, 128, 32],
        [0, 160, 0],
        [0, 0, 0],
        [192, 128, 160],
        [0, 32, 0],
        [0, 128, 128],
        [64, 128, 160],
        [128, 160, 0],
        [0, 128, 0],
        [192, 128, 32],
        [128, 96, 128],
        [0, 0, 128],
        [64, 0, 32],
        [0, 224, 128],
        [128, 0, 0],
        [192, 0, 160],
        [0, 96, 128],
        [128, 128, 128],
        [64, 0, 160],
        [128, 224, 128],
        [128, 128, 64],
        [192, 0, 32],
        [128, 96, 0],
        [128, 0, 192],
        [0, 128, 32],
        [64, 224, 0],
        [0, 0, 64],
        [128, 128, 160],
        [64, 96, 0],
        [0, 128, 192],
        [0, 128, 160],
        [192, 224, 0],
        [0, 128, 64],
        [128, 128, 32],
        [192, 32, 128],
        [0, 64, 192],
        [0, 0, 32],
        [64, 160, 128],
        [128, 64, 64],
        [128, 0, 160],
        [64, 32, 128],
        [128, 192, 192],
        [0, 0, 160],
        [192, 160, 128],
        [128, 192, 0],
        [128, 0, 96],
        [192, 32, 0],
        [128, 64, 128],
        [64, 128, 96],
        [64, 160, 0],
        [0, 64, 0],
        [192, 128, 224],
        [64, 32, 0],
        [0, 192, 128],
        [64, 128, 224],
        [192, 160, 0],
        [0, 192, 0],
        [192, 128, 96],
        [192, 96, 128],
        [0, 64, 128],
        [64, 0, 96],
        [64, 224, 128],
        [128, 64, 0],
        [192, 0, 224],
        [64, 96, 128],
        [128, 192, 128],
        [64, 0, 224],
        [192, 224, 128],
        [128, 192, 64],
        [192, 0, 96],
        [192, 96, 0],
        [128, 64, 192],
        [0, 128, 96],
        [0, 224, 0],
        [64, 64, 64],
        [128, 128, 224],
        [0, 96, 0],
        [64, 192, 192],
        [0, 128, 224],
        [128, 224, 0],
        [64, 192, 64],
        [128, 128, 96],
        [128, 32, 128],
        [64, 0, 192],
        [0, 64, 96],
        [0, 160, 128],
        [192, 0, 64],
        [128, 64, 224],
        [0, 32, 128],
        [192, 128, 192],
        [0, 64, 224],
        [128, 160, 128],
        [192, 128, 0],
        [128, 64, 32],
        [128, 32, 64],
        [192, 0, 128],
        [64, 192, 32],
        [0, 160, 64],
        [64, 0, 0],
        [192, 192, 160],
        [0, 32, 64],
        [64, 128, 128],
        [64, 192, 160],
        [128, 160, 64],
        [64, 128, 0],
        [192, 192, 32],
        [128, 96, 192],
        [64, 0, 128],
        [64, 64, 32],
        [0, 224, 192],
        [192, 0, 0],
        [192, 64, 160],
        [0, 96, 192],
        [192, 128, 128],
        [64, 64, 160],
        [128, 224, 192],
        [192, 128, 64],
        [192, 64, 32],
        [128, 96, 64],
        [192, 0, 192],
        [0, 192, 32],
        [64, 224, 64],
        [64, 0, 64],
        [128, 192, 160],
        [64, 96, 64],
        [64, 128, 192],
        [0, 192, 160],
        [192, 224, 64],
        [64, 128, 64],
        [128, 192, 32],
        [192, 32, 192],
        [64, 64, 192],
        [0, 64, 32],
        [64, 160, 192],
        [192, 64, 64],
        [128, 64, 160],
        [64, 32, 192],
        [192, 192, 192],
        [0, 64, 160],
        [192, 160, 192],
        [192, 192, 0],
        [128, 64, 96],
        [192, 32, 64],
        [192, 64, 128],
        [64, 192, 96],
        [64, 160, 64],
        [64, 64, 0],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names_stuff = self.CLASSES_STUFF
        self.palette = self.PALETTE
        self.ignore_index=255
        self.seg_map_suffix='_labelTrainIds.png'
        self.reduce_zero_label = False
        self.label_map = None
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        # seg_map = img_info['filename'].replace('jpg', 'png')
        # self.ignore_index=255
        # self.seg_map_suffix='_labelTrainIds.png'
        seg_map = img_info['filename'].replace('.jpg', self.seg_map_suffix)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            if idx == 3:
                break
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    # def results2json(self, results, outfile_prefix):
    #     """Dump the detection results to a COCO style json file.

    #     There are 3 types of results: proposals, bbox predictions, mask
    #     predictions, and they have different data types. This method will
    #     automatically recognize the type, and dump them to json files.

    #     Args:
    #         results (list[list | tuple | ndarray]): Testing results of the
    #             dataset.
    #         outfile_prefix (str): The filename prefix of the json files. If the
    #             prefix is "somepath/xxx", the json files will be named
    #             "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
    #             "somepath/xxx.proposal.json".

    #     Returns:
    #         dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
    #             values are corresponding filenames.
    #     """
    #     result_files = dict()
    #     # if isinstance(results[0], list):
    #     if isinstance(results, list):
    #         json_results = self._det2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'
    #         mmcv.dump(json_results, result_files['bbox'])
    #     elif isinstance(results[0], tuple):
    #         json_results = self._segm2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'
    #         result_files['segm'] = f'{outfile_prefix}.segm.json'
    #         mmcv.dump(json_results[0], result_files['bbox'])
    #         mmcv.dump(json_results[1], result_files['segm'])
    #     elif isinstance(results[0], np.ndarray):
    #         json_results = self._proposal2json(results)
    #         result_files['proposal'] = f'{outfile_prefix}.proposal.json'
    #         mmcv.dump(json_results, result_files['proposal'])
    #     else:
    #         raise TypeError('invalid type of results')
    #     return result_files

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def get_gt_seg_maps(self, shape=None, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for data_info in self.data_infos:
            # temporary cityscapes identifier
            if len(self.CLASSES) == 8:
                seg_map_path = osp.join(self.seg_prefix, data_info['segm_file'].replace('_labelIds.png', self.seg_map_suffix))
            else:
                seg_map_path = osp.join(self.seg_prefix, data_info['filename'].replace('.jpg', self.seg_map_suffix))
            seg_map = seg_map_path
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                
                # # Check width vs Height !!!
                # if shape != gt_seg_map.shape:
                #     gt_seg_map = mmcv.imresize(
                #         gt_seg_map,
                #         shape,
                #         interpolation='nearest',
                #         backend='cv2')
                
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast', 'mIoU']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # if 'mIoU' in metrics and 'bbox' in metrics or 'bbox' in metrics and len(results[0]) > 1: # len(results[0]) > 1:
        # if self.old_eval:
        #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if isinstance(results[0], tuple):
            det_results = [result[0][0] for result in results]
            seg_results = [result[1][0] for result in results]
            result_files, tmp_dir = self.format_results(det_results, jsonfile_prefix)
        else:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'mIoU':
                eval_results_segmentation = {}
                # shape = (seg_results[0].shape[1],seg_results[0].shape[0])
                gt_seg_maps = self.get_gt_seg_maps()
                num_stuff_classes = len(self.class_names_stuff)
                # seg_results = [seg_result.to('cpu') for seg_result in seg_results]
                
                ret_metrics = eval_metrics(
                    seg_results,
                    gt_seg_maps,
                    num_stuff_classes,
                    self.ignore_index,
                    metric,
                    label_map=self.label_map,
                    reduce_zero_label=self.reduce_zero_label)
                
                
                 # summary table
                ret_metrics_summary = OrderedDict({
                    ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })
                # each class table
                ret_metrics.pop('aAcc', None)
                ret_metrics_class = OrderedDict({
                    ret_metric: np.round(ret_metric_value * 100, 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })
                ret_metrics_class.update({'Class': self.class_names_stuff})
                ret_metrics_class.move_to_end('Class', last=False)
                
                # for logger
                class_table_data = PrettyTable()
                for key, val in ret_metrics_class.items():
                    class_table_data.add_column(key, val)

                summary_table_data = PrettyTable()
                for key, val in ret_metrics_summary.items():
                    if key == 'aAcc':
                        summary_table_data.add_column(key, [val])
                    else:
                        summary_table_data.add_column('m' + key, [val])

                print_log('per class results:', logger)
                print_log('\n' + class_table_data.get_string(), logger=logger)
                print_log('Summary:', logger)
                print_log('\n' + summary_table_data.get_string(), logger=logger)

                # each metric dict
                for key, value in ret_metrics_summary.items():
                    if key == 'aAcc':
                        eval_results[key] = value / 100.0
                    else:
                        eval_results['m' + key] = value / 100.0

                # ret_metrics_class.pop('Class', None)
                # for key, value in ret_metrics_class.items():
                #     eval_results.update({
                #         key + '.' + str(name): value[idx] / 100.0
                #         for idx, name in enumerate(class_names_stuff)
                #     })
                continue
                
            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
