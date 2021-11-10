# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from shutil import copyfile
from mmcv.runner import load_checkpoint
from torch.nn.functional import threshold, softmax
import torch.nn.functional as F

from .. import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core.bbox.transforms import bbox2result


@DETECTORS.register_module()
class CustomKnowledgeDistillationSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 kd_settings=None,
                 debug=False):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

        
        self.feature_kd = kd_settings.get('features', False)
        self.logit_kd = kd_settings.get('logits', False)
        self.output_kd = kd_settings.get('output', False)

        self.debug = debug



    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)

        

        ### Distillation ###
        if self.output_kd == 'Hard':
            losses = self.hard_loss(x, out_teacher[0], out_teacher[1], img_metas)
        elif self.output_kd == 'Soft':
            losses = self.soft_loss(x, out_teacher[0], out_teacher[1], img_metas)
        elif self.output_kd == 'GT':      
            losses, outs = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                    gt_labels, gt_bboxes_ignore)


        if self.feature_kd is not False:
            feat_loss = self.loss_feat(x, teacher_x)
            losses.update({'kd_feat_loss': feat_loss})

        # if self.
        # kl_div_loss = self.loss_kl_div(outs, out_teacher)
        # losses.update({'kl_div_loss': kl_div_loss})

        ### VISUZALIZATIONS ###
        if self.debug:
            from mmdet.core.visualization.image import vis_att_map
            from mmcv.visualization import imshow_det_bboxes

            with torch.no_grad():
                teacher_cls_scores, teacher_bbox_preds = self.teacher_model.bbox_head.forward(
                    teacher_x)

            for i, img_meta in enumerate(img_metas):
                
                copyfile(img_meta['filename'], 'att_maps/img.png')


                img_vis = mmcv.imdenormalize(img[i,:,:,:].permute(1,2,0).cpu().numpy(), 
                                            mean=img_meta['img_norm_cfg']['mean'], 
                                            std=img_meta['img_norm_cfg']['std'])
                    
                # plot Teacher bboxes
                imshow_det_bboxes(img_vis, 
                            teacher_bboxes[i][0].cpu().numpy(), 
                            teacher_bboxes[i][1].cpu().numpy(), 
                            class_names = self.CLASSES,
                            score_thr=0.3,
                            show=False,
                            out_file='debug_files/bboxes_teacher' + str(i) + '.png')

                ## Extract Feature Attention Maps ##
                for j, teacher_fmap in enumerate(teacher_x):
                    t = 1.0
                    teacher_attention_mask_mean = torch.mean(
                        torch.abs(teacher_fmap[i]), [0], keepdim=False)
                    size = teacher_attention_mask_mean.size()
                    # size = B x 1 x h x W
                    teacher_attention_mask_mean_sm = teacher_attention_mask_mean.view(
                        1, -1)
                    teacher_attention_mask_mean_sm = torch.softmax(
                        teacher_attention_mask_mean / t, dim=1) * size[-1] * size[-2]
                    teacher_attention_mask_mean_sm = teacher_attention_mask_mean.view(
                        size)
                    vis_att_map(teacher_attention_mask_mean_sm,
                                img_path=img_meta['filename'], index=j)

                for j, teacher_cls_score in enumerate(teacher_cls_scores):
                    teacher_cls_score_batch = teacher_cls_score[i]
                    student_cls_score_batch = outs[0][j][i]
                    teacher_cls_all = teacher_cls_score_batch.reshape(
                        self.bbox_head.num_anchors, self.bbox_head.cls_out_channels, teacher_cls_score.shape[2], teacher_cls_score.shape[3])
                    student_cls_all = student_cls_score_batch.reshape(
                        self.bbox_head.num_anchors, self.bbox_head.cls_out_channels, teacher_cls_score.shape[2], teacher_cls_score.shape[3])

                    sigmoid = torch.nn.Sigmoid()
                    threshold = torch.nn.Threshold(0.2, -1)

                    ### SIGMOID ###
                    # teacher_cls_sgm = sigmoid(teacher_cls_all)
                    # teacher_cls_thr = threshold(teacher_cls_sgm)
                    # # class_labels_per_box = torch.argmax(teacher_cls_sgm, dim=1)
                    # class_labels_per_box = torch.argmax(teacher_cls_sgm, dim=1)
                    # mask_zeros = torch.zeros_like(teacher_cls_sgm)
                    # mask_ones = torch.ones_like(teacher_cls_sgm)
                    # mask4d = torch.where(teacher_cls_sgm < 0.1,
                    #                      mask_zeros, mask_ones)
                    # mask3d, _ = torch.max(mask4d, dim=1)
                    # mask2d, _ = torch.max(mask3d, dim=0)

                    # class_labels_per_box_thr = class_labels_per_box * mask

                    # maxval = torch.max(teacher_cls_all)
                    # minval = torch.min(teacher_cls_all)

                    ### SOFTMAX ###
                    teacher_cls_sfm = F.softmax(teacher_cls_all, dim=1)
                    student_cls_sfm = F.log_softmax(student_cls_all / t, dim=1)

                    mask_zeros = torch.zeros_like(teacher_cls_sfm)
                    mask_ones = torch.ones_like(teacher_cls_sfm)
                    mask4d = torch.where((teacher_cls_sfm < 0.2) & (student_cls_sfm < 0.2),
                                         mask_zeros, mask_ones)
                    mask3d, _ = torch.max(mask4d, dim=1)
                    mask2d, _ = torch.max(mask3d, dim=0)

                    mask4d_ = mask2d.expand(
                        mask4d.shape[0], mask4d.shape[1], mask4d.shape[2], mask4d.shape[3])

                    teacher_cls_sfm = teacher_cls_sfm * mask4d_
                    student_cls_sfm = student_cls_sfm * mask4d_

                    loss = F.kl_div(student_cls_sfm,
                                    teacher_cls_sfm, reduction='mean')

                    # labels, _ = torch.max(class_labels_per_box_thr, dim=0)
                    vis_att_map(
                        mask2d, img_path=img_meta['filename'], index=j+50)

        return losses
    def soft_loss(self):
        pass


    def hard_loss(self, x, cls_scores, bbox_scores, img_metas, score_thr=0.3):
        with torch.no_grad():
            teacher_bboxes = self.teacher_model.bbox_head.get_bboxes(cls_scores, bbox_scores, img_metas=img_metas, rescale=False)

        bboxes_t = [result_t[0] for result_t in teacher_bboxes]
        labels_t = [result_t[1] for result_t in teacher_bboxes]

        if score_thr > 0:
            # scores = teacher_bboxes[:, -1]
            # inds = scores > score_thr
            # bboxes = teacher_bboxes[inds, :]
            # labels = teacher_bboxes[inds]

            scores = [batch_bboxes[:, -1] for batch_bboxes in bboxes_t]
            
            inds = [score > score_thr for score in scores]
            # bboxes_t_out = [i for i, bboxes_t_i in enumerate(bboxes_t)]
            bboxes_t = [bboxes_t_i[inds[i],:-1] for i, bboxes_t_i in enumerate(bboxes_t)]
            labels_t = [labels_t[inds[i]] for i, labels_t in enumerate(labels_t)]

        losses = self.bbox_head.forward_train(x, img_metas, bboxes_t,
                                                    labels_t)
        return losses

    def loss_kl_div(self, outs, outs_teacher, t=1.0):
        cls_scores = outs[0]
        teacher_cls_scores = outs_teacher[0]
        loss = 0

        for j, teacher_cls_score in enumerate(teacher_cls_scores):
            teacher_cls_all = teacher_cls_score.reshape(-1,
                                                        self.bbox_head.num_anchors, self.bbox_head.cls_out_channels, teacher_cls_score.shape[2], teacher_cls_score.shape[3])
            student_cls_all = cls_scores[j].reshape(-1,
                                                    self.bbox_head.num_anchors, self.bbox_head.cls_out_channels, teacher_cls_score.shape[2], teacher_cls_score.shape[3])
            teacher_cls_sfm = F.softmax(teacher_cls_all, dim=2)
            student_cls_sfm = F.log_softmax(student_cls_all / t, dim=2)
            
            
            teacher_cls_sfm.detach()
            loss += F.kl_div(student_cls_sfm,
                teacher_cls_sfm, reduction='mean')

            mask_zeros = torch.zeros_like(teacher_cls_sfm)
            mask_ones = torch.ones_like(teacher_cls_sfm)
            mask5d = torch.where((teacher_cls_sfm < 0.2) & (student_cls_sfm < 0.2),
                                 mask_zeros, mask_ones)
            mask4d, _ = torch.max(mask5d, dim=2)
            mask3d, _ = torch.max(mask4d, dim=1)

            # mask5d_ = torch.cat([mask3d[i].expand(
            # mask5d.shape[1], mask5d.shape[2], mask5d.shape[3], mask5d.shape[4]).unsqueeze(0) for i in range(mask3d.shape[0])])

            mask5d_ = mask3d.unsqueeze(1).unsqueeze(1).expand_as(mask5d)

            # mask5d_ = mask3d.expand(
            # -1, mask5d.shape[1], mask5d.shape[2], mask5d.shape[3], mask5d.shape[4])
            # mask5d_ = mask3d.expand_as(mask5d)

            teacher_cls_sfm = teacher_cls_sfm * mask5d_
            student_cls_sfm = student_cls_sfm * mask5d_

            loss += F.kl_div(student_cls_sfm,
                             teacher_cls_sfm, reduction='mean')

        return loss

    def loss_feat(self, x, teacher_x):
        if self.feature_kd is 'All':
            kd_feat_loss = 0
            for _i in range(len(x)):
                kd_feat_loss += torch.dist(x[_i], teacher_x[_i], p=2)

            return kd_feat_loss * 7e-5

        if self.feature_kd is 'SSIM':
            from kornia.losses import ssim_loss
            kd_feat_loss = 0
            for _i in range(len(x)):
                kd_feat_loss += ssim_loss(x[_i], teacher_x[_i], 3)

            return kd_feat_loss #* 7e-5
        # cls_scores = outs[0]
        # teacher_cls_scores = outs_teacher[0]
        # kd_out_feats_loss = 0
        # for _i in range(len(cls_scores)):
            # kd_out_feats_loss += torch.dist(
                # cls_scores[_i], teacher_cls_scores[_i], p=2)
        # return kd_out_feats_loss * 7e-6

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device, )
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
