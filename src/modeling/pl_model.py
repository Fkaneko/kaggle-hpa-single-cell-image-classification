import argparse
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision
import yaml
from kornia.contrib.extract_patches import ExtractTensorPatches
from torchvision.utils import make_grid

from src.dataset.datamodule import IMG_MEAN, IMG_STD
from src.modeling.losses import ArcFaceLoss, FocalLoss
from src.modeling.sc_cam.resnet38_cls import SCCAM, get_pred_head


class LitModel(pl.LightningModule):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        # self.save_hyperparameters()  # type: ignore
        self.hparams = args  # type: ignore[misc]

        if self.hparams.segm_label_dir is not None:
            print("\t >>do segmentation")
            encoder_weights = None if self.hparams.not_imagenet_weight else "imagenet"
            self.model = smp.Unet(
                encoder_name=self.hparams.timm_model_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=self.hparams.num_classes + 1,
            )
        else:
            print("\t >>do classification")
            if self.hparams.load_from_r0_to_r1:
                orig_round_nb = self.hparams.round_nb
                self.hparams.round_nb = 0
            self.model = SCCAM(
                timm_model_name=self.hparams.timm_model_name,
                num_classes=self.hparams.num_classes,
                k_cluster=self.hparams.k_cluster,
                round_nb=self.hparams.round_nb,
                drop_rate=self.hparams.drop_rate,
                pretrained=not self.hparams.not_imagenet_weight,
                use_arc_face=self.hparams.use_arc_face,
                k_arc_center=self.hparams.k_arc_center,
            )
            if self.hparams.load_from_r0_to_r1:
                self.hparams.round_nb = orig_round_nb
                self.model.round_nb = orig_round_nb
        if self.hparams.num_inchannels != 3:
            patch_first_conv(self.model, in_channels=self.hparams.num_inchannels)

        if self.hparams.channels_last:
            # Need to be done once, after model initialization (or load)
            self.model = self.model.to(memory_format=torch.channels_last)

        # self.criterion = torch.nn.BCEWithLogitsLoss()
        if self.hparams.use_arc_face:
            self.criterion = ArcFaceLoss(
                s=30.0,
                m=0.5,
                gamma=self.hparams.arc_gamma,
                classify_loss=self.hparams.arc_cls_loss,
            )
            if self.hparams.round_nb > 0:
                self.criterion_sub = ArcFaceLoss(
                    s=30.0,
                    m=0.5,
                    gamma=self.hparams.arc_gamma,
                    classify_loss=self.hparams.arc_cls_loss,
                )
        elif self.hparams.loss == "focal":
            self.criterion = FocalLoss()  # type:ignore[assignment]
            if self.hparams.round_nb > 0:
                self.criterion_sub = FocalLoss()  # type:ignore[assignment]
        else:
            raise NotImplementedError

        if self.hparams.num_classes == 19:
            self.f1_class = list(set(range(19)) - set([17, 15, 11, 18]))
        else:
            self.f1_class = list(set(range(self.hparams.num_classes)))
        # if self.hparams.num_classes

        self.f1 = pl.metrics.F1(num_classes=len(self.f1_class), average="macro")
        if self.hparams.round_nb > 0:
            self.f1_sub = pl.metrics.F1(num_classes=len(self.f1_class), average="macro")

        self.val_sync_dist = self.hparams.gpus > 1

        self.is_debug = self.hparams.is_debug

        self._set_image_normalization()

        if self.hparams.is_puzzle_cam:
            self.p_n_splits = self.hparams.p_n_splits
            self.p_nrow = np.sqrt(self.p_n_splits).astype(np.int)
            self.split_img = ExtractTensorPatches(
                window_size=self.hparams.input_size // self.p_nrow,
                stride=self.hparams.input_size // self.p_nrow,
            )
            self.l1_loss = torch.nn.L1Loss()
            self.p_rec_weight = 0.0

        # if flip_tta:
        #     self.flip_tta = [
        #         torchvision.transforms.functional.hflip,
        #         torchvision.transforms.functional.vflip,
        #     ]
        # else:
        #     self.flip_tta = []

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--timm_model_name", type=str, default="resnet50")
        parser.add_argument(
            "--channels_last",
            action="store_true",
            help="whether use channels last memory format",
        )
        parser.add_argument(
            "--round_nb",
            default="0",
            type=int,
            help="the round number of the training classifier",
        )
        parser.add_argument(
            "--k_cluster", default="10", type=int, help="the number of the sub-category"
        )
        parser.add_argument("--drop_rate", type=float, default=0.0)
        parser.add_argument("--loss", type=str, default="focal", help="loss function")
        parser.add_argument(
            "--sub_label_dir",
            type=str,
            default="./save_uni/version_17",
            help="sub label directory by SC-CAM",
        )
        parser.add_argument(
            "--segm_label_dir",
            type=str,
            default=None,
            help="psuedo segmentation label directory",
        )
        parser.add_argument(
            "--segm_thresh",
            type=float,
            default=0.15,
            help="confidence threshold of psuedo segmentation label",
        )
        parser.add_argument(
            "--subcls_loss_weight",
            default="5.0",
            type=float,
            help="the weight multiply to the sub-category loss",
        )
        parser.add_argument(
            "--pred_thresh",
            default="0.5",
            type=float,
            help="prediction threshold at test step",
        )
        parser.add_argument(
            "--not_imagenet_weight",
            action="store_true",
            help="whether use imagenet pretrained weight",
        )
        parser.add_argument(
            "--is_puzzle_cam",
            action="store_true",
            help="whether puzzle cam style siamese training",
        )
        parser.add_argument(
            "--p_n_splits",
            default="4",
            type=int,
            help="the number of splits at puzzle cam input image",
        )
        parser.add_argument(
            "--p_cls_weight",
            default="1.0",
            type=float,
            help="loss weight on classification at puzzle cam",
        )
        parser.add_argument(
            "--p_rec_weight_max",
            default="4.0",
            type=float,
            help="max loss weight on reconstructiono at puzzle cam",
        )
        parser.add_argument(
            "--reco_with",
            default="cam",
            choices=["cam", "feat_map"],
            type=str,
            help="target of reconstructiono at puzzle cam",
        )
        parser.add_argument(
            "--apply_p_sub",
            action="store_true",
            help="whether apply puzzle loss on subcenter head",
        )
        parser.add_argument(
            "--use_true_target",
            action="store_true",
            help="whether use True target for cam reconstructiono loss",
        )
        parser.add_argument(
            "--use_arc_face", action="store_true", help="whether use arc face loss"
        )
        parser.add_argument(
            "--k_arc_center",
            default=1,
            type=int,
            help="the number of arcface subsenter",
        )
        parser.add_argument(
            "--arc_gamma",
            default=0.0,
            type=float,
            help="the weight of normal loss on arcface loss",
        )
        parser.add_argument(
            "--arc_cls_loss",
            default="bce",
            type=str,
            help="the weight of normal ce loss on arcface loss",
        )

        return parent_parser

    def forward(self, x):
        x = self.model(x, round_nb=self.hparams.round_nb)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.hparams.channels_last:
            # Need to be done for every input
            inputs = inputs.to(memory_format=torch.channels_last)

        targets = batch["target"]

        if self.hparams.segm_label_dir is not None:
            targets = batch["target_segm"]
            outputs = self.model(inputs)
            pred = outputs.sigmoid()

        elif self.hparams.round_nb == 0:
            # normal forward
            feat_map = self.model.backbone(inputs)
            outputs, _, pred = self.model.head(feat_map, round_nb=self.hparams.round_nb)
            if self.hparams.is_puzzle_cam:
                # puzzle forward
                inputs_split = self._puzzle_split(inputs=inputs)
                feat_map_split = self.model.backbone(inputs_split)
                outputs_split, _, _ = self.model.head(
                    feat_map_split, round_nb=self.hparams.round_nb
                )
                loss_p_cls, loss_rec = self.calc_puzzle_loss(
                    targets=targets,
                    outputs_split=outputs_split,
                    feat_map=feat_map,
                    feat_map_split=feat_map_split,
                    fc_weight=self.model.fc8.weight,
                    reco_with=self.hparams.reco_with,
                    use_true_target=self.hparams.use_true_target,
                )
                self._get_puzzle_log(loss_p_cls, loss_rec, header="train")
        else:
            targets_sub = batch["target_sub"]
            feat_map = self.model.backbone(inputs)
            outputs, _, pred, out_sub, pred_sub = self.model.head(
                feat_map, round_nb=self.hparams.round_nb
            )
            loss_sub = self.criterion_sub(out_sub, targets_sub)
            self.log("train_loss_sub", loss_sub, on_step=True, on_epoch=True)
            self.log(
                "train_f1_sub",
                self.f1_sub(
                    pred_sub[:, [self.f1_class]], targets_sub[:, [self.f1_class]]
                ),
                on_step=True,
                on_epoch=True,
            )
            if self.hparams.is_puzzle_cam:
                # puzzle forward
                inputs_split = self._puzzle_split(inputs=inputs)
                feat_map_split = self.model.backbone(inputs_split)
                outputs_split, _, _, out_sub_split, _ = self.model.head(
                    feat_map_split, round_nb=self.hparams.round_nb
                )
                loss_p_cls, loss_rec = self.calc_puzzle_loss(
                    targets=targets,
                    outputs_split=outputs_split,
                    feat_map=feat_map,
                    feat_map_split=feat_map_split,
                    fc_weight=self.model.fc8_20.weight,
                    reco_with=self.hparams.reco_with,
                    use_true_target=self.hparams.use_true_target,
                )
                self._get_puzzle_log(loss_p_cls, loss_rec, header="train")
                if self.hparams.apply_p_sub:
                    loss_p_cls_sub, loss_rec_sub = self.calc_puzzle_loss(
                        targets=targets_sub,
                        outputs_split=out_sub_split,
                        feat_map=feat_map,
                        feat_map_split=feat_map_split,
                        fc_weight=self.model.fc8_200.weight,
                        reco_with=self.hparams.reco_with,
                        use_true_target=self.hparams.use_true_target,
                    )
                    self._get_puzzle_log(
                        loss_p_cls_sub, loss_rec_sub, header="train_sub"
                    )

        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train_f1",
            self.f1(pred[:, [self.f1_class]], targets[:, [self.f1_class]]),
            on_step=True,
            on_epoch=True,
        )

        if self.hparams.round_nb > 0:
            loss += self.hparams.subcls_loss_weight * loss_sub

        if self.hparams.is_puzzle_cam:
            loss += (
                self.hparams.p_cls_weight * loss_p_cls + self.p_rec_weight * loss_rec
            )
            if self.hparams.apply_p_sub and (self.hparams.round_nb > 0):
                loss += (
                    self.hparams.p_cls_weight * loss_p_cls_sub
                    + self.p_rec_weight * loss_rec_sub
                )
        return loss

    def _puzzle_split(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_split = self.split_img(inputs)
        shape_ = inputs_split.shape
        inputs_split = inputs_split.view(
            shape_[0] * shape_[1], shape_[2], shape_[3], shape_[4]
        )

        if self.is_debug:
            img = make_grid(
                inputs_split[self.p_n_splits : self.p_n_splits * 2],
                nrow=self.p_nrow,
                padding=0,
            )
            assert np.all((img == inputs[1]).cpu().numpy())

            inputs_merge = make_grid(inputs_split, nrow=self.p_nrow, padding=0)
            shape_ = inputs_merge.shape

            inputs_merge = torch.stack(
                torch.split(inputs_merge, shape_[-1], dim=1), dim=0
            )
            assert np.all((inputs_merge == inputs).cpu().numpy())
        return inputs_split

    def _plot_tensor(self, inputs: torch.Tensor) -> None:
        img = (inputs.cpu() * self._img_std + self._img_mean).numpy()
        img = (img.transpose(0, 2, 3, 1)[0][..., :3] * 255).astype(np.uint8)
        plt.imshow(img)

    def _puzzle_merge(self, outputs: torch.Tensor, nrow: int = 2) -> torch.Tensor:
        # (ba*p_n_splits, 2048, map_h//(p_n_splits /2), map_w//(p_n_splits/2))
        outputs_merge = make_grid(outputs, nrow=nrow, padding=0)
        shape_ = outputs_merge.shape
        # (2048, map_h//p_n_splits * ba*p_n_splits, map_w)
        outputs_merge = torch.stack(
            torch.split(outputs_merge, shape_[-1], dim=1), dim=0
        )
        # (2048, map_h//p_n_splits * ba*p_n_splits, map_w)
        return outputs_merge

    def calc_puzzle_loss(
        self,
        targets: torch.Tensor,
        outputs_split: torch.Tensor,
        feat_map: torch.Tensor,
        feat_map_split: torch.Tensor,
        fc_weight: torch.Tensor,
        reco_with: str = "cam",
        use_true_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # puzzle cls loss
        targets_paded = targets.unsqueeze(-1).expand(
            targets.shape[0], targets.shape[1], self.p_n_splits
        )
        targets_paded = (
            targets_paded.transpose(2, 1).contiguous().view(-1, targets.shape[1])
        )
        loss_p_cls = self.criterion(outputs_split, targets_paded)
        # reconstructiono loss
        if use_true_target:
            cam_labels = targets
        else:
            cam_labels = torch.ones_like(targets)

        if reco_with == "cam":
            cam = self.model.calc_cam(feat_map, fc_weight)
            cam_split = self.model.calc_cam(feat_map_split, fc_weight)
            cam_merge = self._puzzle_merge(cam_split, nrow=self.p_nrow)

            cam = self.convert_cam_to_mask(
                cam=cam, label=cam_labels, orig_img_size=None
            )
            cam_merge = self.convert_cam_to_mask(
                cam=cam_merge, label=cam_labels, orig_img_size=None
            )
        elif reco_with == "feat_map":
            cam = feat_map
            cam_merge = self._puzzle_merge(feat_map_split, nrow=self.p_nrow)
        else:
            raise NotImplementedError

        loss_rec = self.l1_loss(cam_merge, cam)
        return loss_p_cls, loss_rec

    def _get_puzzle_log(
        self, loss_p_cls: torch.Tensor, loss_rec: torch.Tensor, header: str = "train"
    ) -> None:
        p_cls_header = header + "_loss_p_cls"
        rec_header = header + "_loss_rec"
        sync_dist = False if header.find("train") > -1 else self.val_sync_dist
        self.log(p_cls_header, loss_p_cls, sync_dist=sync_dist)
        self.log(rec_header, loss_rec, sync_dist=sync_dist)

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        if self.hparams.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        targets = batch["target"]

        if self.hparams.segm_label_dir is not None:
            targets = batch["target_segm"]
            outputs = self.model(inputs)
            pred = outputs.sigmoid()

        elif self.hparams.round_nb == 0:
            # normal forward
            feat_map = self.model.backbone(inputs)
            outputs, _, pred = self.model.head(feat_map, round_nb=self.hparams.round_nb)
            if self.hparams.is_puzzle_cam:
                # puzzle forward
                inputs_split = self._puzzle_split(inputs=inputs)
                feat_map_split = self.model.backbone(inputs_split)
                outputs_split, _, _ = self.model.head(
                    feat_map_split, round_nb=self.hparams.round_nb
                )
                loss_p_cls, loss_rec = self.calc_puzzle_loss(
                    targets=targets,
                    outputs_split=outputs_split,
                    feat_map=feat_map,
                    feat_map_split=feat_map_split,
                    fc_weight=self.model.fc8.weight,
                    reco_with=self.hparams.reco_with,
                    use_true_target=self.hparams.use_true_target,
                )
                self._get_puzzle_log(loss_p_cls, loss_rec, header="val")
            # else:
            #     outputs, _, pred = self.model(inputs, round_nb=self.hparams.round_nb)
        else:
            targets_sub = batch["target_sub"]
            feat_map = self.model.backbone(inputs)
            outputs, _, pred, out_sub, pred_sub = self.model.head(
                feat_map, round_nb=self.hparams.round_nb
            )
            loss_sub = self.criterion_sub(out_sub, targets_sub)
            self.log("val_loss_sub", loss_sub, sync_dist=self.val_sync_dist)
            self.log(
                "val_f1_sub",
                self.f1_sub(
                    pred_sub[:, [self.f1_class]], targets_sub[:, [self.f1_class]]
                ),
                sync_dist=self.val_sync_dist,
            )
            if self.hparams.is_puzzle_cam:
                # puzzle forward
                inputs_split = self._puzzle_split(inputs=inputs)
                feat_map_split = self.model.backbone(inputs_split)
                outputs_split, _, _, out_sub_split, _ = self.model.head(
                    feat_map_split, round_nb=self.hparams.round_nb
                )
                loss_p_cls, loss_rec = self.calc_puzzle_loss(
                    targets=targets,
                    outputs_split=outputs_split,
                    feat_map=feat_map,
                    feat_map_split=feat_map_split,
                    fc_weight=self.model.fc8_20.weight,
                    reco_with=self.hparams.reco_with,
                    use_true_target=self.hparams.use_true_target,
                )
                self._get_puzzle_log(loss_p_cls, loss_rec, header="val")
                if self.hparams.apply_p_sub:
                    loss_p_cls_sub, loss_rec_sub = self.calc_puzzle_loss(
                        targets=targets_sub,
                        outputs_split=out_sub_split,
                        feat_map=feat_map,
                        feat_map_split=feat_map_split,
                        fc_weight=self.model.fc8_200.weight,
                        reco_with=self.hparams.reco_with,
                        use_true_target=self.hparams.use_true_target,
                    )
                    self._get_puzzle_log(loss_p_cls_sub, loss_rec_sub, header="val_sub")

        loss = self.criterion(outputs, targets)

        self.log("val_loss", loss, sync_dist=self.val_sync_dist)
        self.log(
            "val_f1",
            self.f1(pred[:, [self.f1_class]], targets[:, [self.f1_class]]),
            sync_dist=self.val_sync_dist,
        )
        if batch_idx == 0 and (self.hparams.segm_label_dir is not None):
            vis_size = 256
            num_ = 3
            inputs = F.interpolate(
                inputs, vis_size, mode="bilinear", align_corners=False
            )
            pred = F.interpolate(pred, vis_size, mode="bilinear", align_corners=False)
            targets = F.interpolate(
                targets, vis_size, mode="bilinear", align_corners=False
            )

            labeler = (np.arange(targets.shape[1]) + 1)[
                np.newaxis, :, np.newaxis, np.newaxis
            ]
            pred = pred[:num_].detach().cpu().numpy()
            targets = targets[:num_].cpu().numpy()

            # input renormalize and reformat, always use 3 channels for vis
            inputs = (inputs[:num_].cpu() * self._img_std + self._img_mean).numpy()
            inputs = (inputs.transpose(0, 2, 3, 1)[..., :3] * 255).astype(np.uint8)
            inputs = np.concatenate(inputs, axis=0)
            out = np.concatenate(
                [
                    np.sum((targets * labeler)[:, :-1], axis=1, keepdims=True),
                    np.sum(
                        ((pred > 0.5) * labeler)[:, :-1], axis=1, keepdims=True
                    ).clip(0, 16 * 3),
                ],
                axis=-1,
            )
            fig, ax = plt.subplots(
                1, 2, figsize=(10, 10), gridspec_kw=dict(wspace=-0.2)
            )
            ax[0].imshow(inputs)
            ax[1].imshow(np.concatenate(out, axis=-2).squeeze().astype(np.uint8))
            ax[0].axis("off")
            ax[1].axis("off")
            self.logger.experiment.add_figure(
                "prediction_fig",
                fig,
                global_step=self.global_step,
            )
            plt.close()

        elif (
            batch_idx == 0
            and (self.hparams.segm_label_dir is None)
            and (not self.hparams.use_arc_face)
        ):
            # cam visualize
            num_ = 2
            pred_thresh = 0.5
            hm_rate = 0.4
            hm_style = plt.cm.jet
            is_small_size = True
            if self.hparams.round_nb == 0:
                cam = self.model.forward_cam(inputs)
            else:
                cam, _ = self.model.forward_two_cam(inputs)

            cam = cam.detach()[:num_]
            pred = pred.detach()[:num_]
            targets = targets[:num_]
            bkg_label = torch.zeros_like(targets)
            bkg_label[..., -1] = 1
            if is_small_size:
                inputs = F.interpolate(
                    inputs, 256, mode="bilinear", align_corners=False
                )
            # input renormalize and reformat, always use 3 channels for vis
            inputs = (inputs[:num_].cpu() * self._img_std + self._img_mean).numpy()
            inputs = (inputs.transpose(0, 2, 3, 1)[..., :3] * 255).astype(np.uint8)

            # pick class and normarize cam
            cam_gt = self.convert_cam_to_mask(
                cam.clone(), targets, orig_img_size=inputs.shape[2]
            )
            cam_pred = self.convert_cam_to_mask(
                cam.clone(), pred >= pred_thresh, orig_img_size=inputs.shape[2]
            )
            cam_bkg = self.convert_cam_to_mask(
                cam.clone(), bkg_label, orig_img_size=inputs.shape[2]
            )

            out_gt = self.overlay_cam_on_input(
                inputs=inputs,
                cam_mask=cam_gt.cpu().numpy(),
                targets=targets.cpu().numpy(),
                batch_num=num_,
                hm_rate=hm_rate,
                hm_style=hm_style,
                stack_axis=0,
            )
            out_pred = self.overlay_cam_on_input(
                inputs=inputs,
                cam_mask=cam_pred.cpu().numpy(),
                targets=pred.cpu().numpy(),
                batch_num=num_,
                hm_rate=hm_rate,
                hm_style=hm_style,
                stack_axis=0,
                threshold=pred_thresh,
            )
            out_bkg = self.overlay_cam_on_input(
                inputs=inputs,
                cam_mask=cam_bkg.cpu().numpy(),
                targets=bkg_label.cpu().numpy(),
                batch_num=num_,
                hm_rate=hm_rate,
                hm_style=hm_style,
                stack_axis=0,
            )
            inputs = np.concatenate(inputs, axis=0)
            out = np.concatenate((inputs, out_gt, out_pred, out_bkg), axis=1)
            if self.hparams.is_debug:
                plt.imshow(out)
                plt.show()
            self.logger.experiment.add_image(
                "prediction", out, global_step=self.global_step, dataformats="HWC"
            )
        return loss

    def convert_cam_to_mask(
        self,
        cam: torch.Tensor,
        label: torch.Tensor,
        orig_img_size: Optional[Union[int, list, tuple]] = None,
        is_flip: bool = False,
    ) -> torch.Tensor:
        """
        modify from
        https://github.com/Juliachang/SC-CAM/blob/master/infer_cls.py
        """
        cam = cam * label.clone().view(label.shape[0], -1, 1, 1)
        cam = cam / (F.adaptive_max_pool2d(cam, output_size=(1, 1)) + 1.0e-5)

        if orig_img_size is not None:
            cam = F.interpolate(
                cam, orig_img_size, mode="bilinear", align_corners=False
            )

        if is_flip:
            cam = torchvision.transforms.functional.hflip(cam)
        return cam

    @staticmethod
    def overlay_cam_on_input(
        inputs: np.ndarray,
        cam_mask: np.ndarray,
        targets: np.ndarray,
        batch_num: int = 2,
        hm_rate: float = 0.4,
        hm_style: plt.cm = plt.cm.jet,
        is_vstack: bool = False,
        stack_axis: Optional[int] = None,
        threshold: float = 1.0,
    ) -> np.ndarray:
        hm_norm = np.stack([np.zeros_like(cam_mask)] * 3, axis=-1)
        for i in range(batch_num):
            for j, class_ind in enumerate(np.where(targets[i] >= threshold)[0]):
                hm = hm_style(cam_mask[i, class_ind])[:, :, :3] * hm_rate
                hm_norm[i, j] = hm * 255

        hm_norm = np.sum(hm_norm, axis=1)
        if is_vstack:
            stack_axis = 0
        if stack_axis is not None:
            hm_norm = np.concatenate(hm_norm, axis=stack_axis)
            inputs = np.concatenate(inputs, axis=stack_axis)
        # overley heatmap on input
        out_ = (hm_norm + np.array(inputs).astype(np.float)) / (hm_rate + 1.0)
        out_ = (out_ / np.max(out_) * 255).astype(np.uint8)
        return out_

    def _set_image_normalization(self) -> None:
        img_mean = IMG_MEAN[: self.hparams.num_inchannels]  # type: ignore[union-attr]
        img_std = IMG_STD[: self.hparams.num_inchannels]  # type: ignore[union-attr]
        self._img_std = torch.tensor(
            np.array(img_std)[None, :, None, None], device=self.device
        )
        self._img_mean = torch.tensor(
            np.array(img_mean)[None, :, None, None], device=self.device
        )

    def on_test_epoch_start(self):
        self.cam_dir = get_cam_dir(self.hparams.ckpt_path)
        self.cam_dir.mkdir(exist_ok=True, parents=True)
        print('>> make directory for cam: "{}"'.format(str(self.cam_dir)))

    def test_step(self, batch, batch_idx, save_npy=True):
        inputs = batch["image"]
        if self.hparams.channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        if self.hparams.segm_label_dir is not None:
            outputs = self.model(inputs)
            cam = outputs.sigmoid()
            pred = torch.empty(0)
        else:
            feat_map = self.model.backbone(inputs)
            if self.hparams.round_nb == 0:
                cam = self.model.calc_cam(feat_map, self.model.fc8.weight)
                outputs, _, pred = self.model.head(
                    feat_map, round_nb=self.hparams.round_nb
                )
            else:
                cam = self.model.calc_cam(feat_map, self.model.fc8_20.weight)
                outputs, _, pred, out_sub, pred_sub = self.model.head(
                    feat_map, round_nb=self.hparams.round_nb
                )

        # if len(self.flip_tta) > 0:
        #     for flip_trans in self.flip_tta:
        #         tta_inputs = flip_trans(inputs)
        #         tta_outputs = self.model(tta_inputs)
        #         tta_outputs = tta_outputs.softmax(dim=1)
        #         outputs += flip_trans(tta_outputs)
        #     outputs *= torch.tensor(
        #         1.0 / (len(self.flip_tta) + 1.0), device=self.device
        #     )

        if not save_npy:
            return cam, pred

        # pick class and normarize cam
        cam_pred = self.convert_cam_to_mask(
            cam.clone(), pred >= self.hparams.pred_thresh, orig_img_size=None
        )

        cam_pred = cam_pred.cpu().numpy()
        pred_np = pred.cpu().numpy()
        cam_size = cam_pred.shape[-1]
        for i, input_id in enumerate(batch["input_id"]):
            cam_path, pred_path = get_cam_pred_path(self.cam_dir, input_id, cam_size)
            np.save(str(cam_path), cam_pred[i])
            np.save(str(pred_path), pred_np[i])

        if self.hparams.is_debug:
            num_ = 4
            hm_rate = 0.4
            hm_style = plt.cm.jet
            cam = cam.detach()[:num_]
            pred = pred.detach()[:num_]
            # input renormalize and reformat, always use 3 channels for vis
            inputs = (inputs[:num_].cpu() * self._img_std + self._img_mean).numpy()
            inputs = (inputs.transpose(0, 2, 3, 1)[..., :3] * 255).astype(np.uint8)

            cam_pred = self.convert_cam_to_mask(
                cam.clone(),
                pred >= self.hparams.pred_thresh,
                orig_img_size=inputs.shape[2],
            )

            out_pred = self.overlay_cam_on_input(
                inputs=inputs,
                cam_mask=cam_pred.cpu().numpy(),
                targets=pred.cpu().numpy(),
                batch_num=num_,
                hm_rate=hm_rate,
                hm_style=hm_style,
                stack_axis=0,
                threshold=self.hparams.pred_thresh,
            )
            inputs = np.concatenate(inputs, axis=0)
            out = np.concatenate((inputs, out_pred), axis=1)
            plt.imshow(out)
            plt.show()

    # def test_epoch_end(self, outputs):
    #     # convert into world coordinates and compute offsets
    #     pred_box3ds = []
    #     for boxes in outputs:
    #         pred_box3ds.extend(boxes)
    #     pred = [b.serialize() for b in pred_box3ds]
    #     pred_json_path = os.path.join(
    #         self.hparams.output_dir, CSV_NAME.replace(".csv", ".json")
    #     )
    #     with open(pred_json_path, "w") as f:
    #         json.dump(pred, f)

    #     sub = {}
    #     for i in tqdm(range(len(pred_box3ds))):
    #         yaw = 2 * np.arccos(pred_box3ds[i].rotation[0])
    #         yaw = math.copysign(yaw, pred_box3ds[i].rotation[-1])
    #         pred = " ".join(
    #             [
    #                 str(pred_box3ds[i].score / 255),
    #                 str(pred_box3ds[i].center_x),
    #                 str(pred_box3ds[i].center_y),
    #                 str(pred_box3ds[i].center_z),
    #                 str(pred_box3ds[i].width),
    #                 str(pred_box3ds[i].length),
    #                 str(pred_box3ds[i].height),
    #                 str(yaw),
    #                 str(pred_box3ds[i].name),
    #                 " ",
    #             ]
    #         )[:-1]
    #         if pred_box3ds[i].sample_token in sub.keys():
    #             sub[pred_box3ds[i].sample_token] += pred
    #         else:
    #             sub[pred_box3ds[i].sample_token] = pred

    #     sample_sub = pd.read_csv(
    #         os.path.join(
    #             self.hparams.bev_config.dataset_root_dir, "sample_submission.csv"
    #         )
    #     )
    #     for token in set(sample_sub.Id.values).difference(sub.keys()):
    #         sub[token] = ""

    #     sub = pd.DataFrame(list(sub.items()))
    #     sub.columns = sample_sub.columns
    #     sub_csv_path = os.path.join(self.hparams.output_dir, CSV_NAME)
    #     sub.to_csv(sub_csv_path, index=False)
    #     print("save submission file on:", sub_csv_path)

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if not self.hparams.find_lr:
            if self.trainer.global_step < self.warmup_steps:
                lr_scale = min(
                    1.0, float(self.trainer.global_step + 1) / self.warmup_steps
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * self.hparams.lr
            else:
                pct = (self.trainer.global_step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                pct = min(1.0, pct)
                global_pct = float(self.trainer.global_step) / self.total_steps
                if (global_pct > self.hparams.swa_epoch_start) and self.hparams.use_swa:
                    # keep learning rate for swa
                    pass
                else:
                    for pg in optimizer.param_groups:
                        pg["lr"] = self._annealing_cos(
                            pct, start=self.hparams.lr, end=0.0
                        )
            if self.hparams.is_puzzle_cam:
                w_scale = min(
                    1.0, float(self.trainer.global_step + 1) / (self.total_steps // 2)
                )
                self.p_rec_weight = self.hparams.p_rec_weight_max * w_scale
        else:
            if self.hparams.is_puzzle_cam:
                self.p_rec_weight = self.hparams.p_rec_weight_max * 0.1
        optimizer.step(closure=closure)
        optimizer.zero_grad()

    def _annealing_cos(self, pct: float, start: float = 0.1, end: float = 0.0) -> float:
        """
        https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
        Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def configure_optimizers(self):
        self.total_steps = (
            self.hparams.dataset_len // self.hparams.batch_size
        ) * self.hparams.max_epochs
        self.warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)

        if self.hparams.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=4e-5,
            )
        elif self.hparams.optim_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError
        # steps_per_epoch = self.hparams.dataset_len // self.hparams.batch_size
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.lr,
        #     max_epochs=self.hparams.max_epochs,
        #     steps_per_epoch=steps_per_epoch,
        # )
        # return [optimizer], [scheduler]
        return optimizer


def get_cam_dir(ckpt_path: str) -> Path:
    ver_dir = Path(ckpt_path).parents[1]
    cam_dir = Path(ver_dir, "cam_out")
    return cam_dir


def get_cam_pred_path(
    cam_dir: Path, input_id: str, cam_size: int = 16
) -> Tuple[Path, Path]:
    # cam_path = Path(cam_dir, input_id + f"_{cam_size}_cam.npy")
    cam_path = Path(cam_dir, input_id + "_cam.npy")
    pred_path = Path(str(cam_path).replace("_cam.npy", "_pred.npy"))
    return cam_path, pred_path


def check_missing_keys(args_hprams: dict) -> dict:
    check_keys = {
        "segm_label_dir": None,
        "round_nb": 0,
        "is_puzzle_cam": False,
        "reco_with": "cam",
        "use_true_target": False,
        "p_n_splits": 4,
        "p_rec_weight_max": 4.0,
        "apply_p_sub": False,
        "use_swa": False,
        "swa_epoch_start": 0.8,
        "use_arc_face": False,
        "k_arc_center": 1,
    }
    for key, value in check_keys.items():
        if key not in args_hprams:
            args_hprams[key] = value
    return args_hprams


def load_trained_pl_model(
    pl_model,
    checkpoint_path: str,
    only_load_yaml: bool = False,
):
    yaml_path = Path(Path(checkpoint_path).parents[1], "hparams.yaml")
    with open(yaml_path, mode="r") as f:
        args_hprams = yaml.load(f)
    # model = LitModel.load_from_checkpoint(checkpoint_path, args=args_hprams)
    args_hprams["not_imagenet_weight"] = True
    args_hprams["channels_last"] = False
    args_hprams["load_from_r0_to_r1"] = False
    args_hprams = check_missing_keys(args_hprams=args_hprams)
    if not only_load_yaml:
        model = pl_model.load_from_checkpoint(checkpoint_path, args=args_hprams)
        model.hparams.ckpt_path = checkpoint_path
    else:
        model = args_hprams

    # model_load = model.model.fc8_20.weight.detach().numpy()
    # stat = torch.load(checkpoint_path)
    # weight_load = stat["state_dict"]['model.fc8_20.weight'].cpu().numpy()
    # assert np.all(model_load == weight_load)
    return model, args_hprams


def patch_first_conv(model, in_channels):
    """
    from segmentation_models_pytorch/encoders/_utils.py
    Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    # reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    elif in_channels == 4:
        # reset = True
        # weight = torch.Tensor(
        #     module.out_channels,
        #     module.in_channels // module.groups,
        #     *module.kernel_size,
        # )
        weight = torch.nn.Parameter(torch.cat([weight, weight[:, -1:, :, :]], dim=1))
        # backbone.conv1[0] = torch.nn.Conv2d(
        #     num_in_channels,
        #     backbone.conv1[0].out_channels,
        #     kernel_size=backbone.conv1[0].kernel_size,
        #     stride=backbone.conv1[0].stride,
        #     padding=backbone.conv1[0].padding,
        #     bias=False,
        # )
    module.weight = weight


def load_model_with_head_repalcement(args: argparse.Namespace) -> LitModel:
    assert args.round_nb > 0
    assert args.load_from_r0_to_r1
    print(
        """\t >> load ckpt and model with round_nb = 0
    then replace pred head for round_nb = 1"""
    )
    model = LitModel.load_from_checkpoint(args.ckpt_path, args=args)
    assert hasattr(model.model, "fc8")
    assert model.model.round_nb == args.round_nb
    assert model.hparams.round_nb == args.round_nb

    fc8_20, fc8_200 = get_pred_head(
        round_nb=args.round_nb,
        k=args.k_cluster,
        final_channel=model.model.fc8.in_channels,
        num_classes=model.model.fc8.out_channels,
        use_arc_face=model.model.use_arc_face,
        arc_feat_dim=512,
        k_arc_center=args.k_arc_center,
    )
    model.model.fc8_20 = model.model.fc8
    model.model.fc8_200 = fc8_200
    del model.model.fc8
    return model

    # module.weight = nn.parameter.Parameter(weight)
    # if reset:
    #     module.reset_parameters()
