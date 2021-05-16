from pathlib import Path
from typing import List, Optional, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F_vision

from src.dataset.utils import get_class_mask_from_ins, use_sc_cam_format


class HPAImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        file_ids: List[str],
        transforms: albu.core.composition.Compose,
        onehot_labels: Optional[List[np.ndarray]] = None,
        sub_labels: Optional[List[np.ndarray]] = None,
        segm_labels: Optional[dict] = None,
        segm_cahche_dir: Optional[str] = None,
        segm_thresh: float = 0.15,
        is_add_true_bkg: bool = True,
        num_channels: int = 3,
        ext_dir: Optional[Union[Path, List[str]]] = None,
        is_jpgs: Optional[List[bool]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.file_ids = file_ids
        self.onehot_labels = onehot_labels
        self.sub_labels = sub_labels
        self.segm_labels = segm_labels
        self.transforms = transforms
        self.num_channels = num_channels
        self.segm_thresh = segm_thresh
        self.is_add_true_bkg = is_add_true_bkg
        self.ext_dir = ext_dir
        if is_jpgs is None:
            self.is_jpgs = np.zeros((len(file_ids),), dtype=np.bool).tolist()
        else:
            self.is_jpgs = is_jpgs

        if segm_cahche_dir is not None:
            pass

    def __len__(self):
        return len(self.file_ids)

    @staticmethod
    def get_image(
        data_dir: Path, input_id: str, num_channels: int = 3, is_jpg: bool = False
    ) -> np.ndarray:
        im_list = []
        colors = ["red", "green", "blue", "yellow"]
        img_ext = ".jpg" if is_jpg else ".png"

        for color in colors:
            im = cv2.imread(
                str(Path(data_dir, input_id + "_" + color + img_ext)),
                cv2.IMREAD_UNCHANGED,
            )
            if im is None:
                return None
            if is_jpg:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if im.dtype == "uint16":
                im = (im / 256).astype("uint8")
            im_list.append(im)

        im = np.stack(im_list[:num_channels], axis=-1)
        return im

    def load_class_mask(
        self,
        input_id: str,
        ann: dict,
        conf_thresh: float = 0.15,
    ) -> np.ndarray:
        target_segm, labeled_classes = get_class_mask_from_ins(
            class_ids=ann["labels"],
            confs=ann["confs"],
            rles=ann["masks"],
            mask_idxs=ann["mask_idxs"],
            conf_thresh=conf_thresh,
            is_add_true_bkg=False,
        )
        label_ = [
            np.ascontiguousarray(target_segm[..., class_id])
            for class_id in labeled_classes
        ]
        return np.stack(label_, axis=-1), labeled_classes, target_segm.shape[-1]

    def make_full_ch_mask(
        self,
        mask_label: np.ndarray,
        labeled_classes: List[int],
        num_classes: int = 20,
        is_add_true_bkg: bool = False,
    ):
        if is_add_true_bkg:
            true_bkg = np.where(mask_label.sum(axis=-1) == 0, 1, 0)
            num_classes += 1

        class_masks = np.zeros(
            (mask_label.shape[0], mask_label.shape[1], num_classes),
            dtype=np.uint8,
            order="C",
        )
        for i, class_id in enumerate(labeled_classes):
            class_masks[..., class_id] = mask_label[..., i]

        if is_add_true_bkg:
            class_masks[..., -1] = true_bkg

        return class_masks

    def __getitem__(self, idx: int) -> dict:
        input_id = self.file_ids[idx]
        im = self.get_image(
            self.data_dir, input_id, self.num_channels, is_jpg=self.is_jpgs[idx]
        )
        if im is None:
            if self.ext_dir is not None:
                if isinstance(self.ext_dir, list):
                    ext_dir = Path(self.ext_dir[idx])
                else:
                    ext_dir = self.ext_dir
                im = self.get_image(
                    ext_dir, input_id, self.num_channels, is_jpg=self.is_jpgs[idx]
                )
            else:
                raise FileNotFoundError
        try:
            w_size, h_size = im.shape[1], im.shape[0]
        except Exception as e:
            print("cannnot load", input_id, e)

        target_sub = torch.empty(0)
        target_segm = torch.empty(0)
        target_onehot = torch.empty(0)
        if self.onehot_labels is not None:
            target_onehot = self.onehot_labels[idx].astype(np.float32)
            target_onehot = torch.from_numpy(target_onehot)
            if self.sub_labels is not None:
                target_sub = self.sub_labels[idx].astype(np.float32)
                target_sub = torch.from_numpy(target_sub)
            if self.segm_labels is not None:
                ann = self.segm_labels[input_id]["ann"]
                target_segm, labeled_classes, num_classes = self.load_class_mask(
                    input_id=input_id,
                    ann=ann,
                    conf_thresh=self.segm_thresh,
                )

        if self.transforms:
            if self.segm_labels is None:
                augmented = self.transforms(image=im)
                im = augmented["image"]
            else:
                augmented = self.transforms(image=im, mask=target_segm)
                im = augmented["image"]
                mask = self.make_full_ch_mask(
                    augmented["mask"],
                    labeled_classes,
                    num_classes=num_classes,
                    is_add_true_bkg=self.is_add_true_bkg,
                )

                target_segm = mask.astype(np.float32).transpose(2, 0, 1)
                assert target_segm.max() == 1.0
                # target_segm = augmented["mask"].astype(np.float32).transpose(2, 0, 1)
                target_segm = torch.from_numpy(target_segm)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        return {
            "input_id": input_id,
            "image": im,
            "target": target_onehot,
            "target_sub": target_sub,
            "target_segm": target_segm,
            "w_size": w_size,
            "h_size": h_size,
        }


class HPAImageDatasetMSF(HPAImageDataset):
    def __init__(
        self,
        data_dir: Path,
        file_ids: List[str],
        onehot_labels: Optional[List[np.ndarray]],
        sub_labels: Optional[List[np.ndarray]],
        transforms: albu.core.composition.Compose,
        num_channels: int = 3,
        scales: List[float] = [0.5, 1.0],
        inter_transform=None,
        unit: int = 1,
        ext_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            file_ids=file_ids,
            onehot_labels=onehot_labels,
            sub_labels=sub_labels,
            transforms=transforms,
            num_channels=num_channels,
            ext_dir=ext_dir,
        )
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        name, img, label = use_sc_cam_format(data)

        rounded_size = (
            int(round(img.shape[1] / self.unit) * self.unit),
            int(round(img.shape[2] / self.unit) * self.unit),
        )

        ms_img_list = []

        # Multi Scale
        for s in self.scales:
            target_size = (round(rounded_size[0] * s), round(rounded_size[1] * s))
            s_img = F_vision.resize(img, target_size)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(F_vision.hflip(ms_img_list[i]).clone())
        return name, msf_img_list, label


class HPAImageDatasetWMask(HPAImageDataset):
    def __init__(
        self,
        data_dir: Path,
        mask_dir: Optional[Path],
        file_ids: List[str],
        onehot_labels: Optional[List[np.ndarray]],
        sub_labels: Optional[List[np.ndarray]],
        transforms: albu.core.composition.Compose,
        num_channels: int = 3,
        scales: List[float] = [0.5, 1.0],
        is_jpgs: Optional[List[bool]] = None,
        ext_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            file_ids=file_ids,
            onehot_labels=onehot_labels,
            sub_labels=sub_labels,
            transforms=transforms,
            num_channels=num_channels,
            ext_dir=ext_dir,
            is_jpgs=is_jpgs,
        )

        self.mask_dir = mask_dir

    @staticmethod
    def get_mask(mask_dir: Path, input_id: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        try:
            cell_dir = str(mask_dir / "hpa_cell_mask")
            nucl_dir = str(mask_dir / "hpa_nuclei_mask")

            cell_mask = np.load(f"{cell_dir}/{input_id}.npz")["arr_0"]
            nucl_mask = np.load(f"{nucl_dir}/{input_id}.npz")["arr_0"]
            is_load = True
        except Exception:
            cell_mask = np.ones((2048, 2048), dtype=np.int32)
            nucl_mask = np.ones((2048, 2048), dtype=np.int32)
            is_load = False
        return cell_mask, nucl_mask, is_load

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if self.mask_dir is not None:
            cell_mask, nucl_mask, is_load = self.get_mask(
                self.mask_dir, input_id=data["input_id"]
            )
            target_size = data["image"].shape[-1]
            cell_mask = albu.resize(cell_mask, target_size, target_size)
            nucl_mask = albu.resize(nucl_mask, target_size, target_size)
        else:
            is_load = False
            cell_mask = np.empty(0)
            nucl_mask = np.empty(0)

        data.update(
            {
                "cell_mask": cell_mask.astype(np.int32),
                "nucl_mask": nucl_mask.astype(np.int32),
                "is_load": is_load,
            }
        )
        return data
