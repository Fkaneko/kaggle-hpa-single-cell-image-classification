import json
import os
import pickle
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import PIL.Image
import pycocotools.mask as coco_mask

from src.config.config import NUM_CLASSES


def use_sc_cam_format(data: dict, with_pillow=False):
    """
    convert input data into SC-CAM format
    """
    name = data["input_id"]
    img = data["image"]
    if with_pillow:
        img = PIL.Image.fromarray(img.numpy(), mode="RGB")
    label = data["target"]
    return name, img, label


def get_onehot_multilabel(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """"""
    onehot: List[np.ndarray] = []
    for y in df["Label"]:
        y = y.split("|")
        y = list(map(int, y))
        y = np.eye(NUM_CLASSES, dtype="float")[y]
        y = y.sum(axis=0).astype(np.int32)
        # for discussion/217806
        y = np.clip(y, 0, 1)
        onehot.append(y)
    df["onehot"] = onehot
    return df, onehot


def get_sub_label_save_path(
    save_folder: str, for_round_nb: int = 1
) -> Tuple[str, str, str]:
    id_list_path = "{}/label/R{}_train_filename_list.json".format(
        save_folder, for_round_nb
    )
    label_200_path = "{}/label/R{}_train_label_200.npy".format(
        save_folder, for_round_nb
    )
    label_20_path = "{}/label/R{}_train_label_20.npy".format(save_folder, for_round_nb)

    return id_list_path, label_20_path, label_200_path


def save_sub_labels(
    train_filename_list: List[str],
    train_label_20: np.ndarray,
    train_label_200: np.ndarray,
    save_folder: str,
    for_round_nb: int = 1,
) -> None:
    id_list_path, label_20_path, label_200_path = get_sub_label_save_path(
        save_folder, for_round_nb
    )
    with open(id_list_path, "w") as f:
        json.dump(train_filename_list, f)
    np.save(label_200_path, train_label_200)
    np.save(label_20_path, train_label_20)


def load_sub_labels(
    save_folder: str, round_nb: int = 1
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    id_list_path, label_20_path, label_200_path = get_sub_label_save_path(
        save_folder, round_nb
    )
    with open(id_list_path, "r") as f:
        train_filename_list = json.load(f)

    train_label_200 = np.load(label_200_path)
    train_label_20 = np.load(label_20_path)

    return train_filename_list, train_label_20, train_label_200


def save_segm_label(
    samples: dict,
    error_samples: list,
    save_folder: str,
    save_name: str = "segm_full.pkl",
) -> None:
    with open(os.path.join(save_folder, save_name), "wb") as f:
        pickle.dump(samples, f)
    with open(
        os.path.join(save_folder, save_name.replace(".pkl", "_error.pkl")), "wb"
    ) as f:
        pickle.dump(error_samples, f)


class ErrorSample(NamedTuple):
    """error sample"""

    ID: str
    csv_idx: int = -1


def load_segm_label(
    save_folder: str, save_name: str = "segm_full.pkl"
) -> Tuple[dict, list]:
    with open(os.path.join(save_folder, save_name), "rb") as f:
        samples = pickle.load(f)
    try:
        with open(
            os.path.join(save_folder, save_name.replace(".pkl", "_error.pkl")), "rb"
        ) as f:
            error_samples = pickle.load(f)
    except FileNotFoundError as e:
        print(f"load hard code error sample for temporay work around {e}")
        error_samples = [ErrorSample("940f418a-bba4-11e8-b2b9-ac1f6b6435d0")]

    return samples, error_samples


def get_class_mask_from_ins(
    class_ids: np.ndarray,
    confs: np.ndarray,
    rles: list,
    mask_idxs: List[int],
    conf_thresh: float = 0.1,
    num_classes: int = 19,
    is_add_true_bkg: bool = True,
    nega_class=18,
) -> Tuple[np.ndarray, List[int]]:

    if is_add_true_bkg:
        num_classes += 1
    if len(rles) == 0:
        return np.empty(0)
    else:
        class_masks = np.zeros(
            rles[0]["size"] + [num_classes], dtype=np.uint8, order="F"
        )

    labeled_idxs = []
    labeled_classes = []
    for class_id in set(class_ids):
        ins_idx = np.where((class_ids == class_id) & (confs >= conf_thresh))[0]
        labeled_idxs.extend(ins_idx.tolist())
        class_mask = [rles[i] for i in ins_idx]
        if len(class_mask) > 0:
            class_masks[..., class_id] = coco_mask.decode(coco_mask.merge(class_mask))
            labeled_classes.append(class_id)

    # plt.imshow(coco_mask.decode(coco_mask.merge(rles)))
    # plt.imshow(class_masks.sum(axis=-1))

    labeled_idxs = set([mask_idxs[i] for i in set(labeled_idxs)])
    un_labeled_idxs = set(mask_idxs) - set(labeled_idxs)
    if len(un_labeled_idxs) > 0:
        ins_idx = [mask_idxs.index(i) for i in un_labeled_idxs]
        class_mask = [rles[i] for i in ins_idx]
        class_masks[..., nega_class] += coco_mask.decode(coco_mask.merge(class_mask))
        class_masks = np.clip(class_masks, 0, 1)
        labeled_classes.append(nega_class)

    # class_masks = np.ascontiguousarray(class_masks)
    if is_add_true_bkg:
        class_masks[..., -1] = np.where(
            class_masks[..., labeled_classes + [nega_class]].sum(axis=-1) == 0, 1, 0
        )
        labeled_classes.append(nega_class + 1)
    return class_masks, list(set(labeled_classes))


def check_sub_label_def(
    train_label_20: np.ndarray, train_label_200: np.ndarray, k_center: int = 10
) -> None:
    cls_200 = np.where(train_label_200 == 1)[0]
    cls_20 = np.where(train_label_20 == 1)[0]
    cls200_parent = cls_200 // k_center
    assert np.all(cls_20 == cls200_parent)


def get_img_path(data_dir: Path, img_id: str, with_ext_data=False) -> Path:
    if with_ext_data:
        raise NotImplementedError
    folder_name = "train"
    img_path = Path(data_dir, folder_name, img_id)
    return img_path
