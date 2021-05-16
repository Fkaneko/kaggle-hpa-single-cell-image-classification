import argparse
import copy
import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd
import pycocotools.mask as coco_mask
from tqdm import tqdm

from run_cam_infer import decode_ascii_mask
from src.dataset.utils import ErrorSample, save_segm_label
from src.utils.util import print_argparse_arguments


def convert_to_train_format(pred_df: pd.DataFrame) -> Tuple[dict, list]:
    samples = {}
    error_samples = []
    for i, row_ in enumerate(tqdm(pred_df.itertuples(), total=len(pred_df))):
        try:
            pred = row_.PredictionString.split(" ")
        except AttributeError as e:
            print(i, row_.ID, e)
            error_samples.append(ErrorSample(ID=row_.ID, csv_idx=i))
            pass

        w_size, h_size = row_.ImageWidth, row_.ImageHeight
        input_id = row_.ID

        class_ids = np.array(pred[0::3], dtype=np.int32)
        confs = np.array(pred[1::3], dtype=np.float32)
        rle_asciis = pred[2::3]

        last_ascii = ""
        rles: list = []
        rles_idxs = []
        rles_idx = -1
        for ins_id, rle_ascii in enumerate(rle_asciis):
            if last_ascii == rle_ascii:
                rles.append(rles[-1].copy())
                rles_idxs.append(rles_idx)
                continue
            else:
                rles_idx += 1
                rles_idxs.append(rles_idx)
                last_ascii = copy.deepcopy(rle_ascii)

            mask_dict = decode_ascii_mask(rle_ascii, w_size, h_size)
            rles.append(mask_dict["rle"])

        bboxes = coco_mask.toBbox(rles)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        input_sample = {
            "filename": input_id,
            "width": w_size,
            "height": h_size,
            "ann": {
                "bboxes": np.array(bboxes, dtype=np.float32),
                "labels": class_ids,
                "confs": confs,
                "masks": rles,
                "mask_idxs": rles_idxs,
            },
        }
        samples[input_id] = input_sample
    return samples, error_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_folder",
        default="./save_uni/version_15/",
        type=str,
        help="the path to save the sub-category label",
    )
    parser.add_argument(
        "--sub_csv",
        type=str,
        default="./save_uni/version_15/submission_full_v15_0407.csv",
        help="submission style csv, or a directory which contains csvs",
    )

    args = parser.parse_args()
    print_argparse_arguments(args)

    if os.path.isdir(args.sub_csv):
        sub_csvs = glob.glob(os.path.join(args.sub_csv, "*.csv"))
        dfs = []
        for path_ in sub_csvs:
            dfs.append(pd.read_csv(path_))
            print(f"load: {path_} \t len: {len(dfs[-1])}")
        pred_df = pd.concat(dfs, axis=0)
        print("csvs are merged:", len(pred_df))
        assert pred_df.duplicated(["ID"]).sum() == 0

    elif os.path.isfile(args.sub_csv):
        pred_df = pd.read_csv(args.sub_csv)
    else:
        raise NotImplementedError

    print(pred_df.head())

    samples, error_samples = convert_to_train_format(pred_df)
    save_segm_label(
        samples=samples, error_samples=error_samples, save_folder=args.save_folder
    )
