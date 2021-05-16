import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader

from src.config.config import CELLLINES, NUM_SPLIT

# from src.bev_processing.bev_utils import SampleMeta
from src.dataset.hpa_dataset import HPAImageDataset, HPAImageDatasetWMask
from src.dataset.utils import (
    check_sub_label_def,
    get_onehot_multilabel,
    load_segm_label,
    load_sub_labels,
)

IMG_MEAN = (0.485, 0.456, 0.406, 0.406)
IMG_STD = (0.229, 0.224, 0.225, 0.225)


class HpaDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        val_fold: int = 0,
        batch_size: int = 64,
        input_size: int = 512,
        num_workers: int = 16,
        aug_mode: int = 0,
        is_debug: bool = False,
        num_inchannels: int = 3,
        round_nb: int = 0,
        sub_label_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        segm_label_dir: Optional[str] = None,
        segm_thresh: float = 0.15,
        use_cached_split: bool = False,
        use_ext_data: bool = True,
        add_celllines: bool = True,
        ext_data_mode: int = 1,
        para_num: Optional[int] = None,
        para_ind: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.num_workers = num_workers
        self.is_debug = is_debug
        self.val_fold = val_fold
        self.input_size = input_size
        self.n_splits = NUM_SPLIT
        self.num_inchannels = num_inchannels
        self.round_nb = round_nb
        self.sub_label_dir = sub_label_dir
        self.mask_dir = mask_dir
        self.segm_label_dir = segm_label_dir
        self.segm_thresh = segm_thresh

        self.img_mean = IMG_MEAN[: self.num_inchannels]
        self.img_std = IMG_STD[: self.num_inchannels]

        self.csv_cache_path = "../input/train_split.csv"
        self.use_cached_split = use_cached_split
        self.use_ext_data = use_ext_data
        self.add_celllines = add_celllines
        self.ext_data_mode = ext_data_mode
        self.para_num = para_num
        self.para_ind = para_ind

    def prepare_data(self):
        # check
        assert self.data_dir.is_dir()

    def _onehot_to_set(self, onehot: np.ndarray):
        return set(np.where(onehot == 1)[0].astype(str).tolist())

    def handle_segm_label(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        if self.segm_label_dir is not None:
            print("\t >> load segm label...")
            segm_samples, error_samples = load_segm_label(
                save_folder=self.segm_label_dir
            )
            error_ids = [sample.ID for sample in error_samples]
            df = df.loc[~df["ID"].isin(error_ids), :]
            assert set(segm_samples.keys()) >= set(df.ID.values.tolist())
        else:
            segm_samples = None  # type: ignore[assignment]
        return df, segm_samples

    def handle_sub_label(self):
        if self.round_nb > 0 and self.sub_label_dir is not None:
            print("\t >> load sub categry label...")
            train_filename_list, train_label_20, train_label_200 = load_sub_labels(
                self.sub_label_dir, round_nb=self.round_nb
            )
            sub_df = pd.DataFrame(
                [train_filename_list, train_label_20, train_label_200]
            ).T
            sub_df.columns = ["ID", "onehot", "target_sub"]
            self.train_df = pd.merge(
                self.train_df, sub_df, left_on="ID", right_on="ID", suffixes=["", "_y"]
            )
            for row_tuple in self.train_df.itertuples():
                check_sub_label_def(row_tuple.onehot, row_tuple.target_sub)

            assert np.all(
                self.train_df["onehot"].apply(self._onehot_to_set)
                == self.train_df["onehot_y"].apply(self._onehot_to_set)
            )
            del self.train_df["onehot_y"]

            return [lab_ for lab_ in self.train_df["target_sub"]]
        else:
            return None

    def downsampling_with_cellline(
        self, df: pd.DataFrame, ratio: float = 0.3
    ) -> pd.DataFrame:
        frequent_lines = ["U-2 OS", "A-431", "U-251 MG"]
        df_rare = df[~df.Cellline.isin(frequent_lines)]
        df_freq = df[df.Cellline.isin(frequent_lines)]
        sample_num = int(len(df) * ratio)
        if sample_num > len(df_rare):
            df_freq = df_freq.sample(n=sample_num - len(df_rare), random_state=0)
            df = pd.concat([df_rare, df_freq])
        else:
            df = df_rare.sample(n=sample_num, random_state=0)

        return df

    def ext_data_mask(self, df: pd.DataFrame, mode: int = 0) -> pd.DataFrame:
        print(f"\t >> EXT_DATA_MODE: {mode}")
        if mode == 0:
            # inclued in public ext data which was provided as kaggle dataset ~ 23000
            mask_ = df["isin_dataset"]
        elif mode == 1:
            mask_ = df["isin_dataset"] | df["is_non_rare"].isin(["Rare"])
        elif mode == 2:
            mask_ = df["isin_dataset"] | df["is_non_rare"].isin(["Rare", "Multi"])
            mask_ = mask_ & (df["Label"] != "0|16")
        elif mode == 3:
            mask_ = df["isin_dataset"] | df["is_non_rare"].isin(["Rare"])
            df_multi_only = df[~df["isin_dataset"] & (df["Label"] == "0|16")]
            df_multi_only = self.downsampling_with_cellline(df=df_multi_only, ratio=0.3)

            mask_ = mask_ | df.ID.isin(df_multi_only.ID)
        elif mode == 4:
            mask_ = df["isin_dataset"] | df["is_non_rare"].isin(["Rare", "Multi"])

        elif mode == 5:
            mask_ = df["isin_dataset"] | df["is_non_rare"].isin(["Rare"])
            df_multi_only = df[~df["isin_dataset"] & (df["Label"] == "0|16")]
            df_multi_only = self.downsampling_with_cellline(df=df_multi_only, ratio=0.3)

            df_only = df[~df["isin_dataset"] & (df["is_non_rare"] == "Only")]
            df_only = self.downsampling_with_cellline(df=df_only, ratio=0.3)

            mask_ = mask_ | df.ID.isin(df_multi_only.ID)
            mask_ = mask_ | df.ID.isin(df_only.ID)
        elif mode == 6:
            mask_ = df["is_non_rare"].isin(["Rare"])

        elif mode == -1:
            # use all data
            return df
        df = df[mask_]
        return df

    def _get_dir(self, dir_: str) -> str:
        name = os.path.basename(dir_)
        return os.path.join("../input", name)

    def load_ext_data(
        self, onehot: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Path, dict]:

        ext_dir = Path("../input/HPA-Challenge-2021-trainset-extra/")
        ext_data_df = pd.read_csv("../input/publichpa-withcellline_processed.csv")

        common_id = check_img_files(ext_dir=ext_dir)

        missing_num = len(ext_data_df) - len(common_id)
        ext_data_df = ext_data_df[ext_data_df.ID.isin(common_id)]
        print(f"There are missing {missing_num} samples on {str(ext_dir)}")
        ext_data_df["Label"] = ext_data_df["Label_idx"]
        ext_data_df = self.ext_data_mask(df=ext_data_df, mode=self.ext_data_mode)
        ext_data_df["is_jpg"] = ext_data_df["file_ext"] == "jpg"

        ext_data_df["dir"] = ext_data_df["dir"].apply(lambda x: self._get_dir(x))
        field = ["ID", "Label", "Cellline", "dir", "is_jpg"]
        ext_data_df = ext_data_df.loc[:, field]

        target, non_target = self.handle_ext_data(ext_data_df)

        onehot = onehot + target["onehot"]
        self.train_df["Cellline"] = CELLLINES[0]
        self.train_df["dir"] = self.data_dir
        self.train_df["is_jpg"] = False
        self.train_df = pd.concat([self.train_df, target["df"]])
        self.train_df.reset_index(inplace=True, drop=True)
        self.n_splits = 8
        return onehot, ext_dir, non_target

    def handle_ext_data(self, df: pd.DataFrame):
        non_target_df = df[~df.Cellline.isin(CELLLINES)]
        target_df = df[df.Cellline.isin(CELLLINES)]
        non_target_df, onehot_non_target = get_onehot_multilabel(df=non_target_df)
        target_df, onehot_target = get_onehot_multilabel(df=target_df)
        non_target_df["fold"] = -1

        return {"df": target_df, "onehot": onehot_target}, {
            "df": non_target_df,
            "onehot": onehot_non_target,
        }

    def split_df_for_parallel_run(
        self, df: pd.DataFrame, n_splits: int = 10, current_ind: int = 0
    ) -> pd.DataFrame:
        assert (
            current_ind < n_splits
        ), f"should be current_ind,{current_ind} < n_splits,{n_splits}"
        df.sort_values(by="ID", inplace=True, ignore_index=True)
        n_total = len(df)
        n_per_split = n_total // n_splits
        split_inds = np.zeros((n_total,), dtype=np.int32)
        for i in range(n_splits):
            lower = i * n_per_split
            upper = (i + 1) * n_per_split
            split_inds[lower:upper] = i
        df["split_ind"] = split_inds
        print(f" ## split results for parallel run:\n{df.split_ind.value_counts()}")
        df = df.loc[df["split_ind"] == current_ind, :]
        print(f"orig df:{n_total} -> current df:{len(df)} with split ind:{current_ind}")
        return df

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders

        if stage == "fit" or stage is None:
            self.train_df = pd.read_csv(str(Path(self.data_dir, "train.csv")))
            # train/val split
            self.train_df, onehot = get_onehot_multilabel(df=self.train_df)
            if self.use_ext_data:
                onehot, self.ext_dir, non_target = self.load_ext_data(onehot=onehot)
            else:
                self.ext_dir = None  # type:ignore[assignment]
                self.train_df["is_jpg"] = False
                self.train_df["dir"] = self.data_dir

            mskf = MultilabelStratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=0
            )
            self.train_df["fold"] = -1
            for i, (train_idx, valid_idx) in enumerate(
                mskf.split(self.train_df, onehot)
            ):
                self.train_df.loc[valid_idx, "fold"] = i

            if self.use_ext_data and self.add_celllines:
                self.train_df = pd.concat([self.train_df, non_target["df"]])

            if self.use_cached_split:
                split_df = pd.read_csv(self.csv_cache_path).loc[:, ["ID", "fold"]]
                new_df = pd.merge(
                    self.train_df, split_df, on="ID", suffixes=["_old", ""]
                )
                assert len(new_df) == len(self.train_df)
                del new_df["fold_old"]
                self.train_df = new_df
                print("\t >>> use cached split info <<")
            else:
                if not Path(self.csv_cache_path).is_file():
                    self.train_df.to_csv(self.csv_cache_path, index=False)

            self.train_df, segm_labels = self.handle_segm_label(self.train_df)
            sub_labels = self.handle_sub_label()

            train_df = self.train_df.loc[self.train_df["fold"] != self.val_fold, :]
            val_df = self.train_df.loc[self.train_df["fold"] == self.val_fold, :]
            if sub_labels is not None:
                train_sub_labels = train_df["target_sub"].tolist()
                val_sub_labels = val_df["target_sub"].tolist()
            else:
                train_sub_labels, val_sub_labels = None, None

            self.train_dataset = HPAImageDataset(
                data_dir=Path(self.data_dir, "train"),
                file_ids=train_df["ID"].tolist(),
                onehot_labels=train_df["onehot"].tolist(),
                sub_labels=train_sub_labels,
                segm_labels=segm_labels,
                segm_cahche_dir=self.segm_label_dir,
                segm_thresh=self.segm_thresh,
                transforms=self.train_transform(),
                num_channels=self.num_inchannels,
                ext_dir=train_df["dir"].tolist(),
                is_jpgs=train_df["is_jpg"].tolist(),
            )
            self.val_dataset = HPAImageDataset(
                data_dir=Path(self.data_dir, "train"),
                file_ids=val_df["ID"].tolist(),
                onehot_labels=val_df["onehot"].tolist(),
                sub_labels=val_sub_labels,
                segm_labels=segm_labels,
                segm_cahche_dir=self.segm_label_dir,
                segm_thresh=self.segm_thresh,
                transforms=self.val_transform(),
                num_channels=self.num_inchannels,
                ext_dir=val_df["dir"].tolist(),
                is_jpgs=val_df["is_jpg"].tolist(),
            )
            self.plot_dataset(self.train_dataset)

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_df = pd.read_csv(
                str(Path(self.data_dir, "sample_submission.csv"))
            )
            self.test_dataset = HPAImageDataset(
                data_dir=Path(self.data_dir, "test"),
                file_ids=self.test_df["ID"].tolist(),
                onehot_labels=None,
                sub_labels=None,
                transforms=self.test_transform(),
                num_channels=self.num_inchannels,
            )
            self.plot_dataset(self.test_dataset)

        if stage == "gen_pseudo":
            self.train_df = pd.read_csv(str(Path(self.data_dir, "train.csv")))

            self.train_df, onehot = get_onehot_multilabel(df=self.train_df)
            if self.use_ext_data:
                onehot, self.ext_dir, non_target = self.load_ext_data(onehot=onehot)
                self.train_df = pd.concat([self.train_df, non_target["df"]])
                print("ext data configuration \n", self.train_df.dir.value_counts())
            else:
                self.train_df["is_jpg"] = False
                self.train_df["dir"] = self.data_dir

            sub_labels = self.handle_sub_label()

            if self.para_num is not None:
                self.train_df = self.split_df_for_parallel_run(
                    df=self.train_df, n_splits=self.para_num, current_ind=self.para_ind
                )

            train_df = self.train_df
            if sub_labels is not None:
                train_sub_labels = train_df["target_sub"].tolist()
            else:
                train_sub_labels = None
            if self.mask_dir is None:
                mask_dir = None
            else:
                mask_dir = Path(self.mask_dir)

            self.train_dataset = HPAImageDatasetWMask(
                data_dir=Path(self.data_dir, "train"),
                mask_dir=mask_dir,
                file_ids=train_df["ID"].tolist(),
                onehot_labels=train_df["onehot"].tolist(),
                sub_labels=train_sub_labels,
                transforms=self.train_transform(),
                num_channels=self.num_inchannels,
                ext_dir=train_df["dir"].tolist(),
                is_jpgs=train_df["is_jpg"].tolist(),
            )
            self.test_dataset = self.train_dataset
            self.test_df = self.train_df
            self.plot_dataset(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def train_transform(self):
        return self.get_transforms(mode=self.aug_mode)

    def val_transform(self):
        return self.get_transforms(mode=0)

    def test_transform(self):
        return self.get_transforms(mode=0)

    def get_transforms(self, mode: int = 0) -> albu.core.composition.Compose:
        if mode == 0:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 1:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 2:
            transforms = [
                albu.RandomResizedCrop(
                    self.input_size,
                    self.input_size,
                    scale=(0.75, 1.0),
                    ratio=(0.75, 1.333),
                    interpolation=1,
                    always_apply=False,
                    p=1.0,
                ),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]

        elif mode == 3:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.GridDropout(
                    ratio=0.4,
                    unit_size_min=self.input_size // 4,
                    unit_size_max=self.input_size // 1,
                    random_offset=True,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.6,
                ),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 4:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.GridDropout(
                    ratio=0.4,
                    unit_size_min=self.input_size // 3,
                    unit_size_max=self.input_size // 1,
                    random_offset=True,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.4,
                ),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        elif mode == 5:
            transforms = [
                albu.Resize(self.input_size, self.input_size, p=1.0),
                albu.RandomResizedCrop(
                    self.input_size,
                    self.input_size,
                    scale=(0.75, 1.0),
                    ratio=(0.75, 1.33333),
                    interpolation=1,
                    always_apply=False,
                    p=1.0,
                ),
                albu.GridDropout(
                    ratio=0.4,
                    unit_size_min=self.input_size // 4,
                    unit_size_max=self.input_size // 1,
                    random_offset=True,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.6,
                ),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomRotate90(p=1),
                albu.Transpose(p=0.5),
                albu.Normalize(mean=self.img_mean, std=self.img_std),
            ]

        else:
            raise NotImplementedError

        return albu.Compose(transforms)

    def plot_dataset(
        self,
        dataset: HPAImageDataset,
        plot_num: int = 10,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        inds = np.random.choice(len(dataset), plot_num)
        for i in inds:
            plt.figure(figsize=(16, 8))
            data = dataset[i]
            im = data["image"].numpy().transpose(1, 2, 0)
            im = im * np.array(self.img_std) + np.array(self.img_mean)
            if im.shape[2] > 3:
                im_yellow = im[..., 3]
                im_yellow = np.repeat(im_yellow[:, :, np.newaxis], 3, axis=-1)
                im = np.hstack([im[..., :3], im_yellow])

            if data["target"].shape[0] != 0:
                orig_target = self.train_df.query(f'ID == "{data["input_id"]}"')

                target = data["target"].numpy().astype(np.int32)
                assert np.all(orig_target["onehot"].iloc[0] == target)

                target_str = self._onehot_to_set(target)
                assert target_str == set(orig_target["Label"].iloc[0].split("|"))
                target_str = "|".join(target_str)

                if data["target_sub"].shape[0] != 0:
                    target_sub = data["target_sub"].numpy().astype(np.int32)
                    target_str_sub = self._onehot_to_set(target_sub)
                    assert np.all(orig_target["target_sub"].iloc[0] == target_sub)
                    target_str += "+" + "|".join(target_str_sub)

                plt.title(target_str)

            if self.is_debug:
                plt.imshow(im.clip(0.0, 1.0))
                plt.show()

            plt.close()
            if data["target_segm"].shape[0] != 0:
                target_segm = data["target_segm"].numpy().astype(np.uint8)
                if self.is_debug:
                    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                    labeler = (np.arange(target_segm.shape[0]) + 1)[
                        :, np.newaxis, np.newaxis
                    ]
                    ax[0].imshow(im)
                    ax[1].imshow((target_segm * labeler)[:-1].sum(axis=0))
                    ax[0].axis("off")
                    ax[1].axis("off")
                    plt.show()
                    plt.close()


def check_img_files(ext_dir: Path, file_ext: str = "png") -> List[str]:
    common_id = []
    for color in ["red", "green", "blue", "yellow"]:
        ext_files = glob.glob(str(ext_dir / f"*{color}.{file_ext}"))
        ext_file_ids = [
            Path(file).name.replace(f"_{color}.{file_ext}", "") for file in ext_files
        ]
        print(f"{color} image files:", len(ext_file_ids))
        if color == "red":
            common_id = ext_file_ids
        else:
            common_id = list(set(common_id) & set(ext_file_ids))
    print("common file id num", len(set(common_id)))
    return common_id
