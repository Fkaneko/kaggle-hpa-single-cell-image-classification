"""
modify SC-CAM, https://github.com/Juliachang/SC-CAM/blob/master/LICENSE
MIT License

Copyright (c) 2020 juliachang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader

from src.dataset.datamodule import HpaDatamodule
from src.dataset.hpa_dataset import HPAImageDatasetMSF
from src.modeling.pl_model import LitModel, load_trained_pl_model
from src.utils.util import print_argparse_arguments

cudnn.enabled = True


def make_folder(save_folder_path: str) -> None:
    if os.path.exists(save_folder_path) is False:
        os.mkdir(save_folder_path)
    if os.path.exists(os.path.join(save_folder_path, "feature")) is False:
        os.mkdir(os.path.join(save_folder_path, "feature"))
    if os.path.exists(os.path.join(save_folder_path, "label")) is False:
        os.mkdir(os.path.join(save_folder_path, "label"))
    if os.path.exists(os.path.join(save_folder_path, "log")) is False:
        os.mkdir(os.path.join(save_folder_path, "log"))
    if os.path.exists(os.path.join(save_folder_path, "weight")) is False:
        os.mkdir(os.path.join(save_folder_path, "weight"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="./lightning_logs/version_17/checkpoints/epoch=14-step=2729.ckpt",
        type=str,
        help="the weight of the model",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--data_dir",
        default="../input/hpa-single-cell-image-classification",
        type=str,
        help="the path to the dataset folder",
    )
    parser.add_argument(
        "--save_folder",
        default="./save_uni",
        type=str,
        help="the path to save the extracted feature",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--is_multi_scale", action="store_true", help="muti_scales input"
    )
    parser.add_argument("--use_ext_data", action="store_true", help="use external data")
    parser.add_argument(
        "--ext_data_mode",
        default=1,
        type=int,
        help="exteranl data sampling mode",
    )

    pl_class = LitModel
    parser = pl_class.add_model_specific_args(parser)
    args = parser.parse_args()
    print_argparse_arguments(args)

    make_folder(args.save_folder)

    img_orig_size = 2048

    model, args_hparams = load_trained_pl_model(pl_class, args.weights)

    model.eval()
    model.cuda()

    dm = HpaDatamodule(
        data_dir=args.data_dir,
        val_fold=0,
        batch_size=8,
        input_size=img_orig_size,
        aug_mode=0,
        num_workers=args.num_workers,
        is_debug=args.is_debug,
        num_inchannels=args_hparams["num_inchannels"],
        round_nb=args_hparams["round_nb"],
        sub_label_dir=None,
        use_ext_data=args.use_ext_data,
        ext_data_mode=args.ext_data_mode,
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    file_ids = dm.train_df["ID"]
    infer_dataset = HPAImageDatasetMSF(
        data_dir=Path(dm.data_dir, "train"),
        file_ids=file_ids.tolist(),
        onehot_labels=dm.train_df["onehot"].tolist(),
        sub_labels=None,
        transforms=dm.val_transform(),
        num_channels=dm.num_inchannels,
        scales=[0.25, 0.3],
        ext_dir=dm.ext_dir,
    )
    data = infer_dataset[1]

    infer_data_loader = DataLoader(
        infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    filename_list: List[str] = []
    image_feature_list: List[np.ndarray] = []

    print(
        "{} Exteacting features from Round-{} ...... {}".format(
            "#" * 20,
            args_hparams["round_nb"],
            "#" * 20,
        )
    )

    model = model.model

    def calc_feature(
        model: torch.nn.Module,
        img_list: List[torch.Tensor],
        is_multi_scale: bool = False,
        from_round_nb: int = 0,
        is_mean: bool = False,
        flip_num: int = 2,
        use_arc_face: bool = False,
    ) -> torch.Tensor:
        if is_multi_scale:
            muti_feat = []
            for i in range(len(img_list) // flip_num):
                assert np.all(
                    img_list[i * flip_num].shape == img_list[i * flip_num + 1].shape
                )
                img = torch.cat(img_list[i * flip_num : i * flip_num + 2], dim=0)
                tmp, feature, _ = model.forward(img.cuda(), from_round_nb)
                if use_arc_face:
                    feature = F.normalize(feature)
                # muti_feat.append(feature/torch.linalg.norm(feature.unsqueeze(dim=0)))
                muti_feat.append(feature)
            feature = torch.cat(muti_feat, dim=0)
            if is_mean:
                feature = torch.mean(feature)
        else:
            tmp, feature, _ = model.forward(img_list[0].cuda(), from_round_nb)
            if use_arc_face:
                feature = F.normalize(feature)
            feature = feature[0]
        return feature

    with torch.no_grad():
        for img_name, img_list, label in tqdm.tqdm(
            infer_data_loader, total=len(infer_data_loader)
        ):
            img_name = img_name[0]

            filename_list.append(img_name)

            # extract feature
            if args_hparams["round_nb"] == 0:
                feature = calc_feature(
                    model=model,
                    img_list=img_list,
                    is_multi_scale=args.is_multi_scale,
                    from_round_nb=args_hparams["round_nb"],
                    use_arc_face=args_hparams["use_arc_face"],
                )
            else:
                tmp, feature, y_20, x_200, y_200 = model.forward(
                    img_list[0].cuda(), args_hparams["round_nb"]
                )
                if args_hparams["use_arc_face"]:
                    feature = F.normalize(feature)
                feature = feature[0]

            # feature = feature[0].cpu().detach().numpy()
            feature = feature.cpu().detach().numpy()
            image_feature_list.append(feature)

    image_feature_list = np.array(image_feature_list)
    # print(image_feature_list.shape)
    # save the extracted feature
    save_feature_folder_path = os.path.join(args.save_folder, "feature")
    feature_save_path = os.path.join(
        save_feature_folder_path, "R{}_feature.npy".format(args_hparams["round_nb"])
    )  # R1 feature is for R2 use
    np.save(feature_save_path, image_feature_list)
    np.save(feature_save_path.replace(".npy", "_ids.npy"), file_ids.values)
