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
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tqdm
from sklearn import preprocessing
from sklearn.cluster import KMeans

from src.dataset.datamodule import HpaDatamodule
from src.dataset.utils import check_sub_label_def, save_sub_labels
from src.utils.util import print_argparse_arguments


def read_train_list():
    with open("./voc12/train_aug.txt", "r") as r:
        filename_list = r.read().split("\n")
    r.close()

    filename_list = filename_list[:-1]
    for num, filename in enumerate(filename_list):
        filename_list[num] = filename[12:23]
    return filename_list


def make_subclass_label(
    class_idx: int,
    labels: np.ndarray,
    subclass_nb: int,
    all_class_kmeans_label: List[List[int]],
    cls_nb: int = 19,
) -> List[List[int]]:
    # final_label = []

    for i in range(len(labels)):
        label = [0] * subclass_nb * cls_nb
        label[class_idx * subclass_nb + labels[i]] = 1
        all_class_kmeans_label.append(label)

    return all_class_kmeans_label


def generate_merged_label(
    repeat_list,
    label_list: List[List[int]],
):
    new_label_list = []
    for num, one in enumerate(repeat_list):
        merged_label = []
        for i in one:
            merged_label.append(label_list[i])
        # merge the kmeans label to make the sub-category label
        merged_label = sum(np.array(merged_label))
        # check the number of sub-category
        nb_subcategory = np.nonzero(merged_label)[0].shape[0]
        if len(one) != nb_subcategory:
            print(f"wrong labeling..{one}")
        else:
            new_label_list.append(merged_label)

        # assert len(one) == nb_subcategory

    return new_label_list


def create_class_key_in_dict(dict, cls_nb):
    for i in range(cls_nb):
        dict[i] = []

    return dict


def make_filename_class_dict(
    filenamelist: List[str], labels: np.ndarray, cls_nb: int = 19
) -> Tuple[Dict[int, List[int]], Dict[int, List[str]], Dict[int, List[np.ndarray]]]:

    filename_idx_class_dict: Dict[int, list] = {}
    filename_idx_class_dict = create_class_key_in_dict(filename_idx_class_dict, cls_nb)

    filename_class_dict: Dict[int, list] = {}
    filename_class_dict = create_class_key_in_dict(filename_class_dict, cls_nb)

    onehot_class_dict: Dict[int, list] = {}
    onehot_class_dict = create_class_key_in_dict(onehot_class_dict, cls_nb)

    for num, one in enumerate(labels):
        gt_labels = np.where(one == 1)[0]
        for class_id in gt_labels:
            filename_idx_class_dict[class_id].append(num)
            filename_class_dict[class_id].append(filenamelist[num])
            onehot_class_dict[class_id].append(one)

    return filename_idx_class_dict, filename_class_dict, onehot_class_dict


def merge_class_dict(
    filename_class_dict: Dict[int, List[str]],
    onehot_class_dict: Dict[int, List[np.ndarray]],
    cls_nb: int = 19,
) -> Tuple[List[str], List[np.ndarray]]:

    merged_filename_list: List[str] = []
    merged_onehot_list: List[np.ndarray] = []

    for i in range(cls_nb):
        class_filename_list = filename_class_dict[i]
        class_onehot_list = onehot_class_dict[i]
        for file_name, onehot in zip(class_filename_list, class_onehot_list):
            merged_filename_list.append(file_name)
            merged_onehot_list.append(onehot)

    return merged_filename_list, merged_onehot_list


def generate_repeat_list(
    filename_list: List[str], use_json: bool = True, is_debug: bool = False
) -> List[List[int]]:
    json_path = Path("./repete.json")

    if use_json and is_debug:
        if json_path.is_file():
            with open(json_path, "r") as f:
                repeat_list = json.load(f)
            return repeat_list

    repeat_list = []
    print("generate repeat list....")
    for num, one in tqdm.tqdm(enumerate(filename_list), total=len(filename_list)):
        repeat_idx_list = []
        for num_, one_ in enumerate(filename_list):
            if one == one_:
                repeat_idx_list.append(num_)
        repeat_list.append(repeat_idx_list)

    if use_json:
        with open(json_path, "w") as f:
            json.dump(repeat_list, f)

    return repeat_list


def remove_duplicate_label(repeat_list: List[List[int]]) -> Tuple[List[int], List[int]]:
    keep_idx_list = []
    remove_idx_list = []
    for repeat_set in repeat_list:
        for num, one in enumerate(repeat_set):
            if num == 0 and one not in keep_idx_list:
                keep_idx_list.append(one)
            else:
                remove_idx_list.append(one)

    return keep_idx_list, remove_idx_list


def create_train_data(
    merge_filename_list: List[str],
    merged_onehot_list: List[np.ndarray],
    new_label_list: List[np.ndarray],
    keep_idx_list: List[int],
    k_center: int = 10,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    # label_20 = np.load("./voc12/20_class_labels.npy")
    print(len(merge_filename_list), len(new_label_list), len(merged_onehot_list))

    train_filename_list = []
    train_label_200 = []
    train_label_20 = []
    for idx in keep_idx_list:
        train_filename_list.append(merge_filename_list[idx])
        train_label_200.append(new_label_list[idx])
        train_label_20.append(merged_onehot_list[idx])
        check_sub_label_def(train_label_20[-1], train_label_200[-1], k_center)

    return train_filename_list, train_label_200, train_label_20


def load_feature(
    feature_folder_path: str,
    for_round_nb: int = 1,
    is_multi_scale: bool = False,
    num_samples: int = 21806,
    is_hflip: bool = True,
) -> np.ndarray:
    print(f"\t >> load feature from: {feature_folder_path}")
    features = np.load(
        "{}/R{}_feature.npy".format(feature_folder_path, for_round_nb - 1)
    )

    def _norm(features: np.ndarray, axis: int = -1) -> np.ndarray:
        norm = np.linalg.norm(features, axis=axis, keepdims=True)
        return features / norm

    if is_multi_scale:
        features = _norm(features, axis=-1)
        if is_hflip:
            features = np.stack(
                np.split(features, features.shape[1] // 2, axis=1), axis=1
            )
            features = np.mean(features, axis=2)
            features = _norm(features, axis=-1)
        features = np.concatenate(
            np.split(features, features.shape[1], axis=1), axis=-1
        ).squeeze(axis=1)
        features = _norm(features, axis=-1)

        # assert features.shape[0] % num_samples == 0
        # feats_per_sample = features.shape[0] // num_samples
        # agg_feat = []
        # for i in range(num_samples):
        #     sample_feat = features[i * feats_per_sample : (i + 1) * feats_per_sample]
        #     sample_feat = np.linalg.norm(sample_feat, axis=-1)
        #         agg_flip = []
        #         for j in range(sample_feat.shape[0] // 2):
        #             agg_flip.append(
        #                 np.linalg.norm(np.mean(sample_feat[j * 2 : (j + 1) * 2]))
        #             )
        #         sample_feat = np.array(agg_flip)

        #     agg_feat.append(np.concatenate(sample_feat, axis=-1))
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k_cluster", default=10, type=int, help="the number of the sub-category"
    )
    parser.add_argument(
        "--for_round_nb",
        default=1,
        type=int,
        help="the round number that the generated pseudo lable will be used",
    )
    parser.add_argument(
        "--save_folder",
        default="./save_uni/version_17/",
        type=str,
        help="the path to save the sub-category label",
    )
    parser.add_argument(
        "--data_dir",
        default="../input/hpa-single-cell-image-classification",
        type=str,
        help="the path to the dataset folder",
    )
    parser.add_argument(
        "--is_scale_feature",
        action="store_true",
        help="scale the feature before kmeans",
    )
    parser.add_argument(
        "--is_multi_scale", action="store_true", help="whether use multi scale feature"
    )
    parser.add_argument("--use_ext_data", action="store_true", help="use external data")
    parser.add_argument(
        "--ext_data_mode",
        default=1,
        type=int,
        help="exteranl data sampling mode",
    )

    parser.add_argument("--is_debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    print_argparse_arguments(args)

    feature_folder_path = os.path.join(args.save_folder, "feature")

    dm = HpaDatamodule(
        data_dir=args.data_dir,
        val_fold=0,
        batch_size=8,
        input_size=2048,
        aug_mode=0,
        num_workers=4,
        is_debug=args.is_debug,
        num_inchannels=3,
        use_ext_data=args.use_ext_data,
        ext_data_mode=args.ext_data_mode,
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    args.num_classes = 19
    filenamelist = dm.train_df["ID"].tolist()
    cls20_label = dm.train_df["onehot"].values
    assert cls20_label[0].shape[-1] == args.num_classes
    # make the class dictionary  {class_id:[filename]}
    (
        filename_idx_class_dict,
        filename_class_dict,
        onehot_class_dict,
    ) = make_filename_class_dict(filenamelist, cls20_label, cls_nb=args.num_classes)
    # make the list with all samples in the class order (consider each object in
    # the image (there could be multiple objects in an image))
    merge_filename_list, merged_onehot_list = merge_class_dict(
        filename_class_dict, onehot_class_dict, cls_nb=args.num_classes
    )
    # # find the repeated data index (consider each image)
    repeat_list = generate_repeat_list(merge_filename_list, is_debug=args.is_debug)
    # # find the keep/remove index list
    keep_idx_list, remove_idx_list = remove_duplicate_label(repeat_list)

    features = load_feature(
        feature_folder_path,
        for_round_nb=args.for_round_nb,
        is_multi_scale=args.is_multi_scale,
        num_samples=len(filenamelist),
    )

    print(len(filenamelist), cls20_label.shape, features.shape)

    all_class_kmeans_label: list = []
    for i in range(args.num_classes):
        filename_idx_list = filename_idx_class_dict[i]

        class_feature_list = []
        for idx in filename_idx_list:
            class_feature_list.append(features[idx])
        print("Class {}: {}".format(i, len(class_feature_list)))

        # apply kmeans
        X = class_feature_list
        k_cluster = args.k_cluster
        max_iter = 300
        if args.is_scale_feature:
            scaler = preprocessing.StandardScaler()
            X = scaler.fit_transform(X)
        k_center = KMeans(n_clusters=k_cluster, random_state=0, max_iter=max_iter).fit(
            X
        )
        labels = k_center.labels_
        centers = k_center.cluster_centers_
        iter_nb = k_center.n_iter_
        distance = k_center.inertia_
        subclass_nb = k_cluster

        # generate the sub-category label
        all_class_kmeans_label = make_subclass_label(
            i, labels, subclass_nb, all_class_kmeans_label
        )

    # merge the kmeans label to make the sub-category label
    new_label_list = generate_merged_label(repeat_list, all_class_kmeans_label)
    print(len(merge_filename_list), len(new_label_list))  # 16458
    assert np.all(
        new_label_list[repeat_list[0][0]] == new_label_list[repeat_list[0][1]]
    )

    # create the parent label and the sub-category label as the training data with
    train_filename_list, train_label_200, train_label_20 = create_train_data(
        merge_filename_list,
        merged_onehot_list,
        new_label_list,
        keep_idx_list,
        args.k_cluster,
    )
    print(len(train_filename_list), len(train_label_200), len(train_label_20))

    train_label_200 = np.array(train_label_200)
    train_label_20 = np.array(train_label_20)

    print(train_label_200.shape, train_label_20.shape)  # (10582, 200) (10582, 20)

    save_sub_labels(
        train_filename_list,
        train_label_20,
        train_label_200,
        args.save_folder,
        args.for_round_nb,
    )
    print(
        "{}k_{} Round-{} pseudo labels are saved at {}/label. {}".format(
            "#" * 20,
            args.k_cluster,
            args.for_round_nb,
            args.save_folder,
            "#" * 20,
        )
    )

    print(
        "{} You can start to train the classification model.{}".format(
            "#" * 20,
            "#" * 20,
        )
    )
