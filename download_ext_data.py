import argparse
import csv
import datetime
import gzip
import io
import os
import pathlib
from typing import List, Tuple

import imageio
import pandas as pd
import requests
import tqdm

from src.config.config import CELLLINES


def tif_gzip_to_png(tif_path):
    """Function to convert .tif.gz to .png and put it in the same folder
    Eg. for working in local work station
    """
    png_path = pathlib.Path(tif_path.replace(".tif.gz", ".png"))
    tf = gzip.open(tif_path).read()
    img = imageio.imread(tf, "tiff")
    imageio.imwrite(png_path, img)


def download_and_convert_tifgzip_to_png(url, target_path, is_jpg: bool = True):
    """Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    """
    if is_jpg:
        url = url.replace(".tif.gz", ".jpg")
        target_path = target_path.replace(".png", ".jpg")
    else:
        pass

    r = requests.get(url)
    # with open(target_path, "wb") as f:
    #     f.write(r.content)
    f = io.BytesIO(r.content)
    # f = r.content
    if is_jpg:
        img = imageio.imread(f, "jpg")
    else:
        tf = gzip.open(f).read()
        img = imageio.imread(tf, "tiff")

    imageio.imwrite(target_path, img)
    # cv2.imwrite("./", img.astype(np.uint8))


def split_label(label: str):
    for la in label.split("|"):
        if int(la) in [0, 16]:
            if label in ["0", "16"]:
                return "Only"
            else:
                return "Multi"
    return "Rare"


# All label names in the public HPA and their corresponding index.
all_locations = dict(
    {
        "Nucleoplasm": 0,
        "Nuclear membrane": 1,
        "Nucleoli": 2,
        "Nucleoli fibrillar center": 3,
        "Nuclear speckles": 4,
        "Nuclear bodies": 5,
        "Endoplasmic reticulum": 6,
        "Golgi apparatus": 7,
        "Intermediate filaments": 8,
        "Actin filaments": 9,
        "Focal adhesion sites": 9,
        "Microtubules": 10,
        "Mitotic spindle": 11,
        "Centrosome": 12,
        "Centriolar satellite": 12,
        "Plasma membrane": 13,
        "Cell Junctions": 13,
        "Mitochondria": 14,
        "Aggresome": 15,
        "Cytosol": 16,
        "Vesicles": 17,
        "Peroxisomes": 17,
        "Endosomes": 17,
        "Lysosomes": 17,
        "Lipid droplets": 17,
        "Cytoplasmic bodies": 17,
        "No staining": 18,
    }
)


def add_label_idx(df, all_locations):
    """Function to convert label name to index"""
    df["Label_idx"] = None
    for i, row in df.iterrows():
        labels = row.Label.split(",")
        idx = []
        for la in labels:
            if la in all_locations.keys():
                idx.append(str(all_locations[la]))
        if len(idx) > 0:
            df.loc[i, "Label_idx"] = "|".join(idx)

        print(df.loc[i, "Label"], df.loc[i, "Label_idx"])
    return df


def load_csv(path: str = "../input/publichpa-withcellline/kaggle_2021.tsv"):
    public_hpa_df = pd.read_csv(path)
    print(f"orig {len(public_hpa_df)}")

    # Remove all images overlapping with Training set
    public_hpa_df = public_hpa_df[~public_hpa_df.in_trainset]
    print(f"over lapped {len(public_hpa_df)}")
    # Remove all images with only labels that are not in this competition
    public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]
    print(f"nan {len(public_hpa_df)}")

    public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(CELLLINES)]
    len(public_hpa_df), len(public_hpa_df_17)

    public_hpa_df["is_non_rare"] = public_hpa_df["Label_idx"].apply(
        lambda X: split_label(X)
    )
    public_hpa_df["is_non_rare"].value_counts()
    return public_hpa_df


def define_download_order(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # remove data in the dataset,
    # https://www.kaggle.com/philculliton/hpa-challenge-2021-extra-train-images
    df = df.loc[~df["isin_dataset"], :]
    df_rare = df.query('is_non_rare == "Rare"')
    df_multi = df.query('is_non_rare == "Multi"')
    df_only = df.query('is_non_rare == "Only"')
    # return df_rare, df_multi, df_only
    return df_multi, df_rare, df_only


def logging_download(
    csv_path: str, mode: str = "w", log_msg: List[str] = ["row_ind", "ID"]
):
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_msg)


def download_ext_data(df: pd.DataFrame, save_dir: str, is_debug: bool = False):
    colors = ["blue", "red", "green", "yellow"]
    df.reset_index(inplace=True, drop=True)
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        try:
            img = row.Image
            for color in colors:
                img_url = f"{img}_{color}.tif.gz"
                save_path = os.path.join(
                    save_dir, f"{os.path.basename(img)}_{color}.png"
                )
                download_and_convert_tifgzip_to_png(img_url, save_path, is_jpg=True)
                # print(f"Downloaded {img_url} as {save_path}")
                logging_download(
                    csv_path=os.path.join(save_dir, "downloaded.csv"),
                    mode="a",
                    log_msg=[i, img_url],
                )

        except Exception as e:
            print(f"failed to download: {img}")
            logging_download(
                csv_path=os.path.join(save_dir, "un_downloaded.csv"),
                mode="a",
                log_msg=[i, img, str(e)],
            )
        if is_debug:
            if i == 5:
                break


def main(args):
    public_hpa_df = pd.read_csv(args.input_csv)

    save_dir = os.path.join(args.save_dir, "publichpa_download")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df_list = define_download_order(df=public_hpa_df)

    logging_download(
        csv_path=os.path.join(save_dir, "downloaded.csv"),
        mode="w",
        log_msg=["row_ind", "Image_url"],
    )

    logging_download(
        csv_path=os.path.join(save_dir, "un_downloaded.csv"),
        mode="w",
        log_msg=["row_ind", "Image", "Error_msg"],
    )

    for i, df in enumerate(df_list):
        print(f"\t #### START {i} {datetime.datetime.now()} ####")
        download_ext_data(df, save_dir=save_dir, is_debug=args.is_debug)
        if args.is_debug:
            if i == 0:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="download extra hpa data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir",
        default="../input",
        type=str,
        help="save directory for extra hpa images",
    )
    parser.add_argument(
        "--input_csv",
        default="../input/publichpa-withcellline/kaggle_2021_processed.csv",
        type=str,
        help="input csv for download",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
