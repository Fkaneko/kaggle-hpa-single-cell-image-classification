import os
import random
from argparse import Namespace
from pathlib import Path
from pprint import pprint
from typing import Union

import numpy as np
from omegaconf import OmegaConf


def set_random_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # set Python random seed
    random.seed(seed)
    # set NumPy random seed
    np.random.seed(seed)


def print_argparse_arguments(p: Namespace, bar: int = 50) -> None:

    """
    from https://github.com/ka10ryu1/jpegcomp
    Visualize argparse arguments.
    Arguments:
        p : parse_arge() object.
        bar : the number of bar on the output.
    """
    print("PARAMETER SETTING")
    print("-" * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if "_" not in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print("{0}[{1}]:".format(i, len(j)))
            for k in j:
                print("\t{}".format(k))
        else:
            print("{0:25}:{1}".format(i, j))
    print("-" * bar)


def save_config_data(conf: Union[dict, Namespace], path: Path) -> None:
    # with fs.open(config_yaml, "w", encoding="utf-8") as fp:
    if isinstance(conf, Namespace):
        conf = vars(conf)
    print("saving the following configuration:", path.name)
    pprint(conf)
    conf = OmegaConf.create(conf)
    # with open(path, "w", encoding="utf-8") as fp:
    with path.open("w", encoding="utf-8") as fp:
        OmegaConf.save(conf, fp, resolve=True)


def make_version_folder(output_dir: Path, ver_prefix: str = "version_") -> Path:
    output_dir.mkdir(exist_ok=True)
    ver_list = list(output_dir.glob(ver_prefix + "*"))
    if len(ver_list) > 0:
        vers = [int(x.name.replace(ver_prefix, "")) for x in ver_list]
        version_cnt = sorted(vers, key=int)[-1] + 1
    else:
        version_cnt = 0
    version_dir = output_dir / Path(ver_prefix + str(version_cnt))
    version_dir.mkdir()
    return version_dir
