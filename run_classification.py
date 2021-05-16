import argparse
import sys
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.swa import StochasticWeightAveraging
from tqdm import tqdm

from run_cam_infer import PRED_THRESH, get_class_mask, get_dm_default_args
from src.config.config import NUM_CLASSES, SEED
from src.dataset.datamodule import HpaDatamodule
from src.modeling.pl_model import (
    LitModel,
    get_cam_dir,
    get_cam_pred_path,
    load_model_with_head_repalcement,
    load_trained_pl_model,
)
from src.utils.util import print_argparse_arguments, set_random_seed


def main(args: argparse.Namespace) -> None:

    set_random_seed(SEED)

    datamodule = HpaDatamodule(
        data_dir=args.data_dir,
        val_fold=args.val_fold,
        batch_size=args.batch_size,
        aug_mode=args.aug_mode,
        num_workers=args.num_workers,
        is_debug=args.is_debug,
        input_size=args.input_size,
        num_inchannels=args.num_inchannels,
        round_nb=args.round_nb,
        sub_label_dir=args.sub_label_dir,
        segm_label_dir=args.segm_label_dir,
        segm_thresh=args.segm_thresh,
        use_cached_split=args.use_cached_split,
        use_ext_data=args.use_ext_data,
        ext_data_mode=args.ext_data_mode,
    )

    datamodule.prepare_data()
    args.num_classes = NUM_CLASSES
    if args.is_test:
        stage = "gen_pseudo"
        scales = [1.2, 1.4, 1.5]
        scales = scales[: args.num_scales]

        print("\t\t ==== TEST MODE ====")
        print("load from: ", args.ckpt_path)
        model, args_hparams = load_trained_pl_model(LitModel, args.ckpt_path)
        mode = "segm" if args_hparams["segm_label_dir"] is not None else "cam"
        assert args.input_size > int(768 * 1.4), "large resolution is needed for tta"

        datamodule = HpaDatamodule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_debug=args.is_debug,
            input_size=args.input_size,
            segm_label_dir=None,
            use_ext_data=args.use_ext_data,
            ext_data_mode=args.ext_data_mode,
            mask_dir=None,
            **get_dm_default_args(),
        )
        datamodule.prepare_data()
        datamodule.setup(stage=stage if not args.test_with_val else "fit")
        test_dataloader = datamodule.test_dataloader()
        model.cuda()
        model.eval()

        cam_dir = get_cam_dir(args.ckpt_path)
        cam_dir.mkdir(exist_ok=True)
        print(f"\t>> make cam cache directory:\n {str(cam_dir)}")
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                data["image"] = data["image"].cuda()
                cam_pred, pred = get_class_mask(
                    data,
                    batch_idx=i,
                    args_hparams=args_hparams,
                    model=model,
                    infer_size=args_hparams["input_size"],
                    pred_thresh=PRED_THRESH,
                    label_find_size=None,
                    stage=stage,
                    mode=mode,
                    tta_mode=args.tta_mode,
                    scales=scales,
                )
                cam_pred = cam_pred.cpu().numpy()
                pred_np = pred.cpu().numpy()
                for i, input_id in enumerate(data["input_id"]):
                    cam_path, pred_path = get_cam_pred_path(cam_dir, input_id)
                    np.save(str(cam_path), cam_pred[i])
                    np.save(str(pred_path), pred_np[i])

    else:
        print("\t\t ==== TRAIN MODE ====")
        datamodule.setup(stage="fit")
        args.dataset_len = len(datamodule.train_dataset)
        print(
            "training samples: {}, valid samples: {}".format(
                args.dataset_len, len(datamodule.val_dataset)
            )
        )
        if args.ckpt_path is not None:
            if args.load_from_r0_to_r1:
                model = load_model_with_head_repalcement(args=args)
            else:
                model = LitModel.load_from_checkpoint(args.ckpt_path, args=args)
        else:
            model = LitModel(args)

        pl.trainer.seed_everything(seed=SEED)

        callbacks = get_callbacks(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            benchmark=True,
            deterministic=False,
        )

        # Run lr finder
        if args.find_lr:
            lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
            lr_finder.plot(suggest=True)
            plt.savefig("./lr_finder.png")
            plt.show()
            sys.exit()

        # Run Training
        trainer.fit(model, datamodule=datamodule)


def get_callbacks(args: argparse.Namespace, ema_decay: float = 0.9) -> list:
    callbacks: List[Any] = []
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_last=True,
        mode="min",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    if args.use_swa:
        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged: (
                1.0 - ema_decay
            )
            * averaged_model_parameter
            + ema_decay * model_parameter
        )
        swa_callback = StochasticWeightAveraging(
            swa_epoch_start=args.swa_epoch_start,
            swa_lrs=args.swa_lrs,
            annealing_epochs=args.max_epochs
            - int(args.swa_epoch_start * args.max_epochs),
            annealing_strategy="cos",
            avg_fn=ema_avg if args.use_ema else None,
        )
        callbacks.append(swa_callback)
    return callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training for hpa ws",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="../input/hpa-single-cell-image-classification",
        type=str,
        help="root directory path for hpa dataset, ",
    )
    parser.add_argument(
        "--num_inchannels",
        default="3",
        type=int,
        help="number of channels for input image",
    )
    parser.add_argument("--batch_size", type=int, default=96, help="batch size")
    parser.add_argument("--input_size", type=int, default=512, help="input image size")
    parser.add_argument("--lr", default=1.0e-2, type=float, help="learning rate")
    parser.add_argument(
        "--optim_name",
        choices=["adam", "sgd"],
        default="adam",
        help="optimizer name",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="warmup ratio for lr scheduler"
    )
    parser.add_argument("--is_test", action="store_true", help="test mode")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path for model checkpoint",
    )
    parser.add_argument(
        "--load_from_r0_to_r1",
        action="store_true",
        help="load ckpt from round_nb = 0 and train round_nb = 1",
    )
    parser.add_argument(
        "--test_with_val", action="store_true", help="test mode with validation data"
    )
    parser.add_argument(
        "--flip_tta", action="store_true", help="test time augmentation h/vflip"
    )
    parser.add_argument(
        "--val_fold",
        default=0,
        type=int,
        help="validation fold configuration for train/val split",
    )
    parser.add_argument(
        "--use_cached_split", action="store_true", help="use cached split info"
    )
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="use StochasticWeightAveraging",
    )
    parser.add_argument(
        "--swa_epoch_start",
        default=0.8,
        type=float,
        help="swa start epoch with float",
    )

    parser.add_argument(
        "--swa_lrs",
        default=None,
        type=float,
        help="swa learning rate, if None use the swa start learning rate instead",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="use exponential movving average at swa"
    )
    parser.add_argument("--use_ext_data", action="store_true", help="use external data")

    parser.add_argument(
        "--ext_data_mode",
        default=1,
        type=int,
        help="exteranl data sampling mode",
    )
    parser.add_argument(
        "--aug_mode",
        default=1,
        type=int,
        help="augmentation mode",
    )
    parser.add_argument("--is_debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="find lr with fast ai implementation",
    )
    parser.add_argument(
        "--num_workers",
        default="6",
        type=int,
        help="number of cpus for DataLoader",
    )
    parser.add_argument(
        "--tta_mode",
        default="scale",
        choices=["scale", "flip", "skip"],
        type=str,
        help="test time augmentation mode",
    )
    parser.add_argument(
        "--num_scales",
        default="2",
        type=int,
        help="number of scales for tta",
    )

    # add model specific args
    parser = LitModel.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if args.is_debug:
        DEBUG = True
        print("\t ---- DEBUG RUN ---- ")
        VAL_INTERVAL_SAMPLES = 5000
        args.batch_size = 16
    else:
        DEBUG = False
        print("\t ---- NORMAL RUN ---- ")
    print_argparse_arguments(args)
    main(args)
