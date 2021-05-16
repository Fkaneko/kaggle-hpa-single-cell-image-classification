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

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.losses import ArcMarginProduct_subcenter

LAST_CHANEL_TABLE = {
    "resnet50": 2048,
    "resnet50d": 2048,
    "resnet101d": 2048,
    "resnet152d": 2048,
    "resnet200d": 2048,
    "resnest50d": 2048,
    "resnest101e": 2048,
    "resnest200e": 2048,
    "resnest269e": 2048,
    "efficientnet_b3a": 1536,
    "tf_efficientnet_b4_ns": 1792,
}


# class Net(network.resnet38d.Net):
class SCCAM(nn.Module):
    def __init__(
        self,
        timm_model_name: str = "resnet50",
        num_classes: int = 19,
        k_cluster: int = 10,
        round_nb: int = 0,
        # final_channel: int = 2048,
        drop_rate: float = 0.5,
        pretrained: bool = True,
        use_arc_face: bool = True,
        k_arc_center: int = 3,
        arc_feat_dim: int = 512,
    ):

        super().__init__()
        self.backbone = timm.create_model(
            timm_model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        final_channel = LAST_CHANEL_TABLE[timm_model_name]
        self.k = k_cluster
        self.round_nb = round_nb
        self.use_arc_face = use_arc_face
        print("k_cluster: {}".format(self.k))
        print("Round: {}".format(self.round_nb))

        self.dropout7 = torch.nn.Dropout2d(drop_rate)

        if self.use_arc_face:
            self.feat = nn.Linear(final_channel, arc_feat_dim)
            self.bn = nn.BatchNorm1d(arc_feat_dim)

        fc8_20, fc8_200 = get_pred_head(
            round_nb=round_nb,
            k=k_cluster,
            final_channel=final_channel,
            num_classes=num_classes,
            use_arc_face=use_arc_face,
            arc_feat_dim=arc_feat_dim,
            k_arc_center=k_arc_center,
        )
        if round_nb > 0:
            self.fc8_20 = fc8_20
            self.fc8_200 = fc8_200
        else:
            self.fc8 = fc8_20

    def head(self, x: torch.Tensor, round_nb: int = 0):
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

        feature = x
        feature = feature.view(feature.size(0), -1)
        if self.use_arc_face:
            feature = self.feat(feature)
            feature = self.bn(feature)
            x = feature

        # class 20
        if round_nb == 0:
            x = self.fc8(x)
            x = x.view(x.size(0), -1)
            y = torch.sigmoid(x)
            return x, feature, y

        # class 20 + class 200
        else:
            x_20 = self.fc8_20(x)
            x_20 = x_20.view(x_20.size(0), -1)
            y_20 = torch.sigmoid(x_20)

            x_200 = self.fc8_200(x)
            x_200 = x_200.view(x_200.size(0), -1)
            y_200 = torch.sigmoid(x_200)

            return x_20, feature, y_20, x_200, y_200

    def forward(self, x: torch.Tensor, round_nb: int = 0):
        # x = super().forward(x)
        x = self.backbone(x)
        return self.head(x, round_nb=round_nb)

    def multi_label(self, x):
        x = torch.sigmoid(x)
        tmp = x.cpu()
        tmp = tmp.data.numpy()
        _, cls = np.where(tmp > 0.5)

        return cls, tmp

    def forward_cam(self, x):
        # x = super().forward(x)
        x = self.backbone(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def forward_two_cam(self, x):
        # x = super().forward(x)
        x_ = self.backbone(x)

        x_20 = F.conv2d(x_, self.fc8_20.weight)
        cam_20 = F.relu(x_20)

        x_200 = F.conv2d(x_, self.fc8_200.weight)
        cam_200 = F.relu(x_200)

        return cam_20, cam_200

    def calc_cam(self, feat_map: torch.Tensor, fc_weight: torch.Tensor) -> torch.Tensor:
        feat_map = F.conv2d(feat_map, fc_weight)
        cam = F.relu(feat_map)
        return cam


def get_pred_head(
    round_nb: int = 0,
    k: int = 10,
    final_channel: int = 2048,
    num_classes: int = 19,
    use_arc_face: bool = False,
    arc_feat_dim: int = 512,
    k_arc_center: int = 3,
):
    # class 20
    if round_nb == 0:
        if use_arc_face:
            fc8 = ArcMarginProduct_subcenter(
                in_features=arc_feat_dim, out_features=num_classes, k=k_arc_center
            )
        else:
            fc8 = nn.Conv2d(  # type: ignore[assignment]
                final_channel, num_classes, 1, bias=False
            )
            torch.nn.init.xavier_uniform_(fc8.weight)
        # class 20 + class 200
        return fc8, None
    else:
        if use_arc_face:
            fc8_20 = ArcMarginProduct_subcenter(
                in_features=arc_feat_dim, out_features=num_classes, k=k_arc_center
            )
            fc8_200 = ArcMarginProduct_subcenter(
                in_features=arc_feat_dim,
                out_features=k * num_classes,
                k=k_arc_center,
            )
        else:
            fc8_20 = nn.Conv2d(  # type: ignore[assignment]
                final_channel, num_classes, 1, bias=False
            )
            fc8_200 = nn.Conv2d(  # type: ignore[assignment]
                final_channel, k * num_classes, 1, bias=False
            )
            torch.nn.init.xavier_uniform_(fc8_20.weight)
            torch.nn.init.xavier_uniform_(fc8_200.weight)
        return fc8_20, fc8_200

    # def get_parameter_groups(self):
    #     groups = ([], [], [], [])

    #     for m in self.modules():

    #         if isinstance(m, nn.Conv2d):

    #             if m.weight.requires_grad:
    #                 if m in self.from_scratch_layers:
    #                     groups[2].append(m.weight)
    #                 else:
    #                     groups[0].append(m.weight)

    #             if m.bias is not None and m.bias.requires_grad:

    #                 if m in self.from_scratch_layers:
    #                     groups[3].append(m.bias)
    #                 else:
    #                     groups[1].append(m.bias)

    #     return groups
