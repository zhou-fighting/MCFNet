import math
from model.pvtv2 import pvt_v2_b2
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

TRAIN_SIZE = 384


class MCFNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.rgb_backbone = pvt_v2_b2(pretrained=True)
        self.depth_backbone = pvt_v2_b2(pretrained=True)


        self.fuse1 = CAM(64)
        self.fuse2 = CAM(128)
        self.fuse3 = CAM(320)
        self.fuse4 = CAM(512)
        #

        self.decoder3 = DBEB(512 + 320, 256)
        self.decoder2 = DBEB(256 + 128, 128)
        self.decoder1 = DBEB(128 + 64, 64)

        self.pred_4 = nn.Conv2d(512, 1, 1)
        self.pred_3 = nn.Conv2d(256, 1, 1)
        self.pred_2 = nn.Conv2d(128, 1, 1)
        self.pred_1 = nn.Conv2d(64, 1, 1)



    def forward(self, x_rgb, x_d):

        rgb_list = self.rgb_backbone.extract_endpoints(x_rgb)
        depth_list = self.depth_backbone.extract_endpoints(x_d)

        rgb_1 = rgb_list['reduction_2']  # 64,H/4,W/4
        rgb_2 = rgb_list['reduction_3']  # 128,H/8,W/8
        rgb_3 = rgb_list['reduction_4']  # 320,H/16,W/16
        rgb_4 = rgb_list['reduction_5']  # 512,H/32,W/32

        d_1 = depth_list['reduction_2']  # 64,H/4,W/4
        d_2 = depth_list['reduction_3']  # 128,H/8,W/8
        d_3 = depth_list['reduction_4']  # 320,H/16,W/16
        d_4 = depth_list['reduction_5']  # 512,H/32,W/32


        fuse_1 = self.fuse1(rgb_1, d_1)
        fuse_2 = self.fuse2(rgb_2, d_2)
        fuse_3 = self.fuse3(rgb_3, d_3)
        fuse_4 = self.fuse4(rgb_4, d_4)

        #
        x3 = self.decoder3(torch.cat([
            F.interpolate(fuse_4, size=fuse_3.shape[2:], mode='bilinear', align_corners=True),
            fuse_3
        ], dim=1))  # [B,256,H/16,W/16]

        x2 = self.decoder2(torch.cat([
            F.interpolate(x3, size=fuse_2.shape[2:], mode='bilinear', align_corners=True),
            fuse_2
        ], dim=1))  # [B,128,H/8,W/8]

        out = self.decoder1(torch.cat([
            F.interpolate(x2, size=fuse_1.shape[2:], mode='bilinear', align_corners=True),
            fuse_1
        ], dim=1))  # [B,64,H/4,W/4]


        # 预测
        pred4 = F.interpolate(self.pred_4(fuse_4), size=(TRAIN_SIZE, TRAIN_SIZE), mode='bilinear', align_corners=True)
        pred3 = F.interpolate(self.pred_3(x3), size=(TRAIN_SIZE, TRAIN_SIZE), mode='bilinear', align_corners=True)
        pred2 = F.interpolate(self.pred_2(x2), size=(TRAIN_SIZE, TRAIN_SIZE), mode='bilinear', align_corners=True)
        pred1 = F.interpolate(self.pred_1(out), size=(TRAIN_SIZE, TRAIN_SIZE), mode='bilinear', align_corners=True)



        return pred1, pred2, pred3, pred4


class Fusion_Branch(nn.Module):
    def __init__(self, inc, add_ln=True):
        super().__init__()
        self.add_ln = add_ln

        # 通道注意力
        self.channel_att = ChannelAttention(inc)

        # 一个轻量 1x1 卷积替代 Linear
        self.proj = nn.Conv2d(inc, inc, kernel_size=1, stride=1)

        self.layernorm = LayerNorm(inc)
        if add_ln:
            self.ln = LayerNorm(inc)

    def forward(self, x):
        B, C, H, W = x.shape

        # depth 分支先做 CA
        x_att = self.channel_att(x)

        # 再做一次映射
        x_proj = self.proj(x)

        # 两个分支相乘（类似原始 gating 思路）
        x = x_att * x_proj

        # LN 保持
        x = x.flatten(2).permute(0, 2, 1)
        x = self.layernorm(x)
        if self.add_ln:
            x = self.ln(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class CAM(nn.Module):
    def __init__(self, inc, stage=1, use_bn=True):
        super().__init__()
        kernel_size_ls = [3, 5, 7]

        # RGB 分支: 空间注意力
        self.rgb_spatial_att = SpatialAttention(kernel_size=7)

        # Depth 分支: 用 Fusion_Branch(含CA)
        self.fusion_branch = Fusion_Branch(inc, add_ln=True)

        # 拼接分支: 2*inc → inc
        self.conv = DWPWConv(
            inc * 2, inc,
            kernel_size=kernel_size_ls[stage - 1],
            padding=kernel_size_ls[stage - 1] // 2
        )

        if use_bn:
            self.norm = nn.BatchNorm2d(inc)
        else:
            self.norm = LayerNorm(inc, data_format="channels_first")

        # 乘法分支: inc → inc
        self.dw_1 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, groups=inc)
        self.ln_1 = LayerNorm(inc, data_format="channels_first")
        self.pw_1 = nn.Conv2d(inc, inc, 1, 1)

        # 融合
        self.pw = nn.Conv2d(inc * 2, inc, 1, 1)
        self.spatial_att = SpatialAttention(kernel_size=3)

    def forward(self, rgb, depth):
        # --- 乘法交互分支 ---
        # attn_map = self.rgb_spatial_att(rgb)
        rgb_depth_multi = rgb * depth   # [B, inc, H, W]
        rgb_depth_multi = self.dw_1(rgb_depth_multi)
        rgb_depth_multi = self.ln_1(rgb_depth_multi)
        rgb_depth_multi = self.rgb_spatial_att(rgb_depth_multi)
        rgb_depth_multi = rgb_depth_multi + rgb
        rgb_depth_multi = self.pw_1(rgb_depth_multi)
        # --- 拼接交互分支 ---
        rgb_depth_cat = torch.cat((rgb, depth), dim=1)  # [B, 2*inc, H, W]
        rgb_depth_cat = self.conv(rgb_depth_cat)
        rgb_depth_cat = self.fusion_branch(rgb_depth_cat)
        rgb_depth_cat = self.norm(rgb_depth_cat)


        # --- 融合 ---
        fuse = torch.cat((rgb_depth_multi, rgb_depth_cat), dim=1)  # [B, 2*inc, H, W]
        fuse = self.pw(fuse)  # [B, inc, H, W]
        # fuse = self.spatial_att(fuse)
        return fuse

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        # 用 max-pool 和 avg-pool 的结果拼接做注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W] → [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)            # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)          # 最大池化
        x_cat = torch.cat([avg_out, max_out], dim=1)            # [B, 2, H, W]
        attn = self.sigmoid(self.conv(x_cat))                   # [B, 1, H, W]
        return x * attn                                         # 广播相乘

class BConv(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DWPWConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=kernel_size, padding=padding, stride=1,
                      groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class DBEB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DBEB, self).__init__()
        self.DWConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.BConv = BConv(out_channel * 2, out_channel, 3, 1, 1)

    def forward(self, x):
        x_left = self.conv(x)
        x_right = self.DWConv(x)
        out = self.BConv(torch.cat((x_left, x_right), dim=1))

        return out






