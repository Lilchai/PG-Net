import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# 频率融合与编码器模块
# ===========================
class LaplacianPyramidFusion(nn.Module):
    def __init__(self, levels: int = 5, sigma: float = 1.0):
        super().__init__()
        self.levels = levels
        self.sigma = sigma
        # 高斯模糊核（用于构建金字塔）
        self.gaussian_kernel = self._create_gaussian_kernel(sigma)

    def _create_gaussian_kernel(self, sigma):
        """创建3D高斯卷积核"""
        kernel_size = 5
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # 3D可分离高斯核
        kernel_3d = (kernel_1d[:, None, None] *
                     kernel_1d[None, :, None] *
                     kernel_1d[None, None, :])
        return kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)

    def gaussian_blur_3d(self, x):
        """3D高斯模糊（用于构建金字塔）"""
        B, C, D, H, W = x.shape
        kernel = self.gaussian_kernel.to(x.device)
        blurred_list = []
        for c in range(C):
            channel_data = x[:, c:c + 1, ...]
            padded = F.pad(channel_data, (2, 2, 2, 2, 2, 2), mode='reflect')
            blurred = F.conv3d(padded, kernel, padding=0)
            blurred_list.append(blurred)
        return torch.cat(blurred_list, dim=1)

    def build_laplacian_pyramid(self, x):
        """构建拉普拉斯金字塔（只在训练时计算）"""
        # x: [B, C, D, H, W]
        # 构建高斯金字塔
        gaussian_pyramid = [x]
        for i in range(self.levels - 1):
            blurred = self.gaussian_blur_3d(gaussian_pyramid[-1])
            downsampled = blurred[:, :, ::2, ::2, ::2]
            gaussian_pyramid.append(downsampled)
        # 构建拉普拉斯金字塔
        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            current = gaussian_pyramid[i]
            downsampled = gaussian_pyramid[i + 1]
            upsampled = F.interpolate(
                downsampled,
                size=current.shape[2:],
                mode='trilinear',
                align_corners=False
            )
            upsampled_blurred = self.gaussian_blur_3d(upsampled)
            laplacian = current - upsampled_blurred  # 高频细节
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

class SimpleFrequencyFusion(nn.Module):
    def __init__(self, conv_channels, raw_channels, fusion_type='gate'):
        super().__init__()
        self.raw_proj = nn.Conv3d(raw_channels, conv_channels, 1)
        self.norm = nn.InstanceNorm3d(conv_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.fusion_type = fusion_type
        if fusion_type == 'gate':
            self.gate_conv = nn.Conv3d(conv_channels * 2, conv_channels, 3, padding=1)
            self.gate_sigmoid = nn.Sigmoid()

    def forward(self, conv_feat, lap_feat):
        if lap_feat.shape[2:] != conv_feat.shape[2:]:
            lap_feat = F.interpolate(lap_feat, size=conv_feat.shape[2:], mode='trilinear')
        raw_proj = self.act(self.norm(self.raw_proj(lap_feat)))
        if self.fusion_type == 'gate':
            gate = self.gate_sigmoid(self.gate_conv(torch.cat([conv_feat, raw_proj], dim=1)))
            return conv_feat + self.alpha * (gate * raw_proj)
        return conv_feat + self.alpha * raw_proj


class MPConvEncoder3DWithLaplacian(nn.Module):
    def __init__(self, in_channels=1, fusion_type='gate'):
        super().__init__()
        self.laplacian = LaplacianPyramidFusion(levels=5)

        # Stage 1: (H/2)
        self.stem = nn.Sequential(nn.Conv3d(in_channels, 32, 3, stride=2, padding=1), nn.InstanceNorm3d(32),
                                  nn.LeakyReLU(0.01))
        self.fusion1 = SimpleFrequencyFusion(32, in_channels, fusion_type)
        self.stage1 = nn.Sequential(MPConv(32, 64), MPConv(64, 64))

        # Stage 2: (H/4)
        self.down2 = nn.Conv3d(64, 128, 2, stride=2)
        self.fusion2 = SimpleFrequencyFusion(128, in_channels, fusion_type)
        self.stage2 = nn.Sequential(MPConv(128, 128), MPConv(128, 128))

        # Stage 3: (H/8)
        self.down3 = nn.Conv3d(128, 256, 2, stride=2)
        self.fusion3 = SimpleFrequencyFusion(256, in_channels, fusion_type)
        self.stage3 = nn.Sequential(MPConv(256, 256), MPConv(256, 256))

        # Stage 4: (H/16)
        self.down4 = nn.Sequential(nn.Conv3d(256, 512, 2, stride=2), nn.InstanceNorm3d(512), nn.LeakyReLU(0.01))
        self.fusion4 = SimpleFrequencyFusion(512, in_channels, fusion_type)
        self.stage4 = nn.Sequential(MPConv(512, 512), MPConv(512, 512))

    def forward(self, x):
        pyramid = self.laplacian.build_laplacian_pyramid(x)
        x1 = self.stage1(self.fusion1(self.stem(x), pyramid[1]))
        x2 = self.stage2(self.fusion2(self.down2(x1), pyramid[2]))
        x3 = self.stage3(self.fusion3(self.down3(x2), pyramid[3]))
        x4 = self.stage4(self.fusion4(self.down4(x3), pyramid[4]))
        return x1, x2, x3, x4

# ===========================
# 2. 基础组件
# ===========================
class MPConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        self.dwconv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(kernel_size, 3, 3),
                                 padding=(kernel_size // 2, 1, 1), groups=in_channels)
        self.dwconv_d = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                  groups=in_channels)
        self.dwconv_h = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 3), padding=(1, 0, 1),
                                  groups=in_channels)
        self.dwconv_w = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0),
                                  groups=in_channels)
        self.pwconv = nn.Conv3d(in_channels * 3, out_channels, kernel_size=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.leakyrelu = nn.LeakyReLU(0.01, inplace=True)
        self.residual_adapter = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = self.residual_adapter(x) if self.residual_adapter else x
        x1 = self.leakyrelu(self.norm(self.dwconv1(x)))
        x_d = self.leakyrelu(self.norm(self.dwconv_d(x1)))
        x_h = self.leakyrelu(self.norm(self.dwconv_h(x1)))
        x_w = self.leakyrelu(self.norm(self.dwconv_w(x1)))
        return self.pwconv(torch.cat([x_d, x_h, x_w], dim=1)) + identity


class VesselGuidedAttentionGate3D(nn.Module):

    def __init__(self, F_g, F_l, F_int, norm_layer=nn.InstanceNorm3d):
        super().__init__()

        self.W_gv = nn.Sequential(
            nn.Conv3d(F_g + 1, F_int, kernel_size=3, padding=1),
            norm_layer(F_int),
            nn.ReLU(inplace=True)
        )


        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_int, F_int // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int // 4, F_int, kernel_size=1),
            nn.Sigmoid()
        )


        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, g, x, vessel_map):

        if vessel_map.shape[2:] != g.shape[2:]:
            vessel_map = F.interpolate(vessel_map, size=g.shape[2:], mode='trilinear', align_corners=False)

        combined = torch.cat([g, vessel_map], dim=1)

        fused = self.W_gv(combined)

        ca = self.channel_attn(fused)
        fused = fused * ca

        psi = self.psi(fused)

        return x * (1 + self.alpha * psi), psi

class UpBlock3D(nn.Module):
    """通用上采样块"""
    def __init__(self, in_up, skip_ch, out_ch, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_up, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_skip = F.interpolate(x_skip, size=x_up.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x_up, x_skip], dim=1))


class UpBlock3DWithVesselAttention(nn.Module):
    def __init__(self, in_up, skip_ch, out_ch, norm_layer=nn.InstanceNorm3d):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_up, out_ch, kernel_size=2, stride=2)
        self.attention_gate = VesselGuidedAttentionGate3D(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2,
                                                          norm_layer=norm_layer)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm_layer(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x_up, x_skip, vessel_map):
        x_up = self.up(x_up)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_skip = F.interpolate(x_skip, size=x_up.shape[2:], mode='trilinear', align_corners=False)

        x_skip, attn = self.attention_gate(x_up, x_skip, vessel_map)
        return self.conv(torch.cat([x_up, x_skip], dim=1)), attn


# ===========================
# 3. 主模型: FMV_Net
# ===========================
class FMV_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, fusion_type='gate', norm_name="instance"):
        super().__init__()
        norm_layer = nn.BatchNorm3d if norm_name == "batch" else nn.InstanceNorm3d

        self.encoder = MPConvEncoder3DWithLaplacian(in_channels, fusion_type)

        # 瓶颈层与上采样块
        self.bottleneck = nn.Sequential(
            nn.Conv3d(512, 1024, kernel_size=3, padding=1, stride=2),
            norm_layer(1024), nn.ReLU(inplace=True),
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            norm_layer(1024), nn.ReLU(inplace=True)
        )

        self.up1 = UpBlock3D(1024, 512, 512, norm_layer)
        self.up2 = UpBlock3D(512, 256, 256, norm_layer)
        self.up3 = UpBlock3DWithVesselAttention(256, 128, 128, norm_layer)
        self.up4 = UpBlock3DWithVesselAttention(128, 64, 64, norm_layer)

        self.final_up = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            norm_layer(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, out_channels, kernel_size=1)
        )

    def forward(self, x, vessel_map=None, return_aux=False):
        # 1. 编码器入口引导 (Stem 之后)
        pyramid = self.encoder.laplacian.build_laplacian_pyramid(x)

        x_stem = self.encoder.stem(x)

        # 2. 编码器后续 Stage
        x1 = self.encoder.stage1(self.encoder.fusion1(x_stem, pyramid[1]))
        x2 = self.encoder.stage2(self.encoder.fusion2(self.encoder.down2(x1), pyramid[2]))
        x3 = self.encoder.stage3(self.encoder.fusion3(self.encoder.down3(x2), pyramid[3]))
        x4 = self.encoder.stage4(self.encoder.fusion4(self.encoder.down4(x3), pyramid[4]))

        # 3. Bottleneck
        feat = self.bottleneck(x4)

        # 4. 解码器与跳跃连接引导
        d1 = self.up1(feat, x4)
        d2 = self.up2(d1, x3)

        d3, attn3 = self.up3(d2, x2, vessel_map)
        d4, attn4 = self.up4(d3, x1, vessel_map)

        # d1 = self.up1(feat, x4)
        # d2, attn2= self.up2(d1, x3, vessel_map)
        #
        # d3, attn3 = self.up3(d2, x2, vessel_map)
        # d4 = self.up4(d3, x1)

        # 5. 输出
        out = self.out(self.final_up(d4))

        if return_aux:
            return out, {'attentions': [attn3, attn4]}
        return out