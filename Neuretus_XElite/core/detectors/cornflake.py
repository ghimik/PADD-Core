import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def softargmax2d(heatmap, beta=100.0):
    """
    heatmap: [B, 1, H, W]
    return:  [B, 2] -> (x, y) in pixel coordinates
    """
    B, _, H, W = heatmap.shape

    heatmap = heatmap.view(B, -1)
    heatmap = F.softmax(heatmap * beta, dim=1)

    xs = torch.linspace(0, W - 1, W, device=heatmap.device)
    ys = torch.linspace(0, H - 1, H, device=heatmap.device)
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")

    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    x = torch.sum(xs * heatmap, dim=1)
    y = torch.sum(ys * heatmap, dim=1)

    return torch.stack([x, y], dim=1)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.out = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])

    def forward(self, feats):
        # feats: [C2, C3, C4, C5]
        results = []
        x = None
        for i in reversed(range(len(feats))):
            lat = self.lateral[i](feats[i])
            if x is not None:
                x = F.interpolate(x, size=lat.shape[-2:], mode="nearest")
                lat = lat + x
            x = self.out[i](lat)
            results.insert(0, x)
        return results


class HeatmapHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class CornerHeatmapNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        backbone = resnet34(pretrained=pretrained)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1  # 64x64
        self.layer2 = backbone.layer2  # 32x32
        self.layer3 = backbone.layer3  # 16x16
        self.layer4 = backbone.layer4  # 8x8

        self.fpn = FPN(
            in_channels=[64, 128, 256, 512],
            out_channels=256
        )

        self.head = HeatmapHead(in_channels=256 * 4)

    def forward(self, x):
        # x: [B, 3, 256, 256]
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        feats = self.fpn([c2, c3, c4, c5])

        feats = [
            F.interpolate(f, size=(64, 64), mode="bilinear", align_corners=False)
            for f in feats
        ]

        x = torch.cat(feats, dim=1)     # [B, 1024, 64, 64]
        x = self.head(x)                # [B, 1, 64, 64]
        x = F.interpolate(x, size=(256, 256),
                          mode="bilinear", align_corners=False)

        return x


class CornerLoss(nn.Module):
    def __init__(self, lambda_coord=1.0, lambda_sharp=0.1):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_sharp = lambda_sharp
        self.mse = nn.MSELoss()

    def forward(self, pred_heatmap, gt_heatmap, gt_xy):
        """
        pred_heatmap: [B,1,256,256]
        gt_heatmap:   [B,1,256,256]
        gt_xy:        [B,2]  (x,y)
        """
        loss_hm = self.mse(pred_heatmap, gt_heatmap)

        pred_xy = softargmax2d(pred_heatmap)
        loss_coord = F.l1_loss(pred_xy, gt_xy)

        peak = torch.amax(pred_heatmap.view(pred_heatmap.size(0), -1), dim=1)
        loss_sharp = torch.mean(1.0 - peak)

        return loss_hm + self.lambda_coord * loss_coord + self.lambda_sharp * loss_sharp



if __name__ == "__main__":
    model = CornerHeatmapNet(pretrained=False)
    loss_fn = CornerLoss()

    x = torch.randn(2, 3, 256, 256)
    gt_heatmap = torch.randn(2, 1, 256, 256)
    gt_xy = torch.tensor([[120.0, 200.0], [64.0, 180.0]])

    pred = model(x)
    loss = loss_fn(pred, gt_heatmap, gt_xy)

    print("Output shape:", pred.shape)
    print("Loss:", loss.item())
