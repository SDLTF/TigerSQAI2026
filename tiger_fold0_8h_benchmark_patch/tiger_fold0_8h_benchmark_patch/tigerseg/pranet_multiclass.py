from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover - resolved by project requirements
    raise ImportError(
        "PraNet-MC requires timm. Install it with: python -m pip install timm"
    ) from exc


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RFBModified(nn.Module):
    """Receptive-field block used by PraNet before partial decoding."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, out_channels, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, (3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, (5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, (1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, (7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7),
        )

        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = BasicConv2d(in_channels, out_channels, 1, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)]
        fused = self.conv_cat(torch.cat(branches, dim=1))
        return self.relu(fused + self.conv_res(x))


class PartialDecoder(nn.Module):
    """Aggregate stride-32, stride-16 and stride-8 RFB features."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.up1 = BasicConv2d(channels, channels, 3, padding=1)
        self.up2 = BasicConv2d(channels, channels, 3, padding=1)
        self.up3 = BasicConv2d(channels, channels, 3, padding=1)
        self.up4 = BasicConv2d(channels, channels, 3, padding=1)
        self.up5 = BasicConv2d(2 * channels, 2 * channels, 3, padding=1)

        self.concat2 = BasicConv2d(2 * channels, 2 * channels, 3, padding=1)
        self.concat3 = BasicConv2d(3 * channels, 3 * channels, 3, padding=1)
        self.refine = BasicConv2d(3 * channels, 3 * channels, 3, padding=1)
        self.classifier = nn.Conv2d(3 * channels, num_classes, 1)

    @staticmethod
    def _resize(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        deepest: torch.Tensor,
        middle: torch.Tensor,
        shallow: torch.Tensor,
    ) -> torch.Tensor:
        deepest_to_middle = self._resize(deepest, middle)
        middle_gated = self.up1(deepest_to_middle) * middle

        deepest_to_shallow = self._resize(deepest, shallow)
        middle_to_shallow = self._resize(middle, shallow)
        shallow_gated = self.up2(deepest_to_shallow) * self.up3(middle_to_shallow) * shallow

        middle_fused = self.concat2(
            torch.cat([middle_gated, self.up4(deepest_to_middle)], dim=1)
        )
        shallow_fused = self.concat3(
            torch.cat([shallow_gated, self.up5(self._resize(middle_fused, shallow))], dim=1)
        )
        return self.classifier(self.refine(shallow_fused))


class UncertaintyReverseAttention(nn.Module):
    """
    Multiclass adaptation of PraNet reverse attention.

    Original PraNet is binary. Here the reverse-attention gate is the per-pixel
    uncertainty ``1 - max(softmax(logits))``. The stage predicts a multiclass
    residual that refines the previous logits.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        self.reduce = BasicConv2d(in_channels, hidden_channels, 1)
        self.refine = nn.Sequential(
            BasicConv2d(hidden_channels, hidden_channels, 3, padding=1),
            BasicConv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, feature: torch.Tensor, previous_logits: torch.Tensor) -> torch.Tensor:
        previous = F.interpolate(
            previous_logits,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence = torch.softmax(previous, dim=1).amax(dim=1, keepdim=True)
        reverse_gate = 1.0 - confidence
        residual = self.classifier(self.refine(self.reduce(feature) * reverse_gate))
        return previous + residual


class MultiClassPraNet(nn.Module):
    """PraNet-style multiclass semantic-segmentation model with deep supervision."""

    def __init__(
        self,
        num_classes: int,
        *,
        backbone_name: str = "res2net50_26w_4s",
        pretrained: bool = True,
        decoder_channels: int = 32,
        reverse_attention_channels: int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        feature_channels: Sequence[int] = self.backbone.feature_info.channels()
        feature_reductions: Sequence[int] = self.backbone.feature_info.reduction()

        if len(feature_channels) != 3:
            raise RuntimeError(
                f"Expected three backbone feature maps, got channels={feature_channels}"
            )
        if tuple(feature_reductions) != (8, 16, 32):
            raise RuntimeError(
                "PraNet-MC expects backbone reductions (8, 16, 32), "
                f"got {tuple(feature_reductions)} from {backbone_name!r}"
            )

        shallow_channels, middle_channels, deep_channels = map(int, feature_channels)
        self.rfb_shallow = RFBModified(shallow_channels, decoder_channels)
        self.rfb_middle = RFBModified(middle_channels, decoder_channels)
        self.rfb_deep = RFBModified(deep_channels, decoder_channels)
        self.partial_decoder = PartialDecoder(decoder_channels, num_classes)

        self.ra_deep = UncertaintyReverseAttention(
            deep_channels, reverse_attention_channels, num_classes
        )
        self.ra_middle = UncertaintyReverseAttention(
            middle_channels, reverse_attention_channels, num_classes
        )
        self.ra_shallow = UncertaintyReverseAttention(
            shallow_channels, reverse_attention_channels, num_classes
        )

    @staticmethod
    def _to_input(logits: torch.Tensor, input_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_size = x.shape[-2:]
        shallow, middle, deep = self.backbone(x)

        rfb_shallow = self.rfb_shallow(shallow)
        rfb_middle = self.rfb_middle(middle)
        rfb_deep = self.rfb_deep(deep)

        coarse = self.partial_decoder(rfb_deep, rfb_middle, rfb_shallow)
        deep_refined = self.ra_deep(deep, coarse)
        middle_refined = self.ra_middle(middle, deep_refined)
        final = self.ra_shallow(shallow, middle_refined)

        return (
            self._to_input(coarse, input_size),
            self._to_input(deep_refined, input_size),
            self._to_input(middle_refined, input_size),
            self._to_input(final, input_size),
        )
