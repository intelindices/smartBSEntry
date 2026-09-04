"""SmartBS AI backbone: causal dilated convolutions (TCN architecture).

Attribute name ``self.tcn`` on SmartBSClassifier is a checkpoint key prefix —
do not rename it or existing ``.pt`` files will fail to load.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Trim the right padding so convolutions stay causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacked causal residual TCN backbone.

    Input shape:  (batch, channels, seq_len)
    Output shape: (batch, num_channels[-1], seq_len)
    """

    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmartBSClassifier(nn.Module):
    """SmartBS temporal conv net + linear head for FLAT / LONG / SHORT classification."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        num_classes: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features, seq_len)
        y = self.tcn(x)
        # Use the last causal timestep
        return self.head(y[:, :, -1])
