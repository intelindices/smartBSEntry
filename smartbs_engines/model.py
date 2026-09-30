"""SmartBS AI backbones: TCN, ModernTCN, Transformer, DualTF (1h+15m).

``SmartBSClassifier`` keeps attribute ``self.tcn`` as a checkpoint key prefix.
``SmartBSEntryV2`` / ``SmartBSTF`` / ``SmartBSDualTF`` are separate backbones
(new checkpoints).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm

BACKBONE_TCN = "tcn"
BACKBONE_SMARTBS_ENTRY = "smartBSEntry"  # alias of classic TCN
BACKBONE_SMARTBS_ENTRY_V2 = "smartBSEntryV2"
BACKBONE_SMARTBS_TF = "smartBSTF"
BACKBONE_SMARTBS_DUAL_TF = "smartBSDualTF"


class Chomp1d(nn.Module):
    """Trim the right padding so convolutions stay causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


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
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
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
    """Stacked causal residual TCN backbone (Bai et al.).

    Input shape:  (batch, channels, seq_len)
    Output shape: (batch, num_channels[-1], seq_len)
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
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
    """Classic TCN + linear head for FLAT / LONG / SHORT (smartBSEntry)."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        num_classes: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.head = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x)
        return self.head(y[:, :, -1])


class _CausalDWConv1d(nn.Module):
    """Depthwise causal 1D conv (left-pad only)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class ModernTCNBlock(nn.Module):
    """DW temporal conv + pointwise ConvFFN (ModernTCN-style), causal."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
    ):
        super().__init__()
        hidden = max(dim * int(ffn_ratio), dim)
        self.norm1 = nn.LayerNorm(dim)
        self.dw = _CausalDWConv1d(dim, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = x.transpose(1, 2)
        y = self.norm1(y).transpose(1, 2)
        x = x + self.dw(y)
        y = x.transpose(1, 2)
        y = self.norm2(y).transpose(1, 2)
        x = x + self.ffn(y)
        return x


class ModernTCN(nn.Module):
    """ModernTCN-style stack: stem + causal DW/FFN stages."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        stages: list[nn.Module] = []
        prev = num_inputs
        for i, width in enumerate(num_channels):
            if prev != width:
                stages.append(nn.Conv1d(prev, width, kernel_size=1))
            stages.append(
                ModernTCNBlock(
                    width,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            prev = width
        self.network = nn.Sequential(*stages)
        self.out_channels = int(num_channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmartBSEntryV2(nn.Module):
    """ModernTCN backbone + linear head (smartBSEntryV2)."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        num_classes: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = ModernTCN(
            num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.head = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        return self.head(y[:, :, -1])


def _n_heads_for(d_model: int, prefer: int = 4) -> int:
    n = int(prefer)
    while n > 1 and d_model % n != 0:
        n //= 2
    return max(1, n)


class SmartBSTF(nn.Module):
    """Causal Transformer encoder + linear head (smartBSTF).

    Input layout matches the TCN models: ``(batch, num_inputs, seq_len)``.
    ``num_channels[-1]`` sets ``d_model``; ``len(num_channels)`` sets layer count.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        num_classes: int = 3,
        kernel_size: int = 1,  # unused; kept for factory signature parity
        dropout: float = 0.2,
        max_len: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        del kernel_size  # not used by attention blocks
        d_model = int(num_channels[-1])
        n_layers = max(2, len(num_channels))
        heads = _n_heads_for(d_model, n_heads)
        self.d_model = d_model
        self.in_proj = nn.Linear(int(num_inputs), d_model)
        self.pos = nn.Parameter(torch.zeros(1, int(max_len), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=max(d_model * 2, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        h = x.transpose(1, 2)
        h = self.in_proj(h) * math.sqrt(self.d_model)
        t = h.size(1)
        if t > self.pos.size(1):
            raise ValueError(f"seq_len {t} > SmartBSTF max_len {self.pos.size(1)}")
        h = h + self.pos[:, :t, :]
        # True = blocked; upper triangle blocks future tokens (causal).
        mask = torch.triu(
            torch.ones(t, t, device=h.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        return self.head(h[:, -1, :])


class SmartBSDualTF(nn.Module):
    """1H TCN + 15M TCN → time-align → Transformer → fusion MLP → 3-class.

    Pipeline (matches architecture diagram)::

        x_1h (B,C1,T1) ──embed──► TCN(k=3,d=1,2,4) ─┐
                                                      ├─ align ─► Transformer ─► state ─► MLP ─► logits
        x_15m(B,C2,T2) ──embed──► TCN(k=3,d=1,2,4) ─┘

    ``T2`` should be ``T1 * tf_ratio`` (default 4) with end-aligned windows so
    each 1h step maps to ``tf_ratio`` consecutive 15m steps (last kept).
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        num_classes: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        *,
        num_inputs_15m: int | None = None,
        tf_ratio: int = 4,
        n_heads: int = 4,
        n_tf_layers: int = 2,
        max_len: int = 256,
    ):
        super().__init__()
        d_model = int(num_channels[-1]) if num_channels else 64
        # Exactly 3 TCN blocks → dilations 1, 2, 4
        tcn_channels = [d_model, d_model, d_model]
        ks = 3 if kernel_size is None else int(kernel_size)
        c15 = int(num_inputs_15m) if num_inputs_15m is not None else int(num_inputs)
        self.d_model = d_model
        self.tf_ratio = max(1, int(tf_ratio))
        self.num_inputs_1h = int(num_inputs)
        self.num_inputs_15m = c15

        self.embed_1h = nn.Sequential(
            nn.Conv1d(self.num_inputs_1h, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.embed_15m = nn.Sequential(
            nn.Conv1d(c15, d_model, kernel_size=1),
            nn.GELU(),
        )
        self.tcn_1h = TemporalConvNet(
            d_model, tcn_channels, kernel_size=ks, dropout=dropout
        )
        self.tcn_15m = TemporalConvNet(
            d_model, tcn_channels, kernel_size=ks, dropout=dropout
        )
        self.align_proj = nn.Linear(d_model * 2, d_model)
        self.pos = nn.Parameter(torch.zeros(1, int(max_len), d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        heads = _n_heads_for(d_model, n_heads)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=max(d_model * 2, 64),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(n_tf_layers)))
        self.norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head = nn.Linear(d_model, int(num_classes))

    def _align_15m(self, h1: torch.Tensor, m15: torch.Tensor) -> torch.Tensor:
        """Map 15m TCN features onto the 1h time grid (B, D, T1)."""
        b, d, t1 = h1.shape
        t15 = m15.size(2)
        r = self.tf_ratio
        need = t1 * r
        if t15 == need:
            return m15.view(b, d, t1, r)[:, :, :, -1]
        if t15 > need:
            m15 = m15[:, :, t15 - need :]
            return m15.view(b, d, t1, r)[:, :, :, -1]
        # Shorter 15m window: right-pad by repeating last step, then pool.
        pad = need - t15
        m15 = torch.cat([m15, m15[:, :, -1:].expand(b, d, pad)], dim=2)
        return m15.view(b, d, t1, r)[:, :, :, -1]

    def forward(
        self,
        x_1h: torch.Tensor,
        x_15m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_15m is None:
            raise TypeError("SmartBSDualTF requires (x_1h, x_15m)")
        # Embed → TCN
        h1 = self.tcn_1h(self.embed_1h(x_1h))
        m15 = self.tcn_15m(self.embed_15m(x_15m))
        m15_a = self._align_15m(h1, m15)
        # Time alignment: concat per-step features → project to d_model
        tokens = torch.cat([h1.transpose(1, 2), m15_a.transpose(1, 2)], dim=-1)
        tokens = self.align_proj(tokens)
        t = tokens.size(1)
        if t > self.pos.size(1):
            raise ValueError(f"seq_len {t} > SmartBSDualTF max_len {self.pos.size(1)}")
        tokens = tokens + self.pos[:, :t, :]
        mask = torch.triu(
            torch.ones(t, t, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        tokens = self.encoder(tokens, mask=mask)
        state = self.norm(tokens[:, -1, :])
        state = self.fusion(state)
        return self.head(state)


def model_forward(model: nn.Module, x, device: torch.device) -> torch.Tensor:
    """Forward helper: single tensor or ``(x_1h, x_15m)`` batch from DataLoader."""
    if isinstance(x, (tuple, list)):
        return model(x[0].to(device), x[1].to(device))
    return model(x.to(device))


def normalize_backbone(name: str | None) -> str:
    raw = str(name or BACKBONE_TCN).strip()
    key = raw.lower().replace("-", "").replace("_", "")
    if key in (
        "smartbsdualtf",
        "dualtf",
        "dual",
        "tcntf",
        "1h15m",
        "multitf",
        "mtf",
    ):
        return BACKBONE_SMARTBS_DUAL_TF
    if raw == BACKBONE_SMARTBS_DUAL_TF:
        return BACKBONE_SMARTBS_DUAL_TF
    if key in ("smartbstfv2", "tfv2", "transformerv2", "attnreason", "signaltf"):
        raise ValueError(
            "smartBSTFV2 was removed; use smartBSTF or smartBSDualTF"
        )
    if key in ("smartbstf", "transformer", "tf", "attn", "attention"):
        return BACKBONE_SMARTBS_TF
    if raw == BACKBONE_SMARTBS_TF:
        return BACKBONE_SMARTBS_TF
    if key in ("smartbsentryv2", "moderntcn", "v2"):
        return BACKBONE_SMARTBS_ENTRY_V2
    if raw == BACKBONE_SMARTBS_ENTRY_V2:
        return BACKBONE_SMARTBS_ENTRY_V2
    if key in ("smartbsentry", "tcn", "classic", "v1", ""):
        return BACKBONE_TCN
    return BACKBONE_TCN


def default_kernel_size(backbone: str | None) -> int:
    bb = normalize_backbone(backbone)
    if bb == BACKBONE_SMARTBS_ENTRY_V2:
        return 7
    if bb == BACKBONE_SMARTBS_TF:
        return 1
    if bb == BACKBONE_SMARTBS_DUAL_TF:
        return 3
    return 3


def build_classifier(
    *,
    num_inputs: int,
    num_channels: list[int],
    num_classes: int = 3,
    kernel_size: int | None = None,
    dropout: float = 0.2,
    backbone: str | None = None,
    num_inputs_15m: int | None = None,
    tf_ratio: int = 4,
) -> nn.Module:
    """Factory: TCN, V2, TF, or DualTF (1h+15m)."""
    bb = normalize_backbone(backbone)
    if bb == BACKBONE_SMARTBS_DUAL_TF:
        ks = 3 if kernel_size is None else int(kernel_size)
        return SmartBSDualTF(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            num_classes=num_classes,
            kernel_size=ks,
            dropout=dropout,
            num_inputs_15m=num_inputs_15m,
            tf_ratio=int(tf_ratio),
        )
    if bb == BACKBONE_SMARTBS_TF:
        ks = 1 if kernel_size is None else int(kernel_size)
        return SmartBSTF(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            num_classes=num_classes,
            kernel_size=ks,
            dropout=dropout,
        )
    if bb == BACKBONE_SMARTBS_ENTRY_V2:
        ks = 7 if kernel_size is None else int(kernel_size)
        return SmartBSEntryV2(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            num_classes=num_classes,
            kernel_size=ks,
            dropout=dropout,
        )
    ks = 3 if kernel_size is None else int(kernel_size)
    return SmartBSClassifier(
        num_inputs=num_inputs,
        num_channels=list(num_channels),
        num_classes=num_classes,
        kernel_size=ks,
        dropout=dropout,
    )
