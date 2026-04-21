import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    Compact CNN for raw EEG classification (Lawhern et al. 2018).
    Input: (batch, 1, n_channels, n_times)
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 64,
        n_times: int = 512,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        flat_size = f2 * ((n_times // 4 // 8))
        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        return self.classifier(x)


class EEGTransformer(nn.Module):
    """
    Transformer encoder for EEG classification.
    Input: (batch, n_channels, n_times) — treats each channel as a token.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 64,
        n_times: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_times, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)            # (batch, channels, d_model)
        x = self.encoder(x)         # (batch, channels, d_model)
        x = x.mean(dim=1)           # global average pool over channels
        return self.classifier(x)
