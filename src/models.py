"""The four architectures under comparison.

All models take a flat flow-feature vector x of shape (B, F) and return logits
(B, C). CNN, LSTM and Transformer view the F features as a length-F sequence;
how each one exploits that view is the subject of the comparison.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward baseline: treats the feature vector as an unordered set."""

    def __init__(self, n_features, n_classes, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    """1D CNN over the feature axis: kernels see windows of adjacent features."""

    def __init__(self, n_features, n_classes, channels=(32, 64, 128), kernel=3, dropout=0.2):
        super().__init__()
        blocks, c_in = [], 1
        for i, c in enumerate(channels):
            blocks += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2), nn.BatchNorm1d(c), nn.ReLU()]
            if i < len(channels) - 1:
                blocks.append(nn.MaxPool1d(2))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(c_in, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    """Bidirectional LSTM reading features one at a time as a sequence.

    Each step gets the scalar feature value plus a learned embedding of which
    feature it is, so the recurrence is not forced to infer identity from position.
    """

    def __init__(self, n_features, n_classes, hidden=64, layers=1, feat_emb=8, dropout=0.2):
        super().__init__()
        self.feat_emb = nn.Parameter(torch.randn(n_features, feat_emb) * 0.1)
        self.lstm = nn.LSTM(1 + feat_emb, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, 64), nn.ReLU(),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        B, F = x.shape
        seq = torch.cat([x.unsqueeze(-1), self.feat_emb.expand(B, F, -1)], dim=-1)
        out, (h, _) = self.lstm(seq)
        last = torch.cat([h[-2], h[-1]], dim=-1)  # final forward + backward states
        return self.head(torch.cat([last, out.mean(dim=1)], dim=-1))


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block that can expose its attention map."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.n2(x)))
        return (x, w) if return_attn else x


class FTTransformer(nn.Module):
    """Feature-tokenizer Transformer (Gorishniy et al., 2021).

    Every feature becomes a token (value * w_f + b_f), a [CLS] token is
    prepended, and self-attention lets any feature attend to any other,
    independent of column order. Classification reads the [CLS] token.
    """

    def __init__(self, n_features, n_classes, d=32, heads=4, layers=2, ff=64, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_features, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_features, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(EncoderBlock(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def tokens(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), t], dim=1)

    def forward(self, x):
        t = self.tokens(x)
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.norm(t[:, 0]))

    @torch.no_grad()
    def cls_attention(self, x):
        """Mean attention from [CLS] to each feature, per layer: (layers, F)."""
        t, maps = self.tokens(x), []
        for blk in self.blocks:
            t, w = blk(t, return_attn=True)
            maps.append(w[:, 0, 1:].mean(0))
        return torch.stack(maps)


MODELS = {"mlp": MLP, "cnn": CNN1D, "lstm": LSTMNet, "transformer": FTTransformer}
DISPLAY = {"mlp": "MLP", "cnn": "1D CNN", "lstm": "LSTM", "transformer": "Transformer"}


def build_model(name, n_features, n_classes):
    return MODELS[name](n_features, n_classes)
