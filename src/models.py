"""Model untuk EKG 12-lead. Input (B, 12, T), output logits (B, n_classes)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Baseline: sinyal diratakan setelah pooling, tanpa konvolusi."""

    def __init__(self, n_classes, in_ch=12, length=1000, hidden=256, dropout=0.3, pool=10):
        super().__init__()
        self.pool = nn.AvgPool1d(pool)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * (length // pool), hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(self.pool(x))


class CNN1D(nn.Module):
    def __init__(self, n_classes, in_ch=12, width=32, dropout=0.3):
        super().__init__()
        chans = [in_ch, width, width * 2, width * 4, width * 4]
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(chans[i], chans[i + 1], 7, padding=3, bias=False),
                       nn.BatchNorm1d(chans[i + 1]), nn.ReLU(), nn.MaxPool1d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(chans[-1] * 2, n_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, 1, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + (x if self.short is None else self.short(x)))


class ResNet1D(nn.Module):
    def __init__(self, n_classes, in_ch=12, width=32, blocks=(1, 1, 1, 1), dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, 2, padding=7, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(3, 2, padding=1))
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * (2 ** min(i, 2))
            for j in range(n):
                layers.append(BasicBlock1d(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin * 2, n_classes))

    def forward(self, x):
        h = self.layers(self.stem(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


def make_model(name, n_classes, width=32, blocks=1, dropout=0.3, length=1000):
    if name == "mlp":
        return MLP(n_classes, length=length, dropout=dropout)
    if name == "cnn1d":
        return CNN1D(n_classes, width=width, dropout=dropout)
    if name == "resnet1d":
        return ResNet1D(n_classes, width=width, blocks=(blocks,) * 4, dropout=dropout)
    raise ValueError(f"model tidak dikenal: {name}")
