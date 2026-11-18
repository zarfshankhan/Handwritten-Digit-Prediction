import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, patch_size=2, embed_dim=32, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = 16

        self.patch_projection = nn.Linear(4, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def create_patches(self, x):
        batch_size = x.shape[0]
        x = x.unfold(2, 2, 2).unfold(3, 2, 2)
        x = x.contiguous().view(batch_size, 1, 16, 4)
        x = x.squeeze(1)
        return x

    def forward(self, x):
        x = self.create_patches(x)
        x = self.patch_projection(x)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x