import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class TinyCNN(nn.Module):
    """Small CNN: Conv2d -> ReLU -> pool -> (optional extras) -> Linear."""

    def __init__(self, img_size=8, n_classes=2):
        super().__init__()
        # TODO: define conv, activation, pooling, and classifier layers
        # Input shape: (N, 1, img_size, img_size)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            # (img-3)/1 + 1 = img-2
            # output shape is in-2 x in-2 x 6
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # (img-4)/2 + 1 = img/2 - 2 + 1 = img/2 - 1
            # output shape is (img/2 - 1) x (img/2 - 1) x 6
        )
        self.fc = nn.Linear((img_size//2 - 1) * (img_size//2 - 1) * 6, n_classes)            
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # TODO: implement forward pass; return logits of shape (N, n_classes)
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def build_model(img_size=8, n_classes=2):
    """Return an instance of your TinyCNN (or equivalent nn.Module)."""
    # TODO: return TinyCNN(img_size=img_size, n_classes=n_classes)
    model = TinyCNN(img_size, n_classes)
    return model


def train_model(model, train_x, train_y, epochs=15, lr=0.01, batch_size=32, seed=0):
    """Train model on train_x/train_y and return the trained model.

    Args:
        model: nn.Module from build_model
        train_x: FloatTensor (N, 1, H, W)
        train_y: LongTensor (N,)
        epochs: number of full passes over the data
        lr: optimizer learning rate
        batch_size: mini-batch size
        seed: RNG seed for shuffling / init determinism

    Returns:
        Trained model (same instance is fine).
    """
    # TODO:
    # - torch.manual_seed(seed)
    # - CrossEntropyLoss + Adam (or SGD)
    # - mini-batch loop for `epochs` epochs
    # - model.train(); zero_grad -> forward -> loss -> backward -> step
    torch.manual_seed(seed)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_data = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    for epoch in range(epochs):
        for X, y in iter(train_loader):
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return model
