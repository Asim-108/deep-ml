import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 6, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 12, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(12*2*2, 16),
                nn.ReLU(),
                nn.Linear(16, 10)
            )
         def forward(self, x):
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    return TinyNet()
