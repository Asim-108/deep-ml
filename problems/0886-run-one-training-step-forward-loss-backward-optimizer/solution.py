import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    # x is input, y is ground truth?
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    return loss.item()
