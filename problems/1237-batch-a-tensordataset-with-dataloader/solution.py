import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = 0
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        if num_batches == 0:
            first_batch_X_shape_tuple = tuple(batch_X.shape)
        num_batches += 1

    return (num_batches, first_batch_X_shape_tuple)