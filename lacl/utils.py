import torch

# Define the contrastive relationship (num_classes * num_classes)
# Each row define the set of negative sample class
# 'False' means the corresponding class should be negative sample.

CONTRAS_TABLE = torch.tensor([
    [True,  False, False, False, False],
    [False, True,  False, False, False],
    [False, False, True,  False, False],
    [False, False, False, True,  False],
    [False, False, False, False, True],
])