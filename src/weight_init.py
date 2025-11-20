# +
import torch
import torch.nn as nn
import numpy as np

def keras_init(m):
    if isinstance(m, nn.Conv1d):
        fin, fout = nn.init._calculate_fan_in_and_fan_out(m.weight)
        a = np.sqrt(6 / (m.in_channels * (fin + fout)))
        torch.nn.init.uniform_(m.weight, a=-a, b=a)
        
        # Type-safe: only zero bias if it exists
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)