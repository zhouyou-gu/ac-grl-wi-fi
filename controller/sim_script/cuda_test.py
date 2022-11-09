import torch
print("torch.version",torch.version.__version__)

print("torch.cuda.is_available()",torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.current_device()",torch.cuda.current_device())

import torch_geometric
print("torch_geometric.__version__",torch_geometric.__version__)