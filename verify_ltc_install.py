import torch
from ncps.torch import LTC

print("-- Verifying LTC / PyTorch stack ---------------------------")
T, F, H = 20, 10, 32

ltc     = LTC(F, H)
dummy   = torch.randn(T, F)            # (Time, Features)

outputs, hidden = ltc(dummy)

assert outputs.shape == (T, H),  f"unexpected outputs {outputs.shape}"
assert hidden.shape  == (H,),    f"unexpected hidden {hidden.shape}"  # ❗ one 1-D vector

print("✅  LTC forward pass ok")
print(f"    outputs shape {outputs.shape}")
print(f"    hidden  shape {hidden.shape}")