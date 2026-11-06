"""Fail fast if we did not actually get a 3090.

Every entry script in this codebase does
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
so a faulty/unavailable GPU degrades SILENTLY to CPU: the job still "succeeds",
just ~100x slower and with CPU fp16 semantics instead of Ampere. At 12.24M traces
that is days of wasted queue time and numerics we would not trust anyway.
"""
import sys
import torch

if not torch.cuda.is_available():
    print("PREFLIGHT FAIL: torch.cuda.is_available() is False -- would silently "
          "run on CPU. Check CUDA_VISIBLE_DEVICES / GPU health on this node.",
          file=sys.stderr)
    sys.exit(1)

name = torch.cuda.get_device_name(0)
print(f"PREFLIGHT: CUDA ok, device 0 = {name}", flush=True)

if "3090" not in name:
    print(f"PREFLIGHT FAIL: expected a GeForce RTX 3090, got '{name}'. This "
          "pipeline is fp16-exact and results are only trusted on Ampere.",
          file=sys.stderr)
    sys.exit(1)

# prove the device actually works, not just that it enumerates
try:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.matmul(x, x).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    print(f"PREFLIGHT FAIL: GPU enumerated but unusable: {e}", file=sys.stderr)
    sys.exit(1)

print("PREFLIGHT: 3090 verified and functional", flush=True)
