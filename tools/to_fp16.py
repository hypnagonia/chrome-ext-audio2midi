import sys, json, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
sd = load_file(src)
out = {k: (v if k.startswith('condition_provider.') else v.half()).contiguous() for k, v in sd.items()}
save_file(out, dst, metadata={'format': 'pt', 'note': 'fp16 conversion of MuScriptor weights; condition_provider tensors kept fp32'})
print(dst, sum(v.numel() * v.element_size() for v in out.values()) / 1e6, 'MB')
