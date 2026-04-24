#!/usr/bin/env python3
"""Strip visual (vision encoder) tensors from a safetensors file.
The vision weights are already saved separately as mmproj_vision_model.safetensors.
Leaves only the language model tensors for convert_hf_to_gguf."""
import sys
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
with safe_open(src, framework='pt', device='cpu') as f:
    tensors = {}
    kept = dropped = 0
    for k in f.keys():
        if k.startswith(('model.visual.', 'visual.')):
            dropped += 1
            continue
        tensors[k] = f.get_tensor(k)
        kept += 1
    print(f'Kept {kept} lang tensors, dropped {dropped} visual')
save_file(tensors, dst)
print(f'Wrote {dst}')
