#!/usr/bin/env python3
"""
Inject NVFP4_AWQ pre_quant_scale tensors (channel-wise, one per weight)
from a ModelOpt safetensors file into an existing llama.cpp NVFP4 GGUF.

The convert_hf_to_gguf.py pipeline explicitly skips .pre_quant_scale,
which defeats AWQ quantization — the channel-wise per-input-column
scale is never applied at inference, producing garbage output.

Usage: inject_pqs.py <in.gguf> <safetensors.st> <out.gguf>

Emits one new tensor per weight in the GGUF:
  blk.{i}.<name>.pre_quant_scale   shape [in_dim] dtype F32
"""
import sys, os
import numpy as np

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(2)

in_gguf_path, st_path, out_gguf_path = sys.argv[1:]

# Try to reuse the in-tree gguf-py.
root = os.path.dirname(os.path.abspath(__file__))
for cand in [root, os.path.join(root, '../gguf-py'), '/home/jarvis/llama-cpp-v5/gguf-py']:
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, 'gguf')):
        sys.path.insert(0, cand)
        break

import gguf
from safetensors import safe_open

print(f'reading {in_gguf_path}...')
reader = gguf.GGUFReader(in_gguf_path, 'r')

print(f'reading {st_path}...')
st = safe_open(st_path, framework='pt', device='cpu')

# Build map from HF weight name prefix → HF pre_quant_scale tensor.
pqs_map = {}
for k in st.keys():
    if k.endswith('.pre_quant_scale'):
        prefix = k.removesuffix('.pre_quant_scale')  # e.g. model.language_model.layers.0.mlp.down_proj
        pqs_map[prefix] = k
print(f'  found {len(pqs_map)} pre_quant_scale tensors in safetensors')

# HF layer name → GGUF block name mapping for Qwen3.5/3.6 (qwen35 arch).
# HF layout examples:
#   model.language_model.layers.{i}.self_attn.qkv_proj    → blk.{i}.attn_qkv
#   model.language_model.layers.{i}.self_attn.gate_proj   → blk.{i}.attn_gate
#   model.language_model.layers.{i}.self_attn.o_proj      → blk.{i}.attn_output
#   model.language_model.layers.{i}.self_attn.q_proj/k_proj/v_proj → blk.{i}.attn_q/k/v
#   model.language_model.layers.{i}.linear_attn.in_proj_qkv → blk.{i}.attn_qkv (for SSM variant)
#   model.language_model.layers.{i}.linear_attn.out_proj   → blk.{i}.ssm_out
#   model.language_model.layers.{i}.linear_attn.in_proj_a  → blk.{i}.ssm_alpha
#   model.language_model.layers.{i}.linear_attn.in_proj_b  → blk.{i}.ssm_beta
#   model.language_model.layers.{i}.linear_attn.in_proj_z  → blk.{i}.attn_gate (z = the gate)
#   model.language_model.layers.{i}.mlp.gate_proj/up_proj/down_proj → blk.{i}.ffn_{gate,up,down}

HF_TO_GGUF_SUFFIX = {
    'self_attn.qkv_proj': 'attn_qkv',
    'self_attn.gate_proj': 'attn_gate',
    'self_attn.o_proj': 'attn_output',
    'self_attn.q_proj': 'attn_q',
    'self_attn.k_proj': 'attn_k',
    'self_attn.v_proj': 'attn_v',
    'linear_attn.in_proj_qkv': 'attn_qkv',
    'linear_attn.in_proj_z':  'attn_gate',
    'linear_attn.in_proj_a':  'ssm_alpha',
    'linear_attn.in_proj_b':  'ssm_beta',
    'linear_attn.out_proj':   'ssm_out',
    'mlp.gate_proj': 'ffn_gate',
    'mlp.up_proj':   'ffn_up',
    'mlp.down_proj': 'ffn_down',
}

def hf_to_gguf(hf_name: str) -> str | None:
    # Expect "model.language_model.layers.{i}.<suffix>"
    prefix = 'model.language_model.layers.'
    if not hf_name.startswith(prefix):
        return None
    rest = hf_name[len(prefix):]
    idx_str, _, suf = rest.partition('.')
    if not idx_str.isdigit():
        return None
    idx = int(idx_str)
    gguf_suffix = HF_TO_GGUF_SUFFIX.get(suf)
    if gguf_suffix is None:
        return None
    return f'blk.{idx}.{gguf_suffix}'

# Build list of (gguf_name, pqs_values) to append.
extras = []  # each: (full_gguf_tensor_name, f32 numpy array)
for hf_prefix, pqs_key in pqs_map.items():
    gguf_prefix = hf_to_gguf(hf_prefix)
    if gguf_prefix is None:
        print(f'  skip (unknown mapping): {hf_prefix}')
        continue
    t = st.get_tensor(pqs_key).float().numpy().astype(np.float32)
    name = f'{gguf_prefix}.pre_quant_scale'
    extras.append((name, t))

print(f'  mapped {len(extras)} pre_quant_scale tensors')
# Dump a few to sanity check.
for name, v in extras[:4]:
    print(f'    {name} shape={v.shape} first4={v[:4].tolist()}')

print(f'writing {out_gguf_path}...')

# Copy the existing gguf's KVs and tensors into a new writer, then append extras.
writer = gguf.GGUFWriter(out_gguf_path, reader.fields['general.architecture'].parts[-1].tobytes().decode('utf-8'))

# Carry over every KV field.
for key, field in reader.fields.items():
    # Skip tensor count — writer sets it internally.
    if key in ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
        continue
    # Reconstruct the value from the field.
    # gguf-py exposes field.types[0] as the GGUFValueType and field.parts as raw bytes.
    # Easiest: use writer.add_key_value which handles the type dispatch.
    v = field.contents()
    vt = field.types[0]
    writer.add_key_value(key, v, vt)

from gguf.quants import quant_shape_to_byte_shape, quant_shape_from_byte_shape
from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES

# Copy all existing tensors.
for tensor in reader.tensors:
    logical = tuple(int(x) for x in tensor.shape)
    tt = tensor.tensor_type
    # For F32/F16/BF16 etc. quant-size=1 byte per element.
    block_size, type_size = GGML_QUANT_SIZES.get(tt, (1, 1))
    is_quantized = (block_size != 1 or type_size not in (1, 2, 4, 8))
    # Bypass GGUFWriter's shape-conversion path entirely and just record
    # the tensor metadata ourselves — we want to carry over the ORIGINAL
    # logical shape bit-for-bit (including non-standard NVFP4 rows like
    # ssm_alpha with row=48 that don't fit the block-64 helper). Raw data
    # bytes come straight from the reader.
    writer.tensors[-1][tensor.name] = gguf.TensorInfo(
        shape=logical,
        dtype=tt,
        nbytes=int(tensor.data.nbytes),
        tensor=tensor.data,
    )

# Append the extras as F32 tensors.
for name, data in extras:
    writer.add_tensor(name, data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print('done.')
