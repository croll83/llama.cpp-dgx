#!/usr/bin/env python3
"""
Re-quantize the abliterated Qwen3.6-27B BF16 model to *plain* NVFP4
(NVFP4_DEFAULT_CFG, no AWQ) so llama.cpp can use it natively.

The previous `quantize_nvfp4.py` used NVFP4_AWQ_LITE_CFG, which emits
weights paired with a channel-wise `.pre_quant_scale` that llama.cpp
does not apply at inference — result: garbage output. Plain NVFP4
folds the scaling into the weights themselves during quantization.

Usage:
  python3 quantize_nvfp4_plain.py \
    /home/jarvis/qwopus36-27b-nvfp4/abliterated \
    /home/jarvis/qwopus36-27b-nvfp4/nvfp4-plain
"""
import torch, sys, os, shutil

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

SOURCE_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/jarvis/qwopus36-27b-nvfp4/abliterated"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/jarvis/qwopus36-27b-nvfp4/nvfp4-plain"

def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    import modelopt.torch.quantization as mtq

    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_DIR, trust_remote_code=True)

    print("\n=== Loading VLM model (BF16) ===")
    model = AutoModelForImageTextToText.from_pretrained(
        SOURCE_DIR, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Save vision model BEFORE quantization so mmproj stays untouched.
    print("\n=== Saving vision model (mmproj) ===")
    vision_model = model.model.visual
    vision_state = {k: v.cpu().clone() for k, v in vision_model.state_dict().items()}
    vision_params = sum(p.numel() for p in vision_model.parameters()) / 1e6
    print(f"Vision model params: {vision_params:.1f}M")

    calib_prompts = [
        "The theory of relativity describes how space and time are interconnected.",
        "Machine learning algorithms can be broadly categorized into supervised and unsupervised.",
        "The chemical composition of water is H2O, consisting of hydrogen and oxygen.",
        "Shakespeare wrote many famous plays including Hamlet, Macbeth, and Romeo and Juliet.",
        "The human genome contains approximately 3 billion base pairs of DNA.",
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "Neural networks are inspired by the biological structure of the brain.",
        "The French Revolution began in 1789 and led to major political changes.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The stock market is influenced by various economic and political factors.",
        "Climate change is driven by the increasing concentration of greenhouse gases.",
        "Artificial intelligence is transforming industries from healthcare to finance.",
        "The periodic table organizes chemical elements by their atomic number.",
        "Democracy is a form of government where power is held by the people.",
        "The Internet has revolutionized communication and access to information.",
        "Evolution by natural selection was proposed by Charles Darwin.",
    ]

    def forward_loop(model):
        for i, text in enumerate(calib_prompts):
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
            del inputs
            torch.cuda.empty_cache()
            print(f"  calib {i+1}/{len(calib_prompts)}")
        print("Calibration complete")

    print("\n=== Quantizing to plain NVFP4 (NVFP4_DEFAULT_CFG) ===")
    # NVFP4_DEFAULT_CFG: per-block UE4M3 scale + per-tensor FP32 scale2,
    # no AWQ channel-wise pre_quant_scale. llama.cpp's NVFP4 kernels
    # (post commit 8bae7173a) handle this recipe natively.
    quant_cfg = mtq.NVFP4_DEFAULT_CFG
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    print("Quantization complete")

    print("\n=== Exporting quantized model ===")
    from modelopt.torch.export import export_hf_checkpoint
    export_hf_checkpoint(model, export_dir=OUTPUT_DIR)
    print(f"Exported to {OUTPUT_DIR}")

    tokenizer.save_pretrained(OUTPUT_DIR)

    mmproj_path = os.path.join(OUTPUT_DIR, "mmproj_vision_model.safetensors")
    from safetensors.torch import save_file
    save_file(vision_state, mmproj_path)
    print(f"Vision model saved: {mmproj_path}")

    for fn in ["config.json", "processor_config.json", "chat_template.jinja",
               "generation_config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(SOURCE_DIR, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fn))
            print(f"Copied: {fn}")

    print(f"\nDone. Plain NVFP4 model at {OUTPUT_DIR}")
    print("Next:")
    print(f"  python3 convert_hf_to_gguf.py {OUTPUT_DIR} --outfile {OUTPUT_DIR}/Nvfp4-Plain-27B-NVFP4.gguf")

if __name__ == "__main__":
    main()
