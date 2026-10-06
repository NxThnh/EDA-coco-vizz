#!/usr/bin/env python3
"""Run a locked COCO HGAI-aligned baseline when LLaVA weights are available.

Requires a local Hugging Face LLaVA-1.5-7B conversion with model weights.
Never silently downloads 7B weights; the present workspace has only tokenizer
and CLIP vision weights, so this script will stop until the checkpoint is added.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
DETAIL = 'Please help me describe the image in detail.'
CONCISE = 'Describe the visible content in one concise sentence. If the image is too unclear to identify reliably, say "I cannot tell from this image."'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local llava-hf/llava-1.5-7b-hf converted checkpoint with weights")
    parser.add_argument('--partition', choices=['dev_common_unmatched500','eval_common_unmatched500','dev500','eval_random500','eval_unmatched500'], default='dev_common_unmatched500')
    parser.add_argument('--condition', choices=['detailed512','detailed64','concise64'], default='detailed512')
    parser.add_argument('--image-mode', choices=['hgai_crop','pad'], default='hgai_crop')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not any(args.model_path.glob("model*.safetensors")) and not any(args.model_path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"No LLaVA model weights in {args.model_path}; pilot not run")
    from hgai_cpu_audit import prepare
    lock = prepare()
    if args.partition in {'dev_common_unmatched500', 'eval_common_unmatched500'}:
        from lock_common_unmatched import prepare_common
        lock = prepare_common()
    manifest_path = ROOT/'data_prepared/hgai'/(args.partition+'.jsonl')
    out = args.output or ROOT/'eda_results/hgai'/f'llava_baseline_{args.partition}_{args.condition}_{args.image_mode}.jsonl'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite model output: {out}')
    prompt = SYSTEM+' USER: <image>\n'+(CONCISE if args.condition=='concise64' else DETAIL)+' ASSISTANT:'
    max_tokens = 512 if args.condition=='detailed512' else 64
    import torch
    from PIL import Image, ImageOps
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(args.model_path, local_files_only=True)
    # Apply padding only through the explicit manual pad control below.
    if hasattr(processor.image_processor, 'do_pad'):
        processor.image_processor.do_pad = False
    if hasattr(processor, 'patch_size'):
        processor.patch_size = 14
        processor.vision_feature_select_strategy = 'default'
        processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True).eval()
    selected = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    torch.manual_seed(927)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for row in selected:
            with Image.open(ROOT / row["image_path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            if args.image_mode == 'pad':
                side = max(image.size)
                mean = tuple(int(x * 255) for x in processor.image_processor.image_mean)
                padded = Image.new("RGB", (side, side), mean)
                padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
                image = padded
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(model.device, dtype=torch.float16) if key == 'pixel_values'
                      else value.to(model.device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
            generated = output[0, inputs["input_ids"].shape[1]:]
            answer = processor.decode(generated, skip_special_tokens=True).strip()
            record = {"dataset": row["dataset"], "image_id": row["image_id"],
                      "file_name": row["file_name"], "checkpoint": str(args.model_path.resolve()),
                      "preprocessing": args.image_mode,
                      "prompt": prompt, "max_new_tokens": max_tokens,
                      "partition": args.partition, "condition": args.condition,
                      "caption": answer,
                      "manifest_sha256": lock['manifest_sha256'][manifest_path.name],
                      "protocol_note": 'HGAI-aligned baseline; HF conversion and sample differ from published experiment.',
                      "do_sample": False, "num_beams": 1, "answer": answer,
                      "output_tokens": len(generated),
                      "explicit_refusal": "i cannot tell from this image" in answer.casefold()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved {len(selected)} outputs to {out}")


if __name__ == "__main__":
    main()
