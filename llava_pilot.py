#!/usr/bin/env python3
"""Run a fixed 100+100 dev-image caption pilot when LLaVA weights are available.

Requires a local Hugging Face LLaVA-1.5-7B conversion with model weights.
Never silently downloads 7B weights; the present workspace has only tokenizer
and CLIP vision weights, so this script will stop until the checkpoint is added.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEV = ROOT / "data_prepared/dev.jsonl"
OUT = ROOT / "eda_results/vlm_extensions/llava_pilot_100x2.jsonl"
PROMPT = ('USER: <image>\nDescribe the visible content in one concise sentence. '
          'If the image is too unclear to identify reliably, say "I cannot tell from this image." ASSISTANT:')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Local llava-hf/llava-1.5-7b-hf converted checkpoint with weights")
    args = parser.parse_args()
    if not any(args.model_path.glob("model*.safetensors")) and not any(args.model_path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"No LLaVA model weights in {args.model_path}; pilot not run")
    import torch
    from PIL import Image, ImageOps
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    processor = LlavaProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True).eval()
    manifests = [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(42)
    selected = []
    for dataset in ("COCO", "VizWiz"):
        pool = sorted((row for row in manifests if row["dataset"] == dataset), key=lambda row: int(row["image_id"]))
        selected.extend(rng.sample(pool, 100))
    with OUT.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for row in selected:
            with Image.open(ROOT / row["image_path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            side = max(image.size)
            mean = tuple(round(x * 255) for x in processor.image_processor.image_mean)
            padded = Image.new("RGB", (side, side), mean)
            padded.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            inputs = processor(text=PROMPT, images=padded, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1)
            generated = output[0, inputs["input_ids"].shape[1]:]
            answer = processor.decode(generated, skip_special_tokens=True).strip()
            record = {"dataset": row["dataset"], "image_id": row["image_id"],
                      "file_name": row["file_name"], "checkpoint": str(args.model_path.resolve()),
                      "preprocessing": "manual square pad with CLIP image_mean before processor resize/crop",
                      "prompt": PROMPT, "max_new_tokens": 64,
                      "do_sample": False, "num_beams": 1, "answer": answer,
                      "output_tokens": len(generated),
                      "explicit_refusal": "i cannot tell from this image" in answer.casefold()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    print(f"Saved {len(selected)} outputs to {OUT}")


if __name__ == "__main__":
    main()
