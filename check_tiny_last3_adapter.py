"""Check the pinned TinyLLaVA SDPA HGAI adapter on five dev images."""
import json

import torch
from PIL import Image, ImageOps

from hgai_attention import HGAIAdapter
from measure_layer_curves100 import ROOT, OUT, PREP, inputs_for, load


def main():
    rows = [json.loads(line) for line in
            (PREP / 'dev_common_unmatched500.jsonl').read_text(encoding='utf-8').splitlines()[:5]]
    model, processor, tokenizer, image_id, prompt = load('tiny', backend='sdpa')
    results = []
    with HGAIAdapter(model, range(29, 32), .02) as adapter, torch.inference_mode():
        for row in rows:
            with Image.open(ROOT / row['image_path']) as raw:
                image = ImageOps.exif_transpose(raw).convert('RGB')
            inputs, positions = inputs_for('tiny', model, processor, tokenizer,
                                           image, image_id, prompt)
            adapter.set_image_positions(positions)
            try:
                tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                        num_beams=1, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id)
                result = {'image_id': row['image_id'], 'nonfinite': False,
                          'new_tokens': int(tokens.shape[-1])}
            except ValueError as exc:
                if 'Non-finite HGAI attention' not in str(exc):
                    raise
                result = {'image_id': row['image_id'], 'nonfinite': True,
                          'error': str(exc)}
            results.append(result)
            print(result, flush=True)
            del inputs
            torch.cuda.empty_cache()
    output = {'model': 'bczhou/TinyLLaVA-3.1B', 'images': 5,
              'layers_inclusive': [29, 31], 'alpha': .02,
              'nonfinite_images': sum(item['nonfinite'] for item in results),
              'per_image': results}
    (OUT / 'tiny_last3_adapter5_validation.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('SAVED', output['nonfinite_images'], flush=True)


if __name__ == '__main__':
    main()
