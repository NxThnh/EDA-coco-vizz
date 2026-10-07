"""Validate first-token VAR attention on 20 paired dev images per model."""
import json
import math

import torch
from PIL import Image, ImageOps
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward as llama_eager
from transformers.models.phi.modeling_phi import eager_attention_forward as phi_eager

from hgai_attention import decoder_layers
from measure_layer_curves100 import ROOT, OUT, PREP, inputs_for, load


def inspect(model_key):
    model, processor, tokenizer, image_id, prompt = load(model_key, backend='sdpa')
    decoder = decoder_layers(model)
    lookup = {id(layer.self_attn): index for index, layer in enumerate(decoder)}
    original = llama_eager if model_key == 'smol' else phi_eager
    all_records = []
    rows = [json.loads(line) for line in
            (PREP / 'dev_common_unmatched500.jsonl').read_text(encoding='utf-8').splitlines()[:20]]
    for item in rows:
        with Image.open(ROOT / item['image_path']) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        inputs, positions = inputs_for(model_key, model, processor, tokenizer,
                                       image, image_id, prompt)
        with torch.inference_mode():
            prefix = model(**inputs, use_cache=True, return_dict=True)
            chosen = prefix.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            mask = torch.ones((1, inputs['attention_mask'].shape[-1] + 1),
                              device=chosen.device, dtype=torch.long)
        recorded = []

        def capture(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
            output, eager = original(module, query, key, value, attention_mask,
                                     dropout=dropout, scaling=scaling, **kwargs)
            layer = lookup.get(id(module))
            if layer is None:
                return output, eager
            grouped = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
            score = query[:, :, -1:, :].float() @ grouped.float().transpose(-2, -1)
            score *= scaling if scaling is not None else query.shape[-1] ** -.5
            if attention_mask is not None:
                chosen_mask = attention_mask[..., -1:, :score.shape[-1]]
                score = score.masked_fill(~chosen_mask, -math.inf) if chosen_mask.dtype == torch.bool else score + chosen_mask.float()
            reconstructed = score.softmax(-1)
            eager_row = eager[:, :, -1:, :].float()
            if layer <= 28:
                assert torch.isfinite(eager_row).all() and torch.isfinite(reconstructed).all()
                recorded.append({'layer': layer,
                                 'row_sum_error': float((eager_row.sum(-1) - 1).abs().max()),
                                 'eager_sdpa_abs_error': float((eager_row - reconstructed).abs().max())})
            return output, eager

        AttentionInterface.register('audit_eager', capture)
        for layer in decoder:
            layer.self_attn.config._attn_implementation = 'audit_eager'
        try:
            with torch.inference_mode():
                model(input_ids=chosen, attention_mask=mask,
                      past_key_values=prefix.past_key_values, use_cache=False,
                      output_attentions=True, return_dict=True)
        finally:
            for layer in decoder:
                layer.self_attn.config._attn_implementation = 'sdpa'
        assert len(recorded) == (24 if model_key == 'smol' else 29)
        all_records.append({'image_id': item['image_id'], 'image_positions': len(positions),
                            'first_position': min(positions), 'last_position': max(positions),
                            'max_row_sum_error': max(r['row_sum_error'] for r in recorded),
                            'max_eager_sdpa_abs_error': max(r['eager_sdpa_abs_error'] for r in recorded)})
        print(model_key, len(all_records), '/ 20', all_records[-1], flush=True)
        del inputs, prefix
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    assert max(r['max_row_sum_error'] for r in all_records) < 5e-4
    if model_key == 'tiny':
        assert {r['image_positions'] for r in all_records} == {729}
    return {'model': 'SmolVLM2-2.2B' if model_key == 'smol' else 'bczhou/TinyLLaVA-3.1B',
            'images': len(all_records),
            'image_positions': sorted(set(r['image_positions'] for r in all_records)),
            'first_position': sorted(set(r['first_position'] for r in all_records)),
            'last_position': sorted(set(r['last_position'] for r in all_records)),
            'max_row_sum_error': max(r['max_row_sum_error'] for r in all_records),
            'max_eager_sdpa_abs_error': max(r['max_eager_sdpa_abs_error'] for r in all_records),
            'per_image': all_records}


def main():
    output = {'models': [inspect('smol'), inspect('tiny')]}
    (OUT / 'layer_attention_validation.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
