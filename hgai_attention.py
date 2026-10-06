"""HGAI for pinned Transformers Llama/Phi attention, without replacing RoPE/cache.

Only the final query is modified, as in the published HGAI code. Other queries
use native SDPA. Alpha zero delegates directly to SDPA, preserving baseline bits.
Image positions are supplied per input; grid/boundary/padding tokens are excluded.
"""
from contextlib import AbstractContextManager
import math

import torch
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import repeat_kv


def decoder_layers(model):
    for path in ['model.text_model.layers', 'model.language_model.layers',
                 'language_model.model.layers', 'model.layers', 'model.model.layers']:
        obj = model
        try:
            for part in path.split('.'):
                obj = getattr(obj, part)
            if len(obj) and hasattr(obj[0], 'self_attn'):
                return obj
        except AttributeError:
            pass
    raise TypeError('Unsupported decoder; do not guess layer or token positions.')


def fractional_layers(depth, bounds):
    start, end = bounds
    if not 0 <= start <= end <= 1:
        raise ValueError('Fractional bounds must satisfy 0 <= start <= end <= 1')
    # Inclusive, zero-based; floor converts fractional depth to concrete indices.
    return list(range(min(depth-1, math.floor(start*depth)), min(depth-1, math.floor(end*depth))+1))


def intervention_attention(module, query, key, value, attention_mask, *, alpha,
                           image_positions, scaling, dropout=0.0):
    """Reference last-query computation, including GQA and mask preservation."""
    if query.shape[0] != 1 or dropout or module.training:
        raise ValueError('HGAI runner supports eval mode, batch=1 and dropout=0 only.')
    groups = query.shape[1] // key.shape[1]
    if query.shape[1] % key.shape[1]:
        raise ValueError('Incompatible GQA head counts')
    keys = repeat_kv(key, groups)
    values = repeat_kv(value, groups)
    positions = torch.as_tensor(image_positions, device=query.device, dtype=torch.long)
    if positions.numel() == 0 or positions.min() < 0 or positions.max() >= keys.shape[-2]:
        raise ValueError('Invalid or missing visual token positions')
    if positions.unique().numel() != positions.numel():
        raise ValueError('Duplicate visual token positions')
    logits = torch.matmul(query[:, :, -1:, :], keys.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        if attention_mask.ndim != 4:
            raise ValueError('Expected the model-provided additive/boolean 4-D mask')
        mask = attention_mask[..., -1:, :keys.shape[-2]]
        allowed = mask if mask.dtype == torch.bool else mask >= 0
        if not allowed[..., positions].all():
            raise ValueError('A supplied image position is masked; exclude padding tokens.')
        logits = logits.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else logits + mask
    boost = logits[..., positions].abs().mean(dim=1, keepdim=True)
    logits[..., positions] = logits[..., positions] + alpha * boost
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    if not torch.isfinite(weights).all():
        raise ValueError('Non-finite HGAI attention; stop rather than score invalid generation.')
    return torch.matmul(weights, values).transpose(1, 2).contiguous(), weights


class HGAIAdapter(AbstractContextManager):
    def __init__(self, model, layers, alpha):
        self.layers = decoder_layers(model)
        self.indices = list(layers)
        if len(set(self.indices)) != len(self.indices) or not self.indices:
            raise ValueError('Select unique concrete decoder layer indices')
        if any(i < 0 or i >= len(self.layers) for i in self.indices):
            raise ValueError('Layer index outside decoder depth')
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError('Alpha must be finite and non-negative')
        self.alpha = alpha
        self.positions = []
        self.calls = 0
        self.modified_calls = 0
        self.previous = None

    def set_image_positions(self, positions):
        self.positions = [int(i) for i in positions]

    def __enter__(self):
        self.previous = ALL_ATTENTION_FUNCTIONS['sdpa']
        if getattr(self.previous, '_hgai_dispatch', False):
            raise RuntimeError('Nested/concurrent HGAI registry contexts are unsupported')
        selected = {id(self.layers[i].self_attn) for i in self.indices}
        for i in self.indices:
            if self.layers[i].self_attn.config._attn_implementation != 'sdpa':
                raise ValueError('Load language decoder with attn_implementation=sdpa')
        def dispatch(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
            output, weights = self.previous(module, query, key, value, attention_mask,
                                            dropout=dropout, scaling=scaling, **kwargs)
            if id(module) not in selected:
                return output, weights
            self.calls += 1
            if self.alpha == 0:
                return output, weights
            last, _ = intervention_attention(module, query, key, value, attention_mask,
                alpha=self.alpha, image_positions=self.positions,
                scaling=scaling if scaling is not None else query.shape[-1]**-0.5,
                dropout=dropout)
            output[:, -1:, :, :] = last
            self.modified_calls += 1
            return output, None
        dispatch._hgai_dispatch = True
        AttentionInterface.register('sdpa', dispatch)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        AttentionInterface.register('sdpa', self.previous)
        return False
