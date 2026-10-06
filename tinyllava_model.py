"""Explicit Transformers-4.53 port of the pinned legacy TinyLLaVA Phi/SigLIP model.

All checkpoint tensor names are retained; no remote model code is executed.
Legacy SigLIP deletes its 27th layer and pooling head, then uses hidden_states[-1]
(before post-layernorm). Vision and projector weights are embedded
in the full TinyLLaVA checkpoint and must not be replaced by generic SigLIP.
"""
import json

import torch
from torch import nn
from transformers import PhiConfig, PhiForCausalLM, SiglipVisionConfig, SiglipVisionModel


class TinyPhiConfig(PhiConfig):
    model_type = 'tiny_llava_phi'


class TinyPhiForCausalLM(PhiForCausalLM):
    config_class = TinyPhiConfig

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.mm_projector = nn.Sequential(nn.Linear(1152, config.hidden_size), nn.GELU(),
                                               nn.Linear(config.hidden_size, config.hidden_size))
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.vision_tower = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
            num_attention_heads=16, image_size=384, patch_size=14,
            hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-6))
        del self.model.vision_tower.vision_tower.vision_model.encoder.layers[-1:]
        self.model.vision_tower.vision_tower.vision_model.head = nn.Identity()
        self.post_init()

    def prepare_visual_inputs(self, tokenizer, pixel_values, prompt):
        # Exactly one image, no truncation or padding; preserve the legacy template.
        before = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
        after = '\n' + prompt + ' ASSISTANT:'
        left = tokenizer(before, add_special_tokens=True, return_tensors='pt').input_ids
        right = tokenizer(after, add_special_tokens=False, return_tensors='pt').input_ids
        device = self.model.embed_tokens.weight.device
        vision = self.model.vision_tower.vision_tower
        visual = vision(pixel_values.to(device=device, dtype=self.dtype), output_hidden_states=True).hidden_states[-1]
        visual = self.model.mm_projector(visual)
        if visual.shape[1] != 729:
            raise ValueError('Legacy TinyLLaVA must produce exactly 729 patch embeddings')
        embeddings = torch.cat([self.model.embed_tokens(left.to(device)), visual,
                                self.model.embed_tokens(right.to(device))], dim=1)
        if embeddings.shape[1] > self.config.max_position_embeddings:
            raise ValueError('Multimodal prefix exceeds Phi context')
        positions = list(range(left.shape[1], left.shape[1]+729))
        return {'inputs_embeds':embeddings,
                'attention_mask':torch.ones(embeddings.shape[:2],dtype=torch.long,device=device)},positions


def validate_checkpoint_mapping(model, path):
    index=json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))
    expected=set(index['weight_map'])
    # Quantized state_dict also contains absmax/quant_map metadata, not source tensors.
    actual=set(dict(model.named_parameters()))
    if expected!=actual:
        raise ValueError('Tiny checkpoint mapping mismatch; missing='+str(sorted(expected-actual)[:10])+
                         '; extra='+str(sorted(actual-expected)[:10]))
    return len(expected)
