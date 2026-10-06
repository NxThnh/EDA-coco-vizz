"""Semantic checks: original formula, GQA, masked keys, alpha=0 and KV-cache."""
import json
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PhiConfig, PhiForCausalLM
from hgai_attention import HGAIAdapter, intervention_attention
from hgai_cpu_audit import ROOT, sha


def main():
    torch.set_num_threads(2)
    torch.manual_seed(927)
    dummy = torch.nn.Module().eval()
    q, k, v = torch.randn(1,4,7,8), torch.randn(1,2,7,8), torch.randn(1,2,7,8)
    positions = [1,3,5]
    mask = torch.zeros(1,1,7,7)
    mask[...,6] = float('-inf')
    output, weights = intervention_attention(dummy,q,k,v,mask,alpha=.5,image_positions=positions,scaling=8**-.5)
    kr, vr = k.repeat_interleave(2,dim=1), v.repeat_interleave(2,dim=1)
    logits = q[:,:,-1:] @ kr.transpose(-2,-1) / 8**.5 + mask[:,:,-1:]
    for p in positions:
        logits[:,:,:,p] += .5*logits[:,:,:,p].abs().mean(dim=1,keepdim=True)
    expected = logits.softmax(-1) @ vr
    torch.testing.assert_close(output,expected.transpose(1,2),rtol=1e-6,atol=1e-6)
    assert weights[...,6].eq(0).all()
    results=[]
    for cls, cfg in [
        (LlamaForCausalLM,LlamaConfig(vocab_size=48,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8)),
        (PhiForCausalLM,PhiConfig(vocab_size=48,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,partial_rotary_factor=.5)),
    ]:
        cfg._attn_implementation='sdpa'
        model=cls(cfg).eval()
        ids=torch.tensor([[2,3,4,5,6,7]])
        with torch.inference_mode():
            baseline=model(ids,use_cache=True)
            with HGAIAdapter(model,[0,1],0) as adapter:
                adapter.set_image_positions([1,3])
                zero=model(ids,use_cache=True)
                assert torch.equal(baseline.logits,zero.logits), cls.__name__
                assert adapter.calls==2 and adapter.modified_calls==0
            with HGAIAdapter(model,[0,1],.5) as adapter:
                adapter.set_image_positions([1,3])
                prefill=model(ids,use_cache=True)
                cached=model(torch.tensor([[8]]),past_key_values=prefill.past_key_values,use_cache=True)
                assert torch.isfinite(cached.logits).all() and adapter.modified_calls==4
            with HGAIAdapter(model,[0,1],.5) as adapter:
                adapter.set_image_positions([1,3])
                full=model(torch.tensor([[2,3,4,5,6,7,8]]),use_cache=False)
                # Final-query-only intervention changes prefill KV states; compare
                # cached/no-cache *generation* with identical intervened prefix is
                # tested separately on actual models rather than falsely assuming equality.
                assert torch.isfinite(full.logits).all()
            restored=model(ids,use_cache=False)
            assert torch.equal(restored.logits,baseline.logits)
        results.append({'architecture':cls.__name__,'alpha_zero_bit_exact':True,
                        'cached_generation_finite':True,'gqa_tested':True,'registry_restored':True})
    report={'status':'passed','original_formula_and_mask':True,'models':results,
            'adapter_sha256':sha(ROOT/'hgai_attention.py'),'test_sha256':sha(ROOT/'test_hgai_attention.py')}
    path=ROOT/'eda_results/hgai/hgai_adapter_validation.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
