"""Compare HF SigLIP preprocessing against the inspected legacy TinyLLaVA processor."""
import ast
import os
os.environ['USE_TF']='0'
from functools import partial, reduce
from typing import Dict

import numpy as np
from PIL import Image,ImageOps
from transformers import SiglipImageProcessor
from transformers.image_processing_utils import BatchFeature,get_size_dict
from transformers.image_transforms import convert_to_rgb,normalize,rescale,resize,to_channel_dimension_format
from transformers.image_utils import ChannelDimension,PILImageResampling,to_numpy_array

from hgai_cpu_audit import ROOT,OUT,records,sha
import json


def main():
    source=ROOT/'eda_resources/tinyllava_source/legacy_siglip_encoder.py'
    # Only the short, inspected processor class is executed, not the downloaded model module.
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SigLipImageProcessor')
    module=ast.Module(body=[cls],type_ignores=[])
    namespace=dict(globals())
    exec(compile(module,str(source),'exec'),namespace)
    legacy=namespace['SigLipImageProcessor']()
    hf=SiglipImageProcessor.from_pretrained(ROOT/'model_assets/siglip-so400m-patch14-384',local_files_only=True)
    checks=[]
    for row in list(records())[:20]:
        with Image.open(ROOT/row['image_path']) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
        side=max(im.size)
        pad=Image.new('RGB',(side,side),(127,127,127))
        pad.paste(im,((side-im.width)//2,(side-im.height)//2))
        a=legacy.preprocess(pad,return_tensors='np')['pixel_values']
        b=hf(images=pad,return_tensors='np')['pixel_values']
        difference=float(np.max(np.abs(a-b)))
        assert np.allclose(a,b,rtol=0,atol=1e-6),difference
        checks.append({'image_id':row['image_id'],'max_absolute_pixel_tensor_difference':difference})
    result={'images':len(checks),'legacy_source_sha256':sha(source),'processor_class_only_executed':True,
            'checks':checks,'max_absolute_difference':max(c['max_absolute_pixel_tensor_difference'] for c in checks),
            'limitation':'Processor equivalence on 20 fixed images; no TinyLLaVA language/vision forward pass.'}
    (OUT/'tiny_processor_equivalence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TinyLLaVA processor equivalence:',result['images'],'images; max delta',result['max_absolute_difference'])


if __name__=='__main__':main()
