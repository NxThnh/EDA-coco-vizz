"""Record unavailable external sources separately from implemented experiments."""
import json
from datetime import datetime,timezone
from hgai_cpu_audit import ROOT,OUT,sha


def main():
    api_path=ROOT/'model_assets/SmolVLM2-2.2B-Instruct/weights_api.json'
    inventory=json.loads(api_path.read_text(encoding='utf-8'))
    recipe=json.loads((OUT/'smol_training_evidence.json').read_text(encoding='utf-8'))
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),
            'direction_documents':{'status':'unavailable_user_confirmed','originals_read':False,
                'basis':'User states they do not have the two original documents.',
                'working_scope':'COCO primary; VizWiz context; TinyLLaVA-3.1B provisional checkpoint'},
            'smol_complete_training_manifest':{'status':'not_published_in_inspected_sources',
                'checkpoint_revision':inventory['sha'],'repository_inventory_sha256':sha(api_path),
                'published_files':[r['rfilename'] for r in inventory['siblings']],
                'recipe_revision':recipe['revision'],'recipe_entries':recipe['mixture_entries'],
                'recipe_source_paths':'Author-local /fsx JSON paths, not complete selected-ID manifests.',
                'remaining_unknown':recipe['remaining_unknown'],
                'sources':['https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/tree/main',
                           'https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct#training-data',
                           'https://github.com/huggingface/smollm/tree/'+recipe['revision']+'/vision/smolvlm2'],
                'conclusion':'Zero audited matches does not establish zero actual training exposure.'}}
    (OUT/'source_availability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Original direction documents unavailable; exact Smol training manifest remains external and unverified.')


if __name__=='__main__':main()
