"""Versioned sampling fix: two new splits unmatched across every audited ID pool.

The version-8 lock and its three manifests are immutable. This lock filters the
source ID pools, not model outputs; it does not claim globally clean pretraining.
"""
import hashlib
import json
import random
from collections import defaultdict

import pandas as pd

from hgai_cpu_audit import ROOT, OUT, PREP, EXT, prepare, sha, csv_write


FLAGS = ['llava_sft_union', 'share_sft_union', 'share_pretrain_union',
         'share_cap100k_union', 'smol_checked_upstream_candidate']


def audit_fingerprint(frame):
    # Partition labels may change; source-ID evidence must remain unchanged.
    rows = frame[['image_id', *FLAGS]].sort_values('image_id').astype(int)
    return hashlib.sha256(rows.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def prepare_common():
    old = prepare()
    frame = pd.read_csv(OUT / 'small_vlm_training_overlap.csv')
    sources = {'version8_lock': sha(PREP / 'manifest_lock.json'),
               'audited_id_flags': audit_fingerprint(frame),
               **old['source_sha256']}
    path = PREP / 'manifest_lock_common_v9.json'
    if path.exists():
        lock = json.loads(path.read_text(encoding='utf-8'))
        if lock['source_sha256'] != sources:
            raise ValueError('Audited sources changed: create another version, do not resample v9.')
        for name, digest in lock['manifest_sha256'].items():
            if sha(PREP / name) != digest:
                raise ValueError('Locked v9 manifest altered: ' + name)
        return lock
    if any((ROOT / 'experiment_outputs').glob('**/*.jsonl')):
        raise ValueError('Model outputs exist; inspect them before locking a new experiment.')
    hashes = pd.read_csv(EXT / 'val_image_hashes.csv')
    hashes = hashes[hashes.dataset == 'COCO'].set_index('image_id')
    caps = json.loads((ROOT / 'coco/annotations/captions_val2014.json').read_text())
    images = {r['id']: r for r in caps['images']}
    captions = defaultdict(list)
    for r in caps['annotations']:
        captions[r['image_id']].append(r)
    used_ids, used_hashes = set(), set()
    for name in old['manifest_sha256']:
        for line in (PREP / name).read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            used_ids.add(r['image_id']); used_hashes.add(r['md5'])
    unmatched = frame[frame[FLAGS].eq(0).all(axis=1)]
    pool = sorted(i for i in unmatched.image_id.astype(int)
                  if i not in used_ids and hashes.loc[i, 'decode_ok'] == 1
                  and hashes.loc[i, 'md5'] not in used_hashes)
    # Deduplicate the eligible pool before drawing; preserve deterministic ID order.
    unique_pool, seen = [], set()
    for i in pool:
        digest = hashes.loc[i, 'md5']
        if digest not in seen:
            unique_pool.append(i); seen.add(digest)
    if len(unique_pool) < 1000:
        raise ValueError('Not enough disjoint images for eval500 plus dev500.')
    summary, manifests = [], {}
    for partition, seed in [('eval_common_unmatched500', 45), ('dev_common_unmatched500', 46)]:
        chosen = sorted(random.Random(seed).sample(unique_pool, 500))
        selected = set(chosen)
        unique_pool = [i for i in unique_pool if i not in selected]
        rows = []
        for i in chosen:
            image = images[i]
            rows.append({'dataset': 'COCO', 'partition': partition, 'image_id': i,
                         'file_name': image['file_name'], 'image_path': 'coco/val2014/' + image['file_name'],
                         'width': image['width'], 'height': image['height'], 'md5': hashes.loc[i, 'md5'],
                         'llava_direct_coco': False, 'llava_vg_gqa_alias': False,
                         'llava_known_match': False,
                         **{flag: False for flag in FLAGS},
                         'valid_captions': [r['caption'] for r in sorted(captions[i], key=lambda r: r['id'])]})
        name = partition + '.jsonl'
        (PREP / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        manifests[name] = sha(PREP / name)
        csv_write(PREP / (partition + '.csv'), [{k: v for k, v in r.items() if k != 'valid_captions'} for r in rows])
        summary.append({'partition': partition, 'images': 500, 'seed': seed,
                        'direct_coco': 0, 'vg_gqa_alias': 0, 'known_union': 0,
                        'unmatched': 500, 'sha256': manifests[name]})
    lock = {'schema_version': 2, 'created_date': '2026-10-06', 'source_sha256': sources,
            'manifest_sha256': manifests, 'partitions': summary,
            'filter_columns': FLAGS, 'total_common_unmatched': len(unmatched),
            'eligible_unused_id_pool': len(pool), 'remaining_unique_pool': len(unique_pool),
            'notes': ['Version-8 manifests retained byte-for-byte; no model outputs existed at sampling.',
                      'All five manifests are pairwise disjoint in image ID and MD5.',
                      'Zero matches across audited pools only; unknown SmolVLM2 selection/base data remain unverified.',
                      'Filtered evaluation is separate from random500 paper-protocol comparison.']}
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_write(OUT / 'manifest_summary.csv', [*old['partitions'], *summary])
    return lock


if __name__ == '__main__':
    locked = prepare_common()
    print('Common-unmatched lock verified:', [(r['partition'], r['images']) for r in locked['partitions']],
          '; audited unmatched pool:', locked['total_common_unmatched'])
