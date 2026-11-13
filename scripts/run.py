"""Run decision rules on a group file and write one JSON line per question.

  python scripts/run.py --model Qwen/Qwen3-VL-8B-Instruct --data groups.json \
      --methods m1 affirm --out results/qwen3vl8b

Model calls are cached in <out>/cache.jsonl, so interrupted runs resume and methods share passes.
Use --shard k --nshards N to split the groups over several processes (merge with `cat`)."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfanchor import methods as M
from selfanchor.vlm import VLM, Cache, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Hugging Face id or local path')
    p.add_argument('--data', required=True, help='group file, relative to the repository root')
    p.add_argument('--methods', nargs='+', default=['m1'], choices=list(M.METHODS))
    p.add_argument('--out', required=True); p.add_argument('--split', default=None, help="keep only groups with this 'split'")
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--device-map', default='cuda', help="'auto' splits a large model over all visible GPUs")
    p.add_argument('--nn-index', default=None, help='image embeddings from scripts/nn_index.py (needed for the NN rules of str_bank)')
    a = p.parse_args()
    M.DATA_FILE = a.data
    M.NN_INDEX = a.nn_index
    groups = [g for g in json.loads((ROOT/a.data).read_text()) if a.split is None or g.get('split') == a.split]
    groups = groups[a.shard::a.nshards]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stag = f'.s{a.shard}of{a.nshards}' if a.nshards > 1 else ''
    cache = Cache(out/f'cache{stag}.jsonl', reuse=[c for c in out.glob('cache*.jsonl') if c.name != f'cache{stag}.jsonl'])
    vlm = VLM(a.model, device_map=a.device_map)
    files = {m: out/f'{m}@{Path(a.data).stem}{stag}.jsonl' for m in a.methods}
    done = {m: {json.loads(l)['qid'] for l in f.read_text().splitlines()} if f.exists() else set() for m, f in files.items()}
    fh = {m: f.open('a') for m, f in files.items()}; t0 = time.time()
    for i, g in enumerate(groups):
        for m in a.methods:
            for q in g['questions']:
                if q['id'] in done[m]: continue
                r = M.METHODS[m](vlm, cache, g, q)
                rec = dict(qid=q['id'], group=g['group_id'], answer=q.get('answer'), **r)
                rec['pred'] = 'T' if rec['margin'] > 0 else 'F'
                fh[m].write(json.dumps(rec) + '\n'); fh[m].flush()
        if i % 20 == 0: print(f'{i+1}/{len(groups)} groups, {(time.time()-t0)/(i+1):.2f} s/group', flush=True)
    for f in fh.values(): f.close()


if __name__ == '__main__':
    main()
