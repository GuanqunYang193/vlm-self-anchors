"""SigLIP scores for M2 (and the CLIP/SigLIP baselines): minimum pairwise cosine of the query group, the positive anchor and
the negative anchors, using exactly the anchor images of M1.

  python scripts/embed.py --data groups.json --encoder siglip --out results/embed"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfanchor import methods as M
from selfanchor.vlm import ROOT

ENCODERS = {'clip': ('ViT-L-14', 'openai'), 'siglip': ('ViT-SO400M-14-SigLIP', 'webli')}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--encoder', default='siglip', choices=list(ENCODERS))
    p.add_argument('--out', required=True); a = p.parse_args()
    import open_clip
    M.DATA_FILE = a.data
    G = json.loads((ROOT/a.data).read_text())
    views = {g['group_id']: dict(q=g['images'], pos=[g['images'][0]] + [M.augment(g['images'][0], k) for k in range(1, len(g['images']))],
                                 neg=[g['images'][:-1] + [M.distractor(g, k)] for k in range(M.N_NEG)]) for g in G}
    paths = sorted({x for v in views.values() for x in v['q'] + v['pos'] + [y for n in v['neg'] for y in n]})
    model, _, tf = open_clip.create_model_and_transforms(*ENCODERS[a.encoder]); model = model.eval().cuda(); E = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(paths), 64):
            b = paths[i:i+64]
            e = torch.nn.functional.normalize(model.encode_image(torch.stack([tf(Image.open(ROOT/x).convert('RGB')) for x in b]).cuda()).float(), dim=-1)
            E.update(zip(b, e.cpu().numpy()))
    smin = lambda ps: min(float(E[x] @ E[y]) for i, x in enumerate(ps) for y in ps[i+1:])
    res = {gid: dict(s=smin(v['q']), s_pos=smin(v['pos']), s_neg=[smin(n) for n in v['neg']]) for gid, v in views.items()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f'{a.encoder}@{Path(a.data).stem}.json').write_text(json.dumps(res)); print('wrote', len(res), 'groups')

if __name__ == '__main__':
    main()
