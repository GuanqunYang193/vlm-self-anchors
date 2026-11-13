"""Image embeddings for the nearest-neighbour distractors of Str.-pool-NN and Str.-NP-NN: the mean of the L2-normalized
patch features of the Qwen3-VL-4B vision tower, L2-normalized. The same encoder is used for every scored model.

  python scripts/nn_index.py --data data/groups.json --out work/nn@groups.pt"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfanchor.vlm import VLM, ROOT


@torch.inference_mode()
def embed(vlm, path):
    x = vlm.proc(text=['<|vision_start|><|image_pad|><|vision_end|>'], images=[vlm.image(path)], return_tensors='pt').to('cuda')
    v = vlm.model.model.visual(x.pixel_values.to(torch.bfloat16), grid_thw=x.image_grid_thw)
    v = v[0] if isinstance(v, tuple) else v
    if hasattr(v, 'pooler_output'): v = v.pooler_output
    return F.normalize(F.normalize(v.float(), dim=-1).mean(0), dim=0).cpu()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct'); a = p.parse_args()
    G = [g for g in json.loads((ROOT/a.data).read_text()) if g['category'] in ('oc-cpr', 'pc-cpr')]
    paths = sorted({x for g in G for x in g['images']})
    vlm = VLM(a.model); E = {}
    for i, x in enumerate(paths):
        E[x] = embed(vlm, x)
        if i % 200 == 0: print(f'{i+1}/{len(paths)}', flush=True)
    out = ROOT/a.out; out.parent.mkdir(parents=True, exist_ok=True); torch.save(E, out); print('wrote', len(E), 'embeddings')


if __name__ == '__main__':
    main()
