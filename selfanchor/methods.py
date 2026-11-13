"""Decision rules. Every method has the signature fn(vlm, cache, group, question) -> dict(margin=..., ...);
the answer to `question` is 'T' iff margin > 0.

A group is a dict with 'group_id', 'category', 'images' (paths relative to ROOT) and two 'questions': the statement
"... are the same ..." and its negation "... are not the same ...". Both questions of a group derive from one decision.

Per-query rules               : str_anchor (recommended), m1 (two-view rule specified first in the paper)
Pooled rules                  : str_bank records, per query, the scores that the pooled rules need; Str.-pool, Str.-pool-NN
                                and Str.-NP-NN are then computed over the unlabeled batch by scripts/evaluate.py
Ingredients                   : affirm, pair_struct, anchored
Baselines                     : direct, negsym_repeat, cc, dc   (BC and M2 are computed offline, see scripts/evaluate.py)
MuirBench adapter             : muir_mcq (native baseline), muir_anchor
"""
import json, os, random
from pathlib import Path
import numpy as np
from scipy.special import log_expit, logsumexp
from PIL import Image, ImageOps, ImageEnhance
from .vlm import ROOT

LAMBDA = 0.3          # per-query rules: threshold position between the negative (0) and positive (1) anchor scores
LAMBDA_POOL = 0.2     # Str.-pool (random distractors)
LAMBDA_POOL_NN = 0.15 # Str.-pool-NN (nearest-neighbour distractors)
ALPHA_NP = 0.2        # Str.-NP-NN: target false-positive rate
GAP_FLOOR = 1.0       # minimum anchor gap, in logits
N_NEG = 2             # negative anchors per query
WORK = os.environ.get('SELFANCHOR_WORK', 'work')   # where augmented / synthetic anchor images are written (under ROOT)
DATA_FILE = None      # group file of the current run; negative-anchor distractors are drawn from it (set by scripts/run.py)
NN_INDEX = None       # image embeddings for nearest-neighbour distractors (scripts/nn_index.py; set by scripts/run.py)
_POOL = None
_NN = None


# ----------------------------------------------------------------------------------------------- statements
def _negate(question):
    if ' are not the same ' in question: return question.replace(' are not the same ', ' are the same ')
    if ' are the same ' in question: return question.replace(' are the same ', ' are not the same ')
    return None

def _same_q(g):
    return [q for q in g['questions'] if ' are the same ' in q['question']][0]

def _signed(m, q, sq):
    return m if q is sq else -m

def _noun(g):
    """object noun of the statement, used to keep distractors in the same category (e.g. 'person', 'chair')."""
    if g['category'] == 'pc-cpr': return 'person'
    s = g['questions'][0]['question'].split("'")[1].split(' in these images')[0].removeprefix('The ')
    return s[:-1] if s.endswith('s') else s


# ----------------------------------------------------------------------------------------------- scoring
def direct(vlm, cache, g, q):
    """T/F margin of the statement as given (native prompt)."""
    return dict(margin=cache.get(['direct', g['images'], q['question']], lambda: vlm.score(g['images'], q['question']))['margin'])

def affirm(vlm, cache, g, q, base=direct):
    """Affirmative rewriting: always score the affirmative statement; the negated statement gets the opposite answer."""
    sq = q if ' are not the same ' not in q['question'] else dict(q, question=_negate(q['question']))
    m = base(vlm, cache, g, sq)['margin']
    return dict(margin=m if sq is q else -m)

def _pairs(vlm, cache, g, sq):
    I = g['images']
    if len(I) == 2: return [direct(vlm, cache, g, sq)['margin']]
    return [direct(vlm, cache, dict(g, images=[a, b]), sq)['margin'] for i, a in enumerate(I) for b in I[i+1:]]

def structured(pair_margins, n):
    """Log-odds of 'all images match' against 'exactly one image is the odd one out' (uniform over which one, prior 1/2),
    treating the pair margins as independent log-odds. For n=2 this is the pair margin itself."""
    pm = np.asarray(pair_margins, float)
    if n == 2: return float(pm[0])
    P = [(i, j) for i in range(n) for j in range(i+1, n)]
    log_same = log_expit(pm).sum() + np.log(.5)
    log_odd = [sum(log_expit(-m) if o in p else log_expit(m) for p, m in zip(P, pm)) + np.log(.5/n) for o in range(n)]
    return float(log_same - logsumexp(log_odd))

def pair_struct(vlm, cache, g, q):
    """Structured pairwise view with the zero threshold (no anchors)."""
    sq = _same_q(g); pm = _pairs(vlm, cache, g, sq)
    return dict(margin=_signed(structured(pm, len(g['images'])), q, sq), pair_margins=pm)


# ----------------------------------------------------------------------------------------------- anchors
def _pool():
    global _POOL
    if _POOL is None:
        _POOL = {}
        for x in json.loads((ROOT/DATA_FILE).read_text()):
            if x['category'] in ('oc-cpr', 'pc-cpr'): _POOL.setdefault((x['category'], _noun(x)), []).append(x)
    return _POOL

def augment(path, k):
    """Mild, seeded augmentation used for the positive anchor: random crop (75-90%), mirror for even k, brightness."""
    out = ROOT/WORK/'aug'/f'{Path(path).stem}_{k}.jpg'
    if not out.exists():
        out.parent.mkdir(exist_ok=True, parents=True)
        im = Image.open(ROOT/path).convert('RGB'); rng = random.Random(f'{path}{k}'); w, h = im.size
        s = rng.uniform(0.75, 0.9); cw, ch = int(w*s), int(h*s); x, y = rng.randint(0, w-cw), rng.randint(0, h-ch)
        im = im.crop((x, y, x+cw, y+ch))
        if k % 2 == 0: im = ImageOps.mirror(im)
        ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15)).save(out, quality=92)
    return str(out.relative_to(ROOT))

def distractor(g, k=0):
    """Seeded unrelated image of the same category: first image of another group of the evaluation file. No labels read."""
    cands = [x for x in _pool()[(g['category'], _noun(g))] if x['group_id'] != g['group_id']]
    d = random.Random(g['group_id'] + str(k)).choice(cands)['images'][0]; t = 0
    while d in g['images']:
        t += 1; d = random.Random(g['group_id'] + str(k) + '_' + str(t)).choice(cands)['images'][0]
    return d

def nn_distractor(g, k=0):
    """k-th nearest neighbour of the group's last image among the images of the other groups of the same category, by cosine
    similarity of a fixed image embedding (mean-pooled Qwen3-VL-4B vision tower, see scripts/nn_index.py). No labels read."""
    global _NN
    if _NN is None:
        import torch
        _NN = torch.load(ROOT/NN_INDEX)
    cands = [p for x in _pool()[(g['category'], _noun(g))] if x['group_id'] != g['group_id'] for p in x['images']]
    cands = [p for p in cands if p not in g['images']]
    E = np.stack([_NN[p].numpy() for p in cands]).astype(float); q = _NN[g["images"][-1]].numpy().astype(float)
    return cands[int(np.argsort(-(E @ q), kind='stable')[k])]

def anchored(vlm, cache, g, base):
    """Scores of the query, the positive anchor (first image + augmented copies of it) and the negative anchors (the last
    image replaced by a distractor), all with the affirmative statement and the same aggregation `base`."""
    sq = _same_q(g); I = g['images']; n = len(I)
    s = base(vlm, cache, g, sq)['margin']
    s_pos = base(vlm, cache, dict(g, images=[I[0]] + [augment(I[0], k) for k in range(1, n)]), sq)['margin']
    s_neg = [base(vlm, cache, dict(g, images=I[:-1] + [distractor(g, k)]), sq)['margin'] for k in range(N_NEG)]
    return dict(s=s, s_pos=s_pos, s_neg=s_neg)

def z_score(a, lam=LAMBDA, floor=GAP_FLOOR):
    """Distance of the query score from the anchored threshold, in units of the anchor gap."""
    sn = sum(a['s_neg']) / len(a['s_neg']); gap = max(a['s_pos'] - sn, floor)
    return (a['s'] - (sn + lam*(a['s_pos'] - sn))) / gap

def str_anchor(vlm, cache, g, q):
    """Str.-anchor: the anchored structured pairwise view alone (as accurate as M1 in the paper, with fewer calls)."""
    sq = _same_q(g); b = anchored(vlm, cache, g, pair_struct); zb = z_score(b)
    return dict(margin=_signed(zb, q, sq), z_str=zb, str=b)

def _neg_anchor(vlm, cache, g, sq, d):
    """Structured score of the negative anchor with distractor d, and the mean margin of its pairs (I_i, d), i < n."""
    I = g['images']; A = I[:-1] + [d]
    pm = _pairs(vlm, cache, dict(g, images=A), sq); n = len(I)
    P = [(i, j) for i in range(n) for j in range(i+1, n)]
    return structured(pm, n), float(np.mean([m for (i, j), m in zip(P, pm) if j == n-1]))

def str_bank(vlm, cache, g, q):
    """Per-query record for the pooled rules. The decision written here is Str.-anchor; scripts/evaluate.py pools the
    pair-level anchors over all queries of the file (one domain, unlabeled) to obtain Str.-pool, Str.-pool-NN and Str.-NP-NN."""
    sq = _same_q(g); I = g['images']; n = len(I)
    r = str_anchor(vlm, cache, g, q)
    pos_pair = direct(vlm, cache, dict(g, images=[I[0], augment(I[0], 1)]), sq)['margin']
    neg = [_neg_anchor(vlm, cache, g, sq, distractor(g, k)) for k in range(N_NEG)]
    r.update(n=n, s_str=r['str']['s'], pos_pair=pos_pair, neg_pair=float(np.mean([p for _, p in neg])))
    if NN_INDEX:
        nn = [_neg_anchor(vlm, cache, g, sq, nn_distractor(g, k)) for k in range(N_NEG)]
        r.update(nn_neg_pair=float(np.mean([p for _, p in nn])), nn_neg_str=[s for s, _ in nn])
    return r

def m1(vlm, cache, g, q):
    """M1: average of the anchored holistic view and the anchored structured pairwise view."""
    sq = _same_q(g)
    a = anchored(vlm, cache, g, direct); b = anchored(vlm, cache, g, pair_struct)
    za, zb = z_score(a), z_score(b)
    return dict(margin=_signed(0.5*(za + zb), q, sq), z_hol=za, z_str=zb, hol=a, str=b)


# ----------------------------------------------------------------------------------------------- baselines
def _repeat(vlm, cache, g, q):
    P = g['images'] + g['images']; n = len(g['images'])
    L = [f'Image {i+1}:' for i in range(n)] + [f'Image {i+1} (shown again for comparison):' for i in range(n)]
    return dict(margin=cache.get(['repeat', P, q['question']], lambda: vlm.score(P, q['question'], labels=L))['margin'])

def negsym_repeat(vlm, cache, g, q):
    """NegSym-R: images shown twice; (margin of the statement - margin of its negation) / 2."""
    m1_ = _repeat(vlm, cache, g, q)['margin']; m2_ = _repeat(vlm, cache, g, dict(q, question=_negate(q['question'])))['margin']
    return dict(margin=(m1_ - m2_) / 2)

def _blank():
    out = ROOT/WORK/'aug'/'_blank_gray.jpg'
    if not out.exists():
        out.parent.mkdir(exist_ok=True, parents=True); Image.new('RGB', (448, 448), (128, 128, 128)).save(out, quality=95)
    return str(out.relative_to(ROOT))

def _cc_direct(vlm, cache, g, q):
    m0 = direct(vlm, cache, dict(g, images=[_blank()] * len(g['images'])), q)['margin']
    return dict(margin=direct(vlm, cache, g, q)['margin'] - m0, m0=m0)

def _dc_image(g, k, j):
    out = ROOT/WORK/'aug'/f"dc_{g['group_id']}_{k}_{j}.jpg"
    if not out.exists():
        rng = random.Random(f"dc{g['group_id']}{k}{j}")
        cands = [x for x in _pool()[(g['category'], _noun(g))] if x['group_id'] != g['group_id']]
        src = Image.open(ROOT/rng.choice(cands)['images'][0]).convert('RGB').resize((448, 448))
        tiles = [src.crop((x*112, y*112, x*112+112, y*112+112)) for y in range(4) for x in range(4)]; rng.shuffle(tiles)
        im = Image.new('RGB', (448, 448))
        for t, tl in enumerate(tiles): im.paste(tl, ((t % 4)*112, (t // 4)*112))
        im.save(out, quality=92)
    return str(out.relative_to(ROOT))

def _dc_direct(vlm, cache, g, q):
    n = len(g['images'])
    m0 = [direct(vlm, cache, dict(g, images=[_dc_image(g, k, j) for j in range(n)]), q)['margin'] for k in range(3)]
    return dict(margin=direct(vlm, cache, g, q)['margin'] - sum(m0)/3, m0=m0)

def cc(vlm, cache, g, q):
    """Contextual calibration (visual adaptation): subtract the margin on n blank gray images; affirmative statement."""
    return affirm(vlm, cache, g, q, base=_cc_direct)

def dc(vlm, cache, g, q):
    """Domain-context calibration (visual adaptation): subtract the mean margin on 3 groups of patch-shuffled in-domain
    images; affirmative statement."""
    return affirm(vlm, cache, g, q, base=_dc_direct)


# ----------------------------------------------------------------------------------------------- MuirBench Visual Retrieval
MUIR_STMT = ("Judge the following statement based on the input images: 'The buildings in these images are the same building.'. "
             "Provide only one correct answer: 'T'(True) or 'F'(False). Respond with either 'T' or 'F'.")

def muir_mcq(vlm, cache, g, q):
    """Native multiple-choice baseline: query and all candidates in one prompt, answer = highest option-letter logit."""
    L = [chr(65 + i) for i in range(len(g['options']))]; imgs = [g['images'][0]]; labels = ['Query image:']; lines = []; k = 1
    for Lt, o in zip(L, g['options']):
        if o == '<image>':
            imgs.append(g['images'][k]); labels.append(f'Option {Lt}:'); lines.append(f'{Lt}. (image shown as Option {Lt})'); k += 1
        else: lines.append(f'{Lt}. {o}')
    pr = (g['questions'][0]['question'].replace('<image>', 'the query image') + '\nOptions:\n' + '\n'.join(lines)
          + "\nAnswer with the option's letter from the given choices directly.")
    sc = cache.get(['muir_mcq', imgs, pr], lambda: vlm.score_letters(imgs, pr, L, labels=labels))
    return dict(margin=0.0, choice=max(sc, key=sc.get), scores=sc)

def muir_anchor(vlm, cache, g, q):
    """Query-vs-candidate pair margins with the anchored threshold; answer = best candidate with z > 0, else 'None'.
    Positive anchor: augmented query; negative anchors: candidates of two other questions (not this one or its counterpart)."""
    L = [chr(65 + i) for i in range(len(g['options']))]; Q = g['images'][0]
    pair = lambda b: cache.get(['direct', [Q, b], MUIR_STMT], lambda: vlm.score([Q, b], MUIR_STMT))['margin']
    cand = {}; k = 1
    for Lt, o in zip(L, g['options']):
        if o == '<image>': cand[Lt] = pair(g['images'][k]); k += 1
    allg = json.loads((ROOT/DATA_FILE).read_text())
    pool = [p for x in allg if x['group_id'] not in (g['group_id'], g.get('counterpart')) for p in x['images'][1:]]
    pool = [p for p in pool if p not in g['images']]
    s_pos = pair(augment(Q, 1)); s_neg = [pair(random.Random(g['group_id'] + str(j)).choice(pool)) for j in range(N_NEG)]
    z = {Lt: z_score(dict(s=m, s_pos=s_pos, s_neg=s_neg)) for Lt, m in cand.items()}
    none = [Lt for Lt, o in zip(L, g['options']) if o != '<image>']
    best = max(z, key=z.get)
    return dict(margin=0.0, choice=best if (z[best] > 0 or not none) else none[0], cand=cand, s_pos=s_pos, s_neg=s_neg, none=none)


METHODS = dict(direct=direct, affirm=affirm, negsym_repeat=negsym_repeat, cc=cc, dc=dc, pair_struct=pair_struct, m1=m1,
               str_anchor=str_anchor, str_bank=str_bank,
               muir_mcq=muir_mcq, muir_anchor=muir_anchor)
