"""Paired accuracy (a group counts as correct only if both of its statements are answered correctly) for every run file
in a results directory, plus the offline rules:
  bc      batch calibration: affirmative margin minus the mean affirmative margin of the set
  m2      M1 + SigLIP: decide 'same' iff z_M1 + z_SigLIP > 0, SigLIP z with the same anchors (gap floor 1e-3)
  str_pool, str_pool_nn, str_np_nn   pooled rules from a str_bank run (the file is one domain; its queries form the
          unlabeled batch, or a random bank of --bank queries)
and exact McNemar tests of a reference rule (--ref, default str_anchor if present, else m1) against every other rule.

  python scripts/evaluate.py --results results/qwen3vl8b --data groups.json [--embed results/embed]"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selfanchor.methods import z_score, LAMBDA_POOL, LAMBDA_POOL_NN, ALPHA_NP
from selfanchor.vlm import ROOT

def main():
    p = argparse.ArgumentParser(); p.add_argument('--results', required=True); p.add_argument('--data', required=True)
    p.add_argument('--embed', default=None); p.add_argument('--ref', default=None)
    p.add_argument('--bank', type=int, default=0, help='pool over a random bank of this many queries (per group size for '
                   'str_np_nn) instead of the whole file'); p.add_argument('--seed', type=int, default=0); a = p.parse_args()
    stem = Path(a.data).stem
    G = {g['group_id']: g for g in json.loads((ROOT/a.data).read_text())}
    same = {gid: [q for q in g['questions'] if ' are the same ' in q['question']][0]['answer'] == 'T' for gid, g in G.items()}
    correct, runs = {}, {}
    for f in sorted(Path(a.results).glob(f'*@{stem}.jsonl')):
        m = f.name.split('@')[0]; R = [json.loads(l) for l in f.read_text().splitlines()]; runs[m] = R
        ok = collections.defaultdict(lambda: True)
        for r in R: ok[r['group']] &= r['pred'] == r['answer']
        correct[m] = dict(ok)
    if 'affirm' in runs:
        P = {r['group']: r['margin'] for r in runs['affirm'] if r['qid'].endswith('_p')}; mu = np.mean(list(P.values()))
        correct['bc'] = {gid: ((v - mu) > 0) == same[gid] for gid, v in P.items()}
    if 'm1' in runs and a.embed:
        E = json.loads((Path(a.embed)/f'siglip@{stem}.json').read_text())
        Z = {r['group']: 0.5*(r['z_hol'] + r['z_str']) for r in runs['m1'] if r['qid'].endswith('_p')}
        correct['m2'] = {gid: (z + z_score(E[gid], floor=1e-3) > 0) == same[gid] for gid, z in Z.items()}
    if 'str_bank' in runs:
        correct.update(pooled({r['group']: r for r in runs['str_bank'] if r['qid'].endswith('_p')}, same, a.bank, a.seed))
    if 'str_bank' in correct: correct.setdefault('str_anchor', correct.pop('str_bank'))   # its decision is Str.-anchor
    ref = a.ref or ('str_anchor' if 'str_anchor' in correct else 'm1')
    print(f'{stem}: {len(G)} groups; reference rule {ref}')
    for m, c in correct.items():
        neg = [g for g in c if not same[g]]
        line = f'  {m:14s} {100*np.mean(list(c.values())):5.1f}%   FPR {100*np.mean([not c[g] for g in neg]) if neg else float("nan"):5.1f}%'
        if m != ref and ref in correct:
            w = sum(correct[ref][g] and not c[g] for g in c); l = sum(c[g] and not correct[ref][g] for g in c)
            line += f'   {ref} wins/losses {w}/{l}, McNemar p={binomtest(w, w + l).pvalue:.2g}' if w + l else ''
        print(line)


def pooled(R, same, bank=0, seed=0):
    """Str.-pool / Str.-pool-NN: median pair-level anchored threshold t over the batch, 'same' iff f_str > (n-1) t.
    Str.-NP-NN: for each group size n, 'same' iff f_str > the (1-alpha) quantile of the structured scores of the
    nearest-neighbour negative anchors of the batch queries of that size. Labels are used only for scoring."""
    rng = np.random.default_rng(seed); ids = sorted(R); out = {}
    pick = lambda xs: xs if not bank or len(xs) <= bank else [xs[i] for i in sorted(rng.choice(len(xs), bank, replace=False))]
    def pool_rule(lam, key):
        if any(key not in R[g] for g in ids): return None
        B = pick(ids); t = np.median([R[g][key] + lam*(R[g]['pos_pair'] - R[g][key]) for g in B])
        return {g: (R[g]['s_str'] > (R[g]['n'] - 1)*t) == same[g] for g in ids}
    out['str_pool'] = pool_rule(LAMBDA_POOL, 'neg_pair'); out['str_pool_nn'] = pool_rule(LAMBDA_POOL_NN, 'nn_neg_pair')
    if all('nn_neg_str' in R[g] for g in ids):
        th = {}
        for n in sorted({R[g]['n'] for g in ids}):
            B = pick([g for g in ids if R[g]['n'] == n])
            th[n] = np.quantile([s for g in B for s in R[g]['nn_neg_str']], 1 - ALPHA_NP)
        out['str_np_nn'] = {g: (R[g]['s_str'] > th[R[g]['n']]) == same[g] for g in ids}
    return {k: v for k, v in out.items() if v is not None}

if __name__ == '__main__':
    main()
