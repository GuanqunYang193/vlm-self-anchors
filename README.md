# vlm-self-anchors

Code for **"Calibrating a Frozen Vision-Language Model Against Itself: Label-Free Anchors for Multi-Image Identity
Verification"** (G. Yang, submitted to *IEEE Access*).

Open vision-language models often rank matching image groups above non-matching ones well, yet answer "are these the same
object?" badly, because the threshold that turns their True/False margin into a decision is unknown, varies by model and
domain, and flips under negation. This repository implements a training-free decision rule that calibrates every query
against the model itself:

1. **Affirmative scoring.** Always score the affirmative statement; the negated statement gets the opposite answer.
2. **Two views of the group.** The holistic margin of all images, and a structured reading of all pairwise margins
   (log-odds of "all match" against "exactly one image is the odd one out").
3. **Self-anchors.** For each query, score a *positive anchor* (the first image plus augmented copies of it) and two
   *negative anchors* (the last image replaced by an unrelated image of the same category) with the same prompt and view.
   The threshold sits at λ = 0.3 of the way from the negative to the positive anchor score, and the distance to it is
   measured in units of the anchor gap: `z = (s − (s̄⁻ + λ(s⁺ − s̄⁻))) / max(s⁺ − s̄⁻, 1)`.
4. **Per-query rules.** **Str.-anchor** decides "same" iff the anchored structured z is positive; it is as accurate as
   **M1** (the average z of the two views) at fewer calls and is the per-query rule we recommend. **M2** adds the z of a
   SigLIP similarity computed with the same anchor images.
5. **Pooled rules.** The structured score obeys a group-size law (a pair-level bias b shifts it by (n − 1)b), so a
   pair-level threshold can be pooled over unlabeled queries of a domain:
   - **Str.-pool**: t = median over the batch of the pair-level anchored threshold (λ = 0.2); "same" iff f_str > (n − 1)t.
   - **Str.-pool-NN**: the same with the two nearest neighbours of the replaced image as distractors (λ = 0.15).
   - **Str.-NP-NN**: for each group size, "same" iff f_str exceeds the 0.8 quantile of the structured scores of the
     nearest-neighbour negative anchors (target false-positive rate α = 0.2; an operating point, not a guarantee).

No target labels and no model updates are used. Each rule has one parameter, fixed once on labeled development data. The
pooled rules are transductive: they use the unlabeled queries of the deployment domain.

## Installation

Python 3.12 and a CUDA GPU. Install the PyTorch/torchvision build that matches your CUDA version
([pytorch.org](https://pytorch.org/get-started/locally/)), then

```bash
pip install -r requirements.txt
```

## Input format

Put your images and a group file under `data/` (ignored by git). A group file is a JSON list; each group holds 2-4 images
and the pair of opposite statements about them. Image paths are relative to the repository root (set `SELFANCHOR_ROOT` to
use another root):

```json
[
  {
    "group_id": "faces_0",
    "category": "pc-cpr",
    "images": ["data/images/a.jpg", "data/images/b.jpg", "data/images/c.jpg"],
    "questions": [
      {"id": "faces_0_p", "answer": "T",
       "question": "Judge the following statement based on the input images: 'The individuals in these images are the same person.'. Provide only one correct answer: 'T'(True) or 'F'(False). Respond with either 'T' or 'F'."},
      {"id": "faces_0_n", "answer": "F",
       "question": "Judge the following statement based on the input images: 'The individuals in these images are not the same person.'. Provide only one correct answer: 'T'(True) or 'F'(False). Respond with either 'T' or 'F'."}
    ]
  }
]
```

- Keep the statement template: `'The <plural noun> in these images are [not] the same <noun>.'`. The code finds the
  affirmative and the negated statement by `are the same` / `are not the same`, and reads the noun to pick distractors
  of the same kind.
- `category` is `pc-cpr` for people and `oc-cpr` for objects.
- Negative anchors use the first images of *other groups in the same file* with the same category and noun, so a file
  should contain a reasonable number of groups (we used hundreds). No labels are read for this.
- `answer` is only used by `evaluate.py`; it can be omitted at inference. `group_id` seeds the anchor sampling.

## Running

```bash
# M1 and baselines (model calls are cached in results/<name>/cache.jsonl, so runs resume)
python scripts/run.py --model Qwen/Qwen3-VL-8B-Instruct --data data/groups.json \
    --methods m1 affirm direct negsym_repeat cc dc --out results/qwen3vl8b

# pooled rules: embeddings for nearest-neighbour distractors (Qwen3-VL-4B vision tower), then the per-query records
python scripts/nn_index.py --data data/groups.json --out work/nn@groups.pt
python scripts/run.py --model Qwen/Qwen3-VL-8B-Instruct --data data/groups.json \
    --methods str_bank --nn-index work/nn@groups.pt --out results/qwen3vl8b

# SigLIP scores with the same anchors (needed for M2)
python scripts/embed.py --data data/groups.json --encoder siglip --out results/embed

# paired accuracy, false-positive rate, BC, M2, the pooled rules and exact McNemar tests against Str.-anchor
python scripts/evaluate.py --results results/qwen3vl8b --data data/groups.json --embed results/embed [--bank 16]
```

A group file is treated as one domain: its queries form the unlabeled batch of the pooled rules (or a random bank of
`--bank` queries). Run one file per domain.

Supported model families: Qwen3-VL (`Qwen/Qwen3-VL-*-Instruct`), InternVL3.5 (`OpenGVLab/InternVL3_5-*-HF`) and
LLaVA-OneVision (`llava-hf/llava-onevision-qwen2-7b-ov-hf`). For a model that does not fit on one GPU add `--device-map auto`;
split a run over processes with `--shard k --nshards N`.

`run.py` writes one JSON line per statement to `results/<name>/<method>@<file>.jsonl` (`margin`, `pred` = T/F; for `m1` also
the two view scores `z_hol`, `z_str`; for `str_bank` the structured score and the pair-level anchor margins). Augmented positive-anchor images and other synthetic inputs go to `work/`
(`SELFANCHOR_WORK`). For M2 the SigLIP anchor gap is floored at 1e-3 instead of 1 logit.

| Path | Contents |
|---|---|
| `selfanchor/methods.py` | affirmative scoring, structured pairwise view, random and nearest-neighbour anchors, Str.-anchor, M1, per-query records for the pooled rules, baselines (Direct, NegSym-R, CC, DC) |
| `selfanchor/vlm.py` | frozen VLM wrapper (T/F logit margin) and call cache |
| `scripts/run.py`, `scripts/nn_index.py`, `scripts/embed.py`, `scripts/evaluate.py` | run methods, nearest-neighbour index, SigLIP anchor scores, evaluation and pooled rules |

## Citation

```bibtex
@article{yang2026selfanchors,
  title   = {Calibrating a Frozen Vision-Language Model Against Itself: Label-Free Anchors for Multi-Image Identity Verification},
  author  = {Yang, Guanqun},
  journal = {IEEE Access (submitted)},
  year    = {2026}
}
```

## License

MIT.
