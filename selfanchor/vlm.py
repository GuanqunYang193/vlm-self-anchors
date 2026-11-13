"""Frozen VLM wrapper: next-token T/F log-probability margin, with an on-disk cache of model calls."""
import hashlib, json, os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration

ROOT = Path(os.environ.get('SELFANCHOR_ROOT', Path(__file__).resolve().parents[1]))   # image paths are relative to ROOT


class VLM:
    """Supports Qwen3-VL, InternVL3.5 (-HF checkpoints) and LLaVA-OneVision. Every image costs roughly 256 visual tokens:
    Qwen3-VL images are resized to 65,536-262,144 pixels, InternVL uses one 448 px tile (no dynamic tiling),
    LLaVA-OneVision uses its base view plus one 384 px tile."""

    def __init__(self, model, max_pixels=262144, min_pixels=65536, device_map='cuda'):
        name = str(model)
        self.llava = 'llava' in name.lower()
        self.intern = 'InternVL' in name or self.llava    # both take float pixel tensors
        kw = dict(dtype=torch.bfloat16, device_map=device_map, attn_implementation='sdpa')
        if self.llava:
            from transformers import LlavaOnevisionForConditionalGeneration
            self.proc = AutoProcessor.from_pretrained(model)
            self.proc.image_processor.image_grid_pinpoints = [[384, 384]]
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model, **kw).eval()
            self.model.config.image_grid_pinpoints = [[384, 384]]
        elif self.intern:
            self.proc = AutoProcessor.from_pretrained(model)
            self.model = AutoModelForImageTextToText.from_pretrained(model, **kw).eval()
        else:
            self.proc = AutoProcessor.from_pretrained(model, min_pixels=min_pixels, max_pixels=max_pixels)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model, **kw).eval()
        tok = self.proc.tokenizer
        self.ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ['T', 'F']}
        self._imgs = {}

    def image(self, p):
        if p not in self._imgs:
            if len(self._imgs) > 64: self._imgs.clear()
            self._imgs[p] = Image.open(ROOT/p).convert('RGB')
        return self._imgs[p]

    def inputs(self, paths, prompt, labels=None):
        content = []
        labels = labels or [f'Image {i+1}:' for i in range(len(paths))]
        for lab, _ in zip(labels, paths): content += [dict(type='text', text=lab), dict(type='image')]
        content.append(dict(type='text', text=prompt))
        text = self.proc.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=True)
        images = [self.image(p) for p in paths] or None
        if not self.intern:
            return self.proc(text=[text], images=images, return_tensors='pt').to('cuda')
        kw = {} if self.llava else dict(crop_to_patches=False)
        x = self.proc(text=[text], images=images, return_tensors='pt', **kw)
        return {k: (v.to('cuda', torch.bfloat16) if v.dtype.is_floating_point else v.to('cuda')) for k, v in x.items()}

    @torch.inference_mode()
    def score(self, paths, prompt, labels=None):
        """log P('T') - log P('F') for the first answer token."""
        lp = torch.log_softmax(self.model(**self.inputs(paths, prompt, labels)).logits[0, -1].float(), -1)
        a, b = lp[self.ids['T']].item(), lp[self.ids['F']].item()
        return dict(margin=a - b, la=a, lb=b)

    @torch.inference_mode()
    def score_letters(self, paths, prompt, letters, labels=None):
        """log-probabilities of single-letter options (native multiple-choice baseline)."""
        lp = torch.log_softmax(self.model(**self.inputs(paths, prompt, labels)).logits[0, -1].float(), -1)
        tok = self.proc.tokenizer
        return {L: lp[tok.encode(L, add_special_tokens=False)[0]].item() for L in letters}


class Cache:
    """Append-only JSONL cache keyed by the call (method tag, image paths, prompt). Lets runs resume and share passes."""

    def __init__(self, path, reuse=()):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self.d = {}
        for p in [*map(Path, reuse), self.path]:
            if p.exists():
                for line in p.read_text().splitlines():
                    r = json.loads(line); self.d[r['k']] = r['v']

    def get(self, key, fn):
        k = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()
        if k not in self.d:
            self.d[k] = fn()
            with self.path.open('a') as f: f.write(json.dumps(dict(k=k, v=self.d[k])) + '\n')
        return self.d[k]
