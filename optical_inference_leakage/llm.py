"""Apply independent matrix noise to Qwen3 products and score consecutive WikiText-2 windows."""
import os, math, zlib
import numpy as np, torch, torch.nn.functional as F
from . import paths
WIN = 2048

def load_model(name, dtype=torch.bfloat16, device='cuda'):
    """Load the pinned public model in the requested arithmetic and device."""
    from transformers import AutoModelForCausalLM
    repo, rev = paths.LLM_REVS[name]
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=dtype)
    m.to(device).eval()
    return m

def load_tokens(split='test', name='Qwen3-0.6B'):
    """Join raw WikiText-2 lines with blank lines and discard the incomplete token window."""
    import pandas as pd
    from transformers import AutoTokenizer
    from .fetch import fetch
    p = fetch(split)
    text = '\n\n'.join(pd.read_parquet(p)['text'].tolist())
    repo, rev = paths.LLM_REVS[name]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    ids = tok(text)['input_ids']
    n = len(ids) // WIN
    return torch.tensor(ids[:n * WIN]).view(n, WIN)

def weight_matrices(model):
    """List matrices used in products, counting a shared embedding once."""
    out = {}
    emb = model.get_input_embeddings().weight
    out['embed'] = emb
    for i, layer in enumerate(model.model.layers):
        a, m = (layer.self_attn, layer.mlp)
        for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            out[f'L{i}.{k}'] = getattr(a, k).weight
        for k in ['gate_proj', 'up_proj', 'down_proj']:
            out[f'L{i}.{k}'] = getattr(m, k).weight
    head = model.lm_head.weight
    if head.data_ptr() != emb.data_ptr():
        out['head'] = head
    return out

def tied(model):
    return model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()

class NoiseState:
    """Carry matrix SNRs and reproducible random generators for ordinary or antithetic draws."""

    def __init__(self, gamma=None, seed=0, device='cuda'):
        self.gamma = gamma or {}
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.device = device
        self.keyed = False
        self.base = 0
        self.sign = 1.0

    def gen_for(self, key):
        if not self.keyed:
            return self.gen
        g = torch.Generator(device=self.device)
        g.manual_seed((self.base * 1000003 + zlib.crc32(key.encode())) % 2 ** 63)
        return g

    def sd(self, key, w):
        g = self.gamma.get(key)
        if g is None or math.isinf(g):
            return None
        return float(w.detach().float().pow(2).mean().sqrt()) / math.sqrt(g)

def _noisy(w, sd, gen, sign=1.0):
    z = torch.randn(w.shape, generator=gen, device=w.device, dtype=torch.float32).mul_(sd * sign)
    return (w.float() + z).to(w.dtype)

def install(model, state):
    """Replace projection calls with products using a fresh noisy matrix per window."""
    for i, layer in enumerate(model.model.layers):
        for mod, ks in [(layer.self_attn, ['q_proj', 'k_proj', 'v_proj', 'o_proj']), (layer.mlp, ['gate_proj', 'up_proj', 'down_proj'])]:
            for k in ks:
                lin = getattr(mod, k)
                lin._optical_inference_leakage_key = f'L{i}.{k}'
                lin._optical_inference_leakage_state = state
                lin.forward = _make_fwd(lin)
    model.lm_head._optical_inference_leakage_key = 'head'
    model.lm_head._optical_inference_leakage_state = state
    model.lm_head.forward = _make_fwd(model.lm_head)

def _make_fwd(lin):

    def fwd(x):
        st = lin._optical_inference_leakage_state
        sd = st.sd(lin._optical_inference_leakage_key, lin.weight)
        w = lin.weight if sd is None else _noisy(lin.weight, sd, st.gen_for(lin._optical_inference_leakage_key), st.sign)
        return F.linear(x, w, lin.bias)
    return fwd

def noisy_embed(model, ids, state):
    """Share one noisy embedding row across repeated occurrences of a token in a window."""
    E = model.get_input_embeddings().weight
    sd = state.sd('embed', E)
    if sd is None:
        return E[ids]
    uniq, inv = torch.unique(ids, return_inverse=True)
    rows = _noisy(E[uniq], sd, state.gen_for('embed'), state.sign)
    return rows[inv]

@torch.no_grad()
def window_nll(model, ids, state):
    """Return summed next-token loss in nats and the number of predicted tokens."""
    emb = noisy_embed(model, ids.unsqueeze(0), state)
    out = model(inputs_embeds=emb, use_cache=False)
    logits = out.logits[0, :-1].float()
    return (F.cross_entropy(logits, ids[1:], reduction='sum').item(), ids.numel() - 1)

@torch.no_grad()
def eval_ppl(model, tokens, state, windows=None, return_per_window=False):
    """Exponentiate the token-weighted mean negative log likelihood."""
    idx = range(tokens.shape[0]) if windows is None else windows
    tot, cnt, per = (0.0, 0, [])
    for w in idx:
        s, c = window_nll(model, tokens[w].to(model.device), state)
        tot += s
        cnt += c
        per.append(s / c)
    ppl = math.exp(tot / cnt)
    return (ppl, per) if return_per_window else ppl
