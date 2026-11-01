"""Validation tuning and test measurement for the restricted uniform-noise protocol."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import llm, calib, floors, cases, paths, cutfloor as cf

def main():
    """Run validation tuning and, when requested, the held-out test calculation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=cases.LLMS)
    parser.add_argument('level', type=float, choices=cases.LEVELS)
    parser.add_argument('output', type=Path)
    parser.add_argument('--final', action='store_true', help='Evaluate test quality, gradients, spectra and floors after validation tuning')
    parser.add_argument('--from-run', type=Path, help='Reuse the noise allocation from a matching run')
    parser.add_argument('--smoke', action='store_true', help='Exercise the pipeline on a small random CPU model and random tokens; no measured result')
    args = parser.parse_args()
    name, level, out = (args.model, args.level, str(args.output))
    if args.output.resolve().is_relative_to(paths.ROOT / 'data'):
        parser.error('Write new measurements outside data/')
    small = args.smoke
    ntune, nhut, iters, rtune, rfinal = (2, 2, 1, 2, 5) if small else (24, 64, 4, 4, 5)
    final = args.final
    dev = 'cpu' if small else 'cuda'
    if not small and (not torch.cuda.is_available()):
        parser.error('Full measurements require a CUDA GPU with BF16 support')
    if not small and (not torch.cuda.is_bf16_supported()):
        parser.error('This GPU does not support BF16')
    dtype = torch.float32 if small else torch.bfloat16
    args.output.parent.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    if small:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=256, num_hidden_layers=1, hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(cfg).to(dev).eval()
    else:
        model = llm.load_model(name, dtype=dtype, device=dev)

    def tokens(split):
        if small:
            return torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(11 if split == 'validation' else 12))
        return llm.load_tokens(split, name)
    for p in model.parameters():
        p.requires_grad_(False)
    mats = llm.weight_matrices(model)
    TIED = llm.tied(model)
    keys = list(mats.keys()) + (['head'] if TIED else [])
    Wk = lambda k: mats['embed'] if k == 'head' and TIED else mats[k]
    s = np.array([float(Wk(k).detach().float().pow(2).mean()) for k in keys])
    Nk = np.array([Wk(k).numel() for k in keys], float)
    Ntot = float(sum((p.numel() for p in mats.values())))
    st = llm.NoiseState({}, device=dev)
    llm.install(model, st)
    tv = tokens('validation')
    wins = list(range(min(ntune, tv.shape[0])))
    p0 = llm.eval_ppl(model, tv, st, wins)
    res = {'model': name, 'level': level, 'Ntot': Ntot, 'tied': TIED, 'keys': keys, 's': s.tolist(), 'N': Nk.tolist(), 'tune_digital_ppl': p0, 'iters': []}
    print(name, 'digital tune ppl', p0, 't', round(time.time() - T0), flush=True)

    def gam(v):
        return {k: float(sk / vk) for k, sk, vk in zip(keys, s, v)}

    def mean_nll(toks, ws, v, R, seed):
        vals = []
        for r in range(R):
            st.gamma = gam(v) if v is not None else {}
            st.gen.manual_seed(seed + r)
            tot, cnt = (0.0, 0)
            for w in ws:
                a, c = llm.window_nll(model, toks[w].to(dev), st)
                tot += a
                cnt += c
            vals.append(tot / cnt)
        st.gamma = {}
        return vals

    def capacity(v, K):
        """Count each matrix once; add SNRs of independently perturbed lookup and head views."""
        tot = 0.0
        for i, k in enumerate(keys):
            if TIED and k == 'embed':
                continue
            snr = K * s[i] / v[i]
            if TIED and k == 'head':
                j = keys.index('embed')
                snr = K * (s[i] / v[i] + s[j] / v[j])
            tot += Nk[i] * 0.5 * math.log2(1 + snr)
        return tot

    def stein(toks, ws, v, nprobe, seed):
        """Estimate derivatives of expected NLL with antithetic Gaussian perturbations and immediate gradient reduction."""
        params = {k: Wk(k) for k in keys if not (TIED and k == 'head')}
        head_orig = None
        if TIED:
            head_orig = model.lm_head.weight
            params['head'] = torch.nn.Parameter(mats['embed'].detach().clone(), requires_grad=False)
            model.lm_head.weight = params['head']
        est = {k: [] for k in params}
        for p_i in range(nprobe):
            ids = toks[ws[p_i % len(ws)]].to(dev)
            uniq = torch.unique(ids)
            d = {}
            for sign in (1.0, -1.0):
                st.gamma = gam(v)
                st.keyed, st.base, st.sign = (True, seed + p_i, sign)
                dots, hks = ({}, [])

                def mk(k):

                    def f(P):
                        gen = st.gen_for(k)
                        if k == 'embed':
                            xi = torch.randn((uniq.numel(), P.shape[1]), generator=gen, device=P.device, dtype=torch.float32)
                            dots[k] = float((P.grad[uniq].float() * xi).sum())
                        else:
                            xi = torch.randn(P.shape, generator=gen, device=P.device, dtype=torch.float32)
                            dots[k] = float((P.grad.float() * xi).sum())
                        P.grad = None
                    return f
                for k, P in params.items():
                    P.requires_grad_(True)
                    P.grad = None
                    hks.append(P.register_post_accumulate_grad_hook(mk(k)))
                emb = llm.noisy_embed(model, ids.unsqueeze(0), st)
                o = model(inputs_embeds=emb, use_cache=False)
                F.cross_entropy(o.logits[0, :-1].float(), ids[1:]).backward()
                for h in hks:
                    h.remove()
                for P in params.values():
                    P.requires_grad_(False)
                    P.grad = None
                del o, emb
                d[sign] = dots
            st.keyed, st.sign, st.gamma = (False, 1.0, {})
            for k in params:
                est[k].append((d[1.0][k] - d[-1.0][k]) / (4 * math.sqrt(v[keys.index(k)])))
        if TIED:
            model.lm_head.weight = head_orig
        g, se = ([], [])
        for k in keys:
            e = np.array(est[k])
            g.append(e.mean())
            se.append(e.std(ddof=1) / math.sqrt(len(e)) if len(e) > 1 else abs(e.mean()))
        return (np.array(g), np.array(se))
    GRADF = stein

    def alloc(g, lam, K=1.0):
        gg = np.maximum(g, 1e-30)
        c = Nk * K * s / (2 * math.log(2) * lam * gg)
        return 2 * c / (K * s + np.sqrt((K * s) ** 2 + 4 * c))
    TYPE = [k if k in ('embed', 'head') else k.split('.')[-1] for k in keys]

    def robust(g, se):
        """Shrink noisy matrix gradients towards their type mean before allocating noise."""
        out = np.array(g, float)
        for t in set(TYPE):
            I = [i for i, tt in enumerate(TYPE) if tt == t]
            gm = float(np.mean(g[I]))
            sm = float(np.sqrt(np.mean(se[I] ** 2)))
            base = max(gm, sm / math.sqrt(len(I)), 1e-30)
            tau2 = max(0.0, float(np.var(g[I])) - sm ** 2) if len(I) > 1 else float('inf')
            for i in I:
                w = 1.0 if tau2 == float('inf') else tau2 / (tau2 + se[i] ** 2 + 1e-300)
                out[i] = max(w * g[i] + (1 - w) * gm, 0.2 * base)
        return out

    def calibrate(vshape, seed):
        """Search the largest variance multiplier satisfying the validation margin."""

        def ok_at(mlt):
            vals = [math.exp(min(x, 700.0)) for x in mean_nll(tv, wins, vshape * mlt, rtune, seed)]
            return calib.meets(vals, p0, 'ppl', calib.TUNE * level)[0] if len(vals) > 1 else vals[0] <= p0 * (1 + calib.TUNE * level)
        lo, hi = (1.0, 1.0)
        if ok_at(1.0):
            while hi < 1000000000000.0 and ok_at(hi * 8):
                hi *= 8
            lo, hi = (hi, hi * 8)
        else:
            while lo > 1e-12 and (not ok_at(lo / 8)):
                lo /= 8
            if lo <= 1e-12:
                return None
            lo, hi = (lo / 8, lo)
        for _ in range(3 if small else 30):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if ok_at(mid) else (lo, mid)
            if hi / lo < 1.01:
                break
        return vshape * lo
    prev = str(args.from_run) if args.from_run else ''
    if prev:
        P_ = json.load(open(prev))
        assert P_['model'] == name and abs(P_['level'] - level) < 1e-12, '--from: other model or level'
        res['iters'] = P_['iters']
        best = (np.array(P_['v']), None, P_['best_iter'])
    else:
        v = calibrate(s / 1000.0, 300)
        if v is None:
            raise RuntimeError('No initial variance allocation met validation quality')
        best = None
    for it in range(0 if prev else iters + 1):
        leak1 = capacity(v, 1) / Ntot
        res['iters'].append({'it': it, 'gamma': list(gam(v).values()), 'leak1': leak1, 'leakK': capacity(v, cases.counts(name)[1][1]) / Ntot, 't': time.time() - T0})
        print('iter', it, 'leak1', leak1, 'leakK', res['iters'][-1]['leakK'], 't', round(time.time() - T0), flush=True)
        if best is None or leak1 < best[1]:
            best = (v.copy(), leak1, it)
        json.dump(res, open(out, 'w'), indent=1)
        if it == iters:
            break
        g, gse = GRADF(tv, wins, v, nhut, 100 + it)
        gr = robust(g, gse)
        res['iters'][-1].update({'g': g.tolist(), 'g_se': gse.tolist(), 'g_robust': gr.tolist()})
        print('gradient', it, 'g', np.round(g, 3).tolist()[:9], 'se', np.round(gse, 3).tolist()[:9], 't', round(time.time() - T0), flush=True)
        a, b = (-200.0, 200.0)
        for _ in range(200):
            mid = 0.5 * (a + b)
            a, b = (mid, b) if np.mean(np.log(alloc(gr, math.exp(mid)))) > np.mean(np.log(v)) else (a, mid)
        vk = alloc(gr, math.exp(0.5 * (a + b)))
        vn = calibrate(np.exp(0.5 * np.log(v) + 0.5 * np.log(vk)), 300 + it + 1)
        if vn is None:
            print('calibration failed at iteration', it + 1, flush=True)
            break
        v = vn
    v = best[0]
    res['best_iter'] = best[2]
    res['measurement_kind'] = 'random_cpu_smoke' if small else 'bf16_public_inputs'
    res['v'] = v.tolist()
    if final:
        tt = tokens('test')
        allw = list(range(tt.shape[0]))
        if small:
            allw = allw[:2]
        nll0 = mean_nll(tt, allw, None, 1, 0)[0]
        pd = math.exp(nll0)
        st.gamma = {k: float('inf') for k in keys}
        tot, cnt = (0.0, 0)
        for w in allw:
            a, c = llm.window_nll(model, tt[w].to(dev), st)
            tot += a
            cnt += c
        st.gamma = {}
        p_zero = math.exp(tot / cnt)
        draws = [math.exp(min(x, 700.0)) for x in mean_nll(tt, allw, v, rfinal if not small else 2, 5000)]
        ok, u = calib.meets(draws, pd, 'ppl', level) if len(draws) > 1 else (draws[0] <= pd * (1 + level), draws[0])
        nll_draws = [math.log(d) for d in draws]
        nll_lo = np.mean(nll_draws) - (1.96 * np.std(nll_draws, ddof=1) / math.sqrt(len(nll_draws)) if len(nll_draws) > 1 else 0.0)
        target = math.log(pd * (1 + level))
        nhutf, gseed = (2 if small else 256, 9000)
        gt, gtse = GRADF(tt, allw, v, nhutf, gseed)
        Kup = cases.counts(name)[1][1]
        Kfl = 10 ** 6 // llm.WIN
        slack = max(target - nll_lo, 0.0)
        tied_i = (keys.index('embed'), keys.index('head')) if TIED else None
        THETAS = [0.0, 0.25, 0.5, 0.75, 1.0] if TIED else [None]

        def wts(theta):
            """Use a convex mixture of tied views as a separable information lower bound."""
            w = np.ones(len(keys))
            if TIED:
                w[tied_i[0]], w[tied_i[1]] = (theta, 1 - theta)
            return w
        zb = cf.bonferroni_z(len(keys))
        g_lo, g_hi = (gt - zb * gtse, gt + zb * gtse)
        WEIGHTS = [wts(th) for th in THETAS]

        def floor_K(K):
            """Minimise K-copy information over the conditional tangent cut."""
            tabs = [cf.capacity_tab(Nk[i], s[i], v[i], K) for i in range(len(keys))]
            return cf.cut_dual_floor(tabs, v, g_lo, g_hi, slack, weights=WEIGHTS) / Ntot
        fl = {'1': floor_K(1), '1e6': floor_K(Kfl)}
        J = sum((Wk(k).shape[0] if k == 'embed' else Wk(k).shape[1] for k in keys))
        th3 = floors.theorem3_floor_info(fl['1'] * Ntot, J, 0.0001 * llm.WIN) / Ntot
        bound = {'1': capacity(v, 1) / Ntot, '1e6': capacity(v, Kup) / Ntot}
        conv = {}
        cw = allw[:2 if small else 32]
        Rc = 2 if small else 4
        groups = {'ray': np.ones(len(keys)), 'attn': np.array([1.0 if any((t in k for t in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))) else 0.0 for k in keys]), 'mlp': np.array([1.0 if any((t in k for t in ('gate_proj', 'up_proj', 'down_proj'))) else 0.0 for k in keys])}
        for nm, d in groups.items():
            up = np.array(mean_nll(tt, cw, v * np.exp(0.3 * d), Rc, 6100))
            dn = np.array(mean_nll(tt, cw, v * np.exp(-0.3 * d), Rc, 6100))
            ce = np.array(mean_nll(tt, cw, v, Rc, 6100))
            dd = up + dn - 2 * ce
            se_ = float(dd.std(ddof=1) / math.sqrt(len(dd))) if len(dd) > 1 else float('nan')
            conv[nm] = {'second_diff': float(dd.mean()), 'se': se_, 'convex_95': bool(len(dd) > 1 and dd.mean() - 1.645 * se_ >= 0)}
        print('convexity', json.dumps(conv), 't', round(time.time() - T0), flush=True)
        INPUT_OF = {'q_proj': 'attn_in', 'k_proj': 'attn_in', 'v_proj': 'attn_in', 'o_proj': 'o_in', 'gate_proj': 'mlp_in', 'up_proj': 'mlp_in', 'down_proj': 'down_in'}
        SRC_MOD = {'attn_in': ('self_attn', 'q_proj'), 'o_in': ('self_attn', 'o_proj'), 'mlp_in': ('mlp', 'gate_proj'), 'down_in': ('mlp', 'down_proj')}
        gtxt = torch.Generator().manual_seed(4242)
        nq = 1 if small else 8
        texts = [torch.randperm(min(151643, model.config.vocab_size), generator=gtxt)[:16 if small else llm.WIN] for _ in range(nq)]

        class _Stop(Exception):
            pass

        def layer_inputs(layer, ids, srcs):
            """Collect noisy activations at selected products without retaining a full forward graph."""
            got, hks = ({}, [])

            def mk(src, last):

                def f(mod, inp, outp):
                    got[src] = inp[0].detach()[0].float()
                    if last:
                        raise _Stop
                return f
            if layer is None:
                hks.append(model.lm_head.register_forward_hook(mk('head_in', True)))
            else:
                lay = model.model.layers[layer]
                for j, src in enumerate(srcs):
                    a, b = SRC_MOD[src]
                    hks.append(getattr(getattr(lay, a), b).register_forward_hook(mk(src, j == len(srcs) - 1)))
            st.gamma = gam(v)
            st.gen.manual_seed(777)
            try:
                with torch.no_grad():
                    model(inputs_embeds=llm.noisy_embed(model, ids.to(dev).unsqueeze(0), st), use_cache=False)
            except _Stop:
                pass
            finally:
                for h in hks:
                    h.remove()
                st.gamma = {}
            return got

        def span_basis(X):
            Xh = X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)
            U_, S_, Vh = torch.linalg.svd(Xh, full_matrices=False)
            k = int((S_ > 1e-06 * S_[0]).sum())
            return (Vh[:k].T.contiguous(), k)

        def compress(mu, nb=256):
            """Replace each ordered spectral group by its smallest eigenvalue and multiplicity."""
            mu = np.sort(np.clip(mu, 0, None))
            mu = mu[mu > 1e-09]
            if mu.size == 0:
                return (np.zeros(0), np.zeros(0))
            idx = np.array_split(np.arange(mu.size), min(nb, mu.size))
            return (np.array([mu[i[0]] for i in idx]), np.array([float(len(i)) for i in idx]))
        ranks1, specK = ({}, {})
        srcs_all = ['attn_in', 'o_in', 'mlp_in', 'down_in']
        for layer in list(range(len(model.model.layers))) + [None]:
            srcs = srcs_all if layer is not None else ['head_in']
            acc_M, kk = ({}, {})
            for ti, ids in enumerate(texts):
                got = layer_inputs(layer, ids, srcs)
                for src in srcs:
                    Q, k = span_basis(got[src])
                    kk.setdefault(src, []).append(k)
                    acc_M[src] = acc_M.get(src, 0) + Q @ Q.T
                del got
            for src in srcs:
                key = src if layer is None else f'L{layer}.{src}'
                ranks1[key] = kk[src]
                mu = torch.linalg.eigvalsh(acc_M[src]).cpu().numpy()
                specK[key] = compress(mu)
            del acc_M
        print('ranks (min over texts) of layer 0:', {s_: min(ranks1[f'L0.{s_}']) for s_ in srcs_all}, 'head', min(ranks1['head_in']), 't', round(time.time() - T0), flush=True)

        def src_of(k):
            return 'head_in' if k == 'head' else f"{k.split('.')[0]}.{INPUT_OF[k.split('.')[1]]}"
        Vrows, dmod = Wk('embed').shape
        Veff = min(Vrows, 151643)
        grid = cf.GRID

        def H_tab(i, which):
            """Evaluate honest-view information over the variance grid."""
            k = keys[i]
            G = s[i] / (v[i] * grid)
            if k == 'embed':
                if which == '1':
                    U = min(llm.WIN, Vrows)
                    return U * dmod / 2 * np.log2(1 + G * Vrows / U)
                r = Kfl * llm.WIN // Veff
                return Veff * dmod / 2 * np.log2(1 + r * G * Vrows / Veff)
            m_rows, n_cols = Wk(k).shape
            if which == '1':
                kr = max(ranks1[src_of(k)])
                return m_rows * kr / 2 * np.log2(1 + G * n_cols / kr)
            lv, cnt = specK[src_of(k)]
            rM = cnt.sum()
            return m_rows / 2 * (cnt[None, :] * np.log2(1 + Kfl / nq * (n_cols / rM) * G[:, None] * lv[None, :])).sum(1)

        def honest_floor(which):
            return cf.cut_dual_floor([H_tab(i, which) for i in range(len(keys))], v, g_lo, g_hi, slack, weights=WEIGHTS) / Ntot
        hon1 = honest_floor('1')
        honK = honest_floor('K')
        floor_table = {}
        for b in cases.BUDGETS:
            for lab in ['1', '1e6']:
                if b == 0.0:
                    f_ = fl[lab]
                elif b == 0.0001:
                    f_ = th3 if lab == '1' else max(th3, honK)
                else:
                    f_ = hon1 if lab == '1' else max(hon1, honK)
                floor_table[f'{b}|{lab}'] = f_
        ratios = {kk: bound[kk.split('|')[1]] / f_ if f_ > 0 else None for kk, f_ in floor_table.items()}
        res['test'] = {'digital_ppl': pd, 'zero_noise_ppl': p_zero, 'zero_noise_rel': abs(p_zero / pd - 1), 'draws': draws, 'unfavourable': float(u), 'meets': bool(ok), 'g_test': gt.tolist(), 'g_test_se': gtse.tolist(), 'slack_nll': slack, 'bound': bound, 'floor_b0': fl, 'floor_1e-4_one': th3, 'floor_honest_one': hon1, 'floor_honest_K': honK, 'floor': floor_table, 'ratio': ratios, 'K_bound': Kup, 'K_floor': Kfl, 'convexity_test': conv, 'ranks_one_text': ranks1, 'nq_texts': nq, 'nhutf': nhutf, 'gseed': gseed, 'windows': len(allw), 'spectra_K': {k: [a.tolist(), b.tolist()] for k, (a, b) in specK.items()}}
        print('TEST', json.dumps({k: res['test'][k] for k in ['digital_ppl', 'zero_noise_rel', 'unfavourable', 'meets', 'bound', 'floor', 'ratio']}), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    print('done', round(time.time() - T0), flush=True)
    if final and not res['test']['meets']:
        raise SystemExit('Test quality did not meet the requested allowance; measurement saved')
if __name__ == '__main__':
    main()
