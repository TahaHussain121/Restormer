"""Latent-block residual scales for E0 and E1 at iteration 5000.

WHY. E1's `injection_ratio = ||P(D)|| / ||F_latent||` plateaus around 4. Whether
that is distorted depends on a reference nobody has measured: how large are the
residual contributions the latent Transformer blocks already add to their own
stream, and are E1's per-block scales shaped like E0's?

WHAT. Each latent block is

    x     = x + self.attn(self.norm1(x))        -> x_mid
    x_out = x_mid + self.ffn(self.norm2(x_mid))

so, per block:

    attn_ratio = ||attn(norm1(x))||    / ||x||        x = block input
    ffn_ratio  = ||ffn(norm2(x_mid))|| / ||x_mid||    the stream it is added to
    abs_norm   = ||x||                                absolute, not a ratio

Frobenius norm over the whole [B,C,H,W] tensor -- the same convention
`injection_ratio` uses, so every number here is directly comparable to the ~4
plateau. For E1, block 0's input IS the guided latent (inp_enc_level4 + P(D)).

Both arms see the IDENTICAL batches (built once from a fixed seed) so the
comparison is like-for-like.

INFERENCE ONLY. No training, no optimizer, no config touched, no experiment
identity started or resumed, no checkpoint written. Reads two checkpoints and
the train split; writes one JSON to a throwaway path.
"""

import argparse
import copy
import datetime
import json
import os
import random
import re
import sys

import numpy as np
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
for p in (_REPO, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs import define_network              # noqa: E402
from basicsr.data import create_dataset, create_dataloader   # noqa: E402


def train_batches(cfg, n_batches, seed):
    """Real training batches: the project's dataset + train.py's crop block."""
    opt = copy.deepcopy(cfg['datasets']['train'])
    opt['phase'] = 'train'
    opt['scale'] = cfg['scale']
    opt['dist'] = False
    ds = create_dataset(opt)
    loader = create_dataloader(ds, opt, num_gpu=0, dist=False, sampler=None,
                               seed=seed)
    gt_size, crop = opt['gt_size'], opt['gt_sizes'][0]
    mini_bs, bs = opt['mini_batch_sizes'][0], opt['batch_size_per_gpu']
    out = []
    for data in loader:
        lq = data['lq']
        if mini_bs < bs:
            lq = lq[random.sample(range(bs), k=mini_bs)]
        if crop < gt_size:
            x0 = int((gt_size - crop) * random.random())
            y0 = int((gt_size - crop) * random.random())
            lq = lq[:, :, x0:x0 + crop, y0:y0 + crop]
        out.append(lq)
        if len(out) >= n_batches:
            break
    return out


def probe(label, config_path, weights, batches, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    net = define_network(copy.deepcopy(cfg['network_g']))
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt.get('params', ckpt), strict=True)
    net = net.to(device).eval()
    if hasattr(net, 'set_dino_mode'):
        net.set_dino_mode('train128')          # explicit: these are 128 crops
    print(f'\n== {label} ==\n  {weights}\n  arch {cfg["network_g"]["type"]}, '
          f'{len(net.latent)} latent blocks')

    cap = {}
    handles = []
    for i, blk in enumerate(net.latent):
        handles.append(blk.register_forward_pre_hook(
            lambda _m, inp, i=i: cap.__setitem__((i, 'x'), inp[0].detach())))
        handles.append(blk.attn.register_forward_hook(
            lambda _m, _i, o, i=i: cap.__setitem__((i, 'a'), o.detach())))
        handles.append(blk.ffn.register_forward_hook(
            lambda _m, _i, o, i=i: cap.__setitem__((i, 'f'), o.detach())))

    per_batch = []
    with torch.no_grad():
        for b, lq in enumerate(batches, 1):
            cap.clear()
            net(lq.to(device))
            rec = []
            for i in range(len(net.latent)):
                x = cap[(i, 'x')].float()
                a = cap[(i, 'a')].float()
                f = cap[(i, 'f')].float()
                x_mid = x + a
                rec.append({'block': i,
                            'abs_norm': float(x.norm()),
                            'attn_ratio': float(a.norm() / x.norm()),
                            'ffn_ratio': float(f.norm() / x_mid.norm())})
            per_batch.append(rec)
            print(f'    batch {b}/{len(batches)}', flush=True)
    for h in handles:
        h.remove()

    per_block = []
    for i in range(len(net.latent)):
        rows = [pb[i] for pb in per_batch]
        per_block.append({
            'block': i,
            'attn_ratio_mean': float(np.mean([r['attn_ratio'] for r in rows])),
            'attn_ratio_std': float(np.std([r['attn_ratio'] for r in rows], ddof=1)),
            'ffn_ratio_mean': float(np.mean([r['ffn_ratio'] for r in rows])),
            'ffn_ratio_std': float(np.std([r['ffn_ratio'] for r in rows], ddof=1)),
            'abs_norm_mean': float(np.mean([r['abs_norm'] for r in rows])),
            'abs_norm_std': float(np.std([r['abs_norm'] for r in rows], ddof=1)),
        })
    flat = {k: [r[k] for pb in per_batch for r in pb]
            for k in ('attn_ratio', 'ffn_ratio', 'abs_norm')}
    across = {f'{k}_mean': float(np.mean(v)) for k, v in flat.items()}
    across.update({f'{k}_std': float(np.std(v, ddof=1)) for k, v in flat.items()})
    return {'label': label, 'config': os.path.abspath(config_path),
            'weights': os.path.abspath(weights),
            'arch': cfg['network_g']['type'],
            'per_block': per_block, 'across_blocks': across,
            'per_batch': per_batch}


def val_psnr_at(exp_name, want_iter):
    """Validation PSNR/SSIM at a given iteration, from the existing run logs."""
    d = os.path.join(_REPO, 'experiments', exp_name)
    if not os.path.isdir(d):
        return None
    iter_re = re.compile(r'iter:\s*([0-9,]+)')
    out, last = {}, None
    for name in sorted(f for f in os.listdir(d)
                       if f.startswith('train_') and f.endswith('.log')):
        with open(os.path.join(d, name), errors='ignore') as f:
            for line in f:
                m = iter_re.search(line)
                if m:
                    last = int(m.group(1).replace(',', ''))
                m = re.search(r'#\s*psnr:\s*([0-9.]+).*#\s*ssim:\s*([0-9.]+)', line)
                if m and last is not None:
                    out[last] = (float(m.group(1)), float(m.group(2)))
    return out.get(want_iter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e0-config', default=os.path.join(
        _PHASE3, 'configs', 'E0_fixed128_baseline.yml'))
    ap.add_argument('--e0-weights', default=os.path.join(
        _REPO, 'experiments', 'DIAG_E0_lossref', 'models', 'net_g_5000.pth'))
    ap.add_argument('--e1-config', default=os.path.join(
        _PHASE3, 'configs', 'E1_addition_noisy_fixed128_spatial_B6_latent.yml'))
    ap.add_argument('--e1-weights', default=os.path.join(
        _REPO, 'experiments', 'DIAG_E1_injection_ratio', 'models', 'net_g_5000.pth'))
    ap.add_argument('--batches', type=int, default=5)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation', 'latent_residual_scales.json'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    with open(args.e0_config) as f:
        cfg0 = yaml.safe_load(f)
    batches = train_batches(cfg0, args.batches, args.seed)
    print(f'  {len(batches)} identical training batches for both arms, '
          f'shape {list(batches[0].shape)}')

    arms = [probe('E0', args.e0_config, args.e0_weights, batches, args.device),
            probe('E1', args.e1_config, args.e1_weights, batches, args.device)]

    print('\n' + '=' * 96)
    print(f'{"":>7}{"attn_ratio (mean+/-std)":>26}{"ffn_ratio (mean+/-std)":>26}'
          f'{"abs_norm ||x|| (mean+/-std)":>30}')
    for a in arms:
        print(f'\n{a["label"]}')
        def row(tag, d):
            attn = '{:.4f} +/- {:.4f}'.format(d['attn_ratio_mean'],
                                              d['attn_ratio_std'])
            ffn = '{:.4f} +/- {:.4f}'.format(d['ffn_ratio_mean'],
                                             d['ffn_ratio_std'])
            absn = '{:.1f} +/- {:.1f}'.format(d['abs_norm_mean'],
                                              d['abs_norm_std'])
            print('  {:<6}{:>20}{:>26}{:>30}'.format(tag, attn, ffn, absn))

        for s in a['per_block']:
            row(f'blk {s["block"]}', s)
        row('ALL', a['across_blocks'])

    val = {'DIAG_E0_lossref': val_psnr_at('DIAG_E0_lossref', 2500),
           'DIAG_E1_injection_ratio': val_psnr_at('DIAG_E1_injection_ratio', 2500)}
    print('\nvalidation @ iteration 2500 (same 339-image val set, same protocol)')
    for k, v in val.items():
        print(f'  {k:<28} ' + (f'psnr {v[0]:.4f} dB   ssim {v[1]:.4f}'
                               if v else 'not found'))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'purpose': 'reference scale for E1 injection_ratio; per-block '
                       'residual-branch magnitudes for E0 vs E1 at iter 5000',
            'definition': {
                'attn_ratio': '||attn(norm1(x))|| / ||x||, x = block input',
                'ffn_ratio': '||ffn(norm2(x_mid))|| / ||x_mid||, x_mid = x + attn_out',
                'abs_norm': '||x|| of the block input, absolute',
                'norm': 'Frobenius over the whole [B,C,H,W] tensor, identical to '
                        'the injection_ratio convention',
                'note': 'for E1, block 0 input IS the guided latent '
                        '(inp_enc_level4 + P(D_centered))'},
            'n_batches': len(batches), 'batch_shape': list(batches[0].shape),
            'seed': args.seed, 'device': args.device,
            'identical_batches_for_both_arms': True,
            'val_psnr_at_2500': val,
            'created': datetime.datetime.now().astimezone().isoformat(),
            'hostname': os.uname().nodename,
            'arms': arms}, f, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
