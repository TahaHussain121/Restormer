"""Architectural smoke test for aca-L6-nosa — the F_sa ablation.

Run it BEFORE submitting. It builds aca-L6 and aca-L6-nosa under the SAME
seed and checks the claims the arm's whole argument rests on:

  A. step-0 output is bit-identical to E0 (project_out is still zero-init)
  B. the trunk is byte-identical to E0's, tensor by tensor
  C. the SA branch is GONE from state_dict and from the optimizer's view
  D. the parameter delta vs aca-L6 is EXACTLY -452,742, the measured size of
     the removed branch, and nothing else moved
  E. the CROSS branch is byte-identical to aca-L6's at init -- this is what
     makes the comparison one-factor rather than merely similar
  F. the attention matrix is CHANNEL-shaped and token-count independent, the
     property crossattn-render lacked
  G. forward() runs at both regimes and returns the documented 3-tuple

Nothing here needs a GPU or the dataset.
"""

import argparse
import json
import os
import sys
import datetime

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_PHASE3))
sys.path.insert(0, _REPO)

from basicsr.models.archs import define_network            # noqa: E402

CFG = os.path.join(_PHASE3, 'configs')
SA_PREFIXES = ('aca.to_q.', 'aca.to_k.', 'aca.to_v.', 'aca.temperature_sa')
CROSS_PREFIXES = ('aca.to_q_cross.', 'aca.to_k_cross.', 'aca.to_v_cross.')

checks = []


def check(name, ok, detail=''):
    checks.append({'name': name, 'passed': bool(ok), 'detail': str(detail)})
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    return bool(ok)


def build(config_name, seed=100):
    with open(os.path.join(CFG, config_name)) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(seed)
    return define_network(dict(cfg['network_g'])), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        _PHASE3, 'results', 'wo2_implementation',
        'smoke_results_aca_render_fixed128_L6_nosa_latent.json'))
    args = ap.parse_args()

    print('building E0, aca-L6 and aca-L6-nosa under seed 100 ...')
    e0, _ = build('E0_fixed128_baseline.yml')
    l6, _ = build('aca_render_fixed128_L6_latent.yml')
    nosa, cfg = build('aca_render_fixed128_L6_nosa_latent.yml')
    for m in (e0, l6, nosa):
        m.eval()

    sd_e0, sd_l6, sd_no = e0.state_dict(), l6.state_dict(), nosa.state_dict()
    n_e0 = sum(p.numel() for p in e0.parameters())
    n_l6 = sum(p.numel() for p in l6.parameters())
    n_no = sum(p.numel() for p in nosa.parameters())

    print('\nC. the self-attention branch is gone')
    sa_keys_l6 = [k for k in sd_l6 if k.startswith(SA_PREFIXES)]
    sa_keys_no = [k for k in sd_no if k.startswith(SA_PREFIXES)]
    check('aca-L6 HAS the SA branch (control)', len(sa_keys_l6) > 0,
          f'{len(sa_keys_l6)} keys')
    check('aca-L6-nosa has ZERO SA keys in state_dict', not sa_keys_no,
          f'found {sa_keys_no}')
    opt_names = {n for n, _ in nosa.named_parameters()}
    check('aca-L6-nosa has ZERO SA keys among named_parameters',
          not [n for n in opt_names if n.startswith(SA_PREFIXES)])
    check('aca-L6-nosa has no to_q / to_k / to_v attributes',
          not any(hasattr(nosa.aca, a) for a in ('to_q', 'to_k', 'to_v')))
    check('aca-L6-nosa has no temperature_sa attribute',
          not hasattr(nosa.aca, 'temperature_sa'))

    print('\nD. the parameter delta is exactly the removed branch')
    sa_size = sum(sd_l6[k].numel() for k in sa_keys_l6)
    check('removed branch measures 452,742 parameters', sa_size == 452742,
          f'{sa_size:,}')
    check('aca-L6-nosa == aca-L6 minus exactly that branch',
          n_no == n_l6 - sa_size, f'{n_no:,} vs {n_l6:,} - {sa_size:,}')
    check('aca-L6-nosa is SMALLER than aca-L6', n_no < n_l6,
          f'{n_no:,} < {n_l6:,}')
    check('every non-SA key of aca-L6 is present in aca-L6-nosa',
          set(sd_l6) - set(sa_keys_l6) == set(sd_no),
          f'symmetric difference '
          f'{len((set(sd_l6) - set(sa_keys_l6)) ^ set(sd_no))}')
    print(f'    E0 {n_e0:,} | aca-L6 {n_l6:,} (+{n_l6 - n_e0:,}) | '
          f'nosa {n_no:,} (+{n_no - n_e0:,})')

    print('\nE. the CROSS branch is byte-identical to aca-L6 at init')
    worst, worst_k = 0.0, None
    for k in sd_no:
        if k.startswith(CROSS_PREFIXES):
            d = (sd_no[k].double() - sd_l6[k].double()).abs().max().item()
            if d > worst:
                worst, worst_k = d, k
    check('to_*_cross weights identical to aca-L6 (same draws)', worst == 0.0,
          f'max abs diff {worst:.3e}' + (f' at {worst_k}' if worst_k else ''))
    for k in ('aca.alpha_logit', 'aca.temperature_ca',
              'aca.project_out.weight', 'aca.norm_f.body.weight',
              'aca.norm_d.body.weight'):
        if k in sd_no and k in sd_l6:
            d = (sd_no[k].double() - sd_l6[k].double()).abs().max().item()
            check(f'{k} identical to aca-L6', d == 0.0, f'{d:.3e}')
    check('project_out is still ZERO-init',
          float(sd_no['aca.project_out.weight'].abs().max()) == 0.0)
    check('alpha_logit is -2.0 -> alpha ~ 0.119',
          abs(float(sd_no['aca.alpha_logit']) + 2.0) < 1e-9,
          f'alpha = {torch.sigmoid(sd_no["aca.alpha_logit"]).item():.5f}')

    print('\nB. the trunk is byte-identical to E0')
    same, diff = 0, []
    for k, v in sd_e0.items():
        if k in sd_no and sd_no[k].shape == v.shape:
            if torch.equal(sd_no[k].cpu(), v.cpu()):
                same += 1
            else:
                diff.append(k)
    check('every shared E0 tensor is byte-identical', not diff,
          f'{same} identical, {len(diff)} differ' +
          (f' e.g. {diff[:3]}' if diff else ''))

    print('\nF. the attention is CHANNEL-shaped, not token-shaped')
    s16, s32 = nosa.aca.attn_shape(16, 16), nosa.aca.attn_shape(32, 32)
    check('attn_shape identical at 16x16 and 32x32 tokens', s16 == s32,
          f'{s16} vs {s32}')
    check('attn_shape matches aca-L6', s16 == l6.aca.attn_shape(16, 16),
          f'{s16}')

    print('\nA/G. forward runs and step-0 output equals E0')
    torch.manual_seed(0)
    res = {}
    for mode, hw in (('train128', 128), ('eval256', 256)):
        nosa.set_dino_mode(mode)
        e0.eval()
        radar = torch.rand(1, 1, hw, hw)
        render = torch.rand(1, 1, hw, hw)
        stacked = torch.cat([radar, render], dim=1)
        with torch.no_grad():
            y_no = nosa(stacked)
            y_e0 = e0(radar)
        d = (y_no.double() - y_e0.double()).abs().max().item()
        res[mode] = d
        check(f'{mode}: forward runs, output shape {tuple(y_no.shape)}',
              y_no.shape == y_e0.shape)
        check(f'{mode}: step-0 output bit-identical to E0', d == 0.0,
              f'max abs deviation {d:.3e}')

    print('\nG. the block returns the documented 3-tuple')
    feat = torch.randn(1, nosa.latent_channels, 8, 8)
    prior = torch.randn(1, nosa.latent_channels, 8, 8)
    g, ca, st = nosa.aca(feat, prior, collect_stats=True)
    check('returns (guided, ca_term, stats)',
          g.shape == feat.shape and ca.shape == feat.shape
          and isinstance(st, dict))
    check('guided == feat at init (zero project_out)',
          (g.double() - feat.double()).abs().max().item() == 0.0)
    check('stats report aca_ca_to_sa_ratio as exactly 0.0',
          st.get('aca_ca_to_sa_ratio', None) == 0.0,
          f'{st.get("aca_ca_to_sa_ratio")}')
    # NB: match the tag PREFIXES, not the substring '_sa_' -- that also matches
    # 'aca_ca_to_sa_ratio', which is emitted here on purpose.
    sa_tags = [k for k in st
               if k.startswith(('aca_temp_sa_', 'aca_entropy_sa_'))]
    check('stats carry no SA temperature or entropy tags', not sa_tags,
          f'{sa_tags}')
    check('stats DO carry the CA temperature and entropy tags',
          len([k for k in st if k.startswith('aca_temp_ca_')]) == nosa.aca_heads
          and len([k for k in st
                   if k.startswith('aca_entropy_ca_')]) == nosa.aca_heads,
          f'{nosa.aca_heads} heads expected')

    n_failed = sum(1 for c in checks if not c['passed'])
    rec = {
        'created': datetime.datetime.now().astimezone().isoformat(),
        'name': cfg['name'],
        'config': os.path.join(CFG, 'aca_render_fixed128_L6_nosa_latent.yml'),
        'device': 'cpu',
        'params_e0': n_e0, 'params_aca_l6': n_l6, 'params_nosa': n_no,
        'param_delta_over_E0': n_no - n_e0,
        'param_delta_vs_aca_l6': n_no - n_l6,
        'removed_sa_branch_params': sa_size,
        'step0_max_deviation_from_E0': res,
        'cross_branch_max_diff_vs_aca_l6': worst,
        'n_checks': len(checks), 'n_failed': n_failed,
        'checks': checks,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'\n{len(checks) - n_failed}/{len(checks)} passed  ->  {args.out}')
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
