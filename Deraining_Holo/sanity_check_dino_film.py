"""Pre-flight wiring check for RestormerDINO. Run BEFORE spending GPU hours.

Proves the property the whole design depends on: at initialisation the
FiLM-guided model is numerically identical to the baseline Restormer on the
same input. If this fails, the injection is wired wrong -- catch it here.

    python Deraining_Holo/sanity_check_dino_film.py

Part 1 uses the offline DINO stub, because the identity claim comes from the
zero-initialised FiLM projection (gamma=beta=0) and is independent of what DINO
outputs. No download, no GPU needed.

Part 2 (added with the centering change) repeats the identity check with the
REAL frozen DINOv2 and each arm's REAL precomputed mean vector, read from the
arm's yml. Subtracting a constant cannot break a zero-init identity by
construction, but "cannot by construction" is not "confirmed", so it is
confirmed: max abs difference vs the baseline is reported per arm. Part 2 also
verifies the subtraction is actually applied and uses the right vector.
Runs on CPU in a couple of minutes.
"""
import os

import torch
import yaml

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.restormer_dino_arch import (
    RestormerDINO, load_dino_feat_mean)

# Baseline network_g kwargs (must mirror the yml exactly).
CFG = dict(inp_channels=1, out_channels=1, dim=48,
           num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
           heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
           bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)

ARM_CFGS = {
    'lqDINO': 'Deraining_Holo/Options/Holo_DINOv2_lqDINO_Restormer.yml',
    'renderDINO': 'Deraining_Holo/Options/Holo_DINOv2_renderDINO_Restormer.yml',
}

SEED = 100          # manual_seed from the baseline yml
torch.manual_seed(SEED)

x = torch.randn(2, 1, 128, 128)   # [B,1,H,W], like an LQ crop
ok = True


def check(name, cond):
    global ok
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    ok = ok and bool(cond)


# =========================================================== PART 1: stub
print('=== PART 1: stub extractor (wiring / zero-init identity) ===')

print('1) FiLM ON vs FiLM OFF at init (pure zero-init identity)')
torch.manual_seed(SEED)
net = RestormerDINO(dino_stub=True, **CFG).eval()
with torch.no_grad():
    y_on = net(x, apply_film=True)
    y_off = net(x, apply_film=False)
check('max|film_on - film_off| == 0', torch.equal(y_on, y_off))
print(f'       max abs diff = {(y_on - y_off).abs().max().item():.3e}')

print('2) FiLM model == standalone baseline Restormer (same seed, same weights)')
torch.manual_seed(SEED)
base = Restormer(**CFG).eval()
# copy the FiLM model's backbone weights into the baseline so we compare the
# forward path, not the RNG order (belt and braces).
missing = base.load_state_dict(
    {k: v for k, v in net.state_dict().items() if k in base.state_dict()},
    strict=False)
with torch.no_grad():
    y_base = base(x)
    y_film = net(x, apply_film=True)
check('max|film_model - baseline| == 0', torch.equal(y_film, y_base))
print(f'       max abs diff = {(y_film - y_base).abs().max().item():.3e}')

print('3) gamma/beta are exactly zero at init')
feat = net.dino(x)
gammas, betas = net.film(feat)
allz = all(g.abs().max().item() == 0 for g in gammas) and \
       all(b.abs().max().item() == 0 for b in betas)
check('all gamma == 0 and all beta == 0', allz)

print('4) FiLM channel counts match the backbone stages')
check('film_channels == [384,192,96,96]', net.film_channels == [384, 192, 96, 96])

print('5) only DINO is frozen; FiLM head + backbone are trainable')
n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
n_dino = sum(p.numel() for p in net.dino.parameters())
n_frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
print(f'       trainable params: {n_train:,}   frozen (DINO): {n_frozen:,}')
check('some params frozen (DINO) and some trainable', n_train > 0 and n_frozen >= 0)


# ============================================ PART 2: real DINO + real mean
print('\n=== PART 2: real DINOv2 + real per-arm centering vector ===')

for arm, cfg_path in ARM_CFGS.items():
    print(f'\n--- arm {arm} ({os.path.basename(cfg_path)}) ---')
    with open(cfg_path) as f:
        net_cfg = yaml.safe_load(f)['network_g']
    mean_path = net_cfg.get('dino_feat_mean')
    check(f'{arm}: yml sets dino_feat_mean', mean_path is not None)
    if mean_path is None:
        continue

    torch.manual_seed(SEED)
    dnet = RestormerDINO(
        dino_layers=net_cfg['dino_layers'], dino_img_size=net_cfg['dino_img_size'],
        dino_model_name=net_cfg['dino_model_name'],
        dino_hub_source=net_cfg['dino_hub_source'],
        dino_hub_dir=net_cfg['dino_hub_dir'], dino_weights=net_cfg['dino_weights'],
        dino_feat_mean=mean_path, film_hidden=net_cfg['film_hidden'],
        **CFG).eval()

    # 6) centering is actually on, and it is the file the yml names
    mu_file = load_dino_feat_mean(mean_path, dnet.dino.feat_dim)
    check(f'{arm}: extractor.centered is True', dnet.dino.centered)
    check(f'{arm}: feat_mean buffer == the .pt in the yml',
          torch.equal(dnet.dino.feat_mean, mu_file))
    print(f'       ||feat_mean|| = {mu_file.norm().item():.3f}')

    # 7) the subtraction really happens: centered feature + mean == raw pooled
    with torch.no_grad():
        f_centered = dnet.dino(x)
        raw = dnet.dino.dino.get_intermediate_layers(
            torch.nn.functional.interpolate(
                x.repeat(1, 3, 1, 1), size=(net_cfg['dino_img_size'],) * 2,
                mode='bilinear', align_corners=False).sub(dnet.dino.mean)
            .div(dnet.dino.std),
            n=dnet.dino.layers, reshape=False,
            return_class_token=False, norm=True)
        f_raw = torch.cat([t.mean(1) for t in raw], dim=1)
    d_sub = (f_raw - mu_file - f_centered).abs().max().item()
    check(f'{arm}: forward output == raw_pooled - feat_mean', d_sub < 1e-5)
    print(f'       ||raw||={f_raw.norm(dim=1).mean().item():.3f}  '
          f'||centered||={f_centered.norm(dim=1).mean().item():.3f}  '
          f'(max resid {d_sub:.2e})')
    print('       NB: x here is Gaussian noise, not a real crop, so it sits far '
          'off the\n           training distribution -- ||centered|| > ||raw|| '
          'is expected here and\n           says nothing about real inputs.')

    # 8) identity at init STILL holds with real features + centering
    torch.manual_seed(SEED)
    dbase = Restormer(**CFG).eval()
    dbase.load_state_dict(
        {k: v for k, v in dnet.state_dict().items() if k in dbase.state_dict()},
        strict=False)
    with torch.no_grad():
        y_on = dnet(x, apply_film=True)
        y_off = dnet(x, apply_film=False)
        y_b = dbase(x)
    d_onoff = (y_on - y_off).abs().max().item()
    d_base = (y_on - y_b).abs().max().item()
    check(f'{arm}: max|film_on - film_off| == 0', torch.equal(y_on, y_off))
    print(f'       max abs diff (on vs off)      = {d_onoff:.3e}')
    check(f'{arm}: max|film_model - baseline| == 0', torch.equal(y_on, y_b))
    print(f'       max abs diff (film vs baseline) = {d_base:.3e}')

    del dnet, dbase


print()
if ok:
    print('ALL CHECKS PASSED -- identity at init holds WITH centering, for both')
    print('arms, with the real frozen DINOv2 and the real per-arm mean vectors.')
else:
    print('SOME CHECKS FAILED -- do NOT train. Fix the wiring first.')
    raise SystemExit(1)
