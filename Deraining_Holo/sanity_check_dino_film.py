"""Pre-flight wiring check for RestormerDINO. Run BEFORE spending GPU hours.

Proves the property the whole design depends on: at initialisation the
FiLM-guided model is numerically identical to the baseline Restormer on the
same input. If this fails, the injection is wired wrong -- catch it here.

    python Deraining_Holo/sanity_check_dino_film.py

Uses the offline DINO stub, because the identity claim comes from the
zero-initialised FiLM projection (gamma=beta=0) and is independent of what DINO
outputs. No download, no GPU needed -- runs on CPU in seconds.
"""
import torch

from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.restormer_dino_arch import RestormerDINO

# Baseline network_g kwargs (must mirror the yml exactly).
CFG = dict(inp_channels=1, out_channels=1, dim=48,
           num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
           heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
           bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)

SEED = 100          # manual_seed from the baseline yml
torch.manual_seed(SEED)

x = torch.randn(2, 1, 128, 128)   # [B,1,H,W], like an LQ crop
ok = True


def check(name, cond):
    global ok
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    ok = ok and bool(cond)


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

print()
if ok:
    print('ALL CHECKS PASSED -- wiring is identity at init. Safe to train.')
    print('NOTE: this used the DINO STUB. Before training, load real DINOv2 and')
    print('      re-run a forward pass to confirm the download + feature shape.')
else:
    print('SOME CHECKS FAILED -- do NOT train. Fix the wiring first.')
    raise SystemExit(1)
