"""Prove the DINOv2 weights are REALLY loaded, not silently random.

A random-init ViT runs fine, outputs [B,3072], and passes shape+determinism
checks -- so those are not enough. This script:
  (a) loads with strict=True and prints missing/unexpected keys (must be empty);
  (b) determinism: same image twice -> max abs diff (must be 0, eval mode);
  (c) discrimination: pairwise cosine sim of features for {object, other object,
      mostly-empty} crops;
  (d) runs (b)+(c) again on a deliberately RANDOM-INIT DINO as a control, side by
      side, so you can see the checks actually separate loaded from random.

Run:  python Deraining_Holo/verify_dino_weights.py
Reads the cache paths the configs use. No training. No internet.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from basicsr.models.archs.restormer_dino_arch import (
    DINOv2Extractor, dino_preprocess, _IMAGENET_MEAN, _IMAGENET_STD)

TH = '/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub'
REPO = f'{TH}/hub/facebookresearch_dinov2_main'
WEIGHTS = f'{TH}/hub/checkpoints/dinov2_vitb14_pretrain.pth'
DS = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
LAYERS = (0, 3, 7, 11)
MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)


def load_img(path):
    a = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W] in [0,1]


@torch.no_grad()
def pool(dino, img):
    """EXACT mirror of DINOv2Extractor.forward pooling."""
    x = dino_preprocess(img, 224, MEAN, STD)
    feats = dino.get_intermediate_layers(
        x, n=LAYERS, reshape=False, return_class_token=False, norm=True)
    return torch.cat([f.mean(1) for f in feats], dim=1)   # [B,3072]


def cos(a, b):
    return float(F.cosine_similarity(a, b, dim=1).item())


def bg_frac(img):
    return float((img.mean(1) <= 0.05).float().mean().item())


# ---- three probe inputs: object, a DIFFERENT object, a mostly-empty crop -----
obj1 = load_img(f'{DS}/renders_blackbg/0001.png')
obj2 = load_img(f'{DS}/renders_blackbg/0500.png')
# mostly-empty: top strip of a render (space above the furniture is black)
empty = load_img(f'{DS}/renders_blackbg/0002.png')[:, :, 0:90, :]
probes = [('object_0001', obj1), ('object_0500', obj2), ('empty_crop', empty)]
print('probe background fractions:',
      {n: f'{bg_frac(im):.0%}' for n, im in probes})


def discrimination(dino, tag):
    feats = {n: pool(dino, im) for n, im in probes}
    print(f'  [{tag}] pairwise cosine similarity:')
    names = list(feats)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f'      {names[i]:<12} vs {names[j]:<12} = '
                  f'{cos(feats[names[i]], feats[names[j]]):+.4f}')


def determinism(dino, tag):
    a = pool(dino, obj1)
    b = pool(dino, obj1)
    d = (a - b).abs().max().item()
    print(f'  [{tag}] determinism max|f(x)-f(x)| = {d:.3e}  '
          f'({"bit-identical" if d == 0 else "NONDETERMINISTIC"})')


print('\n================ (A) LOADED (pretrained, strict=True) ================')
ext = DINOv2Extractor(layers=LAYERS, hub_source='local',
                      hub_dir=REPO, weights=WEIGHTS)
ext.dino.eval()
r = ext.load_result
print('  strict load_state_dict result:')
print('    missing_keys   :', list(r.missing_keys))
print('    unexpected_keys:', list(r.unexpected_keys))
print('    -> both empty  :', len(r.missing_keys) == 0 and len(r.unexpected_keys) == 0)
print('  training mode? (want False):', ext.dino.training)
determinism(ext.dino, 'loaded')
discrimination(ext.dino, 'loaded')

print('\n================ (B) CONTROL: random-init (pretrained=False, no weights) ==')
rand = torch.hub.load(REPO, 'dinov2_vitb14', source='local', pretrained=False)
rand.eval()
determinism(rand, 'random')
discrimination(rand, 'random')

print('\n  NOTE: within-model cosine sims above are HIGH for BOTH models and do NOT')
print('  cleanly separate loaded from random -- the probes are grayscale-on-black')
print('  (globally similar, OOD for DINO) and mean-pooling adds a common component.')
print('  So this check is INCONCLUSIVE on this data. The two checks below ARE')
print('  definitive and do not depend on OOD feature behaviour.')

print('\n================ (C) PARAMETER-LEVEL PROOF (definitive) ================')
sd_file = torch.load(WEIGHTS, map_location='cpu')
loaded_sd = ext.dino.state_dict()
rand_sd = rand.state_dict()
probe_keys = ['cls_token', 'blocks.0.attn.qkv.weight', 'blocks.11.mlp.fc2.weight']
all_match_file, all_diff_rand = True, True
for k in probe_keys:
    same_as_file = torch.equal(loaded_sd[k], sd_file[k])
    diff_from_rand = (loaded_sd[k] - rand_sd[k]).abs().max().item()
    all_match_file &= same_as_file
    all_diff_rand &= (diff_from_rand > 0)
    print(f'  {k:<32} loaded==file: {same_as_file} | '
          f'max|loaded-random| = {diff_from_rand:.3f}')
print(f'  -> every probed param equals the .pth file : {all_match_file}')
print(f'  -> every probed param differs from random  : {all_diff_rand}')

print('\n================ (D) CROSS-MODEL FEATURES (definitive) ================')
print('  cosine(loaded(x), random(x)) per probe -- must be LOW if weights matter:')
for n, im in probes:
    print(f'      {n:<12}: {cos(pool(ext.dino, im), pool(rand, im)):+.4f}')

print('\n================ VERDICT ================')
ok = (len(r.missing_keys) == 0 and len(r.unexpected_keys) == 0
      and all_match_file and all_diff_rand)
print('  strict keys empty + params == file + params != random =>',
      'WEIGHTS GENUINELY LOADED' if ok else 'PROBLEM -- DO NOT TRAIN')
