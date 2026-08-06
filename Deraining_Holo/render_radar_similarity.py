"""Corrected discrimination: does DINO link a RENDER to the SAME object's radar?

Replaces the synthetic-Gaussian version. The real question for the renderDINO
arm: is DINO(render_i) more similar to DINO(radar_i) [same object, 1e7 verynoisy
heatmap] than to DINO(radar_j) [different object]? No synthetic noise -- the
"other view" is the actual radar image.

  same_i = cos( f(render_i[R]),  f(verynoisy_i[R]) )   # same object, render<->radar
  diff_ij= cos( f(render_i[R]),  f(verynoisy_j[R']) )  # different object
  d = mean(same) - mean(diff)

render and verynoisy share the filename/view, cropped at the same region R.
Reported raw and centered (render feats by render-mean, radar feats by
radar-mean, each domain- and size-matched), mean +/- std over N pairs, crops
128 and 256.  CPU, batched.
"""
import glob
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from basicsr.models.archs.restormer_dino_arch import (
    DINOv2Extractor, dino_preprocess, _IMAGENET_MEAN, _IMAGENET_STD)

TH = '/home/woody/iwnt/iwnt174h/thesis_dino/code/torch_hub'
DS = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
LAYERS = (0, 3, 7, 11)
MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
N_PAIRS, N_MU = 100, 150
random.seed(0); np.random.seed(0); torch.manual_seed(0)

ext = DINOv2Extractor(layers=LAYERS, hub_source='local',
                      hub_dir=f'{TH}/hub/facebookresearch_dinov2_main',
                      weights=f'{TH}/hub/checkpoints/dinov2_vitb14_pretrain.pth')
ext.dino.eval()
FILES = [f.split('/')[-1] for f in sorted(glob.glob(f'{DS}/train_clean/*.png'))]


def load_render(name):
    a = cv2.imread(f'{DS}/train_renders_blackbg/{name}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)


def load_radar(name):  # 1e7 verynoisy heatmap, uint16
    a = cv2.imread(f'{DS}/train_verynoisy/{name}', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    return torch.from_numpy(a).unsqueeze(0)


def rcrop(t, S, loc=None):
    _, H, W = t.shape
    if H <= S:
        return t, (0, 0)
    if loc is None:
        loc = (random.randint(0, H - S), random.randint(0, W - S))
    y, x = loc
    return t[:, y:y + S, x:x + S], (y, x)


@torch.no_grad()
def feats(imgs, bs=16):
    xs = torch.stack([dino_preprocess(im.unsqueeze(0), 224, MEAN, STD)[0] for im in imgs])
    out = []
    for i in range(0, len(xs), bs):
        fb = ext.dino.get_intermediate_layers(
            xs[i:i + bs], n=LAYERS, reshape=False, return_class_token=False, norm=True)
        out.append(torch.cat([f.mean(1) for f in fb], dim=1))
    return torch.cat(out, 0)


def run(S):
    R_imgs, V_same, V_diff = [], [], []
    for _ in range(N_PAIRS):
        i, j = random.sample(FILES, 2)
        ri, loc = rcrop(load_render(i), S)
        vi, _ = rcrop(load_radar(i), S, loc)      # same object, same region
        vj, _ = rcrop(load_radar(j), S)           # different object
        R_imgs.append(ri); V_same.append(vi); V_diff.append(vj)
    fR, fVs, fVd = feats(R_imgs), feats(V_same), feats(V_diff)
    mu_r = feats([rcrop(load_render(random.choice(FILES)), S)[0] for _ in range(N_MU)]).mean(0, keepdim=True)
    mu_v = feats([rcrop(load_radar(random.choice(FILES)), S)[0] for _ in range(N_MU)]).mean(0, keepdim=True)

    print(f'\n  --- render<->radar @ crop {S} ---')
    for tag, (r, vs, vd) in [('raw', (fR, fVs, fVd)),
                             ('centered', (fR - mu_r, fVs - mu_v, fVd - mu_v))]:
        same = F.cosine_similarity(r, vs, dim=1).numpy()
        diff = F.cosine_similarity(r, vd, dim=1).numpy()
        d = same - diff
        print(f'    [{tag:8}] sim(render_i,radar_i)={same.mean():+.4f}+/-{same.std():.4f}  '
              f'sim(render_i,radar_j)={diff.mean():+.4f}+/-{diff.std():.4f}  '
              f'd={d.mean():+.4f}+/-{d.std():.4f}  (SE {d.std()/10:.3f})')


if __name__ == '__main__':
    print('render<->radar cross-modal, same-object vs different-object (no synthetic noise)')
    run(128)
    run(256)
    print('\nDONE')
