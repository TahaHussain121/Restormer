"""Does the pooled DINO feature actually separate our objects? (pre-E1)

Answers the question raised by the weights-verification: the pooled vector's
cosine sims are ~0.9+ and each object looked closer to an empty crop than to
another object. Is that a large shared offset hiding a real residual, or is
there just no separation?

  1. FEATURE CENTERING: mean feature over ~150 random training crops, subtracted;
     redo the 3-probe pairwise cosines. Raw vs centered.
  2. DISCRIMINATION SCORE d = sim(A, A_noisy) - sim(A, B), same object under noise
     vs a different object. Per arm, raw and centered, mean AND std over N pairs.
       arm A: DINO sees the clean render.  A=render_i[R], A_noisy=render_i[R]+N(0,sigma)
              (renders have no natural noisy pair, so synthetic noise -- FLAGGED)
       arm B: DINO sees the noisy heatmap. A=verynoisy_i[R], A_noisy=clean_i[R]
              (real clean/noisy pair, SAME crop region -- no synthetic noise)
       B in both = a different object in the same (anchor) domain.
  3. crop sizes 128 and 256 (the dominant training crops).

Centering mean is domain- and size-matched (arm A: render crops; arm B:
verynoisy crops). CPU, batched.
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
REPO = f'{TH}/hub/facebookresearch_dinov2_main'
WEIGHTS = f'{TH}/hub/checkpoints/dinov2_vitb14_pretrain.pth'
DS = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
LAYERS = (0, 3, 7, 11)
MEAN = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
SIGMA = 0.10          # synthetic noise for arm-A A_noisy
N_PAIRS = 100
N_MU = 150
random.seed(0); np.random.seed(0); torch.manual_seed(0)

ext = DINOv2Extractor(layers=LAYERS, hub_source='local', hub_dir=REPO, weights=WEIGHTS)
ext.dino.eval()
FILES = [f.split('/')[-1] for f in sorted(glob.glob(f'{DS}/train_clean/*.png'))]


def load_render(name):
    a = cv2.imread(f'{DS}/train_renders_blackbg/{name}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    return torch.from_numpy(a).permute(2, 0, 1)                       # [3,H,W]


def load16(folder, name):
    a = cv2.imread(f'{DS}/{folder}/{name}', cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
    return torch.from_numpy(a).unsqueeze(0)                           # [1,H,W]


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
    return torch.cat(out, 0)                                          # [B,3072]


def cospair(A, B):
    return F.cosine_similarity(A, B, dim=1).numpy()


# ------------------------------------------------------------------- ITEM 1
def item1():
    print('\n==== ITEM 1: 3-probe pairwise cosine, raw vs centered ====')
    n1, n2, n3 = FILES[0], FILES[len(FILES) // 2], FILES[1]
    obj1 = load_render(n1); obj2 = load_render(n2)
    empty, _ = rcrop(load_render(n3), 90, (0, 0))
    probes = {f'object_{n1[:-4]}': obj1, f'object_{n2[:-4]}': obj2, 'empty_crop': empty}
    P = feats(list(probes.values()))
    # centering mean over random full renders
    mu = feats([load_render(random.choice(FILES)) for _ in range(N_MU)]).mean(0, keepdim=True)
    names = list(probes)
    for tag, M in [('RAW', P), ('CENTERED', P - mu)]:
        print(f'  [{tag}]')
        for i in range(3):
            for j in range(i + 1, 3):
                c = float(F.cosine_similarity(M[i:i+1], M[j:j+1], dim=1))
                print(f'      {names[i]:<12} vs {names[j]:<12} = {c:+.4f}')


# ------------------------------------------------------------------- ITEM 2/3
def discrimination(arm, S):
    A_imgs, AN_imgs, B_imgs = [], [], []
    for _ in range(N_PAIRS):
        i, j = random.sample(FILES, 2)
        if arm == 'A':
            ri, _ = rcrop(load_render(i), S)
            a = ri
            a_noisy = torch.clamp(ri + SIGMA * torch.randn_like(ri), 0, 1)
            b, _ = rcrop(load_render(j), S)
        else:  # arm B: real clean/noisy pair, same region
            vi_full = load16('train_verynoisy', i); ci_full = load16('train_clean', i)
            a, loc = rcrop(vi_full, S)
            a_noisy, _ = rcrop(ci_full, S, loc)
            b, _ = rcrop(load16('train_verynoisy', j), S)
        A_imgs.append(a); AN_imgs.append(a_noisy); B_imgs.append(b)

    fA, fAN, fB = feats(A_imgs), feats(AN_imgs), feats(B_imgs)
    # centering mean over random crops of the ANCHOR domain at this size
    if arm == 'A':
        mu = feats([rcrop(load_render(random.choice(FILES)), S)[0] for _ in range(N_MU)]).mean(0, keepdim=True)
    else:
        mu = feats([rcrop(load16('train_verynoisy', random.choice(FILES)), S)[0] for _ in range(N_MU)]).mean(0, keepdim=True)

    out = {}
    for tag, (a, an, b) in [('raw', (fA, fAN, fB)), ('centered', (fA - mu, fAN - mu, fB - mu))]:
        same = cospair(a, an); diff = cospair(a, b); d = same - diff
        out[tag] = (same, diff, d)
    return out


def item23():
    print('\n==== ITEM 2/3: discrimination d = sim(A,A_noisy) - sim(A,B) ====')
    print(f'  N={N_PAIRS} pairs; mean +/- std.  arm A noise sigma={SIGMA} (synthetic);'
          f' arm B uses real clean/noisy pair.')
    for S in (128, 256):
        for arm in ('A', 'B'):
            r = discrimination(arm, S)
            print(f'\n  --- arm {arm} ({"renderDINO" if arm=="A" else "lqDINO"}) @ crop {S} ---')
            for tag in ('raw', 'centered'):
                same, diff, d = r[tag]
                print(f'    [{tag:8}] sim(A,A_noisy)={same.mean():+.4f}+/-{same.std():.4f}  '
                      f'sim(A,B)={diff.mean():+.4f}+/-{diff.std():.4f}  '
                      f'd={d.mean():+.4f}+/-{d.std():.4f}')


if __name__ == '__main__':
    item1()
    item23()
    print('\nDONE')
