"""Shared constants and helpers for ONE isolated curiosity experiment.

    Holo_E0_frozen_noisy_output_residual_refiner

A small U-Net sees the noisy radar X and the output Y0 of the FROZEN E0-Fixed
radar-only Restormer, and predicts a signed residual: Y = Y0 + Refiner([X, Y0]).
Radar-only at inference: no render, no DINO, no clean-target-derived input, and
E0 is never modified or fine-tuned.

Proposed in a ChatGPT-assisted discussion and authorised by the author as one
bounded run. It is not a supervisor instruction.

Isolation: nothing outside dino_analysis_phases/refiner_e0/ imports this, and it
changes no shared file. The shared E0 loader (predict_phase3.build_model) is
IMPORTED read-only so E0 is built exactly as its recorded evaluation built it.
"""

import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
P3 = os.path.join(REPO, 'dino_analysis_phases', 'phase3_restoration')
for _p in (REPO, os.path.join(P3, 'scripts'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXP = 'Holo_E0_frozen_noisy_output_residual_refiner'
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
EXP_DIR = os.path.join(REPO, 'experiments', EXP)
CACHE_DIR = os.path.join(EXP_DIR, 'e0_cache')
MODEL_DIR = os.path.join(EXP_DIR, 'models')
RESULTS = os.path.join(HERE, 'results')          # gitignored (**/results/)
CONFIG = os.path.join(HERE, 'configs', 'refiner.yml')
CASES_JSON = os.path.join(HERE, 'qualitative_cases.json')

E0_NAME = 'Holo_E0_fixed128_baseline'
E0_CONFIG = os.path.join(P3, 'configs', 'E0_fixed128_baseline.yml')
E0_BEST_JSON = os.path.join(P3, 'results', E0_NAME, 'metadata',
                            'best_checkpoint.json')
E0_KEPT_JSON = os.path.join(P3, 'KEPT_CHECKPOINTS.json')
E0_REF_PRED = os.path.join(P3, 'results', E0_NAME, 'predictions',
                           'full256_{split}', 'raw')
E0_REF_CSV = os.path.join(P3, 'results', E0_NAME, 'metrics',
                          'full256_{split}_per_image.csv')

# Foreground definition = Deraining_Holo/masked_metrics.py as run_evaluate.sh
# runs it: gt > 0.01, no closing, no dilation. Background is its complement.
FG_THRESHOLD = 0.01


def now():
    return datetime.datetime.now().astimezone().isoformat()


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=REPO).decode().strip()
    except Exception:                                       # noqa: BLE001
        return 'unknown'


def file_digest(path, algo='md5', chunk=1 << 22):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def resolve_e0_checkpoint():
    """E0's VALIDATION-selected checkpoint, from its recorded metadata only.

    Cross-checked against KEPT_CHECKPOINTS.json by md5, so a replaced or
    corrupted file cannot pass silently.
    """
    with open(E0_BEST_JSON) as f:
        best = json.load(f)
    if best['experiment'] != E0_NAME:
        raise SystemExit(f'{E0_BEST_JSON} names {best["experiment"]}')
    path = best['best_checkpoint']
    if not os.path.isfile(path):
        raise SystemExit(f'selected E0 checkpoint missing: {path}')
    with open(E0_KEPT_JSON) as f:
        kept = json.load(f)[E0_NAME]
    if kept['selected'] != [best['best_iter']]:
        raise SystemExit(f'KEPT_CHECKPOINTS selects {kept["selected"]}, '
                         f'best_checkpoint.json selects {best["best_iter"]}')
    fname = os.path.basename(path)
    expected = [k['md5'] for k in kept['kept'] if k['file'] == fname]
    if len(expected) != 1:
        raise SystemExit(f'{fname} not recorded in KEPT_CHECKPOINTS.json')
    md5 = file_digest(path, 'md5')
    if md5 != expected[0]:
        raise SystemExit(f'{fname}: md5 {md5} != recorded {expected[0]}')
    return {'name': E0_NAME, 'config': E0_CONFIG, 'checkpoint': path,
            'best_iter': best['best_iter'],
            'selection_metric': best['selection_metric'],
            'best_val_psnr_8bit_in_loop': best['best_val_psnr'],
            'md5': md5, 'md5_recorded': expected[0],
            'metadata': E0_BEST_JSON}


def build_frozen_e0(device):
    """E0 exactly as predict_phase3 builds it, then frozen: eval mode and
    requires_grad False on EVERY parameter. It is never given to an optimizer.
    """
    import yaml
    import predict_phase3                                   # read-only import
    info = resolve_e0_checkpoint()
    with open(E0_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if cfg['name'] != E0_NAME or cfg['network_g'].get('dino_enabled'):
        raise SystemExit('E0 config is not the radar-only E0-Fixed baseline')
    net = predict_phase3.build_model(cfg, info['checkpoint'], device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, info


def split_ids(split):
    gt_dir = os.path.join(DATASET, f'{split}_clean')
    lq_dir = os.path.join(DATASET, f'{split}_verynoisy')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                 if f.endswith('.png'))
    lq = sorted(os.path.splitext(f)[0] for f in os.listdir(lq_dir)
                if f.endswith('.png'))
    if ids != lq:
        raise SystemExit(f'{split}: clean and verynoisy id sets differ')
    return ids


def load_uint16(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'failed to read {path}')
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint16:
        raise SystemExit(f'{path}: dtype {img.dtype}, expected uint16')
    return img


def load_split_uint16(split, ids):
    """Noisy X and target, uint16 [N,H,W]. The target is loaded for the LOSS and
    for METRICS only; it never reaches the refiner as an input."""
    lq = np.stack([load_uint16(os.path.join(DATASET, f'{split}_verynoisy',
                                            f'{i}.png')) for i in ids])
    gt = np.stack([load_uint16(os.path.join(DATASET, f'{split}_clean',
                                            f'{i}.png')) for i in ids])
    if lq.shape != gt.shape:
        raise SystemExit(f'{split}: lq {lq.shape} != gt {gt.shape}')
    return lq, gt


def to_unit(u16):
    """The established normalisation: uint16 / 65535 in float32 (identical to
    basicsr imfrombytes_uint16 and to predict_phase3)."""
    return u16.astype(np.float32) / 65535.


def cache_paths(split):
    return {'npy': os.path.join(CACHE_DIR, f'e0_raw_float32_full256_{split}.npy'),
            'ids': os.path.join(CACHE_DIR, f'ids_{split}.txt'),
            'prov': os.path.join(CACHE_DIR, f'provenance_{split}.json')}


def load_cache(split, ids=None, mmap=False):
    """Cached E0 raw float32 outputs, validated against their provenance."""
    cp = cache_paths(split)
    with open(cp['prov']) as f:
        prov = json.load(f)
    if not prov.get('complete'):
        raise SystemExit(f'{split} cache is not marked complete')
    info = resolve_e0_checkpoint()
    if prov['e0']['md5'] != info['md5']:
        raise SystemExit(f'{split} cache was made with a different E0 checkpoint')
    with open(cp['ids']) as f:
        cids = f.read().split()
    if ids is not None and cids != list(ids):
        raise SystemExit(f'{split} cache id order differs from the dataset')
    arr = np.load(cp['npy'], mmap_mode='r' if mmap else None)
    if arr.shape[0] != len(cids) or arr.dtype != np.float32:
        raise SystemExit(f'{split} cache shape/dtype {arr.shape} {arr.dtype}')
    return arr, cids, prov


# --- quantisation and metrics: the established evaluation convention --------

def quantize_np(raw):
    """float32 prediction -> uint16, exactly predict_phase3's path:
    clamp(0,1) in float32, *65535, round half-to-even, cast."""
    c = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).clamp(0, 1)
    return (c.numpy() * 65535.).round().astype(np.uint16)


def quantize_torch(y):
    """Same operation on a float32 torch tensor; returns float32 integer levels."""
    return (y.clamp(0, 1) * 65535.).round()


def psnr_u16(gt_u16, pred_u16):
    """skimage peak_signal_noise_ratio(gt, pred, data_range=1.0) on /65535
    float64 images, written out so it can run on a whole split quickly."""
    g = gt_u16.astype(np.float64) / 65535.
    p = pred_u16.astype(np.float64) / 65535.
    mse = np.mean((g - p) ** 2)
    return float('inf') if mse == 0 else 10. * math.log10(1. / mse)


def psnr_levels_torch(gt_levels, pred_levels, mask=None):
    """Per-image PSNR in float64 from integer-level tensors [B,1,H,W]."""
    g = gt_levels.double() / 65535.
    p = pred_levels.double() / 65535.
    se = (g - p) ** 2
    if mask is None:
        mse = se.flatten(1).mean(1)
    else:
        m = mask.double()
        mse = (se * m).flatten(1).sum(1) / m.flatten(1).sum(1)
    return 10. * torch.log10(1. / mse)


def background_metrics(gt_u16, pred_u16):
    """Background = gt <= 0.01 (complement of the masked_metrics foreground).
    Error and false-structure indicators, declared before training."""
    g = gt_u16.astype(np.float64) / 65535.
    p = pred_u16.astype(np.float64) / 65535.
    bg = ~(g > FG_THRESHOLD)
    if not bg.any():
        return None
    e = p[bg] - g[bg]
    return {'bg_frac': float(bg.mean()),
            'bg_mae': float(np.abs(e).mean()),
            'bg_rmse': float(np.sqrt((e ** 2).mean())),
            'bg_mean_pred': float(p[bg].mean()),
            'bg_frac_pred_gt_0p05': float((p[bg] > 0.05).mean()),
            'bg_frac_pred_gt_0p10': float((p[bg] > 0.10).mean())}


def cosine_lr(t, lr0, lr_min, total):
    """Cosine decay from lr0 (update t=0) towards lr_min (t=total), no warmup."""
    return lr_min + 0.5 * (lr0 - lr_min) * (1. + math.cos(math.pi * t / total))


def strict_fp32():
    """float32 throughout: Ampere GPUs otherwise run convolutions and matmuls
    in TF32 by default, which is not float32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_record(device):
    rec = {'hostname': os.uname().nodename, 'torch': torch.__version__,
           'numpy': np.__version__, 'git_commit': git_commit(),
           'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
           'cudnn_benchmark': torch.backends.cudnn.benchmark,
           'cudnn_deterministic': torch.backends.cudnn.deterministic,
           'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
           'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32}
    if str(device).startswith('cuda') and torch.cuda.is_available():
        rec['gpu'] = torch.cuda.get_device_name(0)
        rec['cuda'] = torch.version.cuda
        rec['cudnn'] = torch.backends.cudnn.version()
    return rec


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
