"""Shared helpers for ONE isolated diagnostic experiment:

    Holo_E0_oracle_masked_completion_diagnostic

Isolation contract (enforced here, not just documented):
  * Every write goes through a helper that resolves symlinks and REFUSES any
    destination outside this experiment root.
  * Bytecode, matplotlib, torch hub and temporary files are redirected into the
    root BEFORE torch / cv2 / matplotlib are imported.
  * Nothing from the existing experiments is imported. The utilities below are
    COPIES (adapted) of dino_analysis_phases/refiner_e0/refiner_common.py, so no
    existing module is executed and no existing path is written.
  * Existing data, splits, the frozen E0 checkpoint and the E0 output caches are
    read ONLY, through read-only handles.

The frozen E0 network is never built here: its full-frame float32 outputs come
from the existing validated cache, so no E0 parameter can reach an optimizer.
"""

import os as _os

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# --- redirect every cache/tmp path into the root, BEFORE heavy imports -------
_os.environ['PYTHONPYCACHEPREFIX'] = _os.path.join(ROOT, 'tmp', 'pycache')
_os.environ['MPLCONFIGDIR'] = _os.path.join(ROOT, 'tmp', 'mplconfig')
_os.environ['TMPDIR'] = _os.path.join(ROOT, 'tmp')
_os.environ['TEMP'] = _os.environ['TMP'] = _os.environ['TMPDIR']
_os.environ['XDG_CACHE_HOME'] = _os.path.join(ROOT, 'tmp', 'xdg_cache')
_os.environ['TORCH_HOME'] = _os.path.join(ROOT, 'tmp', 'torch_home')
_os.environ['HF_HOME'] = _os.path.join(ROOT, 'tmp', 'hf_home')
_os.environ['CUDA_CACHE_PATH'] = _os.path.join(ROOT, 'tmp', 'cuda_cache')
for _d in ('pycache', 'mplconfig', 'xdg_cache', 'torch_home', 'hf_home', 'cuda_cache'):
    _os.makedirs(_os.path.join(ROOT, 'tmp', _d), exist_ok=True)
import sys as _sys
_sys.pycache_prefix = _os.environ['PYTHONPYCACHEPREFIX']
_sys.dont_write_bytecode = True                 # belt and braces

import datetime          # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import subprocess        # noqa: E402

import cv2               # noqa: E402
import numpy as np       # noqa: E402
import torch             # noqa: E402

os = _os
sys = _sys

EXP = 'Holo_E0_oracle_masked_completion_diagnostic'
REPO = '/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer'
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
P3 = os.path.join(REPO, 'dino_analysis_phases', 'phase3_restoration')

# --- EXISTING artifacts, READ ONLY ------------------------------------------
E0_NAME = 'Holo_E0_fixed128_baseline'
E0_BEST_JSON = os.path.join(P3, 'results', E0_NAME, 'metadata', 'best_checkpoint.json')
E0_KEPT_JSON = os.path.join(P3, 'KEPT_CHECKPOINTS.json')
E0_REF_PRED = os.path.join(P3, 'results', E0_NAME, 'predictions', 'full256_{split}', 'raw')
E0_CACHE_DIR = os.path.join(REPO, 'experiments',
                            'Holo_E0_frozen_noisy_output_residual_refiner', 'e0_cache')
REF_E0_DIR = os.path.join(REPO, 'dino_analysis_phases', 'refiner_e0')
CASES_JSON = os.path.join(REF_E0_DIR, 'qualitative_cases.json')
REFINER_E0_VAL_PRED = os.path.join(REF_E0_DIR, 'results', 'predictions', 'full256_val', 'refined')
FGBAL_VAL_PRED = os.path.join(REPO, 'dino_analysis_phases', 'refiner_fgbal', 'results',
                              'predictions', 'full256_val', 'refined_fgbal')

# --- destinations, ALL inside the root --------------------------------------
CODE = os.path.join(ROOT, 'code')
CONFIG = os.path.join(ROOT, 'configs', 'completion.yml')
RESULTS = os.path.join(ROOT, 'results')
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
CACHE = os.path.join(ROOT, 'cache')
LOGS = os.path.join(ROOT, 'logs')

FG_THRESHOLD = 0.01          # the study's foreground, unchanged
MISS_GT, MISS_E0 = 0.05, 0.02   # the documented missing-structure rule


# ---------------------------------------------------------------- write guard
class OutsideRootError(RuntimeError):
    pass


def assert_inside_root(path):
    """Resolve symlinks on the path AND on every existing parent, then refuse
    any destination that is not under the experiment root."""
    root = os.path.realpath(ROOT)
    p = os.path.realpath(os.path.abspath(path))
    if p != root and not p.startswith(root + os.sep):
        raise OutsideRootError(f'write destination outside the experiment root: '
                               f'{path} -> {p} (root {root})')
    # a component that is a symlink pointing out of the root would already be
    # resolved above; check the existing parent chain too
    parent = os.path.dirname(p)
    while parent and parent != root and parent.startswith(root + os.sep):
        if os.path.islink(parent):
            raise OutsideRootError(f'symlinked parent inside root: {parent}')
        parent = os.path.dirname(parent)
    return p


def ensure_dir(path):
    p = assert_inside_root(path)
    os.makedirs(p, exist_ok=True)
    return p


def open_exclusive(path, mode='w'):
    p = assert_inside_root(path)
    ensure_dir(os.path.dirname(p))
    return open(p, 'x' + ('b' if 'b' in mode else ''))


def write_json_exclusive(path, obj):
    with open_exclusive(path) as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
    return path


def write_json_overwrite_log(path, obj):
    """For append-style logs that a single run rewrites (train/val logs only)."""
    p = assert_inside_root(path)
    ensure_dir(os.path.dirname(p))
    with open(p, 'w') as f:
        json.dump(obj, f, indent=1, default=str)
    return p


def imwrite_guarded(path, img):
    p = assert_inside_root(path)
    ensure_dir(os.path.dirname(p))
    if os.path.exists(p):
        raise FileExistsError(f'refusing to overwrite {p}')
    if not cv2.imwrite(p, img):
        raise IOError(f'cv2.imwrite failed for {p}')
    return p


def save_checkpoint_exclusive(path, obj):
    with open_exclusive(path, 'wb') as f:
        torch.save(obj, f)
    return path


def verify_write_paths(paths):
    """Called before training: every planned destination must resolve inside
    the root, symlinks included. Aborts on the first violation."""
    ok = []
    for p in paths:
        ok.append({'path': p, 'resolved': assert_inside_root(p)})
    return ok


# --------------------------------------------------------------- provenance
def now():
    return datetime.datetime.now().astimezone().isoformat()


def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
    except Exception:                                             # noqa: BLE001
        return 'unknown'


def file_digest(path, algo='md5', chunk=1 << 22):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def resolve_e0_checkpoint():
    """E0's validation-selected checkpoint from its recorded metadata, md5 cross
    checked against KEPT_CHECKPOINTS.json. Identical logic to
    refiner_common.resolve_e0_checkpoint (copied, read-only)."""
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
        raise SystemExit('KEPT_CHECKPOINTS / best_checkpoint.json disagree')
    fname = os.path.basename(path)
    expected = [k['md5'] for k in kept['kept'] if k['file'] == fname]
    if len(expected) != 1:
        raise SystemExit(f'{fname} not recorded in KEPT_CHECKPOINTS.json')
    md5 = file_digest(path, 'md5')
    if md5 != expected[0]:
        raise SystemExit(f'{fname}: md5 {md5} != recorded {expected[0]}')
    return {'name': E0_NAME, 'checkpoint': path, 'best_iter': best['best_iter'],
            'md5': md5, 'metadata': E0_BEST_JSON}


# --------------------------------------------------------------------- data
def split_ids(split):
    gt_dir = os.path.join(DATASET, f'{split}_clean')
    lq_dir = os.path.join(DATASET, f'{split}_verynoisy')
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.png'))
    lq = sorted(os.path.splitext(f)[0] for f in os.listdir(lq_dir) if f.endswith('.png'))
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
    lq = np.stack([load_uint16(os.path.join(DATASET, f'{split}_verynoisy', f'{i}.png'))
                   for i in ids])
    gt = np.stack([load_uint16(os.path.join(DATASET, f'{split}_clean', f'{i}.png'))
                   for i in ids])
    if lq.shape != gt.shape:
        raise SystemExit(f'{split}: lq {lq.shape} != gt {gt.shape}')
    return lq, gt


def to_unit(u16):
    return u16.astype(np.float32) / 65535.


def cache_paths(split):
    return {'npy': os.path.join(E0_CACHE_DIR, f'e0_raw_float32_full256_{split}.npy'),
            'ids': os.path.join(E0_CACHE_DIR, f'ids_{split}.txt'),
            'prov': os.path.join(E0_CACHE_DIR, f'provenance_{split}.json')}


def load_cache(split, ids=None, mmap=True):
    """The EXISTING frozen-E0 output cache, read-only, validated against its
    provenance and against the resolved E0 checkpoint md5."""
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


# ------------------------------------------------- evaluation convention
def quantize_np(raw):
    c = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).clamp(0, 1)
    return (c.numpy() * 65535.).round().astype(np.uint16)


def quantize_torch(y):
    return (y.clamp(0, 1) * 65535.).round()


def psnr_levels_torch(gt_levels, pred_levels, mask=None):
    g = gt_levels.double() / 65535.
    p = pred_levels.double() / 65535.
    se = (g - p) ** 2
    if mask is None:
        mse = se.flatten(1).mean(1)
    else:
        m = mask.double()
        mse = (se * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)
    return 10. * torch.log10(1. / mse.clamp(min=1e-30))


def psnr_from_mse(mse):
    return float('inf') if mse <= 0 else 10. * math.log10(1. / mse)


def cosine_lr(t, lr0, lr_min, total):
    return lr_min + 0.5 * (lr0 - lr_min) * (1. + math.cos(math.pi * t / total))


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_record(device):
    rec = {'hostname': os.uname().nodename, 'torch': torch.__version__,
           'numpy': np.__version__, 'opencv': cv2.__version__,
           'git_commit': git_commit(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
           'cudnn_benchmark': torch.backends.cudnn.benchmark,
           'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
           'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
           'pycache_prefix': os.environ.get('PYTHONPYCACHEPREFIX'),
           'tmpdir': os.environ.get('TMPDIR')}
    if str(device).startswith('cuda') and torch.cuda.is_available():
        rec['gpu'] = torch.cuda.get_device_name(0)
        rec['cuda'] = torch.version.cuda
    return rec
