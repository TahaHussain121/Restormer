"""Record (or audit) the state of EXISTING repository files and reused artifacts.

Read-only. Writes ONLY inside the experiment root. Run twice: once before any
other action (--mode before) and once at the end (--mode after), then --mode
diff to compare.

Coverage, stated honestly:
  * full sha256 for every file <= 64 MiB under the listed trees;
  * for larger files: size, mtime and sha256 of the first and last 8 MiB
    (a partial fingerprint, not a full hash);
  * the dataset tree is fingerprinted by size+mtime only for the train split
    (12,202 files) and fully hashed for val/test clean+verynoisy.
Anything outside the listed trees is NOT covered.
"""
import argparse, hashlib, json, os, subprocess, sys, time

REPO = '/home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer'
DATASET = '/home/woody/iwnt/iwnt174h/thesis_dino/holographic_image_dataset'
BIG = 64 << 20
HEAD_TAIL = 8 << 20

TREES = [
    os.path.join(REPO, 'dino_analysis_phases'),
    os.path.join(REPO, 'Deraining_Holo'),
    os.path.join(REPO, 'basicsr'),
    os.path.join(REPO, 'experiments', 'Holo_E0_fixed128_baseline'),
    os.path.join(REPO, 'experiments', 'Holo_E0_frozen_noisy_output_residual_refiner'),
    os.path.join(REPO, 'experiments', 'Holo_E0_frozen_noisy_output_fgbalanced_refiner'),
]
HASH_ONLY_DIRS = [os.path.join(DATASET, d) for d in
                  ('val_clean', 'val_verynoisy', 'test_clean', 'test_verynoisy')]
STAT_ONLY_DIRS = [os.path.join(DATASET, d) for d in
                  ('train_clean', 'train_verynoisy')]


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if limit is None:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        else:
            h.update(f.read(limit))
            size = os.path.getsize(path)
            if size > limit:
                f.seek(max(0, size - limit))
                h.update(f.read(limit))
    return h.hexdigest()


def entry(path, stat_only=False):
    st = os.lstat(path)
    rec = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'mode': oct(st.st_mode)}
    if os.path.islink(path):
        rec['symlink_to'] = os.readlink(path)
        return rec
    if stat_only:
        rec['hash'] = 'stat-only'
    elif st.st_size <= BIG:
        rec['hash'] = 'sha256:' + sha256_file(path)
    else:
        rec['hash'] = 'sha256-headtail8M:' + sha256_file(path, HEAD_TAIL)
    return rec


def walk(root, stat_only=False, out=None):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            try:
                out[os.path.relpath(p, REPO if p.startswith(REPO) else DATASET)] = \
                    entry(p, stat_only)
            except (OSError, IOError) as e:                       # noqa: PERF203
                out[p] = {'error': str(e)}
    return out


def collect():
    files = {}
    t0 = time.time()
    for tree in TREES:
        walk(tree, False, files)
    for d in HASH_ONLY_DIRS:
        walk(d, False, files)
    for d in STAT_ONLY_DIRS:
        walk(d, True, files)
    git = {}
    for name, cmd in (('head', ['git', 'rev-parse', 'HEAD']),
                      ('status_porcelain', ['git', 'status', '--porcelain']),
                      ('branch', ['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
                      ('index_hash', ['git', 'hash-object', os.path.join(REPO, '.git', 'index')])):
        try:
            git[name] = subprocess.check_output(cmd, cwd=REPO).decode().rstrip('\n')
        except Exception as e:                                    # noqa: BLE001
            git[name] = f'error: {e}'
    return {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'repo': REPO,
            'dataset': DATASET, 'n_files': len(files),
            'seconds': round(time.time() - t0, 1), 'git': git, 'files': files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['before', 'after', 'diff'], required=True)
    ap.add_argument('--audit-dir', required=True)
    a = ap.parse_args()
    before = os.path.join(a.audit_dir, 'snapshot_before.json')
    after = os.path.join(a.audit_dir, 'snapshot_after.json')
    if a.mode in ('before', 'after'):
        path = before if a.mode == 'before' else after
        snap = collect()
        with open(path, 'x') as f:            # exclusive: never overwrite
            json.dump(snap, f, indent=1, sort_keys=True)
        print(f'{a.mode}: {snap["n_files"]} files in {snap["seconds"]}s -> {path}')
        return
    with open(before) as f:
        b = json.load(f)
    with open(after) as f:
        c = json.load(f)
    bf, cf = b['files'], c['files']
    changed = sorted(k for k in bf.keys() & cf.keys()
                     if bf[k].get('hash') != cf[k].get('hash')
                     or bf[k].get('size') != cf[k].get('size')
                     or bf[k].get('symlink_to') != cf[k].get('symlink_to'))
    mtime_only = sorted(k for k in bf.keys() & cf.keys()
                        if k not in changed and bf[k].get('mtime_ns') != cf[k].get('mtime_ns'))
    rep = {'removed': sorted(bf.keys() - cf.keys()), 'added': sorted(cf.keys() - bf.keys()),
           'content_or_size_changed': changed, 'mtime_changed_content_same': mtime_only,
           'git_before': b['git'], 'git_after': c['git'],
           'git_identical': b['git'] == c['git'],
           'n_files_before': b['n_files'], 'n_files_after': c['n_files'],
           'coverage_note': __doc__.strip()}
    out = os.path.join(a.audit_dir, 'audit_diff.json')
    with open(out, 'x') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in rep.items()
                      if k != 'coverage_note'}, indent=1))
    print(f'-> {out}')
    bad = rep['removed'] or rep['content_or_size_changed'] or not rep['git_identical']
    sys.exit(2 if bad else 0)


if __name__ == '__main__':
    main()
