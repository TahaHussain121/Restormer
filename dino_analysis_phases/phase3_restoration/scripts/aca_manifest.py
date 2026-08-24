"""Append-only manifest for the ACA layer ladder.

Exists so a future session can reconstruct the whole thing WITHOUT anyone
remembering anything: per arm, the config path, experiment name, every job id in
order, state, iterations reached, and the exact command to resume it by hand.

APPEND-ONLY BY CONSTRUCTION: every call adds a line to a .jsonl and rewrites the
human-readable .md view from the full history. Nothing is ever edited or
deleted, so a wrong entry stays visible next to its correction.

WHY A HAND-RESUME COMMAND MATTERS. The chain queues its successor from INSIDE
the job, before training. If the cluster drains hard, or a node is killed
without the epilogue running, the job can die WITHOUT its wrapper getting the
chance to queue anything -- and the chain silently stops. Re-running the arm's
chain script is always safe: chain_core refuses a second trainer on a live
experiment, treats a stale lock from a non-running job as "take over", and
basicsr auto-resumes from the highest training_states/*.state.

Usage:
  aca_manifest.py --arm <config-stem> --job <id> --event submitted|started|ended
  aca_manifest.py --refresh          # rescan disk, rewrite the .md view
"""
import argparse, datetime, glob, json, os, re, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3 = os.path.dirname(_HERE)
_REPO = os.path.dirname(os.path.dirname(_P3))
JSONL = os.path.join(_P3, 'results', 'wo2_implementation', 'aca_manifest.jsonl')
MD = os.path.join(_P3, 'results', 'wo2_implementation', 'ACA_MANIFEST.md')

ARMS = [
    ('aca_render_fixed128_L6_latent',    'Holo_aca_render_fixed128_L6_latent',    '{6}',      1),
    ('aca_render_fixed128_L36_latent',   'Holo_aca_render_fixed128_L36_latent',   '{3,6}',    2),
    ('aca_render_fixed128_L6912_latent', 'Holo_aca_render_fixed128_L6912_latent', '{6,9,12}', 3),
]

def resume_cmd(stem):
    return (f'cd {_REPO} && sbatch dino_analysis_phases/phase3_restoration/'
            f'scripts/chain_{stem}.sh')

def scan(exp):
    ed = os.path.join(_REPO, 'experiments', exp)
    cd = os.path.join(_REPO, 'experiments', f'Phase3_chain_state_{exp}')
    ck = [int(x.split('net_g_')[1].split('.')[0])
          for x in glob.glob(f'{ed}/models/net_g_*.pth')
          if x.split('net_g_')[1].split('.')[0].isdigit()]
    st = [int(os.path.basename(x).split('.')[0])
          for x in glob.glob(f'{ed}/training_states/*.state')
          if os.path.basename(x).split('.')[0].isdigit()]
    it = 0
    for f in sorted(glob.glob(f'{ed}/train_*.log')):
        m = re.findall(r'iter:\s*([\d,]+)', open(f, errors='ignore').read())
        if m: it = max(it, int(m[-1].replace(',', '')))
    return {'exists': os.path.isdir(ed), 'iter': it,
            'last_ckpt': max(ck) if ck else 0, 'last_state': max(st) if st else 0,
            'TRAINING_DONE': os.path.isfile(f'{cd}/TRAINING_DONE'),
            'CHAIN_ABORTED': os.path.isfile(f'{cd}/CHAIN_ABORTED'),
            'STABILITY_FAILURE': len(glob.glob(f'{ed}/STABILITY_FAILURE*')),
            'chain_count': (open(f'{cd}/CHAIN_COUNT').read().strip()
                            if os.path.isfile(f'{cd}/CHAIN_COUNT') else '0')}

def append(rec):
    os.makedirs(os.path.dirname(JSONL), exist_ok=True)
    rec['ts'] = datetime.datetime.now().astimezone().isoformat()
    with open(JSONL, 'a') as f:
        f.write(json.dumps(rec) + '\n')

def history():
    if not os.path.isfile(JSONL): return []
    return [json.loads(l) for l in open(JSONL) if l.strip()]

def sacct_state(job):
    try:
        o = subprocess.check_output(['sacct', '-j', str(job), '--noheader', '-P',
                                     '-o', 'State'], stderr=subprocess.DEVNULL)
        return o.decode().splitlines()[0].strip()
    except Exception:
        return 'unknown'

def refresh():
    h = history()
    L = ['# ACA LADDER MANIFEST', '',
         'Append-only. Regenerated from `aca_manifest.jsonl`; never hand-edited.',
         f'Last refresh: {datetime.datetime.now().astimezone():%Y-%m-%d %H:%M %Z}', '',
         'RUN ORDER IS aca-L6 -> aca-L36 -> aca-L6912. aca-L6 is the one that',
         'matters most: it is ONE factor from addition-render (operator only).',
         'If only one arm ever runs, it must be that one.', '']
    for stem, exp, lset, n in ARMS:
        s = scan(exp)
        jobs = [r for r in h if r.get('arm') == stem and r.get('job')]
        state = ('DONE' if s['TRAINING_DONE'] else
                 'ABORTED' if s['CHAIN_ABORTED'] else
                 'RUNNING/QUEUED' if jobs and s['exists'] else
                 'SUBMITTED' if jobs else 'NOT SUBMITTED')
        L += [f'## {exp}', '',
              f'- layers **{lset}** ({n} layer{"s" if n>1 else ""}, '
              f'{"no AFFM — a 1-layer softmax is a no-op" if n==1 else "AFFM then ACA"})',
              f'- config: `dino_analysis_phases/phase3_restoration/configs/{stem}.yml`',
              f'- state: **{state}**   chain_count {s["chain_count"]}',
              f'- iterations reached: **{s["iter"]:,}** / 300,000   '
              f'(last ckpt {s["last_ckpt"]:,}, last state {s["last_state"]:,})',
              f'- STABILITY_FAILURE files: {s["STABILITY_FAILURE"]}',
              f'- job ids in order: ' + (', '.join(
                  f'{r["job"]} ({r["event"]}, {sacct_state(r["job"])})'
                  for r in jobs) or 'none'),
              '', '  **Resume by hand (safe at any time, incl. after maintenance):**',
              '  ```bash', f'  {resume_cmd(stem)}', '  ```', '']
    L += ['---', '',
          '## If the chain stops without queueing a successor', '',
          'Assume the epilogue may NOT run when the cluster drains. Re-running',
          'the arm\'s chain script above is always safe:', '',
          '- `chain_core` refuses a second trainer while one is genuinely RUNNING,',
          '- it treats a stale `RUNNING_JOB` from a dead job as "take over",',
          '- basicsr auto-resumes from the highest `training_states/*.state`',
          '  (`basicsr/train.py:138-149`), no YAML edit needed.', '',
          '**Do NOT resume an arm whose chain wrote `CHAIN_ABORTED` or whose',
          'experiment holds a `STABILITY_FAILURE*.json`.** Record why instead.', '']
    open(MD, 'w').write('\n'.join(L) + '\n')
    print(f'wrote {MD}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm'); ap.add_argument('--job'); ap.add_argument('--event')
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    if a.arm and a.job:
        append({'arm': a.arm, 'job': int(a.job), 'event': a.event or 'submitted'})
    refresh()
