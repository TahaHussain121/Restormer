"""Assemble this report from saved metrics and PNGs; never run model code.

All writes are restricted to this report directory. No torch or project module
is imported. Existing experiments, predictions and logs are read only.
"""
from pathlib import Path
import os
import csv
import json
import hashlib
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RESULTS = ROOT / 'dino_analysis_phases/phase3_restoration/results'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.matplotlib_cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from PIL import Image
import markdown

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
SOURCES = {}
def record(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read_json(path):
    return json.loads(record(path).read_text())

# Ordered by the experimental questions, not by test score.
ARMS = [
 ('E0', 'Holo_E0_fixed128_baseline', 'Restormer baseline'),
 ('noisy', 'Holo_E1_addition_noisy_fixed128_spatial_B6_latent', 'Addition: noisy radar B6'),
 ('add', 'Holo_E1_addition_render_fixed128_spatial_B6_latent', 'Addition: render B6'),
 ('global', 'Holo_global_addition_render_fixed128_B6_latent', 'Global pooled render B6'),
 ('concat', 'Holo_concat_render_fixed128_spatial_B6_latent', 'Concatenation: render B6'),
 ('gate', 'Holo_gated_render_fixed128_B6_latent', 'Gated addition: render B6'),
 ('gate_noisy', 'Holo_gated_noisy_fixed128_B6_latent', 'Gated addition: noisy B6'),
 ('B3', 'Holo_addition_render_fixed128_B3_latent', 'Addition: render B3'),
 ('B9', 'Holo_addition_render_fixed128_B9_latent', 'Addition: render B9'),
 ('L36', 'Holo_affm_render_fixed128_spatial_L36_latent', 'AFFM addition: B3, B6'),
 ('L369', 'Holo_affm_render_fixed128_spatial_L369_latent', 'AFFM addition: B3, B6, B9'),
 ('L36912', 'Holo_affm_render_fixed128_spatial_L3691_latent', 'AFFM addition: B3, B6, B9, B12'),
 ('aca', 'Holo_aca_render_fixed128_L6_latent', 'ACA: B6'),
 ('aca36', 'Holo_aca_render_fixed128_L36_latent', 'ACA: B3, B6'),
 ('aca6912', 'Holo_aca_render_fixed128_L6912_latent', 'ACA: B6, B9, B12'),
 ('dinolight', 'Holo_dinolight_render_fixed128_L3691_aca_latent', 'DINOLight-inspired: four depths'),
 ('post', 'Holo_postlatent_render_fixed128_B6', 'Addition: B6 post-latent'),
 ('postB3', 'Holo_postlatent_render_fixed128_B3', 'Addition: B3 post-latent'),
 ('aca_post', 'Holo_aca_postlatent_render_fixed128_B6', 'ACA: B6 post-latent'),
 ('ml_add', 'Holo_multilevel_addition_render_fixed128_B6', 'Multi-level addition: B6'),
 ('ml_aca', 'Holo_multilevel_aca_render_fixed128_B6', 'Multi-level ACA: B6'),
]
NAMES = {k: name for k, name, _ in ARMS}
LABELS = {k: label for k, _, label in ARMS}
DATA, PER_IMAGE = {}, {}
max_mean_error = 0.0
for key, name, label in ARMS:
    DATA[key] = {}
    for protocol in ('full256', 'crop128'):
        for split, n in (('val', 339), ('test', 338)):
            cell = f'{protocol}_{split}'
            summary = read_json(RESULTS / name / 'metrics' / f'{cell}_summary.json')
            assert summary['n_images'] == n, (name, cell)
            assert summary['prediction']['experiment'] == name
            DATA[key][cell] = summary
    p = record(RESULTS / name / 'metrics/full256_test_per_image.csv')
    with p.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 338 and len({r['filename'] for r in rows}) == 338
    PER_IMAGE[key] = {r['filename']: r for r in rows}
    for metric in ('psnr_full', 'psnr_mask', 'ssim_full', 'ssim_mask'):
        difference = abs(np.mean([float(r[metric]) for r in rows]) -
                         DATA[key]['full256_test']['metrics'][metric]['mean'])
        max_mean_error = max(max_mean_error, float(difference))
        assert difference < 1e-8, (name, metric, difference)
assert all(set(rows) == set(PER_IMAGE['E0']) for rows in PER_IMAGE.values())

def value(key, cell='full256_test', metric='psnr_full'):
    return DATA[key][cell]['metrics'][metric]['mean']

def table(keys, cols):
    text = '| Configuration | ' + ' | '.join(label for label, _, _, _ in cols) + ' |\n'
    text += '|---|' + '---:|' * len(cols) + '\n'
    for key in keys:
        text += '| ' + LABELS[key] + ' | ' + ' | '.join(
            f'{value(key, cell, metric):.{digits}f}' for _, cell, metric, digits in cols) + ' |\n'
    return text

BASE_COLS = [('Full-image PSNR', 'full256_test', 'psnr_full', 3),
             ('Crop PSNR', 'crop128_test', 'psnr_full', 3)]
FULL_COLS = [('Full-image PSNR', 'full256_test', 'psnr_full', 3),
             ('Foreground PSNR', 'full256_test', 'psnr_mask', 3),
             ('Full-image SSIM', 'full256_test', 'ssim_full', 4),
             ('Crop PSNR', 'crop128_test', 'psnr_full', 3)]

def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)

def dotplot(keys, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.2, .35 * len(keys) + 1.2)),
                             sharey=True, layout='constrained')
    colors = ['#187a64' if k == 'post' else '#296aa4' for k in keys]
    y = np.arange(len(keys))
    for ax, cell, subtitle in zip(axes, ('full256_test', 'crop128_test'),
                                 ('Full image (256 x 256)', 'Matched crop (128 x 128)')):
        x = np.array([value(k, cell) for k in keys])
        ax.scatter(x, y, c=colors, s=50, zorder=3)
        for a, b in zip(x, y):
            ax.annotate(f'{a:.3f}', (a, b), xytext=(7, 0), textcoords='offset points', va='center', fontsize=9)
        span = max(float(np.ptp(x)), .5)
        ax.set_xlim(float(x.min()) - span * .12, float(x.max()) + span * .35)
        ax.set_xlabel('Mean test PSNR (dB); higher is better')
        ax.set_title(subtitle)
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, [LABELS[k] for k in keys])
    axes[0].invert_yaxis()
    fig.suptitle(title + '\nSingle training seed; point estimates, not seed uncertainty', fontsize=12)
    save(fig, name)

dotplot(['E0', 'noisy', 'global', 'add', 'post'], '01_source', 'Prior source and representation')
dotplot(['add', 'B3', 'B9', 'L36', 'L369', 'L36912'], '02_depth', 'DINO depth and feature combinations')
dotplot(['add', 'concat', 'gate', 'aca'], '03_fusion', 'Fusion at matched B6 depth and pre-latent location')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
for ax, cell, title in zip(axes, ('full256_val', 'full256_test', 'crop128_test'),
                          ('Full-image validation', 'Full-image test', 'Matched-crop test')):
    for keys, label, color in [(['add', 'post'], 'Addition', '#187a64'),
                               (['aca', 'aca_post'], 'ACA', '#b35b2d')]:
        vals = [value(k, cell) for k in keys]
        ax.plot([0, 1], vals, 'o-', label=label, color=color)
        for x, y in enumerate(vals):
            offset = -14 if label == 'Addition' and x == 0 and cell == 'full256_test' else 8
            ax.annotate(f'{y:.3f}', (x,y), xytext=(0,offset), textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks([0, 1], ['Before latent', 'After latent'])
    ax.set_xlim(-.18, 1.18)
    ax.margins(y=.3)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(title)
    ax.grid(alpha=.2)
axes[0].legend(loc='best')
fig.suptitle('Placement interacts with the fusion operator')
save(fig, '04_placement')

pair = read_json(RESULTS / 'comparisons/final_matched_pair.json')
fig, ax = plt.subplots(figsize=(9, 3.2), layout='constrained')
for i, (code, label) in enumerate([
 ('A', 'Addition: multi-level minus single-site'),
 ('B', 'ACA: multi-level minus single-site'),
 ('C', 'Multi-level: ACA minus addition')]):
    item = pair['comparisons'][code]['full256_test']
    mu = item['mean']; lo, hi = item['ci95']
    ax.errorbar(mu, i, xerr=[[mu-lo], [hi-mu]], fmt='o', color='#296aa4', capsize=5)
    ax.text(.98, i+.22, f'{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]', transform=ax.get_yaxis_transform(), ha='right', fontsize=9)
ax.set_yticks(range(3), ['Addition: extra decoder sites', 'ACA: extra decoder sites', 'Multi-level ACA vs addition'])
ax.invert_yaxis(); ax.axvline(0, color='grey', linestyle='--')
ax.set_ylim(2.55, -.4)
ax.set_xlim(-.72, .29); ax.set_xlabel('Paired full-image test PSNR difference (dB)')
ax.set_title('Final layout comparisons: mean differences and recorded 95% intervals\nIntervals describe image sampling, not training-seed variability', fontsize=11)
save(fig, '05_multilevel')

shift = read_json(RESULTS / 'render_misalignment/misalignment_summary.json')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
for ax, cell, title in zip(axes, ('full256', 'crop128'), ('Full-image validation', 'Matched-crop validation')):
    x = [r['shift_px'] for r in shift[cell]]; y = [r['psnr'] for r in shift[cell]]
    ax.plot(x, y, 'o-', color='#296aa4', label='Render-guided B6 addition')
    ax.axhline(value('E0', cell+'_val'), color='#888888', linestyle='--', label='E0 baseline')
    ax.set_xticks(x); ax.set_xlabel('Horizontal render displacement (pixels)'); ax.set_ylabel('PSNR (dB)')
    ax.set_title(title); ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Alignment sensitivity of the original pre-latent addition model')
save(fig, '06_alignment')

# Cases selected numerically before rendering: a spread by E0 difficulty and
# explicit extremes of final-model improvement. No manual image selection.
e0 = PER_IMAGE['E0']; post = PER_IMAGE['post']
ranked = sorted(e0, key=lambda f: (float(e0[f]['psnr_full']), f))
typical = [ranked[round(q * (len(ranked)-1))] for q in (.1, .5, .9)]
delta = {f: float(post[f]['psnr_full']) - float(e0[f]['psnr_full']) for f in e0}
ranked_delta = sorted(e0, key=lambda f: (delta[f], f))
extremes = [ranked_delta[0], ranked_delta[len(ranked_delta)//2], ranked_delta[-1]]
predmeta = DATA['post']['full256_test']['prediction']
gt_dir, in_dir = Path(predmeta['gt_dir']), Path(predmeta['input_dir'])

def load_image(path):
    with Image.open(record(path)) as im:
        array = np.array(im)
    assert array.shape == (256, 256) and np.issubdtype(array.dtype, np.integer), (path, array.shape, array.dtype)
    assert array.min() >= 0 and array.max() <= 65535
    return array.astype(np.float64) / 65535.0

def get_images(filename):
    pics = [load_image(in_dir / filename), load_image(gt_dir / filename)]
    for key in ('E0', 'add', 'post', 'postB3'):
        pics.append(load_image(RESULTS / NAMES[key] / 'predictions/full256_test/raw' / filename))
    return pics

def panel(cases, descriptions, name, title):
    fig, axes = plt.subplots(len(cases), 6, figsize=(15, 8.2), layout='constrained')
    cols = ['Noisy radar', 'Reference target', 'E0 baseline', 'B6 before latent', 'B6 after latent', 'B3 after latent']
    norm = PowerNorm(gamma=.5, vmin=0, vmax=1)
    for r, (filename, description) in enumerate(zip(cases, descriptions)):
        for c, (ax, pic) in enumerate(zip(axes[r], get_images(filename))):
            shown = ax.imshow(pic, cmap='inferno', norm=norm, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if c >= 2:
                key = ('E0', 'add', 'post', 'postB3')[c-2]
                ax.set_xlabel(f"{float(PER_IMAGE[key][filename]['psnr_full']):.2f} dB", fontsize=9)
            if r == 0: ax.set_title(cols[c], fontsize=10)
        axes[r,0].set_ylabel(description + '\n' + filename, fontsize=9)
    fig.colorbar(shown, ax=axes, shrink=.65, label='Normalized intensity (shared gamma 0.5 display)')
    fig.suptitle(title + '\nSaved full-image test predictions; identical [0,1] display scale for every panel', fontsize=12)
    save(fig, name)

panel(typical, ['E0 10th percentile', 'E0 median', 'E0 90th percentile'], '07_examples', 'Examples selected by baseline difficulty, not by guided-model gain')
panel(extremes, ['Largest regression', 'Median gain', 'Largest gain'], '08_failures_and_gains', 'Post-hoc range of outcomes: B6 post-latent relative to E0')

fig, axes = plt.subplots(3, 3, figsize=(9, 9), layout='constrained')
for r, filename in enumerate(extremes):
    pics = get_images(filename); gt = pics[1]
    for c, idx in enumerate((2,3,4)):
        ax=axes[r,c]
        shown=ax.imshow(np.abs(pics[idx]-gt), cmap='magma', vmin=0, vmax=.3, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(['E0 error','B6 pre-latent error','B6 post-latent error'][c])
    axes[r,0].set_ylabel(['Largest regression','Median gain','Largest gain'][r]+'\n'+filename)
fig.colorbar(shown, ax=axes, shrink=.7, label='Absolute error; values above 0.30 saturate')
fig.suptitle('Target error on the same outcome-range cases\nLinear shared error scale; no per-panel rescaling')
save(fig, '09_errors')
dotplot([k for k,_,_ in ARMS], '10_all_completed', 'All 21 completed main-study configurations')

provenance = {
 'review_date': '2026-09-17', 'data_cutoff': 'Completed main-study records available on 2026-09-17',
 'scope': '21 completed main-study configurations; excludes refiners and incomplete runs from headline tables',
 'no_model_execution': True, 'training_seeds_per_arm': 1,
 'qualitative_selection': {'baseline_quantiles': [0.1,0.5,0.9], 'baseline_rank_cases': typical,
                          'post_hoc_delta_extremes_and_median': extremes,
                          'deltas_db': {f:delta[f] for f in sorted(set(typical+extremes))}},
 'display': {'normalization':'uint16 / 65535', 'image_gamma':.5,'image_limits':[0,1], 'absolute_error_limits':[0,.3]},
 'verification': {'completed_arms':len(ARMS),'summary_cells':len(ARMS)*4,
                  'test_images_per_arm':338, 'max_summary_vs_csv_mean_difference':max_mean_error},
 'source_sha256': SOURCES}
(OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
with (OUT/'complete_results.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['experiment','label','protocol','split','n','psnr_full','psnr_foreground','ssim_full','ssim_foreground','checkpoint'])
    for key,name,label in ARMS:
        for cell,s in DATA[key].items():
            writer.writerow([name,label,s['protocol'],s['split'],s['n_images'],
                             *[s['metrics'][m]['mean'] for m in ('psnr_full','psnr_mask','ssim_full','ssim_mask')],
                             s['prediction']['weights']])

replacements={
 'SOURCE_TABLE':table(['E0','noisy','add','global'],FULL_COLS),
 'DEPTH_TABLE':table(['add','B3','B9','L36','L369','L36912'],BASE_COLS),
 'FUSION_TABLE':table(['add','concat','gate','aca'],BASE_COLS),
 'ACA_TABLE':table(['aca','aca36','aca6912','dinolight'],BASE_COLS),
 'PLACEMENT_TABLE':table(['add','post','aca','aca_post'],BASE_COLS),
 'STACKING_TABLE':table(['add','B3','post','postB3'],[('Validation full PSNR','full256_val','psnr_full',3)]+BASE_COLS),
 'FINAL_TABLE':table(['E0','add','post','postB3'],FULL_COLS+[('Foreground SSIM','full256_test','ssim_mask',4)]),
 'ALL_TABLE':table([k for k,_,_ in ARMS],FULL_COLS),
 'TYPICAL_CASES':', '.join(typical), 'EXTREME_CASES':', '.join(extremes),
 'IMAGE_COUNTS':f"{sum(v>0 for v in delta.values())}/338 improved; {sum(v<0 for v in delta.values())}/338 worsened",
 'PROVENANCE_TABLE':'\n'.join(f'| {LABELS[k]} | `{name}` |' for k,name,_ in ARMS),
}
body=(OUT/'RESULTS.template.md').read_text()
for key, replacement in replacements.items(): body=body.replace('{{'+key+'}}',replacement)
assert not re.search(r'\{\{[A-Z_]+\}\}',body)
(OUT/'RESULTS.md').write_text(body)
html=markdown.markdown(body,extensions=['tables','fenced_code','toc'])
css='''body{font-family:Georgia,serif;color:#203044;max-width:1080px;margin:40px auto;padding:0 25px;line-height:1.65}h1,h2,h3{font-family:Arial,sans-serif;line-height:1.25;color:#123c55}h1{font-size:2.1em}h2{margin-top:2.2em;border-bottom:1px solid #d6e1e7;padding-bottom:.3em}table{border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:.88em;margin:1.3em 0}th,td{border:1px solid #d5dfe7;padding:8px 10px}th{background:#eaf1f5}tr:nth-child(even){background:#f7f9fa}img{max-width:100%;height:auto;border:1px solid #e0e6ea}a{color:#176383}blockquote{border-left:4px solid #398575;margin:1em 0;padding:.5em 1.2em;background:#f1f7f5}code{font-size:.85em;overflow-wrap:anywhere}p{margin:.8em 0}em{color:#4b5968}@media print{body{font-size:10pt;max-width:none;margin:0}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit;text-decoration:none}}'''
(OUT/'RESULTS.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thesis experimental results</title><style>'+css+'</style></head><body>'+html+'</body></html>')
links=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body)
assert links and all((OUT/p).is_file() for p in links)
print(json.dumps({'report':str(OUT/'RESULTS.md'),'html':str(OUT/'RESULTS.html'),
                  'figures':len(list(FIG.glob('*.png'))),'cases':provenance['qualitative_selection'],
                  'verification':provenance['verification'],'figure_links_valid':True},indent=2))
