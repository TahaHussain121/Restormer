import re, sys
log = open(sys.argv[1]).read()
VF = 2000
ps = [float(x) for x in re.findall(r'# psnr: ([0-9.]+)', log)]
# basicsr runs an extra end-of-training validation, duplicating the last one
if len(ps) > 1 and ps[-1] == ps[-2]:
    ps = ps[:-1]
gam = [(int(i.replace(',', '')), float(g)) for i, g in
       re.findall(r'iter:\s+([\d,]+).*?film_g_absmax: ([0-9.e+-]+)', log)]
m = re.search(r'film_gamma_scale:\s*([0-9.]+)', log)
bound = float(m.group(1)) if m else 0.5

print("  iter    val_psnr")
for i, v in enumerate(ps):
    it = (i + 1) * VF
    tag = ("(warmup: FiLM OFF = baseline)" if it <= 5000 else
           "(ramping)" if it < 10000 else "(FiLM full)")
    print(f"  {it:>6}   {v:7.3f}  {tag}")
print("\n  |gamma|max trajectory (every 1000 iters):")
for it, g in gam:
    if it % 1000 == 0:
        print(f"    iter {it:>7}  |g|max={g:.4e}  ({100*g/bound:5.1f}% of bound)")

warm = [v for i, v in enumerate(ps) if (i + 1) * VF <= 5000]
base, fin = (max(warm) if warm else None), (ps[-1] if ps else None)
c1 = fin is not None and fin >= 18.0
c2 = base is not None and fin is not None and fin >= base - 1.0
# c3: gamma must be neither near the rail NOR still climbing fast. A run one
# step from the cliff must not pass just because its endpoint value looks fine.
last = max((i for i, _ in gam), default=0)
tail = [g for it, g in gam if it >= 0.75 * last]
head = [g for it, g in gam if it <= 0.55 * last]
gmax = max(tail) if tail else None
growth = (max(tail) / max(head)) if (tail and head and max(head) > 0) else None
c3 = gmax is not None and gmax < 0.5 * bound and (growth is None or growth < 20)
print(f"\n  baseline (best warmup val) : {base}")
print(f"  final val PSNR             : {fin}")
print(f"  |gamma|max (final quarter) : {gmax}   bound {bound}")
print(f"  |gamma| growth last/first  : {None if growth is None else round(growth, 1)}x")
print(f"  [1] final >= 18.0                     : {'PASS' if c1 else 'FAIL'}")
print(f"  [2] final >= baseline - 1             : {'PASS' if c2 else 'FAIL'}")
print(f"  [3] gamma < half bound AND not racing : {'PASS' if c3 else 'FAIL'}")
print(f"  VERDICT : {'PASS - safe to launch the full run' if (c1 and c2 and c3) else 'FAIL - do NOT launch'}")
sys.exit(0 if (c1 and c2 and c3) else 1)
