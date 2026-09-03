# affm-render {3,6,9} — a middle point on the ADDITIVE depth curve

`Holo_affm_render_fixed128_spatial_L369_latent`

Scope: **this arm only**. Written **before the run** (2026-09-03), so the
prediction below is a pre-registration, not a description of a result.

---

## WHY THIS ARM EXISTS

See the sibling devlog `affm_render_fixed128_spatial_L36_latent.md` for the
full argument. In one line: the additive operator carries the depth finding and
has only the endpoints {6} and {3,6,9,12}; this arm is the three-depth point of
the nested chain {6} → {3,6} → {3,6,9} → {3,6,9,12}.

## THE DECISION

Config-only. Same class, same AFFM, same zero-init P, same injection point,
same recipe, same seed. The config differs from
`affm_render_fixed128_spatial_L3691_latent.yml` in three places: the name,
`dino_layers: [3, 6, 9]`, and the mean maps listing only B3, B6 and B9.

Parameter delta over E0: **295,296 + 3 × 769 = 297,603** (+0.78% over
addition-render). Verified by `smoke_tests_affm_subset.py`.

## PRE-REGISTERED PREDICTION

Against addition-render on the test split, full256:

* **+0.15 to +0.25 dB** expected. The learned {3,6,9,12} weights put B9 highest
  (0.326) and B12 lowest and least stable (0.201), so removing B12 should cost
  little: this arm is expected to land close to the four-depth arm.
* If {3,6,9} ≈ {3,6,9,12} (within 0.05): B12 is dispensable and the curve
  saturates at three depths.
* If {3,6,9} is clearly below {3,6,9,12} (≥ 0.10 lower): the deepest block
  contributes even though its weight is small.

crop128 must also be reported; the expectation is no significant loss, as for
the four-depth arm.

## THE CO-INFLATION STOP RULE

Not added, for the reason measured and recorded in the sibling devlog: every
healthy finished arm grew its latent norm 9–16× over its 5k reference, so the
specified 5× rule would abort them all. The gate is byte-identical to every
other arm's.

## SMOKE

`smoke_tests_affm_subset.py --config <this> --device cpu`; results in
`results/wo2_implementation/smoke_results_Holo_affm_render_fixed128_spatial_L369_latent.json`
and DEVLOG Step 37. No 6k GPU integration run, for the reason given in the
sibling devlog.

## STATUS

Submitted 2026-09-03 as a self-chaining a100 job (DEVLOG Step 37 has the job
id). a100 nodes were draining for a reboot at submission time.
