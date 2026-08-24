#!/bin/bash
# STAGE 2 of the ACA ladder: arms 2 and 3, submitted TOGETHER.
#
# Run this ONLY after arm 1 (aca-L6) has been confirmed healthy. Arm 1 is the
# diagnostic: it is the one-factor arm that matters most, and it exercises the
# SAME DinoAca block these two use. If that block had a bug, launching all
# three blind would burn ~76 h of compute across three arms before anyone saw
# it.
#
# The two arms are independent: separate experiment dirs, chain state, logs and
# gates. A failure in one cannot block the other.
set -e
cd /home/woody/iwnt/iwnt174h/thesis_dino/code/Restormer
P3=dino_analysis_phases/phase3_restoration
for s in aca_render_fixed128_L36_latent aca_render_fixed128_L6912_latent; do
    jid=$(sbatch --parsable $P3/scripts/chain_$s.sh)
    echo "submitted $s -> job $jid"
    python3 $P3/scripts/aca_manifest.py --arm $s --job $jid --event submitted
done
squeue -u $USER -o "%.10i %.17j %.9T %.11M %.14R"
