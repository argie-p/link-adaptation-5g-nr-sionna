# -*- coding: utf-8 -*-
"""
k4_4_speed_compression.py — Ενότητα 4.4: η ταχύτητα του χρήστη και ο συμπιεσμένος χρόνος.

Παράγει τα δεδομένα των Πινάκων 11 και 12 και των Σχημάτων 28 και 29. Δύο μέρη:

1. Ταχύτητες (Ενότητα 4.4.1, Πίνακας 11, Σχήμα 28): σπόρος 1, κύρια διαδρομή. Ίδιο
   σενάριο με τα 30 km/h, αλλά κάθε θέση κρατά διαφορετικό πλήθος slots: 380 (3 km/h,
   μόνο στα 28 GHz), 115 (10 km/h), 19 (61 km/h) και 10 (115 km/h). Η μετατόπιση
   Doppler κλιμακώνεται με την ταχύτητα. Τα 30 km/h (σπόρος 1) προέρχονται από το
   k4_2_main.py.
2. Συμπιεσμένος χρόνος (Ενότητα 4.4.2, Πίνακας 12, Σχήμα 29): πέντε σπόροι, κύρια
   διαδρομή. Κάθε θέση είναι ένα slot, με στατικό κανάλι μέσα στο slot, όπως στο
   tutorial του Sionna για το link adaptation: ο χρήστης μετακινείται 16 cm ανά 0.5 ms,
   δηλαδή περίπου 1150 km/h. Τα 500 slots κάθε εκτέλεσης τρέχουν σε λίγα δευτερόλεπτα.

Χρόνος: περίπου 3 ώρες σε GPU T4 για τις ταχύτητες, οι μισές για τα 3 km/h, και
περίπου 15 λεπτά για τον συμπιεσμένο χρόνο. Για να τρέξει μόνο το δεύτερο μέρος:
    python k4_4_speed_compression.py compressed
Έξοδος: results/speed.csv, results/compressed.csv και results/hist/.
"""
import sys
from la_common import FREQS, SEEDS, SALAD_REF, spec, run_all

BEST_OLLA = {'3p5GHz': 0.5, '28GHz': 0.25}   # [dB], καλύτερο βήμα στα 30 km/h
SPEED_SEED = 1

speed_specs = []
for f in FREQS:
    for spp in ([380] if f == '28GHz' else []) + [115, 19, 10]:
        mechs = [spec('speed', f, 'ILLA_ideal', SPEED_SEED, spp=spp)]
        if spp != 380:                     # στα 3 km/h έτρεξε μόνο το καλύτερο βήμα του OLLA
            mechs += [spec('speed', f, 'OLLA', SPEED_SEED, spp=spp, delta=1.0)]
        mechs += [spec('speed', f, 'OLLA', SPEED_SEED, spp=spp, delta=BEST_OLLA[f]),
                  spec('speed', f, 'SALAD', SPEED_SEED, spp=spp, salad=SALAD_REF)]
        speed_specs += mechs

COMPRESSED_OLLA = [0.25, 0.5, 1.0, 2.0, 3.0]   # βήματα του OLLA στον συμπιεσμένο χρόνο [dB]
comp_specs = []
for s in SEEDS:
    for f in FREQS:
        comp_specs += [spec('comp', f, 'ILLA_ideal', s, mode='compressed'),
                       spec('comp', f, 'ILLA_noisy', s, mode='compressed')]
        comp_specs += [spec('comp', f, 'OLLA', s, mode='compressed', delta=d) for d in COMPRESSED_OLLA]
        comp_specs += [spec('comp', f, 'SALAD', s, mode='compressed', salad=SALAD_REF),
                       spec('comp', f, 'SALAD', s, mode='compressed', salad=SALAD_REF, aw=True)]

if __name__ == '__main__':
    if 'compressed' not in sys.argv[1:]:
        df = run_all(speed_specs, 'speed.csv')
        print(df.groupby(['freq', 'kmh', 'algorithm'], sort=False)[['SE', 'BLER']].mean().round(3).to_string())
    df = run_all(comp_specs, 'compressed.csv')
    print(df.groupby(['freq', 'algorithm'], sort=False)[['SE', 'BLER']].mean().round(3).to_string())
