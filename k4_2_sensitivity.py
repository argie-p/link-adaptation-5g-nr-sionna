# -*- coding: utf-8 -*-
"""
k4_2_sensitivity.py — Ενότητα 4.2.4: ευαισθησία των μηχανισμών στη ρύθμισή τους.

Παράγει τα δεδομένα του Σχήματος 23 και επιλέγει τις «καλύτερες» ρυθμίσεις του
Πίνακα 7. Κύρια διαδρομή, 30 km/h, και στις δύο συχνότητες, με τον σπόρο 99. Ο σπόρος
99 δεν ανήκει στους πέντε σπόρους της σύγκρισης, ώστε η επιλογή της καλύτερης
ρύθμισης να μη «φουσκώνει» το αποτέλεσμά της. Οι καλύτερες ρυθμίσεις
επιβεβαιώνονται μετά με τους πέντε σπόρους, στο k4_2_main.py.

Εκτελέσεις ανά συχνότητα:
  - ILLA με τέλεια εκτίμηση (σημείο αναφοράς στο σχήμα),
  - OLLA με βήματα 0.25 έως 4 dB,
  - SALAD με ρυθμούς μάθησης 0.6 έως 3, με τις υπόλοιπες παραμέτρους στις τιμές αναφοράς,
  - SALAD με τις ρυθμίσεις που είχαν βρεθεί καλύτερες στον συμπιεσμένο χρόνο
    (για να φανεί αν μεταφέρονται στα 30 km/h).

Χρόνος: 32 εκτελέσεις, περίπου 1.5 ώρα σε GPU T4.
Έξοδος: results/sensitivity.csv και results/hist/sens_*.npz.
"""
from la_common import FREQS, SEL_SEED, SALAD_REF, spec, run_all

OLLA_SWEEP = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]   # [dB]
LR_SWEEP = [0.6, 0.8, 1.2, 1.6, 2.0, 3.0]
# Οι καλύτερες ρυθμίσεις του SALAD στον συμπιεσμένο χρόνο (από grid search εκεί)
SALAD_COMPRESSED_BEST = {'3p5GHz': dict(lr=1.6, probe=0.30, rho=0.50, ke=0.02),
                         '28GHz': dict(lr=2.0, probe=0.15, rho=0.25, ke=0.02)}

specs = []
for f in FREQS:
    specs += [spec('sens', f, 'ILLA_ideal', SEL_SEED)]
    specs += [spec('sens', f, 'OLLA', SEL_SEED, delta=d) for d in OLLA_SWEEP]
    specs += [spec('sens', f, 'SALAD', SEL_SEED, salad=dict(SALAD_REF, lr=lr),
                   label=f'SALAD lr {lr:g}') for lr in LR_SWEEP]
    specs += [spec('sens', f, 'SALAD', SEL_SEED, salad=SALAD_COMPRESSED_BEST[f],
                   label='SALAD, ρύθμιση συμπιεσμένου χρόνου')]

if __name__ == '__main__':
    df = run_all(specs, 'sensitivity.csv')
    for f in FREQS:
        d = df[df.freq == f]
        o = d[d.mech == 'OLLA'].sort_values('SE')
        s = d[(d.mech == 'SALAD') & d.algorithm.str.startswith('SALAD lr')].sort_values('SE')
        print(f'{f}: καλύτερο βήμα OLLA {o.iloc[-1].delta:g} dB (SE {o.iloc[-1].SE:.3f}), '
              f'καλύτερος ρυθμός μάθησης SALAD {s.iloc[-1].lr:g} (SE {s.iloc[-1].SE:.3f})')
