# -*- coding: utf-8 -*-
"""
k4_2_main.py — Ενότητες 4.2.2, 4.2.3 και 4.2.5: η σύγκριση των μηχανισμών στα 30 km/h.

Παράγει τα δεδομένα των Πινάκων 4, 5, 6 (στήλες πέντε σπόρων) και 7 και των
Σχημάτων 19 έως 22 και 24. Τα σχήματα και οι πίνακες φτιάχνονται μετά, με το
k4_figures.py.

Κύρια διαδρομή (y0 = -40 m), 38 slots ανά θέση (30.4 km/h), 19000 slots, και στις δύο
συχνότητες, με τους πέντε σπόρους (42, 1, 7, 123, 2024). Για κάθε σπόρο:
  - ILLA με τέλεια εκτίμηση (αναφορά SINR χωρίς σφάλμα) και με θορυβώδη εκτίμηση,
  - OLLA με βήμα 1 dB (προεπιλογή του Sionna) και με το καλύτερο βήμα,
  - SALAD με τις παραμέτρους αναφοράς, με τον καλύτερο ρυθμό μάθησης και με anti-windup.
Το καλύτερο βήμα του OLLA (0.5 dB στα 3.5 GHz, 0.25 dB στα 28 GHz) και ο καλύτερος
ρυθμός μάθησης του SALAD (0.6) επιλέχθηκαν με τον σπόρο 99 στο k4_2_sensitivity.py.
Σε όλες τις εκτελέσεις του SALAD καταγράφεται και η εσωτερική του κατάσταση
(αθροιστής, στιγμιαίος στόχος BLER, εκτίμηση SINR), από την οποία προκύπτει το Σχήμα 24.

Χρόνος: 70 εκτελέσεις, περίπου 3.5 ώρες σε GPU T4. Αν διακοπεί, συνεχίζει από εκεί
που σταμάτησε.
Έξοδος: results/main.csv και results/hist/main_*.npz.
"""
from la_common import FREQS, SEEDS, SALAD_REF, spec, run_all

BEST_OLLA = {'3p5GHz': 0.5, '28GHz': 0.25}   # [dB], από το k4_2_sensitivity.py
BEST_LR = 0.6                                # από το k4_2_sensitivity.py

specs = []
for s in SEEDS:
    for f in FREQS:
        specs += [spec('main', f, 'ILLA_ideal', s),
                  spec('main', f, 'ILLA_noisy', s),
                  spec('main', f, 'OLLA', s, delta=1.0),
                  spec('main', f, 'OLLA', s, delta=BEST_OLLA[f]),
                  spec('main', f, 'SALAD', s, salad=SALAD_REF),
                  spec('main', f, 'SALAD', s, salad=dict(SALAD_REF, lr=BEST_LR),
                       label=f'SALAD lr {BEST_LR:g}'),
                  spec('main', f, 'SALAD', s, salad=SALAD_REF, aw=True)]

if __name__ == '__main__':
    df = run_all(specs, 'main.csv')
    print(df.groupby(['freq', 'algorithm'], sort=False)[['SE', 'BLER']].mean().round(3).to_string())
