# -*- coding: utf-8 -*-
"""
k4_1_salad_reference.py — Ενότητα 4.1.2: το SALAD στο σενάριο αναφοράς των δημιουργών του.

Αναπαράγει τον Πίνακα 3 και το Σχήμα 18.

Είναι το notebook "Meet SALAD" του αποθετηρίου https://github.com/NVlabs/salad
(notebooks/meet_salad.ipynb, commit 2746947), σε μορφή script, χωρίς αλλαγές στις
παραμέτρους ή στη σειρά των υπολογισμών. Προστέθηκαν μόνο ο υπολογισμός των μέσων
τιμών του Πίνακα 3, σχόλια στα ελληνικά και η αποθήκευση του σχήματος. Όπως ο κώδικας
του SALAD, διατίθεται με την άδεια της NVIDIA (LICENSE-NVIDIA.txt), που επιτρέπει
χρήση μόνο για έρευνα ή αξιολόγηση.

Το σενάριο:
  - Το SINR είναι τεχνητό, ορθογώνιος παλμός: 10 dB, άλμα στα 25 dB και επιστροφή
    στα 10 dB, σε 1000 slots.
  - Τα ACK/NACK φτάνουν ομαδοποιημένα ανά 6 slots.
  - Δεν υπάρχει αναφορά SINR (CQI) από τον χρήστη: ο OLLA και το SALAD μαθαίνουν
    μόνο από τα ACK/NACK.
  - Το «ιδανικό» είναι ο ILLA με γνώση του πραγματικού SINR (oracle).

Δεν χρειάζεται Sionna ούτε GPU. Τρέχει σε λίγα δευτερόλεπτα.
Έξοδος: ο Πίνακας 3 στην οθόνη και figures/fig18_salad_reference.png.
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Γραμματοσειρά όπως στο notebook των δημιουργών
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# --- Ο κώδικας του SALAD (ίδιο commit με το la_common.py) ---
SALAD_DIR = os.environ.get('SALAD_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'salad'))
SALAD_COMMIT = '2746947bb4350f087915bd4512cdef1656159e49'
if not os.path.isdir(SALAD_DIR):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/NVlabs/salad.git', SALAD_DIR], check=True)
    subprocess.run(['git', '-C', SALAD_DIR, 'checkout', '-q', SALAD_COMMIT], check=True)
sys.path.append(SALAD_DIR)

from algos import SALAD, OLLA, ILLA                                      # noqa: E402
from utils import StudentSINREstimator, OLLASINREstimator, run_la, \
    get_bler_sigmoid_params, generate_rect, plot_results                  # noqa: E402
from config import BLER_FILENAME                                          # noqa: E402

# Ίδιος σπόρος με το notebook των δημιουργών
np.random.seed(11)

# --- Παράμετροι της προσομοίωσης ---
bler_target = 0.1              # μακροπρόθεσμος στόχος BLER
table_index = 2                # πίνακας MCS 2 (έως 256QAM)
cbs = 100                      # μέγεθος code block, σταθερό
n_slots = 1000
nack_report_batch_size = 6     # τα ACK/NACK φτάνουν ανά 6 slots

# Πίνακες BLER: σιγμοειδείς προσεγγίσεις και φασματική απόδοση κάθε MCS
df = pd.read_csv(os.path.join(SALAD_DIR, BLER_FILENAME))
bler_sigmoid_params, min_available_mcs = get_bler_sigmoid_params(
    df, cbs, table_index, return_mcs_min_available=True)
mcs_to_se = df[(df['category'] == 'PDSCH') & (df['table_index'] == table_index) &
               (df['CBS_num_info_bits'] == cbs)]['spectral_efficiency'].values

# --- Το SINR: ορθογώνιος παλμός ανάμεσα στα 10 και 25 dB ---
sinr_bounds = [10, 25]
sinr_true = generate_rect(n_slots, n_jumps=2, bounds=sinr_bounds)

# --- SALAD με τις παραμέτρους αναφοράς ---
sinr_est_init = 10             # αρχική εκτίμηση SINR, ίδια για OLLA και SALAD
sinr_estimator_salad = StudentSINREstimator(bler_sigmoid_params, sinr_init=sinr_est_init,
                                            learning_rate_init=1.2)
salad = SALAD(bler_target, bler_sigmoid_params, sinr_estimator_salad,
              rho_calibration=0.25,        # κατώφλι του bias score για probing
              prob_probe=0.15,             # πιθανότητα probing
              bler_target_probe=0.95,      # στόχος BLER στο probing
              k_e=0.05,                    # κέρδος του ελεγκτή (integral gain)
              score_window=15)             # slots για τον υπολογισμό του bias score

is_nack_hist, rate_hist, la_algos, mcs_hist = {}, {}, {}, {}
is_nack_hist['SALAD'], rate_hist['SALAD'], la_algos['SALAD'], mcs_hist['SALAD'] = \
    run_la(salad, sinr_true, bler_sigmoid_params, mcs_to_se=mcs_to_se,
           nack_report_batch_size=nack_report_batch_size)

# --- OLLA με τρία βήματα μετά από NACK (0.5, 1 και 3 dB) ---
for delta_nack in [0.5, 1, 3]:
    sinr_estimator_olla = OLLASINREstimator(bler_target, delta_nack=delta_nack,
                                            sinr_offset_init=sinr_est_init)
    olla = OLLA(bler_target, bler_sigmoid_params, sinr_estimator_olla)
    lab = f'OLLA_delta_{delta_nack}'
    is_nack_hist[lab], rate_hist[lab], la_algos[lab], mcs_hist[lab] = \
        run_la(olla, sinr_true, bler_sigmoid_params, mcs_to_se=mcs_to_se,
               nack_report_batch_size=nack_report_batch_size)

# --- Ιδανικό: ILLA με γνώση του πραγματικού SINR ---
illa = ILLA(bler_target=bler_target, bler_sigmoid_params=bler_sigmoid_params)
mcs_oracle = illa(sinr_true)
se_oracle = mcs_to_se[mcs_oracle] * (1 - bler_target)

# --- Πίνακας 3 ---
ideal = float(np.mean(se_oracle))
rows = [('Ιδανικό (γνώση του πραγματικού SINR)', ideal, None)]
for lab, name in [('SALAD', 'SALAD'), ('OLLA_delta_1', 'OLLA, βήμα 1 dB'),
                  ('OLLA_delta_0.5', 'OLLA, βήμα 0.5 dB'), ('OLLA_delta_3', 'OLLA, βήμα 3 dB')]:
    rows.append((name, float(np.mean(rate_hist[lab])), 100 * float(np.mean(is_nack_hist[lab]))))
print('Πίνακας 3: φασματική απόδοση [bits/s/Hz], BLER και απόσταση από το ιδανικό')
for name, se, bl in rows:
    gap = '' if bl is None else f'{100 * (se / ideal - 1):+.1f}%'
    print(f'  {name:38s} {se:.3f}  {"-" if bl is None else f"{bl:.1f}%":>6s}  {gap}')

# --- Σχήμα 18: οι χρονοσειρές, με τη συνάρτηση σχεδίασης των δημιουργών ---
legends = {'OLLA_delta_1': r"OLLA $\Delta^{\mathrm{NACK}}=1$",
           'OLLA_delta_0.5': r"OLLA $\Delta^{\mathrm{NACK}}=0.5$",
           'OLLA_delta_3': r"OLLA $\Delta^{\mathrm{NACK}}=3$", 'SALAD': 'SALAD'}
colors = {'OLLA_delta_1': 'C0', 'OLLA_delta_0.5': 'C1', 'OLLA_delta_3': 'k', 'SALAD': '#76B900'}
markers = {'OLLA_delta_1': '', 'OLLA_delta_0.5': '', 'OLLA_delta_3': '', 'SALAD': '*'}
linewidths = {k: 1 for k in legends}
fig = plot_results(sinr_true, la_algos, mcs_hist, is_nack_hist, rate_hist, bler_target,
                   mcs_oracle=mcs_oracle, min_available_mcs=min_available_mcs,
                   linewidths=linewidths, legends=legends, colors=colors, markers=markers)
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(FIG_DIR, 'fig18_salad_reference.png'), dpi=200, bbox_inches='tight')
print('Το Σχήμα 18 αποθηκεύτηκε στον φάκελο', FIG_DIR)
