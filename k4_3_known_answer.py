# -*- coding: utf-8 -*-
"""
k4_3_known_answer.py — Ενότητες 4.3.1 και 4.3.2: έλεγχος «γνωστής απάντησης».

Αναπαράγει τον Πίνακα 9 και τα Σχήματα 25 και 26.

Το πείραμα μεταφέρει το τεχνητό προφίλ SINR του σεναρίου των δημιουργών του SALAD
(Ενότητα 4.1.2: 10 dB, άλμα στα 25 dB και επιστροφή στα 10 dB) μέσα στον βρόχο
του Sionna:
  1. Από τη διαδρομή των 28 GHz με αρχή y0 = -50 m επιλέγεται η θέση με το
     υψηλότερο ενεργό SINR (θέση 278, περίπου 25.9 dB), σε συμπιεσμένο χρόνο.
  2. Το κανάλι αυτής της θέσης αντιγράφεται σε 500 slots (στατικό κανάλι) και
     κλιμακώνεται ώστε το SINR να ακολουθεί το προφίλ 10 → 25 → 10 dB. Λόγω της
     μεθόδου EESM σε κανάλι επιλεκτικό ως προς τη συχνότητα, το ενεργό SINR
     σταθεροποιείται περίπου στα 11 και 26 dB.
  3. Τρέχουν, με τους πέντε σπόρους: SALAD με και χωρίς anti-windup, OLLA με βήματα
     0.5, 1 και 3 dB και ILLA με τέλεια εκτίμηση (Ενότητα 4.3.1).
  4. Ο OLLA ξανατρέχει με «παγωμένη» αναφορά SINR: η αναφορά μένει στην αρχική της
     τιμή σε όλα τα slots, οπότε ο OLLA μαθαίνει το άλμα μόνο από τα ACK/NACK, όπως
     στο σενάριο των δημιουργών (Ενότητα 4.3.2). Το SALAD δεν χρησιμοποιεί την
     αναφορά, οπότε τα αποτελέσματά του δεν αλλάζουν.

Χρόνος: περίπου 10 λεπτά σε GPU T4.
Έξοδος: results/known_answer.csv, ο Πίνακας 9 στην οθόνη και, στον φάκελο figures/,
τα fig25_known_answer.png και fig26_known_answer_frozen.png.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats

from la_common import (dev, SEEDS, NUM_POINTS, BLER_TARGET, NOISE_FEEDBACK_STD, P_RE, NO,
                       SALAD_REF, RESULTS_DIR, OuterLoopLinkAdaptation, InnerLoopLinkAdaptation,
                       phy_abs, normal, get_channel, route_sinr_fn, channel_block, run, reset_seeds,
                       make_la, spec, bler_pct)

FREQ, Y0 = '28GHz', -50
N_SLOTS = 500
ch = get_channel(FREQ, Y0)

# --- 1. Θέση αναφοράς: η θέση με το υψηλότερο ενεργό SINR ---
# Το ενεργό SINR κάθε slot προκύπτει από μια εκτέλεση του OLLA (βήμα 1 dB) σε
# συμπιεσμένο χρόνο, χωρίς σφάλμα στην αναφορά SINR.
reset_seeds(42)
o_ref = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=BLER_TARGET)
r_ref = run(o_ref, route_sinr_fn(ch, 1, static=True), NUM_POINTS, 1)
sinr_ok = np.where(np.isfinite(r_ref['sinr']), r_ref['sinr'], -np.inf)
i_ref = int(np.argmax(sinr_ok))
sinr_ref = float(r_ref['sinr'][i_ref])
print(f'Θέση αναφοράς: {i_ref}, ενεργό SINR {sinr_ref:.2f} dB')

# --- 2. Συνθετικό στατικό κανάλι με το προφίλ 10 → 25 → 10 dB ---
target = np.full(N_SLOTS, 10.0)
n3 = N_SLOTS // 3
target[n3:2 * n3] = 25.0
scale = torch.tensor(10 ** ((target - sinr_ref) / 20), dtype=torch.float32, device=dev)  # πλάτος
h_ref = channel_block(ch, 1, i_ref, static=True)[0]          # [13 σύμβολα, 1024 subcarriers]


def syn_sinr(k):
    """SINR ανά resource element του slot k: το κανάλι αναφοράς επί τον συντελεστή του slot."""
    h = h_ref * scale[k]
    s = (P_RE / NO) * (h.real ** 2 + h.imag ** 2)
    return s.reshape(1, *h.shape, 1, 1)


# --- 3. Επίκαιρη αναφορά SINR (Ενότητα 4.3.1) ---
rows, keep = [], {}
for s in SEEDS:
    # Ίδιο σφάλμα αναφοράς SINR για όλους τους μηχανισμούς του σπόρου
    reset_seeds(s)
    nf = normal(size=[N_SLOTS], dtype=torch.float32, std=NOISE_FEEDBACK_STD).to(dev)
    for name, aw in (('SALAD', False), ('SALAD + anti-windup', True)):
        reset_seeds(s)
        la, (core, est) = make_la(spec('ka', FREQ, 'SALAD', s, salad=SALAD_REF, aw=aw))
        r = run(la, syn_sinr, N_SLOTS, 1, noise_feedback=nf)
        rows.append(dict(test='live', algorithm=name, seed=s, SE=float(r['se'].mean()),
                         BLER=bler_pct(r['harq'])))
        if s == 42:
            keep[name] = (r, np.array(est.sinr_hist, dtype=float))
    for d in (0.5, 1.0, 3.0):
        reset_seeds(s)
        o = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=BLER_TARGET)
        o.delta_up = d
        r = run(o, syn_sinr, N_SLOTS, 1, noise_feedback=nf)
        name = f'OLLA {d:g} dB'
        rows.append(dict(test='live', algorithm=name, seed=s, SE=float(r['se'].mean()),
                         BLER=bler_pct(r['harq'])))
        if s == 42:
            keep[name] = (r, None)
    reset_seeds(s)
    il = InnerLoopLinkAdaptation(phy_abs, bler_target=BLER_TARGET)
    r = run(il, syn_sinr, N_SLOTS, 1)
    rows.append(dict(test='live', algorithm='ILLA ideal', seed=s, SE=float(r['se'].mean()),
                     BLER=bler_pct(r['harq'])))
    print(f'επίκαιρη αναφορά, σπόρος {s} ολοκληρώθηκε', flush=True)

# --- 4. «Παγωμένη» αναφορά SINR προς τον OLLA (Ενότητα 4.3.2) ---
# Η αναφορά είναι το ενεργό SINR του προηγούμενου slot συν noise_feedback. Με
# noise_feedback = c0 - (ενεργό SINR της εκτέλεσης του OLLA 1 dB, σπόρος 42), η
# αναφορά μένει πρακτικά σταθερή στην αρχική τιμή c0.
true_hist = keep['OLLA 1 dB'][0]['sinr'].astype(np.float64)
c0 = float(true_hist[0])
frozen = torch.tensor(c0 - true_hist, dtype=torch.float32, device=dev)
print(f'Η αναφορά του OLLA μένει σταθερή στα {c0:.1f} dB')
keep2 = {}
for s in SEEDS:
    for d in (0.5, 1.0, 3.0):
        reset_seeds(s)
        o = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=BLER_TARGET)
        o.delta_up = d
        r = run(o, syn_sinr, N_SLOTS, 1, noise_feedback=frozen)
        name = f'OLLA {d:g} dB, frozen report'
        rows.append(dict(test='frozen', algorithm=name, seed=s, SE=float(r['se'].mean()),
                         BLER=bler_pct(r['harq'])))
        if s == 42:
            keep2[name] = r
    print(f'παγωμένη αναφορά, σπόρος {s} ολοκληρώθηκε', flush=True)

df = pd.DataFrame(rows)
os.makedirs(RESULTS_DIR, exist_ok=True)
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
df.to_csv(os.path.join(RESULTS_DIR, 'known_answer.csv'), index=False)

# --- Πίνακας 9: μέσοι όροι πέντε σπόρων και διαφορά του SALAD ανά σπόρο (95% CI, κατανομή t) ---
def paired(a, b):
    d = (df[df.algorithm == a].set_index('seed').SE - df[df.algorithm == b].set_index('seed').SE)
    d = d.loc[SEEDS].values
    m = d.mean()
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    rel = 100 * m / df[df.algorithm == b].SE.mean()
    return f'{rel:+.1f}% [{m - h:+.3f}, {m + h:+.3f}]'


print(df.groupby('algorithm', sort=False).agg(SE=('SE', 'mean'), SE_std=('SE', 'std'),
                                              BLER=('BLER', 'mean')).round(3).to_string())
print('\nΠίνακας 9: διαφορά του SALAD από κάθε εκδοχή του OLLA')
for d in ('0.5', '1', '3'):
    print(f'  OLLA {d} dB: επίκαιρη αναφορά {paired("SALAD", f"OLLA {d} dB")}, '
          f'παγωμένη αναφορά {paired("SALAD", f"OLLA {d} dB, frozen report")}')

# --- Σχήμα 25: σπόρος 42, επίκαιρη αναφορά ---
cols = {'SALAD': 'C3', 'SALAD + anti-windup': 'C2', 'OLLA 0.5 dB': 'C4', 'OLLA 1 dB': 'C1',
        'OLLA 3 dB': 'C5'}
fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
axs[0].plot(keep['OLLA 1 dB'][0]['sinr'], color='C0', label='true SINR')
axs[0].plot(keep['SALAD'][1], color='C3', lw=1, label='SALAD: sinr_hist')
axs[0].set_ylabel('SINR [dB]')
for name, (r, _) in keep.items():
    c = cols[name]
    axs[1].plot(r['mcs'], color=c, lw=0.8, label=name)
    axs[2].plot(np.cumsum(r['harq'] == 0), color=c, label=name)
    axs[3].plot(np.convolve(r['se'], np.ones(25) / 25, mode='same'), color=c, label=name)
axs[1].set_ylabel('MCS index')
axs[2].set_ylabel('Cumulative NACKs')
axs[3].set_ylabel('SE, moving avg. [bps/Hz]')
axs[3].set_xlabel('Slot')
for ax in axs:
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7)
fig.suptitle('Known-answer test: SINR 10 → 25 → 10 dB, seed 42')
fig.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig25_known_answer.png'), dpi=150)

# --- Σχήμα 26: σπόρος 42, παγωμένη αναφορά προς τον OLLA ---
fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
r_s = keep['SALAD'][0]
axs[0].plot(r_s['mcs'], color='C3', lw=0.8, label='SALAD')
axs[1].plot(np.convolve(r_s['se'], np.ones(25) / 25, mode='same'), color='C3', label='SALAD')
for (name, r), c in zip(keep2.items(), ('C4', 'C1', 'C5')):
    axs[0].plot(r['mcs'], color=c, lw=0.8, label=name)
    axs[1].plot(np.convolve(r['se'], np.ones(25) / 25, mode='same'), color=c, label=name)
axs[0].set_ylabel('MCS index')
axs[1].set_ylabel('SE, moving avg. [bps/Hz]')
axs[1].set_xlabel('Slot')
for ax in axs:
    ax.grid(alpha=0.4)
    ax.legend(fontsize=7)
fig.suptitle('Same scenario, OLLA with frozen SINR report')
fig.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig26_known_answer_frozen.png'), dpi=150)
print('Τα Σχήματα 25 και 26 αποθηκεύτηκαν.')
