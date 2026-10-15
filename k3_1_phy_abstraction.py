# -*- coding: utf-8 -*-
"""
k3_1_phy_abstraction.py — Ενότητα 3.1: αφαίρεση φυσικού επιπέδου (PHY abstraction).

Αναπαράγει τα Σχήματα 1 έως 3 και τους αριθμούς της Ενότητας 3.1. Ακολουθεί το
tutorial του Sionna για το PHY abstraction, με τις ίδιες παραμέτρους.

Βήματα:
  1. Καμπύλες BLER-SINR των δεικτών MCS 15 και 17 (πίνακας MCS 1, άνω ζεύξη) για
     διάφορα μεγέθη code block (Σχήμα 1).
  2. Ανάγνωση μίας τιμής BLER από τους πίνακες: MCS 15, SINR 9 dB, code block 80 bits.
  3. Υπολογισμός νέου πίνακα BLER με προσομοίωση Monte Carlo (ένα MCS, τρία μεγέθη
     code block, 25 τιμές SINR), για να φανεί πόσο χρόνο απαιτεί.
  4. Συνθετικό σενάριο 80 χρηστών: τυχαίο μέσο SINR από -5 έως 30 dB για κάθε χρήστη
     και τυχαία διακύμανση 2 dB σε κάθε resource element. Ο ILLA επιλέγει MCS και το
     PHY abstraction δίνει το αποτέλεσμα κάθε μετάδοσης.
     Σχήμα 2: ενεργό SINR (effective SINR) έναντι του αριθμητικού μέσου.
     Σχήμα 3: επιτευχθείσα φασματική απόδοση έναντι του ορίου Shannon.

Τα βήματα 3 και 4 πρέπει να τρέξουν με αυτή τη σειρά: η προσομοίωση Monte Carlo
καταναλώνει τυχαίους αριθμούς, άρα επηρεάζει τις τιμές SINR των 80 χρηστών.

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.0.1, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Σε CPU οι τυχαίοι αριθμοί είναι διαφορετικοί, οπότε τα νούμερα αλλάζουν ελαφρά.
Χρόνος: περίπου 2 λεπτά σε GPU.
Έξοδος: οι αριθμοί στην οθόνη και, στον φάκελο figures/, τα fig01a_bler_mcs15.png,
fig01b_bler_mcs17.png, fig02_effective_sinr.png και fig03_se_shannon.png.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

import sionna
from sionna.phy.utils import db_to_lin
from sionna.phy.utils.random import uniform, normal
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.precision = 'single'
sionna.phy.config.seed = 42

# ============================================================================
# 1. Σχήμα 1: καμπύλες BLER των MCS 15 και 17
# ============================================================================
phy_abs = PHYAbstraction()

mcs_category = 0      # 0 = άνω ζεύξη (PUSCH), 1 = κάτω ζεύξη (PDSCH)
mcs_table_index = 1   # πίνακας MCS της TS 38.214: 1 ή 2 για την άνω ζεύξη, 1 έως 4 για την κάτω
mcs_index = 15

names = phy_abs.plot(plot_subset={'category': {mcs_category: {'index': {mcs_table_index:
                                  {'MCS': [mcs_index, mcs_index + 2]}}}}},
                     show=False, save_path=FIG_DIR)
for src, dst in zip(names, ['fig01a_bler_mcs15.png', 'fig01b_bler_mcs17.png']):
    os.replace(src, os.path.join(FIG_DIR, dst))

# ============================================================================
# 2. Μία τιμή από τους πίνακες BLER
# ============================================================================
snr_eff_db = 9    # ενεργό SINR [dB]
cb_size = 80      # μέγεθος code block [bits]

# Θέση της τιμής στο πλέγμα παρεμβολής των πινάκων
snr_db_idx = phy_abs.get_idx_from_grid(snr_eff_db, 'snr')
cbs_idx = phy_abs.get_idx_from_grid(cb_size, 'cbs')
bler = phy_abs.bler_table_interp[mcs_category, mcs_table_index - 1, mcs_index, cbs_idx, snr_db_idx]
print(f'Ενότητα 3.1: BLER για MCS {mcs_index}, SINR {snr_eff_db} dB, code block {cb_size} bits: '
      f'{float(bler.cpu()):.4f}')

# ============================================================================
# 3. Νέος πίνακας BLER με προσομοίωση Monte Carlo
# ============================================================================
sim_set = {'category': {1: {'index': {2: {'MCS': [16]}}}}}   # κάτω ζεύξη, πίνακας MCS 2, MCS 16
sinr_dbs = np.linspace(5, 25, 25)
cb_sizes = [24, 200, 3000]

t0 = time.time()
new_table = phy_abs.new_bler_table(sinr_dbs, cb_sizes, sim_set,
                                   max_mc_iter=15,   # το πολύ 15 επαναλήψεις Monte Carlo ανά τιμή SINR
                                   batch_size=10,
                                   verbose=True)
print(f'Ενότητα 3.1: ο νέος πίνακας BLER χρειάστηκε {time.time() - t0:.0f} s')
print('BLER του νέου πίνακα για code block 200 bits:',
      new_table['category'][1]['index'][2]['MCS'][16]['CBS'][200])

# ============================================================================
# 4. Συνθετικό σενάριο 80 χρηστών
# ============================================================================
mcs_table_index = 1
direction = 'uplink'
mcs_category = int(direction == 'downlink')
num_ut = 80
num_ofdm_symbols = 14
num_subcarriers = 1024
num_streams_per_ut = 1
bler_target = .1

# Μέσο SINR κάθε χρήστη [dB] και διακύμανση ανά subcarrier και σύμβολο OFDM
# Διαστάσεις: [num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
sinr_db = uniform([1, 1, num_ut, 1], low=-5, high=30)
sinr_db = sinr_db + normal([num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut],
                           mean=0, std=2)
sinr = db_to_lin(sinr_db)

# Επιλογή MCS από τον ILLA, με γνώση του SINR κάθε resource element
illa = InnerLoopLinkAdaptation(phy_abs, bler_target)
mcs_index = illa(sinr=sinr, mcs_table_index=mcs_table_index)
print('Δείκτες MCS:', mcs_index.cpu().numpy())

# PHY abstraction: bits που αποκωδικοποιήθηκαν, HARQ (1 = ACK, 0 = NACK), ενεργό SINR, BLER
num_decoded_bits, harq_feedback, sinr_eff, bler, tbler = \
    phy_abs(mcs_index, sinr=sinr, mcs_table_index=mcs_table_index, mcs_category=mcs_category)
sinr_eff = sinr_eff.cpu().numpy()
assert np.all(num_decoded_bits.cpu().numpy()[harq_feedback.cpu().numpy() == 0] == 0)

# --- Σχήμα 2: ενεργό SINR έναντι αριθμητικού μέσου (χρήστες σε αύξουσα σειρά ενεργού SINR) ---
ind_sort = np.argsort(sinr_eff)
sinr_t = np.transpose(sinr.cpu().numpy(), [2, 0, 1, 3]).reshape(num_ut, -1)[ind_sort, :]
fig, ax = plt.subplots()
for ut in range(num_ut):
    ax.scatter([ut] * sinr_t.shape[1], 10 * np.log10(sinr_t[ut, :]), c='y', alpha=.3, edgecolors='k',
               label='Per-stream' if ut == 0 else None)
ax.plot(10 * np.log10(sinr_eff[ind_sort]), '-k', linewidth=3, label='Effective')
ax.plot(10 * np.log10(sinr_t.mean(axis=1)), '--r', linewidth=2, label='Arithmetic mean')
ax.set_xlabel('User')
ax.set_ylabel('SINR [dB]')
ax.set_title('Per-stream vs. effective SINR')
ax.grid()
ax.legend(framealpha=1)
fig.savefig(os.path.join(FIG_DIR, 'fig02_effective_sinr.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
gap = 10 * np.log10(sinr_t.mean(axis=1)) - 10 * np.log10(sinr_eff[ind_sort])
print(f'Ενότητα 3.1: ο αριθμητικός μέσος ξεπερνά το ενεργό SINR κατά έως {gap.max():.1f} dB')

# --- Σχήμα 3: φασματική απόδοση έναντι ορίου Shannon ---
# Resource elements με ισχύ από τουλάχιστον ένα stream, ανά χρήστη
num_allocated_re = np.sum(np.sum(sinr.cpu().numpy(), axis=-1) > 0, axis=(-2, -3))
se_achieved = num_decoded_bits.cpu().numpy() / num_allocated_re   # bits ανά resource element
se_shannon = np.log2(1 + sinr_eff)                                 # όριο Shannon από το ενεργό SINR

fig, ax = plt.subplots(figsize=(5, 4))
ind_sort = np.argsort(se_shannon)
ax.plot(se_achieved[ind_sort], label='Achieved')
ax.plot(se_shannon[ind_sort], label='Shannon bound')
ax.set_title('Spectral Efficiency')
ax.set_xlabel('User')
ax.set_ylabel('SE [bits/s/Hz]')
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig03_se_shannon.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# Κορεσμός: ο πίνακας MCS 1 φτάνει έως 64QAM, οπότε η φασματική απόδοση έχει ανώτατο όριο
s = se_achieved[ind_sort]
top = s.max()
first = int(np.argmax(np.isclose(s, top)))
print(f'Ενότητα 3.1: μέγιστη επιτευχθείσα φασματική απόδοση {top:.2f} bits/s/Hz, για {int(np.isclose(s, top).sum())} '
      f'χρήστες, ο πρώτος στη θέση {first + 1} (αύξουσα σειρά του ορίου Shannon). Μέγιστο όριο Shannon '
      f'{se_shannon.max():.2f} bits/s/Hz')
print('Τα σχήματα είναι στον φάκελο', FIG_DIR)
