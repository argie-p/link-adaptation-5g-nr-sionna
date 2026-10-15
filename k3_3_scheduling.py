# -*- coding: utf-8 -*-
"""
k3_3_scheduling.py — Ενότητα 3.3: χρονοδρομολογητής αναλογικής δικαιοσύνης (PF).

Αναπαράγει τα Σχήματα 7 έως 9 και τους αριθμούς της Ενότητας 3.3. Ακολουθεί το
tutorial του Sionna για τον χρονοδρομολογητή, με τις ίδιες παραμέτρους.

Σενάριο: τρεις χρήστες σε ένα μικρό πλέγμα 12 subcarriers x 2 σύμβολα OFDM, για 200
slots. Η εφικτή φασματική απόδοση κάθε χρήστη σε κάθε resource element εξελίσσεται
στον χρόνο ως αυτοπαλίνδρομη διαδικασία πρώτης τάξης, AR(1), γύρω από διαφορετική
μέση τιμή για κάθε χρήστη. Ο χρονοδρομολογητής PF του Sionna (PFSchedulerSUMIMO,
συντελεστής μνήμης beta = 0.95) αναθέτει κάθε resource element σε έναν χρήστη. Για
σύγκριση, οι πόροι ανατίθενται και τυχαία, ομοιόμορφα.

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.0.1, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Σε CPU οι τυχαίοι αριθμοί είναι διαφορετικοί, οπότε τα νούμερα αλλάζουν.
Χρόνος: λιγότερο από ένα λεπτό.
Έξοδος: οι αριθμοί στην οθόνη και, στον φάκελο figures/, τα
fig07_scheduling_subcarrier.png, fig08_allocation_grid.png και fig09_throughput.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

import sionna
from sionna.phy import config, Block
from sionna.phy.utils.random import randint, uniform
from sionna.sys import PFSchedulerSUMIMO

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.seed = 48
sionna.phy.config.precision = 'single'

# ============================================================================
# 1. Παράμετροι και εφικτή φασματική απόδοση
# ============================================================================
num_ut = 3
num_subcarriers = 12
num_ofdm_symbols = 2
num_slots = 200

se_achievable_avg = uniform([num_ut], low=1, high=7)    # μέση εφικτή φασματική απόδοση κάθε χρήστη
rho = uniform([num_ut], low=0.8, high=0.95)             # συντελεστής της AR(1) κάθε χρήστη


class AR1(Block):
    """AR(1): x[t] = rho x[t-1] + (1 - rho) mean + w[t], με w ~ N(0, 1) και x >= 0."""

    def __init__(self, mean, rho, precision=None):
        super().__init__(precision=precision)
        self.mean = mean.to(self.dtype) if isinstance(mean, torch.Tensor) else torch.tensor(mean, dtype=self.dtype)
        self.rho = rho.to(self.dtype) if isinstance(rho, torch.Tensor) else torch.tensor(rho, dtype=self.dtype)
        self.val = self.mean.clone()

    def call(self):
        val_new = self.val * self.rho + self.mean * (1 - self.rho) + torch.randn(self.val.shape, dtype=self.dtype)
        self.val = torch.maximum(torch.zeros(1, dtype=self.dtype), val_new)
        return self.val


# Ίδια μέση τιμή και ίδιο rho σε όλα τα subcarriers ενός χρήστη
se_achievable_avg = torch.tile(se_achievable_avg[None, :], [num_subcarriers, 1])
rho = torch.tile(rho[None, :], [num_subcarriers, 1])
se_achievable_ar = AR1(se_achievable_avg, rho)

# Χρονοσειρά: [num_slots * num_ofdm_symbols, num_subcarriers, num_ut]
se_achievable_hist = np.zeros([num_slots * num_ofdm_symbols, num_subcarriers, num_ut])
for t in range(num_slots * num_ofdm_symbols):
    se_achievable_hist[t, :] = se_achievable_ar().cpu().numpy()

# ============================================================================
# 2. Χρονοδρομολόγηση PF
# ============================================================================
scheduler = PFSchedulerSUMIMO(num_ut, num_subcarriers, num_ofdm_symbols, beta=.95)

se_achieved_hist = np.zeros([num_slots, num_ut])
ut_scheduled_hist = np.zeros([num_slots, num_ofdm_symbols, num_subcarriers])
se_last_slot = torch.zeros([num_ut], dtype=config.dtype, device=config.device)   # ρυθμός του προηγούμενου slot

for slot in range(num_slots):
    # Εφικτή φασματική απόδοση του slot: [num_ofdm_symbols, num_subcarriers, num_ut]
    se_achievable_curr = torch.tensor(se_achievable_hist[slot * num_ofdm_symbols:(slot + 1) * num_ofdm_symbols, :],
                                      dtype=config.dtype, device=config.device)
    # Αποφάσεις: [num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
    is_scheduled = scheduler(se_last_slot, se_achievable_curr)

    # Ρυθμός κάθε χρήστη στο slot: άθροισμα στους πόρους που του ανατέθηκαν
    is_scheduled_re = torch.all(is_scheduled, dim=-1)
    se_last_slot = torch.sum(is_scheduled_re.float() * se_achievable_curr, dim=(-2, -3))

    # Ο χρήστης κάθε resource element
    ut_scheduled = torch.argmax(torch.sum(is_scheduled.int(), dim=-1), dim=-1)
    se_achieved_hist[slot, :] = se_last_slot.cpu().numpy()
    ut_scheduled_hist[slot, :] = ut_scheduled.cpu().numpy()

ut_scheduled_hist = np.reshape(ut_scheduled_hist, [num_slots * num_ofdm_symbols, num_subcarriers])
rate_achieved_pf = se_achieved_hist.sum(axis=0)          # συνολικός ρυθμός κάθε χρήστη
pf_metric_pf = np.sum(np.log(rate_achieved_pf))          # μετρική PF: άθροισμα λογαρίθμων

# --- Σχήμα 7: αποφάσεις σε ένα subcarrier ---
sc = 2
colors = plt.cm.Set1(np.linspace(0, 1, num_ut))
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Slot')
ax.set_ylabel('Spectral efficiency [bps/Hz]')
ax.set_title(f'Scheduling decisions on subcarrier {sc}')
for ut in range(num_ut):
    ax.plot(se_achievable_hist[:, sc, ut], color=colors[ut], label=f'User {ut + 1}')
    ind = ut_scheduled_hist[:, sc] == ut
    ax.plot(np.where(ind)[0], se_achievable_hist[ind, sc, ut], marker='x', color=colors[ut], linestyle='None',
            label=f'User {ut + 1} scheduled')
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig07_scheduling_subcarrier.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# --- Σχήμα 8: ανάθεση των πόρων σε χρόνο και συχνότητα ---
show_up_to_subcarrier = 100
show_up_to_symbol = 200
cmap = plt.get_cmap('Set1', num_ut)
fig = plt.figure(figsize=(12, 5))
plt.imshow(ut_scheduled_hist[np.arange(min(show_up_to_symbol, num_slots * num_ofdm_symbols))[:, np.newaxis],
                             np.arange(min(show_up_to_subcarrier, num_subcarriers))].T, cmap=cmap, aspect='auto')
cbar = plt.colorbar(ticks=np.arange(num_ut))
cbar.ax.set_yticklabels([f'User {ut + 1}' for ut in range(num_ut)])
cbar.set_label('User Allocation', fontsize=12)
plt.gca().invert_yaxis()
plt.xlabel('OFDM symbols')
plt.ylabel('Subcarriers')
fig.savefig(os.path.join(FIG_DIR, 'fig08_allocation_grid.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

for ut in range(num_ut):
    perc_ut = (ut_scheduled_hist == ut).sum() / (num_ofdm_symbols * num_subcarriers * num_slots) * 100
    print(f'Ενότητα 3.3: ο χρήστης {ut + 1} πήρε το {perc_ut:.2f}% των πόρων')

# ============================================================================
# 3. Σύγκριση με τυχαία ανάθεση
# ============================================================================
ut_scheduled_rand_hist = randint(0, num_ut, [num_slots * num_ofdm_symbols, num_subcarriers])
is_scheduled_rand_hist = torch.nn.functional.one_hot(ut_scheduled_rand_hist, num_ut).float()
rate_achieved_rand = np.sum(se_achievable_hist * is_scheduled_rand_hist.cpu().numpy(), axis=(-2, -3))
pf_metric_rand = np.sum(np.log(rate_achieved_rand))

print(f'Ενότητα 3.3: μετρική PF, τυχαία ανάθεση {pf_metric_rand:.2f}, PF {pf_metric_pf:.2f} '
      f'(διαφορά {pf_metric_pf - pf_metric_rand:.2f})')
for ut in range(num_ut):
    print(f'Ενότητα 3.3: χρήστης {ut + 1}, ρυθμός με PF έναντι τυχαίας ανάθεσης: '
          f'{100 * (rate_achieved_pf[ut] / rate_achieved_rand[ut] - 1):+.0f}%')

# --- Σχήμα 9: ρυθμαπόδοση ανά χρήστη, κανονικοποιημένη ως προς τον μέγιστο ρυθμό με PF ---
users = ['User 1', 'User 2', 'User 3']
fig, ax = plt.subplots(figsize=(7, 4))
bar_width = 0.25
index = np.arange(len(users))
norm_fact = max(rate_achieved_pf)
ax.bar(index, rate_achieved_pf / norm_fact, bar_width, label='PF Scheduling')
ax.bar(index + bar_width, rate_achieved_rand / norm_fact, bar_width, label='Random Scheduling')
ax.set_ylabel('(normalized) Throughput')
ax.set_title('Throughput per User for PF and Random Scheduling')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(users)
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig09_throughput.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Τα σχήματα είναι στον φάκελο', FIG_DIR)
