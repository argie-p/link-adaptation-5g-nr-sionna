# -*- coding: utf-8 -*-
"""
k3_2_link_adaptation.py — Ενότητα 3.2: προσαρμογή ζεύξης με ILLA και OLLA.

Αναπαράγει τα Σχήματα 4 έως 6. Ακολουθεί το tutorial του Sionna για το link
adaptation, με τις ίδιες παραμέτρους.

Σενάριο: ένας σταθμός βάσης και ένας χρήστης στη σκηνή simple_street_canyon του
Sionna RT, κάτω ζεύξη, 3.5 GHz. Ο χρήστης κινείται σε ευθεία 80 m (από y = -40 έως
y = 40 m) και η διαδρομή χωρίζεται σε 500 θέσεις. Κάθε θέση αντιστοιχεί σε ένα slot
(ο «συμπιεσμένος χρόνος» του tutorial). Το κανάλι κάθε θέσης υπολογίζεται με ray
tracing. Στόχος BLER 10%, πίνακας MCS 2 (έως 256QAM).

Τρεις εκτελέσεις:
  - ILLA με τέλεια εκτίμηση: η αναφορά SINR είναι το ενεργό SINR του προηγούμενου
    slot, χωρίς σφάλμα (Σχήμα 4),
  - ILLA με θορυβώδη εκτίμηση: στην αναφορά προστίθεται γκαουσιανό σφάλμα τυπικής
    απόκλισης 1.5 dB (Σχήμα 5),
  - OLLA με το ίδιο ακριβώς σφάλμα (Σχήμα 6, μαζί με τον ILLA).

Η συνάρτηση la_step μεταγλωττίζεται με torch.compile, όπως στο tutorial. Αυτό
επηρεάζει την ακολουθία των τυχαίων αριθμών του PHY abstraction: χωρίς torch.compile
οι αποφάσεις ACK/NACK, άρα και οι καμπύλες, αλλάζουν ελαφρά.

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.1.0, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Χρόνος: περίπου 3 λεπτά σε GPU.
Έξοδος: το BLER κάθε εκτέλεσης στην οθόνη και, στον φάκελο figures/, τα
fig04_illa_perfect.png, fig05_illa_noisy.png και fig06_illa_olla.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

import sionna
import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver, subcarrier_frequencies
from sionna.phy import config
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import dbm_to_watt, lin_to_db, db_to_lin, normal
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation
from sionna.phy.nr.utils import decode_mcs_index

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.seed = 42
sionna.phy.config.precision = 'single'

# ============================================================================
# 1. Παράμετροι (κάτω ζεύξη, σταθμός βάσης -> χρήστης)
# ============================================================================
num_slots = 500             # μία θέση της διαδρομής ανά slot
bler_target = .1
num_subcarriers = 1024
subcarrier_spacing = 30e3   # [Hz]
num_ofdm_symbols = 13
mcs_table_index = 2

num_bs = 1
num_bs_ant = 1
num_ut_ant = 1
num_streams_per_bs = num_ut_ant

# Χαμηλή ισχύς αρκεί, αφού δεν υπάρχουν παρεμβολές
bs_power_dbm = 20           # [dBm]
bs_power_watt = dbm_to_watt(bs_power_dbm).item()

temperature = 294           # [K]
no = BOLTZMANN_CONSTANT * temperature * subcarrier_spacing    # θόρυβος ανά subcarrier [W]

# Ισχύς μοιρασμένη ομοιόμορφα σε subcarriers και streams
tx_power = torch.ones([1, num_bs, num_streams_per_bs, num_ofdm_symbols, num_subcarriers],
                      device=config.device)
tx_power *= bs_power_watt / num_streams_per_bs / num_subcarriers

# Ένας χρήστης και ένας σταθμός βάσης
rx_tx_association = np.ones([1, num_bs])
stream_management = StreamManagement(rx_tx_association, num_streams_per_bs)
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing, num_tx=num_bs,
                             num_streams_per_tx=num_streams_per_bs)
frequencies = subcarrier_frequencies(num_subcarriers=num_subcarriers, subcarrier_spacing=subcarrier_spacing)

# ============================================================================
# 2. Σκηνή και ray tracing
# ============================================================================
ut_pos_start = np.array([-23, -40, 1.5])     # αρχή της διαδρομής [m]
ut_pos_end = np.array([-23, 40, 1.5])        # τέλος της διαδρομής [m]
bs_pos = np.array([32.5, 10.5, 23])          # θέση του σταθμού βάσης [m]
bs_look_at = np.array([22, -8, 0])           # σημείο στόχευσης της κεραίας

# Η συχνότητα φέροντος είναι η προεπιλογή της σκηνής, 3.5 GHz
scene = load_scene(sionna.rt.scene.simple_street_canyon)
print(f'Συχνότητα φέροντος της σκηνής: {float(np.array(scene.frequency).ravel()[0]) / 1e9:.1f} GHz')
scene.bandwidth = num_subcarriers * subcarrier_spacing
scene.tx_array = PlanarArray(num_rows=1, num_cols=num_bs_ant, pattern="tr38901", polarization='V')
scene.rx_array = PlanarArray(num_rows=1, num_cols=num_ut_ant, pattern="dipole", polarization='V')
scene.add(Transmitter("bs", position=bs_pos, look_at=bs_look_at, power_dbm=bs_power_dbm, display_radius=3))

# Η κίνηση του χρήστη προσομοιώνεται με έναν δέκτη σε κάθε θέση της διαδρομής
step = (ut_pos_end - ut_pos_start) / (num_slots - 1)
for slot in range(num_slots):
    scene.add(Receiver(f"ut{slot}", position=ut_pos_start + slot * step, display_radius=1, color=[0, 0, 0]))

paths = PathSolver()(scene, max_depth=8, refraction=False)

# Απόκριση συχνότητας (CFR) κάθε θέσης
# [num_slots, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq = paths.cfr(frequencies=frequencies, sampling_frequency=1 / resource_grid.ofdm_symbol_duration,
                   num_time_steps=resource_grid.num_ofdm_symbols, out_type="torch")

# ============================================================================
# 3. Ένα slot και ο βρόχος της προσομοίωσης
# ============================================================================
phy_abs = PHYAbstraction()


@torch.compile
def la_step(la, h, sinr_eff_feedback, harq_feedback):
    """Ένα slot: SINR, επιλογή MCS από τον μηχανισμό και αποτέλεσμα της μετάδοσης."""
    # SINR μετά από RZF precoding και LMMSE equalization (κάτω ζεύξη)
    precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
    h_eff = precoded_channel(h, tx_power=tx_power, alpha=no)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                                  stream_management=stream_management)
    # [num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx]
    sinr = lmmse_posteq_sinr(h_eff, no=no)[0, ...]

    num_allocated_re = torch.sum((sinr > 0).int(), axis=[-4, -3, -1])
    mcs_index = la(num_allocated_re=num_allocated_re, sinr_eff=sinr_eff_feedback,
                   mcs_table_index=mcs_table_index, mcs_category=1,   # 1 = κάτω ζεύξη
                   harq_feedback=harq_feedback)
    _, harq_feedback, sinr_eff, *_ = phy_abs(mcs_index, sinr=sinr, mcs_table_index=mcs_table_index,
                                             mcs_category=1)
    return mcs_index, harq_feedback, sinr_eff


def run(la, noise_feedback=None):
    """Προσομοίωση των num_slots slots. Η αναφορά SINR προς τον μηχανισμό είναι το ενεργό
    SINR του προηγούμενου slot, συν noise_feedback [dB] αν δοθεί."""
    mcs_index_hist = np.zeros(num_slots)
    harq_feedback_hist = np.zeros(num_slots)
    se_la_hist = np.zeros(num_slots)
    se_shannon_hist = np.zeros(num_slots)
    sinr_eff_db_hist = np.zeros(num_slots)

    harq_feedback = - torch.tensor(1, dtype=torch.int32)      # -1: καμία ανάδραση ακόμη
    sinr_eff_db_true = torch.tensor([0.0], device=h_freq.device)
    for i in range(num_slots):
        h = h_freq[i, ...][None, None, ...]
        if noise_feedback is not None:
            sinr_eff_db_feedback = sinr_eff_db_true + noise_feedback[i]
        else:
            sinr_eff_db_feedback = sinr_eff_db_true
        mcs_index, harq_feedback, sinr_eff_true = la_step(la, h, sinr_eff_feedback=db_to_lin(sinr_eff_db_feedback),
                                                          harq_feedback=harq_feedback)
        sinr_eff_db_true = lin_to_db(sinr_eff_true)

        # Φασματική απόδοση: bits ανά σύμβολο x code rate, μόνο αν η μετάδοση πέτυχε
        mod_order, coderate = decode_mcs_index(mcs_index, table_index=mcs_table_index, is_pusch=False)
        if harq_feedback == 1:
            se_la = mod_order.to(dtype=coderate.dtype) * coderate
        else:
            se_la = torch.tensor([0])
        se_shannon = torch.log2(1 + sinr_eff_true)          # όριο Shannon

        mcs_index_hist[i] = mcs_index[0].cpu().numpy()
        harq_feedback_hist[i] = harq_feedback[0].cpu().numpy()
        se_la_hist[i] = se_la[0].cpu().numpy()
        se_shannon_hist[i] = se_shannon[0].cpu().numpy()
        sinr_eff_db_hist[i] = sinr_eff_db_true[0].cpu().numpy()
    return sinr_eff_db_hist, mcs_index_hist, harq_feedback_hist, se_la_hist, se_shannon_hist


def plot(sinr_eff_db_hist, se_la_hist, se_shannon_hist, harq_feedback_hist,
         noise_feedback=None, fig=None, label=None, linestyle='-'):
    """Τρία πλαίσια: SINR, φασματική απόδοση και σωρευτικό BLER. Με fig προστίθεται
    μια ακόμη εκτέλεση στο ίδιο σχήμα."""
    is_first_plot = fig is None
    if fig is not None:
        axs = fig.get_axes()
    else:
        fig, axs = plt.subplots(3, 1, sharex='col', sharey='row', figsize=(5, 9))
        for ax in axs.flat:
            ax.yaxis.set_tick_params(labelleft=True)
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.grid()
    if is_first_plot:
        axs[0].plot(sinr_eff_db_hist, label='true')
        if noise_feedback is not None:
            nf = noise_feedback.cpu().numpy() if torch.is_tensor(noise_feedback) else noise_feedback
            axs[0].plot(sinr_eff_db_hist + nf, ':', label='noisy feedback')
        axs[0].set_ylabel('SINR [dB]')
        axs[0].set_xlabel('Slot')
        axs[0].legend()
    axs[1].plot(se_la_hist[:], linestyle=linestyle, label=label)
    if is_first_plot:
        axs[1].plot(se_shannon_hist, '--k', label='Shannon capacity')
    axs[1].set_ylabel('Spectral efficiency [bps/Hz]')
    axs[1].set_xlabel('Slot')
    axs[1].legend()
    if is_first_plot:
        axs[2].plot([0, num_slots - 1], [bler_target] * 2, '--k', label='BLER target')
    axs[2].plot(1 - np.cumsum(harq_feedback_hist) / np.arange(1, num_slots + 1), linestyle=linestyle, label=label)
    axs[2].set_ylabel('Achieved BLER')
    axs[2].set_xlabel('Slot')
    axs[2].legend()
    fig.tight_layout()
    return fig, axs


def report(name, harq, se):
    """Σωρευτικό BLER στο τέλος της εκτέλεσης και μέση φασματική απόδοση."""
    print(f'{name}: BLER στο τέλος {100 * (1 - harq.mean()):.1f}%, μέση φασματική απόδοση {se.mean():.3f} bits/s/Hz')


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 4. Σχήμα 4: ILLA με τέλεια εκτίμηση
# ============================================================================
illa = InnerLoopLinkAdaptation(phy_abs, bler_target=bler_target)
sinr_eff_db_hist, mcs_illa_ideal, harq_illa_ideal, se_illa_ideal, se_shannon = run(illa)
report('ILLA, τέλεια εκτίμηση', harq_illa_ideal, se_illa_ideal)
plateau = se_illa_ideal[se_illa_ideal > 0]
vals, counts = np.unique(np.round(plateau, 3), return_counts=True)
print(f'  η συχνότερη φασματική απόδοση μετάδοσης: {vals[np.argmax(counts)]:.2f} bits/s/Hz, '
      f'μέγιστο όριο Shannon {se_shannon.max():.1f} bits/s/Hz')
fig, axs = plot(sinr_eff_db_hist, se_illa_ideal, se_shannon, harq_illa_ideal, label='ILLA')
fig.suptitle('Inner Loop Link Adaptation (ILLA)\nPerfect channel estimate', fontsize=16, y=1)
fig.tight_layout()
save(fig, 'fig04_illa_perfect.png')

# ============================================================================
# 5. Σχήμα 5: ILLA με θορυβώδη εκτίμηση (σφάλμα 1.5 dB)
# ============================================================================
np.random.seed(42)
torch.manual_seed(42)
sionna.phy.config.seed = 42
noise_feedback_std = 1.5    # [dB]
noise_feedback = normal(size=[num_slots], dtype=torch.float32, std=noise_feedback_std)

sinr_eff_db_hist, mcs_illa_noisy, harq_illa_noisy, se_illa_noisy, se_shannon = run(illa, noise_feedback=noise_feedback)
report('ILLA, θορυβώδης εκτίμηση', harq_illa_noisy, se_illa_noisy)
fig, axs = plot(sinr_eff_db_hist, se_illa_noisy, se_shannon, harq_illa_noisy, noise_feedback=noise_feedback,
                label='ILLA')
fig.suptitle('Inner Loop Link Adaptation (ILLA)\nImperfect channel estimate', fontsize=16, y=1)
fig.tight_layout()
save(fig, 'fig05_illa_noisy.png')

# ============================================================================
# 6. Σχήμα 6: OLLA με το ίδιο σφάλμα, μαζί με τον ILLA
# ============================================================================
olla = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=bler_target)
np.random.seed(42)
torch.manual_seed(42)
sionna.phy.config.seed = 42
sinr_eff_db_hist, mcs_olla, harq_olla, se_olla, se_shannon = run(olla, noise_feedback=noise_feedback)
report('OLLA (βήμα 1 dB), θορυβώδης εκτίμηση', harq_olla, se_olla)
cum = 1 - np.cumsum(harq_olla) / np.arange(1, num_slots + 1)
print(f'  OLLA: σωρευτικό BLER στο slot 100: {100 * cum[99]:.1f}%, στο slot 200: {100 * cum[199]:.1f}%')

fig, axs = plot(sinr_eff_db_hist, se_illa_noisy, se_shannon, harq_illa_noisy, noise_feedback=noise_feedback,
                label='ILLA')
fig, axs = plot(sinr_eff_db_hist, se_olla, se_shannon, harq_olla, noise_feedback=noise_feedback, fig=fig,
                label='OLLA', linestyle='-.')
fig.suptitle('Imperfect CQI feedback', fontsize=16, y=1)
fig.tight_layout()
save(fig, 'fig06_illa_olla.png')
print('Τα σχήματα είναι στον φάκελο', FIG_DIR)
