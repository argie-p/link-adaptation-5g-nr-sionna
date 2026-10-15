# -*- coding: utf-8 -*-
"""
k3_5_power_control.py — Ενότητα 3.5: έλεγχος ισχύος.

Αναπαράγει τα Σχήματα 13 έως 16 και τους αριθμούς της Ενότητας 3.5. Ακολουθεί το
tutorial του Sionna για τον έλεγχο ισχύος, με τις ίδιες παραμέτρους.

Σενάριο: εξαγωνικό πλέγμα 7 κυψελών με 3 τομείς (21 σταθμοί βάσης) και απόσταση
κέντρων 200 m (προεπιλογή της 3GPP για το UMi), 5 χρήστες ανά τομέα σε απόσταση
έως 80 m από τον σταθμό τους, κανάλι UMi της 3GPP στα 3.5 GHz,
πλέγμα 10 συμβόλων OFDM x 32 subcarriers. Όλοι οι χρήστες εκπέμπουν ταυτόχρονα σε
όλους τους πόρους, με μία ροή ο καθένας.

Άνω ζεύξη: έλεγχος ισχύος ανοιχτού βρόχου με κλασματική αντιστάθμιση των απωλειών,
P = min(P_max, P0 + alpha PL + 10 log10(N_PRB)) σε dBm, με P_max = 26 dBm. Σάρωση 50
τιμών του alpha (0.5 έως 1) και 41 τιμών του P0 (-110 έως -70 dBm), 2050 συνδυασμοί.
  Σχήμα 13: 10ο εκατοστημόριο της φασματικής απόδοσης για κάθε συνδυασμό.
  Σχήμα 14: κατανομή της ισχύος εκπομπής για τρεις συνδυασμούς.
Κάτω ζεύξη: κατανομή των 46 dBm (περίπου 40 W) κάθε σταθμού στους 5 χρήστες του, με
συντελεστή δικαιοσύνης f (0, 0.5, 1) και εγγυημένο λόγο ισχύος rho (0 ή 0.3).
  Σχήμα 15: κατανομή της ισχύος εκπομπής για τρεις συνδυασμούς.
  Σχήμα 16: κατανομή της φασματικής απόδοσης.

Η συνάρτηση του SINR της άνω ζεύξης μεταγλωττίζεται με torch.compile, όπως στο
tutorial. Δεν περιέχει τυχαίους αριθμούς, οπότε το torch.compile επηρεάζει μόνο την
ταχύτητα.

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.0.1, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Σε CPU οι τυχαίοι αριθμοί (θέσεις χρηστών, κανάλι) είναι διαφορετικοί, οπότε τα
νούμερα αλλάζουν.
Χρόνος: λίγα λεπτά σε GPU, περισσότερο σε CPU (2050 υπολογισμοί του SINR).
Έξοδος: οι αριθμοί στην οθόνη και, στον φάκελο figures/, τα fig13_ul_heatmap.png,
fig14_ul_power_cdf.png, fig15_dl_power_cdf.png και fig16_dl_se_cdf.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

import sionna
from sionna.phy.utils import dbm_to_watt
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, EyePrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.channel.tr38901 import UMi, PanelArray
from sionna.phy.mimo import StreamManagement
from sionna.sys import gen_hexgrid_topology, downlink_fair_power_control, open_loop_uplink_power_control
from sionna.sys.utils import spread_across_subcarriers, get_pathloss

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.precision = 'single'
sionna.phy.config.seed = 45

# ============================================================================
# 1. Παράμετροι, τοπολογία και κανάλι
# ============================================================================
batch_size = 1
num_rings = 1
num_ut_per_sector = 5
num_ofdm_sym = 10
num_subcarriers = 32
subcarrier_spacing = 15e3   # [Hz]
carrier_frequency = 3.5e9   # [Hz]
bs_max_power_dbm = 46       # [dBm], περίπου 40 W
ut_max_power_dbm = 26       # [dBm], περίπου 398 mW
ut_max_power = dbm_to_watt(ut_max_power_dbm)
bs_max_power = dbm_to_watt(bs_max_power_dbm)
temperature = 294           # [K]
no = BOLTZMANN_CONSTANT * temperature * subcarrier_spacing     # θόρυβος ανά subcarrier [W]
max_bs_ut_dist = 80         # μέγιστη απόσταση χρήστη από τον σταθμό του [m]

bs_array = PanelArray(num_rows_per_panel=3, num_cols_per_panel=2, polarization='dual', polarization_type='VH',
                      antenna_pattern='38.901', carrier_frequency=carrier_frequency)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V',
                      antenna_pattern='omni', carrier_frequency=carrier_frequency)
num_ut_ant = ut_array.num_ant
num_bs_ant = bs_array.num_ant

topology = gen_hexgrid_topology(batch_size=batch_size, num_rings=num_rings, num_ut_per_sector=num_ut_per_sector,
                                max_bs_ut_dist=max_bs_ut_dist, scenario='umi')
ut_loc, bs_loc, *_ = topology
num_bs = bs_loc.shape[1]
num_ut = ut_loc.shape[1]
num_rx, num_tx = num_bs, num_ut          # άνω ζεύξη: ο χρήστης εκπέμπει, ο σταθμός λαμβάνει

num_streams_per_ut = num_ut_ant
num_streams_per_bs = num_streams_per_ut * num_ut_per_sector

# Κάθε χρήστης εξυπηρετείται από τον σταθμό βάσης του τομέα του
rx_tx_association_ul = np.zeros([num_rx, num_tx])
idx_fair = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                     np.arange(i1 * num_ut_per_sector, (i1 + 1) * num_ut_per_sector)])
rx_tx_association_ul[idx_fair[:, 0], idx_fair[:, 1]] = 1
stream_management_ul = StreamManagement(rx_tx_association_ul, num_streams_per_ut)
rx_tx_association_dl = rx_tx_association_ul.T        # στην κάτω ζεύξη αντιστρέφονται οι ρόλοι
stream_management_dl = StreamManagement(rx_tx_association_dl, num_streams_per_bs)

channel_model = UMi(carrier_frequency=carrier_frequency, o2i_model='low', ut_array=ut_array, bs_array=bs_array,
                    direction='uplink', enable_pathloss=True, enable_shadow_fading=True)
channel_model.set_topology(*topology)

resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym, fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing, num_tx=num_ut_per_sector,
                             num_streams_per_tx=num_streams_per_ut)
ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid)
# Άνω ζεύξη: [batch_size, num_bs, num_rx_ant, num_ut, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq_ul = ofdm_channel(batch_size)
# Κάτω ζεύξη: το ίδιο κανάλι με αντεστραμμένους πομπούς και δέκτες
h_freq_dl = torch.permute(h_freq_ul, [0, 3, 4, 1, 2, 5, 6])

# Όλοι οι χρήστες σε όλους τους πόρους
is_scheduled = torch.full([batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector,
                           num_streams_per_ut], True)
num_allocated_subcarriers = torch.full([batch_size, num_bs, num_ofdm_sym, num_ut_per_sector], num_subcarriers)

# Απώλειες διάδοσης (pathloss) προς τον σταθμό που εξυπηρετεί κάθε χρήστη
_, pathloss_serving_cell = get_pathloss(h_freq_ul,
                                        rx_tx_association=torch.tensor(rx_tx_association_ul, dtype=torch.int32))


def group_by_bs(tensor):
    """[batch, num_ut, num_ofdm_sym] -> [batch, num_bs, num_ofdm_sym, num_ut_per_sector]."""
    tensor = torch.reshape(tensor, [batch_size, num_bs, num_ut_per_sector, num_ofdm_sym])
    return torch.permute(tensor, [0, 1, 3, 2])


pathloss_serving_cell = group_by_bs(pathloss_serving_cell)


def get_cdf(values):
    """Εμπειρική αθροιστική συνάρτηση κατανομής."""
    values = np.array(values).flatten()
    return np.sort(values), np.arange(1, len(values) + 1) / len(values)


def get_se(sinr):
    """Φασματική απόδοση κάθε χρήστη: άθροισμα στις ροές, μέσος όρος στα resource elements."""
    se_per_ut = torch.sum(torch.log2(1 + sinr), axis=[4])
    return torch.mean(se_per_ut, axis=[1, 2])


# ============================================================================
# 2. Άνω ζεύξη: σάρωση των alpha και P0
# ============================================================================
@torch.compile
def get_sinr_uplink(h_freq, pathloss_serving_cell, alpha, p0_dbm, ut_max_power_dbm,
                    num_allocated_subcarriers, is_scheduled):
    """SINR της άνω ζεύξης για δεδομένα alpha και P0, και η ισχύς εκπομπής κάθε χρήστη."""
    # [batch_size, num_bs, num_ofdm_sym, num_ut_per_sector]
    ul_power_per_ut = open_loop_uplink_power_control(pathloss_serving_cell, num_allocated_subcarriers,
                                                     alpha=alpha, p0_dbm=p0_dbm,
                                                     ut_max_power_dbm=ut_max_power_dbm)
    # Ομοιόμορφα σε όλα τα subcarriers
    ul_power = spread_across_subcarriers(ul_power_per_ut, is_scheduled, num_tx=num_ut_per_sector)
    ul_power = torch.reshape(ul_power, [batch_size, num_ut, num_streams_per_ut, num_ofdm_sym, num_subcarriers])
    precoded_channel = EyePrecodedChannel(resource_grid=resource_grid, stream_management=stream_management_ul)
    h_eff = precoded_channel(h_freq, tx_power=ul_power)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                                  stream_management=stream_management_ul)
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
    # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_ut, num_streams_per_ut]
    sinr = torch.reshape(sinr, sinr.shape[:-2] + (num_ut, num_streams_per_ut))
    return sinr, ul_power_per_ut


alpha_vec = np.linspace(.5, 1, 50)          # συντελεστής αντιστάθμισης
p0_dbm_vec = np.arange(-110, -69, 1)        # επιθυμητή ισχύς λήψης [dBm]
se_per_ut_ul_mat = np.zeros([len(p0_dbm_vec), len(alpha_vec), num_ut * batch_size])
ul_power_mat = np.zeros([len(p0_dbm_vec), len(alpha_vec), num_ut * batch_size])
for idx_p0, p0_dbm in enumerate(p0_dbm_vec):
    for idx_alpha, alpha in enumerate(alpha_vec):
        sinr, ul_power_per_ut = get_sinr_uplink(h_freq_ul, pathloss_serving_cell, alpha, p0_dbm, ut_max_power_dbm,
                                                num_allocated_subcarriers, is_scheduled)
        se_per_ut_ul_mat[idx_p0, idx_alpha, :] = get_se(sinr).cpu().numpy().flatten()
        ul_power_mat[idx_p0, idx_alpha, :] = ul_power_per_ut.cpu().numpy().mean(axis=2).flatten()
    print(f'P0 = {p0_dbm} dBm ολοκληρώθηκε', flush=True)

# --- Σχήμα 13: 10ο εκατοστημόριο της φασματικής απόδοσης ---
xth_percentile = 10
metric_mat = np.zeros([len(p0_dbm_vec), len(alpha_vec)])
for idx_p0 in range(len(p0_dbm_vec)):
    for idx_alpha in range(len(alpha_vec)):
        metric_mat[idx_p0, idx_alpha] = np.nanpercentile(se_per_ut_ul_mat[idx_p0, idx_alpha, :], xth_percentile)

fig = plt.figure(figsize=(8, 6))
plt.imshow(metric_mat, aspect='auto')
plt.colorbar()
plt.ylabel(r'Target received power $P_0$ [dBm]')
plt.xlabel(r'Pathloss compensation factor $\alpha$')
plt.title(f'{xth_percentile}-th percentile of user spectral efficiency [bps/Hz]')
plt.yticks(np.arange(len(p0_dbm_vec)), p0_dbm_vec)
plt.gca().invert_yaxis()
show_every = 3
plt.xticks(np.arange(0, len(alpha_vec), show_every),
           [f'{alpha_vec[i]:.2f}' for i in range(0, len(alpha_vec), show_every)], rotation=45)
plt.yticks(np.arange(0, len(p0_dbm_vec), show_every),
           [p0_dbm_vec[i] for i in range(0, len(p0_dbm_vec), show_every)], rotation=45)
fig.savefig(os.path.join(FIG_DIR, 'fig13_ul_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
i, j = np.unravel_index(np.argmax(metric_mat), metric_mat.shape)
print(f'Ενότητα 3.5: {metric_mat.size} συνδυασμοί, το υψηλότερο 10ο εκατοστημόριο {metric_mat[i, j]:.3f} '
      f'bits/s/Hz με alpha = {alpha_vec[j]:.2f} και P0 = {p0_dbm_vec[i]} dBm')

# --- Σχήμα 14: ισχύς εκπομπής των χρηστών για τρεις συνδυασμούς ---
alpha_po_vec = [(.9, -100), (1., -100), (.9, -80)]
p_max_mw = (dbm_to_watt(ut_max_power_dbm) * 1e3).cpu().item()
fig, ax = plt.subplots()
for alpha, po in alpha_po_vec:
    # το πλησιέστερο σημείο του πλέγματος της σάρωσης
    idx_p0, idx_alpha = np.argmin(np.abs(p0_dbm_vec - po)), np.argmin(np.abs(alpha_vec - alpha))
    p_mw = ul_power_mat[idx_p0, idx_alpha, :] * 1e3
    ax.plot(*get_cdf(p_mw), label=fr'$\alpha=${alpha_vec[idx_alpha]:.2f}, $P_0$={p0_dbm_vec[idx_p0]}')
    print(f'Ενότητα 3.5: alpha = {alpha_vec[idx_alpha]:.2f}, P0 = {p0_dbm_vec[idx_p0]} dBm: '
          f'κάτω από 25 mW {100 * (p_mw < 25).mean():.0f}% των χρηστών, '
          f'στη μέγιστη ισχύ {100 * np.isclose(p_mw, p_max_mw, rtol=1e-3).mean():.0f}%')
ax.plot([p_max_mw] * 2, [0, 1], 'k--', label='max UT power')
ax.set_xlabel('Per-user uplink transmit power [mW]')
ax.set_ylabel('CDF')
ax.legend()
ax.grid()
fig.savefig(os.path.join(FIG_DIR, 'fig14_ul_power_cdf.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 3. Κάτω ζεύξη: δικαιοσύνη και εγγυημένη ισχύς
# ============================================================================
# Απώλειες προς όλους τους σταθμούς και παρεμβολές από τους άλλους σταθμούς, με
# μέγιστη ισχύ: [batch_size, num_ut, num_bs, num_ofdm_symbols], [batch_size, num_ut, num_ofdm_symbols]
pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
    h_freq_dl, rx_tx_association=torch.tensor(rx_tx_association_dl, dtype=torch.int32))
one = torch.tensor(1, dtype=pathloss_serving_cell.dtype)
interference_dl = torch.sum(one / pathloss_all_pairs, axis=-2) - one / pathloss_serving_cell
interference_dl *= bs_max_power
pathloss_serving_cell = group_by_bs(pathloss_serving_cell)
interference_dl = group_by_bs(interference_dl)


def get_sinr_downlink(guaranteed_power_ratio, fairness_dl):
    """SINR της κάτω ζεύξης και ισχύς κάθε χρήστη, για δεδομένα rho και f."""
    # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
    dl_power_per_ut, _ = downlink_fair_power_control(pathloss_serving_cell, interference_dl + no,
                                                     num_allocated_subcarriers, bs_max_power_dbm=bs_max_power_dbm,
                                                     guaranteed_power_ratio=guaranteed_power_ratio,
                                                     fairness=fairness_dl)
    dl_power = spread_across_subcarriers(dl_power_per_ut, is_scheduled, num_tx=1)
    dl_power = torch.reshape(dl_power, [batch_size, num_bs, num_streams_per_bs, num_ofdm_sym, num_subcarriers])
    precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management_dl)
    h_eff = precoded_channel(h_freq_dl, tx_power=dl_power, alpha=no)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                                  stream_management=stream_management_dl)
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
    return sinr, dl_power_per_ut


fairness_dl_vec = [0, .5, 1]                 # συντελεστής δικαιοσύνης f
guaranteed_power_ratio_vec = [0, .3]         # εγγυημένος λόγος ισχύος rho
se_per_ut_dl_mat = np.zeros([len(fairness_dl_vec), len(guaranteed_power_ratio_vec), num_ut * batch_size])
dl_power_mat = np.zeros([len(fairness_dl_vec), len(guaranteed_power_ratio_vec), num_ut * batch_size])
for idx_guar, guaranteed_power_ratio in enumerate(guaranteed_power_ratio_vec):
    for idx_fair, fairness_dl in enumerate(fairness_dl_vec):
        sinr, dl_power_per_ut = get_sinr_downlink(guaranteed_power_ratio, fairness_dl)
        se_per_ut_dl_mat[idx_fair, idx_guar, :] = get_se(sinr).cpu().numpy().flatten()
        dl_power_mat[idx_fair, idx_guar, :] = dl_power_per_ut.cpu().numpy().mean(axis=2).flatten()

# --- Σχήμα 15: ισχύς εκπομπής ανά χρήστη ---
p_bs = dbm_to_watt(bs_max_power_dbm).cpu().item()
fig, ax = plt.subplots()
for fairness, guaranteed_power_ratio in [(0, 0), (0, .3), (1, 0)]:
    idx_fair = fairness_dl_vec.index(fairness)
    idx_guar = guaranteed_power_ratio_vec.index(guaranteed_power_ratio)
    p = dl_power_mat[idx_fair, idx_guar, :]
    ax.plot(*get_cdf(p), label=fr'$f$={fairness:.1f}, $\rho$= {guaranteed_power_ratio:.1f}')
    print(f'Ενότητα 3.5: f = {fairness}, rho = {guaranteed_power_ratio}: χρήστες με μηδενική ισχύ '
          f'{100 * (p < 1e-3).mean():.0f}%, με όλη την ισχύ του σταθμού {100 * np.isclose(p, p_bs, rtol=1e-2).mean():.0f}%, '
          f'ελάχιστη {p.min():.1f} W, διάμεσος {np.median(p):.1f} W, μέγιστη {p.max():.1f} W')
ax.plot([p_bs] * 2, [0, 1], 'k--', label='max BS power')
ax.set_xlabel('Per-user downlink transmit power [W]')
ax.set_ylabel('Cumulative density function')
ax.legend()
ax.grid()
fig.savefig(os.path.join(FIG_DIR, 'fig15_dl_power_cdf.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# --- Σχήμα 16: φασματική απόδοση ανά χρήστη ---
fig, axs = plt.subplots(len(guaranteed_power_ratio_vec), figsize=(5, 4 * len(guaranteed_power_ratio_vec)))
for idx_guar, guaranteed_power_ratio in enumerate(guaranteed_power_ratio_vec):
    axs[idx_guar].set_title(r'Guaranteed power ratio $\rho$ =' + f'{guaranteed_power_ratio}')
    for idx_fair, fairness_dl in enumerate(fairness_dl_vec):
        se = se_per_ut_dl_mat[idx_fair, idx_guar, :]
        axs[idx_guar].plot(*get_cdf(se), label=f'Fairness $f$ = {fairness_dl}')
        print(f'Ενότητα 3.5: f = {fairness_dl}, rho = {guaranteed_power_ratio}: χρήστες με φασματική απόδοση '
              f'κάτω από 0.1 bits/s/Hz {100 * (se < 0.1).mean():.0f}%, διάμεσος {np.median(se):.2f} bits/s/Hz')
for ax in axs:
    ax.grid()
    ax.legend()
    ax.set_xlabel('Per-user spectral efficiency [bps/Hz]')
    ax.set_ylabel('Cumulative density function')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig16_dl_se_cdf.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Τα σχήματα είναι στον φάκελο', FIG_DIR)
