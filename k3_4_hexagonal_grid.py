# -*- coding: utf-8 -*-
"""
k3_4_hexagonal_grid.py — Ενότητα 3.4: τοπολογία εξαγωνικού πλέγματος.

Αναπαράγει τα Σχήματα 10 έως 12 και τους αριθμούς της Ενότητας 3.4. Ακολουθεί το
tutorial του Sionna για το εξαγωνικό πλέγμα, με τις ίδιες παραμέτρους.

Σενάριο: 7 εξαγωνικές κυψέλες (ένας δακτύλιος γύρω από την κεντρική) με 3 τομείς
η καθεμία, δηλαδή 21 σταθμοί βάσης, και 5 χρήστες ανά τομέα, ομοιόμορφα τυχαία
(105 χρήστες). Τα Σχήματα 10 και 11 δείχνουν το πλέγμα με απόσταση κέντρων 100 m.
Για το κανάλι η τοπολογία παράγεται ξανά κατά 3GPP, με την απόσταση κέντρων του
σεναρίου UMi (200 m). Κανάλι από το στοχαστικό μοντέλο UMi της 3GPP (TR 38.901) στα
3.5 GHz, με περιτύλιξη (wraparound). Άνω ζεύξη, πλέγμα 10 συμβόλων OFDM x 32
subcarriers, 2 ροές ανά χρήστη. Κάθε ροή εκπέμπει τυχαία σε περίπου 30% των
resource elements (sample_bernoulli με p = 0.3).

Βήματα:
  1. Το πλέγμα με τους τομείς (Σχήμα 10).
  2. Τυχαία θέση χρηστών και η αρχή της περιτύλιξης για έναν χρήστη (Σχήμα 11).
  3. Τοπολογία κατά 3GPP (απόσταση κέντρων 200 m), κανάλι OFDM και SINR μετά από
     LMMSE equalization.
  4. Αθροιστικές κατανομές του SINR και της φασματικής απόδοσης κατά Shannon (Σχήμα 12).
  5. Παράγωγος της μέσης χωρητικότητας ως προς την ισχύ εκπομπής κάθε resource
     element, με ένα backward pass (παραγωγισιμότητα του Sionna).

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.0.1, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Σε CPU οι τυχαίοι αριθμοί είναι διαφορετικοί, οπότε τα νούμερα αλλάζουν.
Χρόνος: λιγότερο από ένα λεπτό σε GPU.
Έξοδος: οι αριθμοί στην οθόνη και, στον φάκελο figures/, τα fig10_hexgrid.png,
fig11_wraparound.png και fig12_sinr_se_cdf.png.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

import sionna
from sionna.phy.utils import flatten_dims, sample_bernoulli, dbm_to_watt
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, EyePrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.mimo import StreamManagement
from sionna.sys import HexGrid, gen_hexgrid_topology
from sionna.sys.utils import spread_across_subcarriers

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.precision = 'single'
sionna.phy.config.seed = 45

# ============================================================================
# 1. Σχήμα 10: το εξαγωνικό πλέγμα
# ============================================================================
isd = 100           # απόσταση των κέντρων δύο γειτονικών κυψελών [m]
num_rings = 1       # ένας δακτύλιος γύρω από την κεντρική κυψέλη: 7 κυψέλες
cell_height = 10    # ύψος των σταθμών βάσης [m]
grid = HexGrid(isd=isd, num_rings=num_rings, cell_height=cell_height)
print(f'Κέντρα των κυψελών (X, Y, Z) [m]:\n{grid.cell_loc.cpu().numpy()}')
fig = grid.show(show_sectors=True)
fig.savefig(os.path.join(FIG_DIR, 'fig10_hexgrid.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 2. Χρήστες και Σχήμα 11: η αρχή της περιτύλιξης
# ============================================================================
batch_size = 1
num_ut_per_sector = 5
min_bs_ut_dist = 0
max_bs_ut_dist = None      # οι χρήστες μπορεί να βρεθούν οπουδήποτε στον τομέα
min_ut_height = 1.5
max_ut_height = 2

# ut_loc: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, 3]
# mirrors_per_ut_loc: για κάθε χρήστη, η θέση του πλησιέστερου αντιγράφου κάθε κυψέλης
ut_loc, mirrors_per_ut_loc, wraparound_dist = grid(batch_size, num_ut_per_sector, min_bs_ut_dist,
                                                    max_bs_ut_dist=max_bs_ut_dist,
                                                    min_ut_height=min_ut_height, max_ut_height=max_ut_height)
ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)

fig = grid.show(show_mirrors=True)
ax = fig.gca()
batch, ut = 0, 30          # ο χρήστης του σχήματος
ax.plot(*ut_loc[batch, ut, :2].cpu(), 'xk', markersize=8, markeredgewidth=2, label='user')
mirrors_per_ut_loc = flatten_dims(mirrors_per_ut_loc, num_dims=3, axis=1)
mirrors_per_ut_xy_loc = mirrors_per_ut_loc[batch, ut, :, :2].cpu()
for cell in range(mirrors_per_ut_xy_loc.shape[0]):
    ax.text(*mirrors_per_ut_xy_loc[cell, :2], str(cell))
    # Βέλος από την κυψέλη στο πλησιέστερο αντίγραφό της, αν αυτό είναι άλλο
    if np.linalg.norm(grid.cell_loc[cell, :2].cpu() - mirrors_per_ut_xy_loc[cell, :2]) > .1:
        ax.arrow(*grid.cell_loc[cell, :2].cpu(), *(mirrors_per_ut_xy_loc[cell, :2] - grid.cell_loc[cell, :2].cpu()),
                 head_length=20, head_width=20, length_includes_head=True, fc='black')
ax.set_title('Wraparound principle')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig11_wraparound.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 3. Τοπολογία κατά 3GPP, κανάλι και SINR
# ============================================================================
# Νέα τοπολογία με τις προεπιλογές της 3GPP για το σενάριο: στο UMi απόσταση
# κέντρων 200 m και ύψος σταθμών 10 m. Το πλέγμα των Σχημάτων 10 και 11 (100 m)
# δεν χρησιμοποιείται για το κανάλι.
scenario = 'umi'           # 'umi', 'uma' ή 'rma'
topology = gen_hexgrid_topology(batch_size=batch_size, num_rings=num_rings, num_ut_per_sector=num_ut_per_sector,
                                min_bs_ut_dist=min_bs_ut_dist, max_bs_ut_dist=max_bs_ut_dist, scenario=scenario,
                                los=True)
ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state, los, bs_virtual_loc = topology
num_bs = bs_loc.shape[1]
num_ut = ut_loc.shape[1]

carrier_frequency = 3.5e9  # [Hz]
num_ofdm_sym = 10
num_subcarriers = 32
subcarrier_spacing = 15e3  # [Hz]
bs_power_dbm = 56          # [dBm], χρησιμοποιείται μόνο στην κάτω ζεύξη
ut_power_dbm = 13          # [dBm]
direction = 'uplink'
temperature = 294          # [K]
o2i_model = 'low'          # μοντέλο απωλειών εξωτερικού-εσωτερικού χώρου
average_street_width = 20.     # μόνο για το RMa
average_building_height = 10.  # μόνο για το RMa

if direction == 'downlink':
    num_rx, num_tx = num_ut, num_bs
else:
    num_rx, num_tx = num_bs, num_ut
ut_power = dbm_to_watt(ut_power_dbm)
bs_power = dbm_to_watt(bs_power_dbm)
no = BOLTZMANN_CONSTANT * temperature * subcarrier_spacing      # θόρυβος ανά subcarrier [W]

# Κεραίες: 12 στοιχεία (3 x 2, διπλή πόλωση) στον σταθμό βάσης, 2 (διπλή πόλωση) στον χρήστη
bs_array = PanelArray(num_rows_per_panel=3, num_cols_per_panel=2, polarization='dual', polarization_type='VH',
                      antenna_pattern='38.901', carrier_frequency=carrier_frequency)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='dual', polarization_type='VH',
                      antenna_pattern='omni', carrier_frequency=carrier_frequency)
num_ut_ant = ut_array.num_ant
num_bs_ant = bs_array.num_ant

if scenario == 'umi':
    channel_model = UMi(carrier_frequency=carrier_frequency, o2i_model=o2i_model, ut_array=ut_array,
                        bs_array=bs_array, direction=direction, enable_pathloss=True, enable_shadow_fading=True)
elif scenario == 'uma':
    channel_model = UMa(carrier_frequency=carrier_frequency, o2i_model=o2i_model, ut_array=ut_array,
                        bs_array=bs_array, direction=direction, enable_pathloss=True, enable_shadow_fading=True)
else:
    channel_model = RMa(carrier_frequency=carrier_frequency, ut_array=ut_array, bs_array=bs_array,
                        direction=direction, average_street_width=average_street_width,
                        average_building_height=average_building_height, enable_pathloss=True,
                        enable_shadow_fading=True)
channel_model.set_topology(*topology)

num_streams_per_ut = num_ut_ant
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym, fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing, num_tx=num_ut_per_sector,
                             num_streams_per_tx=num_streams_per_ut)
ofdm_channel = GenerateOFDMChannel(channel_model, resource_grid)
# [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
h_freq = ofdm_channel(batch_size)

# Κάθε χρήστης εξυπηρετείται από τον σταθμό βάσης του τομέα του
if direction == 'downlink':
    num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
    rx_tx_association = np.zeros([num_rx, num_tx])
    idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
                    np.arange(i2 * num_ut_per_sector, (i2 + 1) * num_ut_per_sector)])
    rx_tx_association[idx[:, 0], idx[:, 1]] = 1
else:
    num_streams_per_tx = num_streams_per_ut
    rx_tx_association = np.zeros([num_rx, num_tx])
    idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                    np.arange(i1 * num_ut_per_sector, (i1 + 1) * num_ut_per_sector)])
    rx_tx_association[idx[:, 0], idx[:, 1]] = 1
stream_management = StreamManagement(rx_tx_association, num_streams_per_tx)

# Κάθε ροή εκπέμπει σε τυχαία resource elements (πιθανότητα 0.3), με ομοιόμορφη ισχύ
# [batch_size, num_ofdm_sym, num_subcarriers, num_ut, num_streams_per_ut]
is_scheduled = sample_bernoulli([batch_size, num_ofdm_sym, num_subcarriers, num_ut, num_streams_per_ut], p=.3)
tx_power_per_ut = ut_power if direction == 'uplink' else bs_power / num_ut_per_sector
tx_power_per_ut = torch.full([batch_size, num_ofdm_sym, num_ut], tx_power_per_ut, device=sionna.phy.config.device)
# [batch_size, num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers]
tx_power = spread_across_subcarriers(tx_power_per_ut, is_scheduled, num_tx=num_tx)

if direction == 'downlink':
    precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
    h_eff = precoded_channel(h_freq, tx_power=tx_power, alpha=no)
else:
    # Χωρίς precoding στην άνω ζεύξη
    precoded_channel = EyePrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
    h_eff = precoded_channel(h_freq, tx_power=tx_power)
lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid, stream_management=stream_management)
sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
# [batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
sinr = torch.reshape(sinr, sinr.shape[:-2] + (num_bs * num_ut_per_sector, num_streams_per_ut))
print(f'Ενότητα 3.4: διαστάσεις του SINR {tuple(sinr.shape)}, {sinr.numel()} τιμές')

# ============================================================================
# 4. Σχήμα 12: κατανομές SINR και φασματικής απόδοσης κατά Shannon
# ============================================================================
# Φασματική απόδοση κάθε χρήστη: άθροισμα στις ροές, μέσος όρος στα resource elements
se_shannon = torch.sum(torch.log2(1 + sinr), axis=-1)
se_shannon = torch.mean(se_shannon, axis=[-2, -3])     # [batch_size, num_ut]


def get_cdf(values):
    """Εμπειρική αθροιστική συνάρτηση κατανομής."""
    values = np.asarray(values.cpu()).flatten()
    return np.sort(values), np.arange(1, len(values) + 1) / len(values)


sinr_db_active = 10 * torch.log10(sinr[sinr > 0])        # μόνο τα resource elements με μετάδοση
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
ax[0].plot(*get_cdf(sinr_db_active))
ax[0].set_xlabel('SINR [dB]')
ax[0].set_ylabel('Cumulative density function')
ax[0].set_title('SINR across Users and Resource Elements')
ax[0].grid()
ax[1].plot(*get_cdf(se_shannon))
ax[1].grid()
ax[1].set_title('Shannon spectral efficiency ')
ax[1].set_ylabel('Cumulative density function')
ax[1].set_xlabel('Spectral Efficiency [bps/Hz]')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig12_sinr_se_cdf.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

s = sinr_db_active.cpu().numpy()
se = se_shannon.cpu().numpy().ravel()
print(f'Ενότητα 3.4: SINR από {s.min():.0f} έως {s.max():.0f} dB, διάμεσος {np.median(s):.1f} dB')
print(f'Ενότητα 3.4: χρήστες με φασματική απόδοση κάτω από 0.1 bits/s/Hz: {100 * (se < 0.1).mean():.0f}%, '
      f'μέγιστη {se.max():.1f} bits/s/Hz')

# ============================================================================
# 5. Παράγωγος της μέσης χωρητικότητας ως προς την ισχύ εκπομπής
# ============================================================================
tx_power_re = torch.ones([batch_size, num_tx, num_streams_per_tx, num_ofdm_sym, num_subcarriers],
                         device=sionna.phy.config.device, requires_grad=True)
if direction == 'downlink':
    precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
    h_eff = precoded_channel(h_freq, tx_power=tx_power_re, alpha=no)
else:
    precoded_channel = EyePrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
    h_eff = precoded_channel(h_freq, tx_power=tx_power_re)
lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid, stream_management=stream_management)
sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
capacity_re_avg = torch.mean(torch.log2(1 + sinr))       # μέση χωρητικότητα ανά ροή και resource element
capacity_re_avg.backward()
d_capacity_d_tx_power = tx_power_re.grad.cpu().numpy()
print(f'Ενότητα 3.4: παράγωγοι της χωρητικότητας ως προς την ισχύ: διαστάσεις {d_capacity_d_tx_power.shape}, '
      f'{d_capacity_d_tx_power.size} τιμές από ένα backward pass')
print('Τα σχήματα είναι στον φάκελο', FIG_DIR)
