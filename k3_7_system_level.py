# -*- coding: utf-8 -*-
"""
k3_7_system_level.py — Ενότητα 3.7: προσομοίωση επιπέδου συστήματος.

Αναπαράγει το Σχήμα 17 και τους αριθμούς της Ενότητας 3.7. Ο προσομοιωτής είναι αυτός
του tutorial του Sionna για την προσομοίωση από άκρο σε άκρο (end-to-end) πολλών
κυψελών, με όλους τους μηχανισμούς του επιπέδου 2 ενεργούς μαζί.

Σενάριο: εξαγωνικό πλέγμα 7 κυψελών με 3 τομείς (21 σταθμοί βάσης), απόσταση κέντρων
200 m (προεπιλογή της 3GPP για το UMi) και περιτύλιξη, 10 χρήστες ανά τομέα (210
χρήστες) σε απόσταση έως 60 m από τον σταθμό τους, κανάλι
UMi της 3GPP στα 3.5 GHz, ένα σύμβολο OFDM x 128 subcarriers ανά slot. Σε κάθε slot:
  1. ενημέρωση του καναλιού: νέα πραγματοποίηση κάθε 100 slots και, σε κάθε slot,
     πολλαπλασιαστική διάλειψη που εξελίσσεται ως διαδικασία AR(1),
  2. χρονοδρομολόγηση PF με βάση την εκτιμώμενη εφικτή απόδοση,
  3. έλεγχος ισχύος (ανοιχτού βρόχου στην άνω ζεύξη, δίκαιη κατανομή στην κάτω),
  4. SINR μετά από precoding (RZF στην κάτω ζεύξη) και LMMSE equalization,
  5. προσαρμογή ζεύξης με OLLA και PHY abstraction (ACK/NACK),
  6. καταγραφή των αποτελεσμάτων και μετακίνηση των χρηστών.

Σενάρια: 84 = 2 κατευθύνσεις (κάτω και άνω ζεύξη) x 7 στόχοι BLER x 6 βήματα του
OLLA, 20 slots το καθένα. Κάθε σενάριο ξεκινά με νέα πραγματοποίηση του καναλιού,
οπότε μέσα στα 20 slots του αλλάζει μόνο η διάλειψη AR(1). Τα στατιστικά
υπολογίζονται μόνο από τα slots όπου ο χρήστης εξυπηρετήθηκε (HARQ διάφορο του -1).

Ο βρόχος της προσομοίωσης μεταγλωττίζεται με torch.compile. Η πρώτη κλήση κάθε
προσομοιωτή (ένας ανά κατεύθυνση) διαρκεί αρκετά λεπτά, γιατί τότε γίνεται η
μεταγλώττιση. Οι επόμενες κλήσεις είναι γρήγορες, εφόσον το πλήθος των slots μένει ίδιο.

Προέλευση: tutorial του Sionna (Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES,
άδεια Apache 2.0). Αλλαγές: μορφή script, σχόλια στα ελληνικά, αποθήκευση των σχημάτων
και εκτύπωση των αριθμών της εργασίας.

Εκδόσεις της εργασίας: Sionna 2.0.1, PyTorch 2.11, GPU NVIDIA T4 (Google Colab).
Σε CPU οι τυχαίοι αριθμοί είναι διαφορετικοί, οπότε τα νούμερα αλλάζουν.
Χρόνος: περίπου 16 λεπτά σε GPU.
Έξοδος: οι αριθμοί στην οθόνη και figures/fig17_topology.png.
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sionna
from sionna.sys.utils import spread_across_subcarriers
from sionna.sys import PHYAbstraction, OuterLoopLinkAdaptation, InnerLoopLinkAdaptation, gen_hexgrid_topology, \
    get_pathloss, open_loop_uplink_power_control, downlink_fair_power_control, get_num_hex_in_grid, \
    PFSchedulerSUMIMO
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import db_to_lin, dbm_to_watt, insert_dims
from sionna.phy import Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, EyePrecodedChannel, LMMSEPostEqualizationSINR

print('Sionna', sionna.__version__, '| PyTorch', torch.__version__, '| GPU:',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'καμία')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sionna.phy.config.seed = 42
torch.manual_seed(42)
np.random.seed(42)
sionna.phy.config.precision = 'single'


# ============================================================================
# 1. Κανάλι: νέα πραγματοποίηση κάθε coherence_time slots και διάλειψη AR(1) σε κάθε slot
# ============================================================================
class ChannelMatrix(Block):
    def __init__(self, resource_grid, batch_size, num_rx, num_tx, coherence_time, precision=None, device=None):
        super().__init__(precision=precision, device=device)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        self.num_rx = num_rx
        self.num_tx = num_tx
        self._init_fading()

    def _init_fading(self):
        """Αρχική κατάσταση της διάλειψης (buffers, ώστε να λειτουργεί με torch.compile)."""
        if not hasattr(self, '_rho_fading_initialized'):
            self.register_buffer('rho_fading', torch.empty([self.batch_size, self.num_rx, self.num_tx],
                                                           dtype=self.dtype, device=self.device).uniform_(0.95, 0.99),
                                 persistent=False)
            self.register_buffer('fading', torch.ones([self.batch_size, self.num_rx, self.num_tx],
                                                      dtype=self.dtype, device=self.device), persistent=False)
            self._rho_fading_initialized = True
        else:
            self.fading.fill_(1.0)

    def reset(self):
        """Επαναφορά της διάλειψης στην αρχή κάθε προσομοίωσης."""
        self._init_fading()

    def __call__(self, channel_model):
        return self.call(channel_model)

    def call(self, channel_model):
        """Νέα πραγματοποίηση του καναλιού OFDM."""
        return GenerateOFDMChannel(channel_model, self.resource_grid)(self.batch_size)

    def update(self, channel_model, h_freq, slot):
        """Νέο κανάλι κάθε coherence_time slots. Η νέα πραγματοποίηση υπολογίζεται σε κάθε
        slot, όπως στο tutorial, ώστε η ακολουθία των τυχαίων αριθμών να μένει ίδια."""
        h_freq_new = self.call(channel_model)
        if (slot % self.coherence_time) == 0:
            h_freq = h_freq_new
        return h_freq

    def apply_fading(self, h_freq):
        """Διάλειψη ως διαδικασία AR(1) πάνω στο κανάλι."""
        noise = torch.empty_like(self.fading).uniform_(-0.1, 0.1)
        new_fading = 1.0 - self.rho_fading + self.rho_fading * self.fading + noise
        new_fading = torch.clamp(new_fading, min=1e-6)
        self.fading.copy_(new_fading)
        fading_expand = insert_dims(self.fading, 1, axis=2)
        fading_expand = insert_dims(fading_expand, 3, axis=4)
        return torch.sqrt(fading_expand).to(self.cdtype) * h_freq


# ============================================================================
# 2. Βοηθητικές συναρτήσεις: ροές, SINR, εφικτή απόδοση, καταγραφή
# ============================================================================
def get_stream_management(direction, num_rx, num_tx, num_streams_per_ut, num_ut_per_sector):
    """Ποιες ροές προορίζονται για κάθε δέκτη: κάθε χρήστης με τον σταθμό του τομέα του."""
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
    return StreamManagement(rx_tx_association, num_streams_per_tx)


def get_sinr(tx_power, stream_management, no, direction, h_freq_fading, num_bs, num_ut_per_sector,
             num_streams_per_ut, resource_grid):
    """SINR μετά από precoding (RZF στην κάτω ζεύξη, κανένα στην άνω) και LMMSE equalization.
    Έξοδος: [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]."""
    s = tx_power.shape
    tx_power = torch.reshape(tx_power, [s[0], s[1] * s[2]] + list(s[3:]))
    if direction == 'downlink':
        precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power, alpha=no)
    else:
        precoded_channel = EyePrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading, tx_power=tx_power)
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid, stream_management=stream_management)
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)
    sinr = torch.reshape(sinr, list(sinr.shape[:-2]) + [num_bs * num_ut_per_sector, num_streams_per_ut])
    sinr = torch.reshape(sinr, list(sinr.shape[:-2]) + [num_bs, num_ut_per_sector, num_streams_per_ut])
    return torch.permute(sinr, [0, 3, 1, 2, 4, 5])


def estimate_achievable_rate(sinr_eff_db_last, num_ofdm_sym, num_subcarriers):
    """Εκτίμηση της εφικτής απόδοσης για τον χρονοδρομολογητή, από το τελευταίο ενεργό SINR."""
    rate = torch.log2(torch.tensor(1.0, dtype=sinr_eff_db_last.dtype) + db_to_lin(sinr_eff_db_last))
    rate = insert_dims(rate, 2, axis=-2)
    return torch.tile(rate, [1, 1, num_ofdm_sym, num_subcarriers, 1])


class TensorList:
    """Προδεσμευμένος πίνακας για την καταγραφή ανά slot (χωρίς λίστες που αλλάζουν
    μέγεθος, κάτι που θα ανάγκαζε το torch.compile να ξαναμεταγλωττίζει)."""

    def __init__(self, size, element_shape, dtype=torch.float32, device=None):
        self.data = torch.zeros([size] + list(element_shape), dtype=dtype, device=device)

    def write(self, index, value):
        self.data[index] = value
        return self

    def stack(self):
        return self.data


def init_result_history(batch_size, num_slots, num_bs, num_ut_per_sector, device=None):
    hist = {}
    for key in ['pathloss_serving_cell', 'tx_power', 'olla_offset', 'sinr_eff', 'pf_metric', 'num_decoded_bits',
                'mcs_index', 'harq', 'num_allocated_re', 'mcs_optimal']:
        hist[key] = TensorList(size=num_slots, element_shape=[batch_size, num_bs, num_ut_per_sector],
                               dtype=torch.float32, device=device)
    return hist


def record_results(hist, slot, sim_failed=False, pathloss_serving_cell=None, num_allocated_re=None,
                   tx_power_per_ut=None, num_decoded_bits=None, mcs_index=None, mcs_optimal=None,
                   harq_feedback=None, olla_offset=None, sinr_eff=None, pf_metric=None, shape=None):
    """Καταγραφή των αποτελεσμάτων του slot."""
    if not sim_failed:
        for key, value in zip(['pathloss_serving_cell', 'olla_offset', 'sinr_eff', 'num_allocated_re', 'tx_power',
                               'num_decoded_bits', 'mcs_index', 'harq', 'mcs_optimal'],
                              [pathloss_serving_cell, olla_offset, sinr_eff, num_allocated_re, tx_power_per_ut,
                               num_decoded_bits, mcs_index, harq_feedback, mcs_optimal]):
            hist[key] = hist[key].write(slot, value.float())
        hist['pf_metric'] = hist['pf_metric'].write(slot, torch.mean(pf_metric, dim=[-2, -3]))
    else:
        nan_tensor = torch.full(shape, float('nan'), dtype=torch.float32)
        for key in hist:
            hist[key] = hist[key].write(slot, nan_tensor)
    return hist


# ============================================================================
# 3. Ο προσομοιωτής επιπέδου συστήματος
# ============================================================================
class SystemLevelSimulator(Block):
    def __init__(self, batch_size, num_rings, num_ut_per_sector, carrier_frequency, resource_grid, scenario,
                 direction, ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm, coherence_time, pf_beta=0.98,
                 max_bs_ut_dist=None, min_bs_ut_dist=None, temperature=294, o2i_model='low',
                 average_street_width=20.0, average_building_height=5.0, precision=None):
        super().__init__(precision=precision)
        assert scenario in ['umi', 'uma', 'rma']
        assert direction in ['uplink', 'downlink']
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.resource_grid = resource_grid
        self.num_ut_per_sector = int(num_ut_per_sector)
        self.direction = direction
        self.bs_max_power_dbm = bs_max_power_dbm
        self.ut_max_power_dbm = ut_max_power_dbm
        self.coherence_time = int(coherence_time)
        num_cells = get_num_hex_in_grid(num_rings)
        self.num_bs = num_cells * 3
        self.num_ut = self.num_bs * self.num_ut_per_sector
        self.num_ut_ant = ut_array.num_ant
        self.num_bs_ant = bs_array.num_ant
        if bs_array.polarization == 'dual':
            self.num_bs_ant *= 2
        if self.direction == 'uplink':
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
            self.num_tx_per_sector = 1
        self.num_streams_per_ut = resource_grid.num_streams_per_tx
        self._mcs_category = 1 if direction == 'downlink' else 0
        self.stream_management = get_stream_management(direction, self.num_rx, self.num_tx, self.num_streams_per_ut,
                                                       num_ut_per_sector)
        self.no = torch.tensor(BOLTZMANN_CONSTANT * temperature * resource_grid.subcarrier_spacing,
                               dtype=self.dtype, device=self.device)
        self.slot_duration = resource_grid.ofdm_symbol_duration * resource_grid.num_ofdm_symbols
        self._setup_channel_model(scenario, carrier_frequency, o2i_model, ut_array, bs_array, average_street_width,
                                  average_building_height)
        self._setup_topology(num_rings, min_bs_ut_dist, max_bs_ut_dist)
        self.phy_abs = PHYAbstraction(precision=self.precision, device=self.device)
        self.olla = OuterLoopLinkAdaptation(self.phy_abs, self.num_ut_per_sector,
                                            batch_size=[self.batch_size, self.num_bs], precision=self.precision,
                                            device=self.device)
        self.illa = InnerLoopLinkAdaptation(self.phy_abs, bler_target=0.1, precision=self.precision,
                                            device=self.device)
        self.scheduler = PFSchedulerSUMIMO(self.num_ut_per_sector, resource_grid.fft_size,
                                           resource_grid.num_ofdm_symbols, batch_size=[self.batch_size, self.num_bs],
                                           num_streams_per_ut=self.num_streams_per_ut, beta=pf_beta,
                                           precision=self.precision, device=self.device)
        self.channel_matrix = ChannelMatrix(self.resource_grid, self.batch_size, self.num_rx, self.num_tx,
                                            self.coherence_time, precision=self.precision, device=self.device)
        self._rx_tx_association = torch.tensor(self.stream_management.rx_tx_association, device=self.device)

    def _setup_channel_model(self, scenario, carrier_frequency, o2i_model, ut_array, bs_array,
                             average_street_width, average_building_height):
        common = dict(carrier_frequency=carrier_frequency, ut_array=ut_array, bs_array=bs_array,
                      direction=self.direction, enable_pathloss=True, enable_shadow_fading=True,
                      precision=self.precision)
        if scenario == 'umi':
            self.channel_model = UMi(o2i_model=o2i_model, **common)
        elif scenario == 'uma':
            self.channel_model = UMa(o2i_model=o2i_model, **common)
        else:
            self.channel_model = RMa(average_street_width=average_street_width,
                                     average_building_height=average_building_height, **common)

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, self.ut_velocities, self.in_state, \
            self.los, self.bs_virtual_loc, self.grid = gen_hexgrid_topology(
                batch_size=self.batch_size, num_rings=num_rings, num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist, max_bs_ut_dist=max_bs_ut_dist, scenario=self.scenario, los=True,
                return_grid=True, precision=self.precision)
        self.channel_model.set_topology(self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations,
                                        self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc)

    def _reset_olla(self, bler_target, olla_delta_up):
        """Νέος στόχος BLER και βήμα του OLLA (εκτός του μεταγλωττισμένου κώδικα)."""
        self.olla.reset()
        self.olla.bler_target = bler_target
        self.olla.delta_up = olla_delta_up
        self.illa.bler_target = bler_target

    def _init_feedback_tensors(self):
        self.channel_matrix.reset()
        last_harq_feedback = -torch.ones([self.batch_size, self.num_bs, self.num_ut_per_sector],
                                         dtype=torch.int32, device=self.device)
        sinr_eff_feedback = torch.ones([self.batch_size, self.num_bs, self.num_ut_per_sector],
                                       dtype=self.dtype, device=self.device)
        num_decoded_bits = torch.zeros([self.batch_size, self.num_bs, self.num_ut_per_sector],
                                       dtype=torch.int32, device=self.device)
        return last_harq_feedback, sinr_eff_feedback, num_decoded_bits

    def _group_by_sector(self, tensor):
        tensor = torch.reshape(tensor, [self.batch_size, self.num_bs, self.num_ut_per_sector,
                                        self.resource_grid.num_ofdm_symbols])
        return torch.permute(tensor, [0, 1, 3, 2])

    def call(self, num_slots, alpha_ul, p0_dbm_ul, bler_target, olla_delta_up, mcs_table_index=1, fairness_dl=0,
             guaranteed_power_ratio_dl=0.5):
        """Μία προσομοίωση num_slots slots με δεδομένο στόχο BLER και βήμα του OLLA."""
        self._reset_olla(bler_target, olla_delta_up)
        return self._run_simulation(num_slots, alpha_ul, p0_dbm_ul, mcs_table_index, fairness_dl,
                                    guaranteed_power_ratio_dl)

    @torch.compile
    def _run_simulation(self, num_slots, alpha_ul, p0_dbm_ul, mcs_table_index=1, fairness_dl=0,
                        guaranteed_power_ratio_dl=0.5):
        hist = init_result_history(self.batch_size, num_slots, self.num_bs, self.num_ut_per_sector,
                                   device=self.device)
        harq_feedback, sinr_eff_feedback, num_decoded_bits = self._init_feedback_tensors()
        h_freq = self.channel_matrix(self.channel_model)
        for slot in range(num_slots):
            # 1. Κανάλι
            h_freq = self.channel_matrix.update(self.channel_model, h_freq, slot)
            h_freq_fading = self.channel_matrix.apply_fading(h_freq)

            # 2. Χρονοδρομολόγηση PF
            rate_achievable_est = estimate_achievable_rate(self.olla.sinr_eff_db_last,
                                                           self.resource_grid.num_ofdm_symbols,
                                                           self.resource_grid.fft_size)
            is_scheduled = self.scheduler(num_decoded_bits, rate_achievable_est)
            num_allocated_sc = torch.minimum(torch.sum(is_scheduled.int(), dim=-1),
                                             torch.ones(1, dtype=torch.int32, device=self.device))
            num_allocated_sc = torch.sum(num_allocated_sc, dim=-2)
            num_allocated_re = torch.sum(is_scheduled.int(), dim=[-1, -3, -4])

            # 3. Έλεγχος ισχύος
            pathloss_all_pairs, pathloss_serving_cell = get_pathloss(h_freq_fading,
                                                                     rx_tx_association=self._rx_tx_association)
            pathloss_serving_cell = self._group_by_sector(pathloss_serving_cell)
            if self.direction == 'uplink':
                tx_power_per_ut = open_loop_uplink_power_control(pathloss_serving_cell, num_allocated_sc,
                                                                 alpha=alpha_ul, p0_dbm=p0_dbm_ul,
                                                                 ut_max_power_dbm=self.ut_max_power_dbm)
            else:
                one = torch.ones(1, dtype=pathloss_serving_cell.dtype, device=self.device)
                rx_power_tot = self._group_by_sector(torch.sum(one / pathloss_all_pairs, dim=-2))
                interference_dl = (rx_power_tot - one / pathloss_serving_cell) * dbm_to_watt(self.bs_max_power_dbm)
                tx_power_per_ut, _ = downlink_fair_power_control(pathloss_serving_cell, interference_dl + self.no,
                                                                 num_allocated_sc,
                                                                 bs_max_power_dbm=self.bs_max_power_dbm,
                                                                 guaranteed_power_ratio=guaranteed_power_ratio_dl,
                                                                 fairness=fairness_dl, precision=self.precision)
            tx_power = spread_across_subcarriers(tx_power_per_ut, is_scheduled, num_tx=self.num_tx_per_sector,
                                                 precision=self.precision)

            # 4. SINR ανά ροή
            sinr = get_sinr(tx_power, self.stream_management, self.no, self.direction, h_freq_fading, self.num_bs,
                            self.num_ut_per_sector, self.num_streams_per_ut, self.resource_grid)

            # 5. Προσαρμογή ζεύξης (OLLA) και PHY abstraction
            mcs_index = self.olla(num_allocated_re, harq_feedback=harq_feedback, sinr_eff=sinr_eff_feedback)
            num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(mcs_index, sinr=sinr,
                                                                           mcs_table_index=mcs_table_index,
                                                                           mcs_category=self._mcs_category)
            # Το MCS που θα διάλεγε ο ILLA αν γνώριζε το πραγματικό SINR
            mcs_optimal = self.illa(sinr=sinr, mcs_table_index=mcs_table_index, mcs_category=self._mcs_category)
            sinr_eff_feedback = torch.where(num_allocated_re > 0, sinr_eff,
                                            torch.zeros(1, dtype=self.dtype, device=self.device))

            # 6. Καταγραφή και κίνηση των χρηστών
            hist = record_results(hist, slot, sim_failed=False,
                                  pathloss_serving_cell=torch.sum(pathloss_serving_cell, dim=-2),
                                  num_allocated_re=num_allocated_re,
                                  tx_power_per_ut=torch.sum(tx_power_per_ut, dim=-2),
                                  num_decoded_bits=num_decoded_bits, mcs_index=mcs_index, mcs_optimal=mcs_optimal,
                                  harq_feedback=harq_feedback, olla_offset=self.olla.offset, sinr_eff=sinr_eff,
                                  pf_metric=self.scheduler.pf_metric)
            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration
            self.channel_model.set_topology(self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations,
                                            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc)
        for key in hist:
            hist[key] = hist[key].stack()
        return hist


# ============================================================================
# 4. Παράμετροι του σεναρίου και οι δύο προσομοιωτές
# ============================================================================
scenario = 'umi'
num_rings = 1
num_ut_per_sector = 10
max_bs_ut_dist = 60         # [m]: χωρίς μεταπομπή, οι χρήστες μένουν κοντά στον σταθμό τους
min_bs_ut_dist = 0          # [m]
carrier_frequency = 3.5e9   # [Hz]
bs_max_power_dbm = 56       # [dBm]
ut_max_power_dbm = 26       # [dBm]
coherence_time = 100        # νέα πραγματοποίηση του καναλιού κάθε 100 slots και στην αρχή κάθε σεναρίου
mcs_table_index = 1
batch_size = 1

bs_array = PanelArray(num_rows_per_panel=2, num_cols_per_panel=3, polarization='dual', polarization_type='VH',
                      antenna_pattern='38.901', carrier_frequency=carrier_frequency)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1, polarization='single', polarization_type='V',
                      antenna_pattern='omni', carrier_frequency=carrier_frequency)

num_ofdm_sym = 1
num_subcarriers = 128
subcarrier_spacing = 15e3   # [Hz]
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym, fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing, num_tx=num_ut_per_sector,
                             num_streams_per_tx=ut_array.num_ant)


def build_simulator(direction):
    return SystemLevelSimulator(batch_size, num_rings, num_ut_per_sector, carrier_frequency, resource_grid,
                                scenario, direction, ut_array, bs_array, bs_max_power_dbm, ut_max_power_dbm,
                                coherence_time, max_bs_ut_dist=max_bs_ut_dist, min_bs_ut_dist=min_bs_ut_dist,
                                temperature=294, o2i_model='low', average_street_width=20.,
                                average_building_height=10.)


sls_dl = build_simulator('downlink')
sls_ul = build_simulator('uplink')

# --- Σχήμα 17: η τοπολογία (προσομοιωτής της κάτω ζεύξης) ---
fig = sls_dl.grid.show()
ax = fig.get_axes()
ut_loc_np = sls_dl.ut_loc.cpu().numpy()
ax[0].plot(ut_loc_np[0, :, 0], ut_loc_np[0, :, 1], 'xk', label='user position')
ax[0].legend()
fig.savefig(os.path.join(FIG_DIR, 'fig17_topology.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 5. Τα 84 σενάρια
# ============================================================================
alpha_ul = 1.0              # έλεγχος ισχύος της άνω ζεύξης: συντελεστής αντιστάθμισης
p0_dbm_ul = -80.0           # και επιθυμητή ισχύς λήψης [dBm]
bler_targets = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
olla_deltas = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
n_slots_run = 20            # σταθερό, ώστε η μεταγλώττιση να γίνει μία φορά ανά προσομοιωτή


def hist_to_dataframe(hist, scenario_label, **params):
    """Το ιστορικό μιας προσομοίωσης ως πίνακας, μία γραμμή ανά (slot, σταθμός, χρήστης)."""
    data = {key: val.cpu().numpy() for key, val in hist.items()}
    n_slots, n_batch, n_bs, n_ut = data['mcs_index'].shape
    slot_idx, batch_idx, bs_idx, ut_idx = np.meshgrid(np.arange(n_slots), np.arange(n_batch), np.arange(n_bs),
                                                      np.arange(n_ut), indexing='ij')
    df = pd.DataFrame({'slot': slot_idx.ravel(), 'bs': bs_idx.ravel(), 'ut': ut_idx.ravel()})
    for key in data:
        df[key] = data[key].ravel()
    df['scenario'] = scenario_label
    for k, v in params.items():
        df[k] = v
    return df


all_dfs, call_secs = [], {}
t_start = time.time()
for dir_label, sim in [('downlink', sls_dl), ('uplink', sls_ul)]:
    print(f'--- {dir_label} ---', flush=True)
    call_secs[dir_label] = []
    for bt in bler_targets:
        for delta in olla_deltas:
            t0 = time.time()
            hist_run = sim(n_slots_run, alpha_ul, p0_dbm_ul, bt, delta)
            all_dfs.append(hist_to_dataframe(hist_run, scenario_label=f'umi_{dir_label}_bler{bt}_delta{delta}',
                                             bler_target=bt, olla_delta_up=delta, direction=dir_label))
            call_secs[dir_label].append(time.time() - t0)
        print(f'  στόχος BLER {bt} ολοκληρώθηκε', flush=True)

df_all = pd.concat(all_dfs, ignore_index=True)
df_clean = df_all[df_all['harq'] != -1].copy()      # μόνο τα slots όπου ο χρήστης εξυπηρετήθηκε

# ============================================================================
# 6. Αριθμοί της Ενότητας 3.7
# ============================================================================
print()
print(f'Ενότητα 3.7: συνολικός χρόνος {time.time() - t_start:.0f} s')
for d, secs in call_secs.items():
    print(f'  {d}: πρώτη κλήση (με μεταγλώττιση) {secs[0]:.0f} s, οι επόμενες έως {max(secs[1:]):.1f} s')
print(f'Ενότητα 3.7: σενάρια {df_all["scenario"].nunique()}, εγγραφές {len(df_all)}, με μετάδοση {len(df_clean)} '
      f'({100 * len(df_clean) / len(df_all):.0f}%)')
mcs_counts = df_clean['mcs_index'].value_counts().sort_index()
print('Κατανομή του MCS:')
print(mcs_counts.to_string())
print(f'Ενότητα 3.7: MCS 27 {int(mcs_counts.get(27.0, 0))} φορές, MCS 3 έως 5 '
      f'{int(mcs_counts.reindex([3.0, 4.0, 5.0]).fillna(0).sum())} φορές')
sinr_db = 10 * np.log10(df_clean['sinr_eff'].clip(lower=1e-10))
print(f'Ενότητα 3.7: ενεργό SINR από {sinr_db.min():.1f} έως {sinr_db.max():.1f} dB, διάμεσος {sinr_db.median():.1f} dB')
print(f'Ενότητα 3.7: ποσοστό επιτυχημένων μεταδόσεων {100 * df_clean["harq"].mean():.1f}%')
print('Ανά κατεύθυνση:')
print(df_clean.groupby('direction').agg(rows=('mcs_index', 'count'), mean_MCS=('mcs_index', 'mean'),
                                        success=('harq', 'mean')).round(3).to_string())
print('Μέσο MCS ανά στόχο BLER:')
print(df_clean.groupby('bler_target')['mcs_index'].agg(['count', 'mean']).round(2).to_string())
print('Το Σχήμα 17 είναι στον φάκελο', FIG_DIR)
