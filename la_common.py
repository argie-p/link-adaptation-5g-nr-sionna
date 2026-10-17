# -*- coding: utf-8 -*-
"""
la_common.py — Κοινός κώδικας για τα πειράματα του Κεφαλαίου 4.

Το αρχείο δεν τρέχει μόνο του. Το εισάγουν τα αρχεία k4_*.py και περιέχει:
  1. τις παραμέτρους του σεναρίου (Πίνακας 4 της εργασίας),
  2. το ray tracing της διαδρομής του χρήστη με το Sionna RT,
  3. το κανάλι κάθε χρονοθυρίδας (slot), με χρονική εξέλιξη Doppler,
  4. το SINR ανά resource element και το PHY abstraction του Sionna,
  5. τους μηχανισμούς προσαρμογής ζεύξης (link adaptation): ILLA, OLLA και SALAD,
     με και χωρίς anti-windup,
  6. τον βρόχο της προσομοίωσης και την αποθήκευση των αποτελεσμάτων.

Το ray tracing της σκηνής, η la_step και ο βρόχος της προσομοίωσης βασίζονται στο
tutorial του Sionna για το link adaptation (Copyright (c) 2021-2026 NVIDIA CORPORATION &
AFFILIATES, άδεια Apache 2.0), με αλλαγές: χρονική εξέλιξη Doppler σε κάθε θέση, SINR
κλειστής μορφής με έλεγχο έναντι του Sionna, SALAD και anti-windup.

Εκδόσεις με τις οποίες παράχθηκαν τα αποτελέσματα της εργασίας:
  Sionna 2.2.0 (sionna-rt 2.2.0, mitsuba 3.9.1, drjit 1.5.0), PyTorch 2.11,
  κώδικας του SALAD από το https://github.com/NVlabs/salad, commit 2746947,
  GPU NVIDIA T4 (Google Colab).
Η ιχνηλάτηση ακτίνων σε CPU δίνει ελάχιστα διαφορετικά κανάλια από τη GPU, οπότε
σε CPU τα νούμερα μπορεί να διαφέρουν από αυτά της εργασίας στο τρίτο δεκαδικό.

Ορολογία που χρησιμοποιείται στα σχόλια:
  slot = χρονοθυρίδα (0.5 ms), θέση = σημείο της διαδρομής όπου έγινε ray tracing,
  SE = spectral efficiency (φασματική απόδοση, bits/s/Hz),
  BLER = block error rate, HARQ = ACK (1) / NACK (0) / καμία μετάδοση (-1).
"""

import os
import sys
import time
import subprocess
import warnings

import numpy as np
import pandas as pd
import torch

import sionna
import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, \
    PathSolver, subcarrier_frequencies
from sionna.phy import config
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, LMMSEPostEqualizationSINR
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import dbm_to_watt, lin_to_db, db_to_lin, normal
from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation, OuterLoopLinkAdaptation
from sionna.phy.nr.utils import decode_mcs_index

warnings.filterwarnings('ignore', category=RuntimeWarning)

sionna.phy.config.seed = 42
sionna.phy.config.precision = 'single'
dev = config.device
if not torch.cuda.is_available():
    print('Προσοχή: δεν βρέθηκε GPU. Η προσομοίωση θα είναι πολύ πιο αργή και τα '
          'κανάλια ελάχιστα διαφορετικά από αυτά της εργασίας.')

# ============================================================================
# 1. Παράμετροι του σεναρίου (Πίνακας 4)
# ============================================================================
NUM_POINTS = 500            # θέσεις της διαδρομής όπου γίνεται ray tracing
SPP_REF = 38                # slots ανά θέση στα 30 km/h (σενάριο αναφοράς)

BLER_TARGET = 0.1           # μακροπρόθεσμος στόχος BLER
NUM_SUBCARRIERS = 1024
SUBCARRIER_SPACING = 30e3   # [Hz]
NUM_OFDM_SYMBOLS = 13
SLOT_DURATION = 0.5e-3      # [s]
SYMBOL_DURATION = SLOT_DURATION / 14
MCS_TABLE_INDEX = 2         # πίνακας MCS 2 της TS 38.214 (έως 256QAM)

NUM_BS = 1
NUM_BS_ANT = 1
NUM_UT_ANT = 1
NUM_STREAMS_PER_BS = NUM_UT_ANT

BS_POWER_DBM = 20
BS_POWER_WATT = dbm_to_watt(BS_POWER_DBM).item()
TEMPERATURE = 294           # [K]
NO = BOLTZMANN_CONSTANT * TEMPERATURE * SUBCARRIER_SPACING   # θόρυβος ανά subcarrier [W]
NOISE_FEEDBACK_STD = 1.5    # [dB], σφάλμα της αναφοράς SINR προς τον ILLA και τον OLLA

BS_POS = np.array([32.5, 10.5, 23])
BS_LOOK_AT = np.array([22, -8, 0])

# Διαδρομή: ευθεία 80 m στο x = -23 m και ύψος 1.5 m, από y0 έως y0 + 80 m
ROUTE_X, ROUTE_Z, ROUTE_LEN = -23, 1.5, 80
MAIN_Y0 = -40               # η κύρια διαδρομή (Ενότητα 4.2)
POINT_SPACING = ROUTE_LEN / (NUM_POINTS - 1)   # περίπου 16 cm

FREQS = {'3p5GHz': 3.5e9, '28GHz': 28e9}
SEEDS = [42, 1, 7, 123, 2024]   # οι πέντε σπόροι της σύγκρισης
SEL_SEED = 99                   # σπόρος για την επιλογή των ρυθμίσεων (σαρώσεις)

# Παράμετροι αναφοράς του SALAD (όπως στο σενάριο των δημιουργών του)
SALAD_REF = dict(lr=1.2, probe=0.15, rho=0.25, ke=0.05)


def kmh(spp):
    """Ταχύτητα του χρήστη [km/h] όταν κάθε θέση της διαδρομής κρατά spp slots."""
    return 3.6 * POINT_SPACING / (spp * SLOT_DURATION)


# Ισχύς εκπομπής ανά resource element, ομοιόμορφα σε όλα τα subcarriers
P_RE = BS_POWER_WATT / NUM_STREAMS_PER_BS / NUM_SUBCARRIERS
tx_power = torch.full([1, NUM_BS, NUM_STREAMS_PER_BS, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS],
                      P_RE, device=dev)

stream_management = StreamManagement(np.ones([1, NUM_BS]), NUM_STREAMS_PER_BS)
resource_grid = ResourceGrid(num_ofdm_symbols=NUM_OFDM_SYMBOLS, fft_size=NUM_SUBCARRIERS,
                             subcarrier_spacing=SUBCARRIER_SPACING, num_tx=NUM_BS,
                             num_streams_per_tx=NUM_STREAMS_PER_BS)
frequencies = subcarrier_frequencies(num_subcarriers=NUM_SUBCARRIERS,
                                     subcarrier_spacing=SUBCARRIER_SPACING)


def to_numpy(x):
    """Μετατροπή πίνακα Dr.Jit, torch ή numpy σε numpy."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, 'numpy'):
        return np.array(x.numpy())
    return np.array(x)


freqs_t = torch.as_tensor(to_numpy(frequencies).astype(np.float64), dtype=torch.float64,
                          device=dev)

# Τα blocks του Sionna για το SINR χρησιμοποιούνται μόνο για έλεγχο (check_sinr)
precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid,
                                      stream_management=stream_management)
lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                              stream_management=stream_management)

# ============================================================================
# 2. Ray tracing της διαδρομής
# ============================================================================
# Σε κάθε θέση της διαδρομής τοποθετείται ένας δέκτης με ταχύτητα κατά μήκος
# του δρόμου. Το Sionna RT δίνει για κάθε διαδρομή διάδοσης p το πλάτος a_p,
# την καθυστέρηση tau_p και τη μετατόπιση Doppler f_D,p. Η ιχνηλάτηση γίνεται
# για την ταχύτητα αναφοράς (30 km/h). Η μετατόπιση Doppler είναι ανάλογη της
# ταχύτητας, οπότε για άλλη ταχύτητα πολλαπλασιάζεται με τον λόγο των ταχυτήτων.


def cfr_points(a_i, tau_i, fd_i, t):
    """Απόκριση συχνότητας μιας θέσης σε χρόνους t [s] -> [T, NUM_SUBCARRIERS]."""
    coef = a_i[None, :] * torch.exp(2j * np.pi * fd_i[None, :] * t[:, None])
    phase = torch.exp(-2j * np.pi * tau_i[:, None] * freqs_t[None, :])
    return coef @ phase


def ray_trace(carrier_frequency_hz, y0=MAIN_Y0):
    """Ray tracing των NUM_POINTS θέσεων της διαδρομής που αρχίζει στο y0.

    Επιστρέφει dict με a [θέσεις, διαδρομές], tau [s] και fd [Hz] (για 30 km/h),
    καθώς και τα σφάλματα δύο ελέγχων έναντι των συναρτήσεων του Sionna.
    """
    start = np.array([ROUTE_X, y0, ROUTE_Z], dtype=float)
    end = np.array([ROUTE_X, y0 + ROUTE_LEN, ROUTE_Z], dtype=float)
    step = (end - start) / (NUM_POINTS - 1)
    speed_ref = np.linalg.norm(step) / (SPP_REF * SLOT_DURATION)   # [m/s], 30.4 km/h

    scene = load_scene(sionna.rt.scene.simple_street_canyon)
    scene.frequency = carrier_frequency_hz
    scene.bandwidth = NUM_SUBCARRIERS * SUBCARRIER_SPACING
    scene.tx_array = PlanarArray(num_rows=1, num_cols=NUM_BS_ANT, pattern="tr38901",
                                 polarization='V')
    scene.rx_array = PlanarArray(num_rows=1, num_cols=NUM_UT_ANT, pattern="dipole",
                                 polarization='V')
    scene.add(Transmitter("bs", position=BS_POS, look_at=BS_LOOK_AT, power_dbm=BS_POWER_DBM))
    velocity = (step / np.linalg.norm(step)) * speed_ref
    for p in range(NUM_POINTS):
        rx = Receiver(f"ut{p}", position=start + p * step)
        rx.velocity = [float(v) for v in velocity]
        scene.add(rx)

    paths = PathSolver()(scene, max_depth=8, refraction=False)
    a, tau = paths.cir(out_type="torch")
    if isinstance(a, (tuple, list)):
        a = torch.complex(a[0], a[1])
    a = a[..., 0].reshape(NUM_POINTS, -1).to(torch.complex128).to(dev)
    tau = torch.as_tensor(to_numpy(tau)).reshape(NUM_POINTS, -1).to(torch.float64).to(dev)
    fd = torch.as_tensor(to_numpy(paths.doppler)).reshape(NUM_POINTS, -1)
    fd = fd.to(torch.float64).to(dev)

    energy = (a.abs() ** 2).sum(dim=1)
    no_path = int((energy == 0).sum().item())            # θέσεις χωρίς καμία διαδρομή
    idx = torch.argsort(energy, descending=True)[:3].tolist()

    # Έλεγχος 1: η απόκριση συχνότητας από τα a, tau συμφωνεί με την paths.cfr του Sionna
    h_ref = paths.cfr(frequencies=frequencies, sampling_frequency=1.0, num_time_steps=1,
                      out_type="torch")
    h_ref = h_ref.reshape(NUM_POINTS, -1).to(torch.complex128).to(dev)
    t0 = torch.zeros(1, dtype=torch.float64, device=dev)
    h_my = torch.stack([cfr_points(a[i], tau[i], fd[i], t0)[0] for i in idx])
    err = (torch.linalg.norm(h_my - h_ref[idx]) / torch.linalg.norm(h_ref[idx])).item()
    assert err < 1e-3, f'Η απόκριση συχνότητας δεν συμφωνεί με την paths.cfr ({err:.1e})'

    # Έλεγχος 2: η χρονική εξέλιξη a_p exp(j2π f_D,p t) συμφωνεί με την paths.cir
    n_chk = 3 * 14
    a_t, _ = paths.cir(sampling_frequency=1 / SYMBOL_DURATION, num_time_steps=n_chk,
                       out_type="torch")
    if isinstance(a_t, (tuple, list)):
        a_t = torch.complex(a_t[0], a_t[1])
    a_t = a_t.reshape(NUM_POINTS, -1, n_chk).to(torch.complex128).to(dev)
    i = idx[0]
    t_chk = torch.arange(n_chk, dtype=torch.float64, device=dev) * SYMBOL_DURATION
    a_my = a[i][:, None] * torch.exp(2j * np.pi * fd[i][:, None] * t_chk[None, :])
    err_t = (torch.linalg.norm(a_my - a_t[i]) / torch.linalg.norm(a_t[i])).item()
    assert err_t < 1e-3, f'Η χρονική εξέλιξη δεν συμφωνεί με την paths.cir ({err_t:.1e})'

    return dict(a=a, tau=tau, fd=fd, no_path=no_path, err_cfr=err, err_cir=err_t,
                max_doppler=fd.abs().max().item(), y0=y0, fc=carrier_frequency_hz)


# ============================================================================
# 3. Κανάλι και SINR κάθε slot
# ============================================================================
# Κάθε θέση της διαδρομής κρατά spp slots. Μέσα σε αυτά, κάθε διαδρομή διάδοσης
# εξελίσσεται ως a_p exp(j2π f_D,p t), σε κάθε σύμβολο OFDM. Με spp = 38 ο χρήστης
# κινείται με 30.4 km/h.
# Ο «συμπιεσμένος χρόνος» του tutorial του Sionna αντιστοιχεί σε spp = 1 με
# στατικό κανάλι μέσα στο slot (static=True): κάθε slot είναι μια νέα θέση.
# Με μία κεραία σε κάθε άκρο (SISO), το SINR μετά από RZF precoding και LMMSE
# equalization ισούται με P_RE |h|^2 / N0. Η check_sinr ελέγχει ότι αυτή η κλειστή
# μορφή συμπίπτει με τα blocks του Sionna.
SYM_TIMES = torch.arange(NUM_OFDM_SYMBOLS, dtype=torch.float64, device=dev) * SYMBOL_DURATION
_phase_cache = {'key': None, 'phase': None}


def phase_matrix(ch, i):
    """Ο όρος exp(-j2π f tau) μιας θέσης, υπολογίζεται μία φορά ανά θέση."""
    key = (id(ch), i)
    if _phase_cache['key'] != key:
        _phase_cache['key'] = key
        _phase_cache['phase'] = torch.exp(-2j * np.pi * ch['tau'][i][:, None] * freqs_t[None, :])
    return _phase_cache['phase']


def channel_block(ch, spp, i, static=False):
    """Κανάλι όλων των slots της θέσης i -> [spp, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS]."""
    if static:
        t = torch.zeros(spp * NUM_OFDM_SYMBOLS, dtype=torch.float64, device=dev)
    else:
        j = torch.arange(spp, dtype=torch.float64, device=dev)
        t = (j[:, None] * SLOT_DURATION + SYM_TIMES[None, :]).reshape(-1)
    fd = ch['fd'][i] * (SPP_REF / spp)          # Doppler ανάλογο της ταχύτητας
    coef = ch['a'][i][None, :] * torch.exp(2j * np.pi * fd[None, :] * t[:, None])
    h = coef @ phase_matrix(ch, i)
    return h.to(torch.complex64).reshape(spp, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)


def sinr_block(ch, spp, i, static=False):
    """SINR ανά resource element όλων των slots της θέσης i -> [spp, 13, 1024, 1, 1]."""
    h = channel_block(ch, spp, i, static)
    s = (P_RE / NO) * (h.real ** 2 + h.imag ** 2)
    return s.reshape(spp, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS, 1, 1)


def check_sinr(ch):
    """Σύγκριση του SINR κλειστής μορφής με RZFPrecodedChannel + LMMSEPostEqualizationSINR."""
    energy = (ch['a'].abs() ** 2).sum(dim=1)
    worst = 0.0
    for i in torch.argsort(energy, descending=True)[:3].tolist():
        h = channel_block(ch, 2, i)[1].reshape(1, 1, 1, 1, 1, NUM_OFDM_SYMBOLS, NUM_SUBCARRIERS)
        h_eff = precoded_channel(h, tx_power=tx_power, alpha=NO)
        s_ref = lmmse_posteq_sinr(h_eff, no=NO)[0, ...]
        s_new = sinr_block(ch, 2, i)[1]
        worst = max(worst, ((s_new - s_ref).abs().max() / s_ref.abs().max()).item())
    assert worst < 1e-4, f'Το SINR κλειστής μορφής διαφέρει από το Sionna ({worst:.1e})'
    return worst


# ============================================================================
# 4. PHY abstraction και ο βρόχος της προσομοίωσης
# ============================================================================
# Σε κάθε slot: ο μηχανισμός επιλέγει MCS με βάση την ανάδραση του προηγούμενου
# slot (αναφορά SINR και HARQ), και το PHY abstraction του Sionna αποφασίζει αν η
# μετάδοση πέτυχε, από το ενεργό SINR (effective SINR) και τους πίνακες BLER.
# Η αναφορά SINR προς τον ILLA και τον OLLA είναι το ενεργό SINR του προηγούμενου
# slot συν γκαουσιανό σφάλμα (noise_feedback, σε dB). Το SALAD δεν τη χρησιμοποιεί.
phy_abs = PHYAbstraction()


def la_step(la, sinr, sinr_eff_feedback, harq_feedback):
    """Ένα slot: επιλογή MCS και αποτέλεσμα της μετάδοσης."""
    num_allocated_re = torch.sum((sinr > 0).int(), axis=[-4, -3, -1])
    mcs_index = la(num_allocated_re=num_allocated_re, sinr_eff=sinr_eff_feedback,
                   mcs_table_index=MCS_TABLE_INDEX, mcs_category=1,   # 1 = downlink
                   harq_feedback=harq_feedback)
    _, harq_feedback, sinr_eff, *_ = phy_abs(mcs_index, sinr=sinr,
                                             mcs_table_index=MCS_TABLE_INDEX, mcs_category=1)
    return mcs_index, harq_feedback, sinr_eff


def run(la, sinr_fn, n_pos, spp, noise_feedback=None):
    """Προσομοίωση n_pos θέσεων με spp slots η καθεμία.

    sinr_fn(i) δίνει το SINR ανά resource element των spp slots της θέσης i.
    Επιστρέφει, για κάθε slot: MCS, HARQ (1 ACK, 0 NACK, -1 χωρίς μετάδοση),
    SE και ενεργό SINR [dB]. Οι τιμές μένουν στη GPU μέχρι το τέλος, ώστε να μην
    περιμένει η GPU σε κάθε slot.
    """
    n = n_pos * spp
    mcs_buf = torch.zeros(n, dtype=torch.int32, device=dev)
    harq_buf = torch.zeros(n, dtype=torch.int32, device=dev)
    se_buf = torch.zeros(n, dtype=torch.float32, device=dev)
    sinr_buf = torch.zeros(n, dtype=torch.float32, device=dev)

    harq_feedback = - torch.tensor(1, dtype=torch.int32, device=dev)   # καμία ανάδραση ακόμη
    sinr_eff_db_true = torch.tensor([0.0], device=dev)
    k = 0
    for i in range(n_pos):
        blk = sinr_fn(i)
        for j in range(spp):
            if noise_feedback is not None:
                sinr_eff_db_feedback = sinr_eff_db_true + noise_feedback[k]
            else:
                sinr_eff_db_feedback = sinr_eff_db_true
            mcs_index, harq_feedback, sinr_eff_true = la_step(
                la, blk[j], sinr_eff_feedback=db_to_lin(sinr_eff_db_feedback),
                harq_feedback=harq_feedback)
            sinr_eff_db_true = lin_to_db(sinr_eff_true)

            # Φασματική απόδοση: bits ανά σύμβολο x code rate, μόνο αν η μετάδοση πέτυχε
            mod_order, coderate = decode_mcs_index(mcs_index, table_index=MCS_TABLE_INDEX,
                                                   is_pusch=False)
            se = torch.where(harq_feedback == 1, mod_order.to(dtype=coderate.dtype) * coderate,
                             torch.zeros_like(coderate))
            mcs_buf[k] = mcs_index[0]
            harq_buf[k] = harq_feedback[0]
            se_buf[k] = se[0]
            sinr_buf[k] = sinr_eff_db_true[0]
            k += 1

    return dict(mcs=mcs_buf.cpu().numpy(), harq=harq_buf.cpu().numpy(),
                se=se_buf.cpu().numpy().astype(np.float64), sinr=sinr_buf.cpu().numpy())


def route_sinr_fn(ch, spp, static=False):
    """Η sinr_fn μιας διαδρομής που προέκυψε από ray_trace."""
    return lambda i: sinr_block(ch, spp, i, static)


# ============================================================================
# 5. SALAD
# ============================================================================
# Ο κώδικας του SALAD είναι των δημιουργών του (NVlabs), χωρίς αλλαγές. Το
# περιτύλιγμα (wrapper) μεταφράζει τα ορίσματα του Sionna σε αυτά του SALAD.
SALAD_DIR = os.environ.get('SALAD_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'salad'))
SALAD_COMMIT = '2746947bb4350f087915bd4512cdef1656159e49'
if not os.path.isdir(SALAD_DIR):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/NVlabs/salad.git', SALAD_DIR],
                   check=True)
    subprocess.run(['git', '-C', SALAD_DIR, 'checkout', '-q', SALAD_COMMIT], check=True)
sys.path.append(SALAD_DIR)

from algos import SALAD                                         # noqa: E402
from utils import StudentSINREstimator, get_bler_sigmoid_params  # noqa: E402

# Σιγμοειδείς προσεγγίσεις των καμπυλών BLER που χρησιμοποιεί εσωτερικά το SALAD,
# για code block 2000 bits (το μεγαλύτερο διαθέσιμο, πλησιέστερο στο grid μας)
_df_salad = pd.read_csv(os.path.join(SALAD_DIR, 'data', 'bler_table_sigmoid_fit.csv'))
PARAMS_2000, MCS_MIN_2000 = get_bler_sigmoid_params(_df_salad, 2000, MCS_TABLE_INDEX,
                                                    return_mcs_min_available=True)


class SaladWrapper:
    """Καλεί το SALAD με τα ορίσματα που δίνει η la_step του Sionna.

    Το SALAD χρειάζεται μόνο αν η προηγούμενη μετάδοση απέτυχε (NACK). Κρατά μια
    ουρά με τα MCS που έχουν σταλεί και δεν έχουν λάβει ακόμη ACK/NACK.
    """

    def __init__(self, salad_obj, device, mcs_offset=0):
        self.salad = salad_obj
        self.device = device
        self.mcs_offset = mcs_offset   # το SALAD αριθμεί από το ελάχιστο διαθέσιμο MCS

    def __call__(self, num_allocated_re=None, sinr_eff=None, mcs_table_index=None,
                 mcs_category=None, harq_feedback=None, **kwargs):
        if harq_feedback is None:
            hf = -1
        elif torch.is_tensor(harq_feedback):
            hf = int(harq_feedback.reshape(-1)[0].item())
        else:
            hf = int(harq_feedback)
        # Αν στο προηγούμενο slot δεν έγινε μετάδοση (HARQ = -1, θέση χωρίς σήμα), το
        # MCS που είχε επιλεγεί δεν στάλθηκε ποτέ και βγαίνει από την ουρά. Αλλιώς κάθε
        # επόμενο ACK/NACK θα αντιστοιχιζόταν σε λάθος MCS. Στην κύρια διαδρομή δεν
        # υπάρχουν τέτοια slots, οπότε εκεί αυτό δεν αλλάζει τίποτα.
        if hf == -1 and len(self.salad.mcs_unackned) > 0:
            self.salad.mcs_unackned.pop()
        is_nack = [] if hf == -1 else [1 - hf]      # το SALAD θέλει 1 = NACK, 0 = ACK
        mcs = int(self.salad(is_nack=is_nack)) + self.mcs_offset
        return torch.tensor([mcs], dtype=torch.int32, device=self.device)


class SaladWrapperAW(SaladWrapper):
    """Το ίδιο περιτύλιγμα με anti-windup: όριο στον αθροιστή του ελεγκτή του SALAD.

    Ο ελεγκτής ορίζει τον στιγμιαίο στόχο BLER ως tau = target + k_i * I, όπου I το
    άθροισμα των σφαλμάτων (target - NACK), και τον περιορίζει στο [0.001, 0.99].
    Το anti-windup κρατά το I στα όρια που αντιστοιχούν σε αυτό το διάστημα.
    """

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        c = self.salad
        i_min = -(c.bler_target - 0.001) / c.k_i
        i_max = (0.99 - c.bler_target) / c.k_i
        c.integral_error = float(np.clip(c.integral_error, i_min, i_max))
        return out


# ============================================================================
# 6. Μηχανισμοί, μία εκτέλεση και αποθήκευση
# ============================================================================
def reset_seeds(s):
    """Ίδιος σπόρος για numpy (SALAD), torch και Sionna (PHY abstraction)."""
    np.random.seed(s)
    torch.manual_seed(s)
    sionna.phy.config.seed = s


def make_la(sp):
    """Δημιουργεί τον μηχανισμό μιας εκτέλεσης. Για το SALAD επιστρέφει και τα
    εσωτερικά του αντικείμενα, ώστε να καταγραφεί η κατάστασή του."""
    if sp['mech'] in ('ILLA_ideal', 'ILLA_noisy'):
        return InnerLoopLinkAdaptation(phy_abs, bler_target=BLER_TARGET), None
    if sp['mech'] == 'OLLA':
        la = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=BLER_TARGET)
        la.delta_up = float(sp['delta'])   # βήμα της διόρθωσης μετά από NACK [dB]
        return la, None
    est = StudentSINREstimator(PARAMS_2000, sinr_init=-5, learning_rate_init=sp['lr'])
    core = SALAD(BLER_TARGET, PARAMS_2000, est, rho_calibration=sp['rho'],
                 prob_probe=sp['probe'], bler_target_probe=0.95, k_e=sp['ke'], score_window=15)
    W = SaladWrapperAW if sp['aw'] else SaladWrapper
    return W(core, device=dev, mcs_offset=MCS_MIN_2000), (core, est)


LABEL = {'ILLA_ideal': 'ILLA τέλεια εκτίμηση', 'ILLA_noisy': 'ILLA θορυβώδης'}


def spec(exp, freq, mech, seed, y0=MAIN_Y0, spp=SPP_REF, mode='realistic', delta=None,
         salad=None, aw=False, label=None):
    """Περιγραφή μιας εκτέλεσης.

    mech: 'ILLA_ideal' (αναφορά SINR χωρίς σφάλμα), 'ILLA_noisy', 'OLLA' ή 'SALAD'.
    mode: 'realistic' (spp slots ανά θέση, με Doppler) ή 'compressed' (ένα slot ανά
    θέση με στατικό κανάλι, όπως στο tutorial του Sionna).
    """
    if mode == 'compressed':
        spp = 1
    sp = dict(exp=exp, freq=freq, mech=mech, seed=seed, y0=y0, spp=spp, mode=mode, aw=aw)
    if mech == 'OLLA':
        sp['delta'] = float(delta)
        code = f'OLLA{delta:g}'
        sp['label'] = label or f'OLLA {delta:g} dB'
    elif mech == 'SALAD':
        sp.update(salad)
        code = (f"SALAD_lr{salad['lr']:g}_p{salad['probe']:g}_r{salad['rho']:g}_k{salad['ke']:g}"
                + ('_AW' if aw else ''))
        sp['label'] = label or ('SALAD + anti-windup' if aw else 'SALAD')
    else:
        code = mech
        sp['label'] = LABEL[mech]
    mode_tag = 'comp_' if mode == 'compressed' else ''
    sp['key'] = (f"{exp}_{freq}_{mode_tag}y{y0}_spp{spp}_s{seed}_{code}"
                 .replace('.', 'p').replace('-', 'm'))
    return sp


def bler_pct(harq, mask=None):
    """BLER [%] στα slots όπου έγινε μετάδοση (HARQ 0 ή 1)."""
    h = harq if mask is None else harq[mask]
    h = h[h >= 0]
    return 100 * float(1 - h.mean()) if len(h) else float('nan')


def run_spec(sp, sinr_fn, n_pos=NUM_POINTS):
    """Μία εκτέλεση: επιστρέφει μια γραμμή αποτελεσμάτων και τις τιμές ανά slot."""
    spp = sp['spp']
    n = n_pos * spp
    # Σειρά των τυχαίων αριθμών: πρώτα το σφάλμα της αναφοράς SINR, μετά, με τον ίδιο
    # σπόρο από την αρχή, η προσομοίωση. Έτσι όλοι οι μηχανισμοί ενός σπόρου βλέπουν
    # το ίδιο σφάλμα αναφοράς.
    reset_seeds(sp['seed'])
    nf = normal(size=[n], dtype=torch.float32, std=NOISE_FEEDBACK_STD).to(dev)
    reset_seeds(sp['seed'])
    la, salad = make_la(sp)
    t0 = time.time()
    r = run(la, sinr_fn, n_pos, spp, noise_feedback=None if sp['mech'] == 'ILLA_ideal' else nf)
    secs = time.time() - t0

    after = np.arange(n) >= 30 * spp           # χωρίς τις πρώτες 30 θέσεις (διακοπή στα 28 GHz)
    row = dict(key=sp['key'], exp=sp['exp'], freq=sp['freq'], mode=sp['mode'], y0=sp['y0'],
               spp=spp, kmh=round(kmh(spp), 1), seed=sp['seed'], algorithm=sp['label'],
               mech=sp['mech'], delta=sp.get('delta'), lr=sp.get('lr'), probe=sp.get('probe'),
               rho=sp.get('rho'), ke=sp.get('ke'), aw=sp.get('aw'),
               SE=float(r['se'].mean()), BLER=bler_pct(r['harq']),
               SE_after30=float(r['se'][after].mean()), BLER_after30=bler_pct(r['harq'], after),
               n_slots=n, seconds=round(secs, 1))
    hist = dict(se=r['se'].astype(np.float32), harq=r['harq'].astype(np.int8),
                mcs=r['mcs'].astype(np.int8), sinr=r['sinr'].astype(np.float32))
    if salad is not None:
        core, est = salad
        tau = np.array(core.tau_inst_hist)
        row['clipped'] = 100 * float(np.mean(tau <= 0.0011))   # % slots με στόχο στο κάτω όριο
        row['probes'] = int(np.sum(core.is_probe_hist))
        hist.update(tau=tau.astype(np.float32),                        # στιγμιαίος στόχος BLER
                    probe=np.array(core.is_probe_hist).astype(np.int8),
                    integ=np.array(core.integral_error_hist, dtype=np.float32),   # αθροιστής
                    est=np.array(est.sinr_hist, dtype=np.float32))     # εκτίμηση SINR του SALAD
    return row, hist


RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
HIST_DIR = os.path.join(RESULTS_DIR, 'hist')

_channels = {}


def get_channel(freq, y0=MAIN_Y0):
    """Ray tracing μιας διαδρομής (μία φορά ανά συχνότητα και διαδρομή) και έλεγχοι."""
    if (freq, y0) not in _channels:
        t0 = time.time()
        ch = ray_trace(FREQS[freq], y0)
        err = check_sinr(ch)
        print(f'=== {freq}, y0 = {y0} m: ray tracing σε {time.time() - t0:.0f} s, '
              f'θέσεις χωρίς διαδρομές {ch["no_path"]}, μέγιστη |Doppler| '
              f'{ch["max_doppler"]:.0f} Hz στα 30 km/h, σφάλματα ελέγχων: cfr {ch["err_cfr"]:.0e}, '
              f'cir {ch["err_cir"]:.0e}, SINR {err:.0e}', flush=True)
        _channels[(freq, y0)] = ch
    return _channels[(freq, y0)]


def run_all(specs, csv_name):
    """Τρέχει όσες εκτελέσεις δεν υπάρχουν ήδη στο results/csv_name και τις αποθηκεύει.

    Μετά από κάθε εκτέλεση γράφεται μια γραμμή στο CSV και οι τιμές ανά slot στο
    results/hist/<key>.npz. Αν η εκτέλεση διακοπεί, ξεκινά από εκεί που σταμάτησε.
    """
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, csv_name)
    rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
    done = {r['key'] for r in rows}
    todo = [sp for sp in specs if sp['key'] not in done]
    print(f'{csv_name}: {len(specs)} εκτελέσεις, έτοιμες {len(specs) - len(todo)}', flush=True)
    for k, sp in enumerate(todo, start=1):
        ch = get_channel(sp['freq'], sp['y0'])
        row, hist = run_spec(sp, route_sinr_fn(ch, sp['spp'], static=(sp['mode'] == 'compressed')))
        rows.append(row)
        tmp = path + '.tmp'
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, path)
        np.savez_compressed(os.path.join(HIST_DIR, sp['key'] + '.npz'), **hist)
        print(f'[{k}/{len(todo)}] {sp["freq"]:6s} y0={sp["y0"]:4d} {row["kmh"]:6.1f} km/h '
              f'σπόρος {sp["seed"]:<4d} {sp["label"]:27s} SE={row["SE"]:.3f} '
              f'BLER={row["BLER"]:5.1f}% ({row["seconds"]:.0f} s)', flush=True)
    return pd.DataFrame(rows)
