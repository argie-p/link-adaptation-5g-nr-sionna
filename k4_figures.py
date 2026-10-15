# -*- coding: utf-8 -*-
"""
k4_figures.py — Οι πίνακες και τα σχήματα του Κεφαλαίου 4, από τα αποτελέσματα των πειραμάτων.

Διαβάζει τα αρχεία που γράφουν στον φάκελο results/ τα:
    k4_2_main.py              -> main.csv
    k4_2_sensitivity.py       -> sensitivity.csv
    k4_3_routes.py            -> routes.csv
    k4_4_speed_compression.py -> speed.csv, compressed.csv
καθώς και τις τιμές ανά slot (results/hist/*.npz), και φτιάχνει:
    Πίνακες 4, 5, 6, 7, 9, 10 και 11   -> figures/tables_k4.md (και στην οθόνη)
    Σχήματα 19 έως 24, 27 και 28        -> figures/fig*.png
    βοηθητικούς πίνακες με τους αριθμούς που αναφέρει το κείμενο (στο ίδιο .md).
Ο Πίνακας 2 και το Σχήμα 18 βγαίνουν από το k4_1_salad_reference.py, ο Πίνακας 8 και
τα Σχήματα 25 και 26 από το k4_3_known_answer.py.

Δεν χρειάζεται GPU ούτε Sionna. Τρέχει σε λιγότερο από ένα λεπτό.
Το main.csv είναι απαραίτητο. Αν λείπει κάποιο από τα υπόλοιπα, παραλείπονται μόνο οι
πίνακες και τα σχήματα που το χρειάζονται.

Ονόματα μηχανισμών μέσα στον κώδικα:
    ILLA_ideal  ILLA με τέλεια εκτίμηση (αναφορά SINR χωρίς σφάλμα)
    ILLA_noisy  ILLA με θορυβώδη εκτίμηση
    OLLA_<β>    OLLA με βήμα β dB μετά από NACK, π.χ. OLLA_1
    SALAD_ref   SALAD με τις παραμέτρους αναφοράς
    SALAD_lr<x> SALAD με ρυθμό μάθησης x και τις υπόλοιπες παραμέτρους αναφοράς
    SALAD_AW    SALAD με τις παραμέτρους αναφοράς και anti-windup
    SALAD_tuned SALAD με τη ρύθμιση που είχε βρεθεί καλύτερη στον συμπιεσμένο χρόνο
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')
FIG_DIR = os.environ.get('FIG_DIR', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================================
# 1. Σταθερές (ίδιες με τα αρχεία των πειραμάτων)
# ============================================================================
SEEDS = [42, 1, 7, 123, 2024]   # σπόροι της σύγκρισης
SEL_SEED = 99                   # σπόρος επιλογής ρυθμίσεων (k4_2_sensitivity.py)
SPEED_SEED = 1                  # σπόρος του πειράματος των ταχυτήτων
ROUTE_SEEDS = [42, 1]           # σπόροι των τριών επιπλέον διαδρομών
SPP = 38                        # slots ανά θέση στα 30 km/h
SLOT = 0.5e-3                   # διάρκεια slot [s]
N_POS = 500                     # θέσεις στη διαδρομή
FREQS = ['3p5GHz', '28GHz']
FLAB = {'3p5GHz': '3.5 GHz', '28GHz': '28 GHz'}
BEST_OLLA = {'3p5GHz': 'OLLA_0.5', '28GHz': 'OLLA_0.25'}      # όπως στο k4_2_main.py
BEST_SALAD = {'3p5GHz': 'SALAD_lr0.6', '28GHz': 'SALAD_lr0.6'}
OUTAGE_POS = 30                 # 28 GHz: στις πρώτες 30 θέσεις το σήμα είναι πολύ ασθενές
SALAD_REF = (1.2, 0.15, 0.25, 0.05)     # ρυθμός μάθησης, probing, rho, k_e

# Τμήματα της κύριας διαδρομής (θέσεις από-έως). Τα όρια ορίστηκαν από το προφίλ του
# ενεργού SINR (επάνω πλαίσιο των Σχημάτων 19 και 21).
SEGMENTS = {
    '3p5GHz': [(0, 26, 'shadowing'), (26, 137, '8–17 dB'), (137, 348, 'LOS plateau'),
               (348, 429, 'drop'), (429, 500, 'weak')],
    '28GHz': [(0, 30, 'outage'), (30, 137, 'weak'), (137, 177, '13–15 dB'), (177, 216, 'LOS'),
              (216, 296, 'fast fading'), (296, 348, '14–16 dB'), (348, 429, '3–14 dB'), (429, 500, 'weak')]}
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']

# Σπόροι για ό,τι υπολογίζεται από τις τιμές ανά slot (Σχήματα 20 και 22, πίνακες
# ανά τμήμα και ανά MCS). Στην εργασία χρησιμοποιήθηκαν οι σπόροι 1, 7, 123 και 2024,
# γιατί από την εκτέλεση του σπόρου 42 για τον ILLA και τον OLLA είχαν κρατηθεί μόνο
# οι μέσες τιμές. Με SEG_SEEDS = SEEDS οι υπολογισμοί γίνονται με τους πέντε σπόρους.
SEG_SEEDS = [1, 7, 123, 2024]

# ============================================================================
# 2. Ανάγνωση των αποτελεσμάτων
# ============================================================================
def mech_key(r):
    """Σύντομο όνομα μηχανισμού και ρύθμισης για μια γραμμή των CSV."""
    if r.mech in ('ILLA_ideal', 'ILLA_noisy'):
        return r.mech
    if r.mech == 'OLLA':
        return f'OLLA_{r.delta:g}'
    assert r.mech == 'SALAD', r.mech
    p = (r.lr, r.probe, r.rho, r.ke)
    if str(r.aw) == 'True':
        assert np.allclose(p, SALAD_REF)
        return 'SALAD_AW'
    if np.allclose(p, SALAD_REF):
        return 'SALAD_ref'
    if np.allclose(p[1:], SALAD_REF[1:]):
        return f'SALAD_lr{r.lr:g}'
    return 'SALAD_tuned'


HAVE, parts = {}, []
for name in ['main', 'sensitivity', 'routes', 'speed', 'compressed']:
    path = os.path.join(RESULTS_DIR, name + '.csv')
    HAVE[name] = os.path.exists(path)
    if not HAVE[name]:
        print(f'Δεν βρέθηκε το {path}: παραλείπονται οι πίνακες και τα σχήματα που το χρειάζονται.')
        continue
    parts.append(pd.read_csv(path).assign(src=name))
assert HAVE['main'], 'Χρειάζεται το results/main.csv (k4_2_main.py).'

ALL = pd.concat(parts, ignore_index=True).rename(columns={'key': 'run', 'BLER_after30': 'BLER_after'})
ALL['key'] = [mech_key(r) for r in ALL.itertuples()]
ALL['file'] = [os.path.join(RESULTS_DIR, 'hist', run + '.npz') for run in ALL.run]
if 'clipped' not in ALL:
    ALL['clipped'] = np.nan
R = ALL[ALL['mode'] == 'realistic'].reset_index(drop=True)     # spp slots ανά θέση, με Doppler
C = ALL[ALL['mode'] == 'compressed'].reset_index(drop=True)    # συμπιεσμένος χρόνος
for d in (R, C):
    dup = d.duplicated(['freq', 'y0', 'spp', 'seed', 'key'], keep=False)
    assert not dup.any(), d[dup]


def hist(r):
    """Οι τιμές ανά slot μιας εκτέλεσης: se, harq, mcs, sinr και, για το SALAD, tau,
    probe, integ (αθροιστής του ελεγκτή) και est (εκτίμηση SINR του SALAD)."""
    with np.load(r['file']) as z:
        return {k: z[k] for k in z.files}


def bler(harq):
    """BLER [%] στα slots όπου έγινε μετάδοση."""
    h = np.asarray(harq)
    return 100 * (h[h >= 0] == 0).mean()


def one(freq, key, seed, spp=SPP, y0=-40):
    """Η μοναδική εκτέλεση με αυτά τα χαρακτηριστικά (στα 30 km/h, εκτός αν δοθεί άλλο spp)."""
    d = R[(R.freq == freq) & (R.key == key) & (R.seed == seed) & (R.spp == spp) & (R.y0 == y0)]
    assert len(d) == 1, (freq, key, seed, spp, y0, len(d))
    return d.iloc[0]


def five(freq, key, field='SE', y0=-40):
    """Οι τιμές των πέντε σπόρων, με τη σειρά του SEEDS."""
    return np.array([one(freq, key, s, y0=y0)[field] for s in SEEDS], float)


def ci95(x):
    """Μέσος όρος και μισό εύρος του 95% διαστήματος εμπιστοσύνης (κατανομή t)."""
    x = np.asarray(x, float)
    return x.mean(), stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


print(ALL.groupby(['src', 'mode']).size().to_string())

# ============================================================================
# 3. Πίνακες
# ============================================================================
M = '−'   # σύμβολο μείον
NAME = {'ILLA_ideal': 'ILLA, τέλεια εκτίμηση', 'ILLA_noisy': 'ILLA, θορυβώδης εκτίμηση',
        'OLLA_1': 'OLLA, βήμα 1 dB', 'SALAD_ref': 'SALAD, παράμετροι αναφοράς',
        'SALAD_AW': 'SALAD + anti-windup'}


def sgn(x, d=3):
    return f'+{x:.{d}f}' if x >= 0 else f'{M}{-x:.{d}f}'


def pct(x, d=1):
    return f'+{x:.{d}f}%' if x >= 0 else f'{M}{-x:.{d}f}%'


def paired(freq, a, b):
    """Διαφορά a − b ανά σπόρο: μέσος όρος, 95% CI, σχετική διαφορά ως προς το b και
    πλήθος σπόρων όπου ο a υπερέχει."""
    d = five(freq, a) - five(freq, b)
    m, h = ci95(d)
    return dict(m=m, lo=m - h, hi=m + h, rel=100 * m / five(freq, b).mean(), wins=int((d > 0).sum()))


def md_table(header, body):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body]
    return '\n'.join(out)


T = {}
# --- Πίνακες 4 και 5: σύγκριση στα 30 km/h, πέντε σπόροι ---
for f, num in [('3p5GHz', 4), ('28GHz', 5)]:
    body = []
    for k in ['ILLA_ideal', 'ILLA_noisy', 'OLLA_1', 'SALAD_ref']:
        se, bl = five(f, k), five(f, k, 'BLER')
        if k == 'SALAD_ref':
            diff = '—'
        else:
            p = paired(f, 'SALAD_ref', k)
            diff = f"{pct(p['rel'])} [{sgn(p['lo'])}, {sgn(p['hi'])}]"
        body.append([NAME[k], f'{se.mean():.3f} ± {se.std(ddof=1):.3f}', f'{bl.mean():.1f}%', diff])
    T[num] = (f'Πίνακας {num}: Αποτελέσματα στα {FLAB[f]} και 30 km/h (μέσος όρος και τυπική απόκλιση πέντε '
              f'σπόρων). Η διαφορά του SALAD δίνεται ως προς κάθε μηχανισμό, με το διάστημα εμπιστοσύνης 95% '
              f'σε αγκύλες (bits/s/Hz).',
              md_table(['Μηχανισμός', 'Φασματική απόδοση [bits/s/Hz]', 'BLER', 'Διαφορά SALAD [95% CI]'], body))

# --- Πίνακας 6: προεπιλεγμένη και καλύτερη ρύθμιση, πέντε σπόροι ---
body = []
for label, kf in [('OLLA, βήμα 1 dB (προεπιλογή του Sionna)', lambda f: 'OLLA_1'),
                  ('OLLA, καλύτερο βήμα', lambda f: BEST_OLLA[f]),
                  ('SALAD, παράμετροι αναφοράς (ρυθμός μάθησης 1.2)', lambda f: 'SALAD_ref'),
                  ('SALAD, καλύτερος ρυθμός μάθησης', lambda f: BEST_SALAD[f])]:
    row = [label]
    for f in FREQS:
        k = kf(f)
        note = ''
        if k.startswith('OLLA_') and k != 'OLLA_1':
            note = f' ({k[5:]} dB)'
        if k.startswith('SALAD_lr'):
            note = f' ({k[8:]})'
        row += [f'{five(f, k).mean():.3f}{note}', f"{five(f, k, 'BLER').mean():.1f}%"]
    body.append(row)
T[6] = ('Πίνακας 6: Επίδραση της ρύθμισης στους δύο μηχανισμούς στα 30 km/h (μέσος όρος πέντε σπόρων). '
        'Οι καλύτερες ρυθμίσεις επιλέχθηκαν με τον σπόρο 99, που δεν ανήκει στους πέντε σπόρους της σύγκρισης.',
        md_table(['Μηχανισμός', '3.5 GHz: SE [bits/s/Hz]', 'BLER', '28 GHz: SE [bits/s/Hz]', 'BLER'], body))

# --- Πίνακας 7: anti-windup, πέντε σπόροι ---
body = [['Φασματική απόδοση [bits/s/Hz]'], ['BLER'], ['BLER μετά τη διακοπή'],
        ['Χρονοθυρίδες με τον στιγμιαίο στόχο στο κάτω όριο']]
for f in FREQS:
    for k in ['SALAD_ref', 'SALAD_AW']:
        body[0].append(f'{five(f, k).mean():.3f}')
        body[1].append(f"{five(f, k, 'BLER').mean():.1f}%")
        body[2].append(f"{five(f, k, 'BLER_after').mean():.1f}%" if f == '28GHz' else '—')
        c = five(f, k, 'clipped').mean()
        body[3].append(f'{c:.2f}%' if c > 99.9 else f'{c:.1f}%')
T[7] = ('Πίνακας 7: Επίδραση του anti-windup στο SALAD με τις παραμέτρους αναφοράς, στα 30 km/h '
        '(μέσος όρος πέντε σπόρων). Το BLER μετά τη διακοπή υπολογίζεται χωρίς τις πρώτες 30 θέσεις της διαδρομής.',
        md_table(['', '3.5 GHz: χωρίς', '3.5 GHz: με', '28 GHz: χωρίς', '28 GHz: με'], body))

# --- Πίνακας 9: τέσσερις διαδρομές ---
ROUTES_Y0 = [-30, -40, -50, -60]


def route_val(f, k, y0, field='SE'):
    """−40 m: οι πέντε σπόροι των Πινάκων 4 και 5. Οι άλλες διαδρομές: σπόροι 42 και 1."""
    if y0 == -40:
        return five(f, k, field).mean()
    d = R[(R.freq == f) & (R.key == k) & (R.y0 == y0) & (R.spp == SPP) & R.seed.isin(ROUTE_SEEDS)]
    if len(d) == 0:
        return np.nan                  # π.χ. anti-windup στα 3.5 GHz, που δεν έτρεξε
    assert len(d) == len(ROUTE_SEEDS), (f, k, y0, len(d))
    return d[field].mean()


if HAVE['routes']:
    body = []
    for f in FREQS:
        for y0 in ROUTES_Y0:
            row = [f'{FLAB[f]}, {M}{-y0} m']
            for k in ['ILLA_ideal', 'OLLA_1', BEST_OLLA[f], 'SALAD_ref', 'SALAD_AW']:
                se, bl = route_val(f, k, y0), route_val(f, k, y0, 'BLER')
                row.append('—' if np.isnan(se) else f'{se:.3f} ({bl:.1f}%)')
            row.append(pct(100 * (route_val(f, 'SALAD_ref', y0) / route_val(f, BEST_OLLA[f], y0) - 1)))
            body.append(row)
    # Μέγιστη σχετική διαφορά ανάμεσα στους δύο σπόρους, για τη λεζάντα
    d = R[(R.y0 != -40) & R.seed.isin(ROUTE_SEEDS) & (R.spp == SPP)]
    g2 = d.groupby(['freq', 'y0', 'key']).SE.agg(['min', 'max', 'mean'])
    ROUTE_SPREAD = (100 * (g2['max'] - g2['min']) / g2['mean']).max()
    T[9] = ('Πίνακας 9: Φασματική απόδοση (bits/s/Hz) και, σε παρένθεση, BLER σε τέσσερις διαδρομές στον ίδιο '
            'δρόμο, με διαφορετικό σημείο εκκίνησης y0, στα 30 km/h. Η διαδρομή των −40 m είναι αυτή της Ενότητας '
            '4.2 (μέσοι όροι πέντε σπόρων). Για τις άλλες τρεις διαδρομές οι τιμές είναι μέσοι όροι των σπόρων 42 και '
            f'1, που διαφέρουν μεταξύ τους λιγότερο από {np.ceil(10 * ROUTE_SPREAD) / 10:.1f}%. Οι χρονοθυρίδες '
            'χωρίς μετάδοση, στις θέσεις όπου δεν φτάνει καμία διαδρομή διάδοσης, δεν μετρούν στο BLER. Η τελευταία '
            'στήλη δίνει τη διαφορά του SALAD από τον OLLA με το καλύτερο βήμα (0.5 dB στα 3.5 GHz και 0.25 dB '
            'στα 28 GHz).',
            md_table(['Διαδρομή', 'ILLA, τέλεια εκτίμηση', 'OLLA, βήμα 1 dB', 'OLLA, καλύτερο βήμα',
                      'SALAD, παράμετροι αναφοράς', 'SALAD + anti-windup', 'SALAD έναντι καλύτερου OLLA'], body))

# --- Πίνακας 10: συμπιεσμένος χρόνος έναντι 30 km/h ---
COMP_KEYS = ['ILLA_ideal', 'ILLA_noisy', 'OLLA_1', 'OLLA_best', 'SALAD_ref', 'SALAD_AW']
if HAVE['compressed']:
    def comp(f, k):
        """Μέση φασματική απόδοση και BLER των πέντε σπόρων στον συμπιεσμένο χρόνο."""
        d = C[(C.freq == f) & (C.key == k)]
        assert sorted(d.seed) == sorted(SEEDS), (f, k, list(d.seed))
        return d.SE.mean(), d.BLER.mean()

    # Στον συμπιεσμένο χρόνο το καλύτερο βήμα του OLLA είναι αυτό με την υψηλότερη μέση
    # φασματική απόδοση στους πέντε σπόρους (δεν υπάρχει χωριστός σπόρος επιλογής).
    COMP_BEST = {f: C[(C.freq == f) & (C.mech == 'OLLA')].groupby('key').SE.mean().idxmax() for f in FREQS}
    body = []
    for k in COMP_KEYS:
        row = [NAME.get(k, 'OLLA, καλύτερο βήμα')]
        for f in FREQS:
            kc = COMP_BEST[f] if k == 'OLLA_best' else k
            kr = BEST_OLLA[f] if k == 'OLLA_best' else k
            c_se, c_bl = comp(f, kc)
            se, bl = five(f, kr).mean(), five(f, kr, 'BLER').mean()
            if k == 'OLLA_best':
                row += [f'{c_se:.3f} / {c_bl:.1f}% ({kc[5:]} dB)', f'{se:.3f} / {bl:.1f}% ({kr[5:]} dB)']
            else:
                row += [f'{c_se:.3f} / {c_bl:.1f}%', f'{se:.3f} / {bl:.1f}%']
        body.append(row)
    T[10] = ('Πίνακας 10: Φασματική απόδοση (bits/s/Hz) και BLER στον συμπιεσμένο χρόνο (μία θέση ανά χρονοθυρίδα, '
             'περίπου 1150 km/h) και στα 30 km/h (38 χρονοθυρίδες ανά θέση), μέσοι όροι πέντε σπόρων. Σε παρένθεση '
             'το καλύτερο βήμα του OLLA σε κάθε περίπτωση.',
             md_table(['Μηχανισμός', '3.5 GHz: συμπιεσμένος', '3.5 GHz: 30 km/h', '28 GHz: συμπιεσμένος',
                       '28 GHz: 30 km/h'], body))

# --- Πίνακας 11: ταχύτητες (σπόρος 1) ---
SPEEDS = [(380, 3.0), (115, 10.0), (38, 30.4), (19, 60.8), (10, 115.4)]    # (slots ανά θέση, km/h)


def speed_cell(f, k, spp):
    d = R[(R.freq == f) & (R.key == k) & (R.seed == SPEED_SEED) & (R.spp == spp) & (R.y0 == -40)]
    return f'{d.SE.iloc[0]:.3f} / {d.BLER.iloc[0]:.1f}%' if len(d) else '—'


if HAVE['speed']:
    body = []
    for spp, v in SPEEDS:
        for f in FREQS:
            if not len(R[(R.freq == f) & (R.spp == spp) & (R.seed == SPEED_SEED)]):
                continue                       # τα 3 km/h προσομοιώθηκαν μόνο στα 28 GHz
            body.append([f'{v:g} km/h', FLAB[f]] + [speed_cell(f, k, spp) for k in
                                                    ['ILLA_ideal', 'OLLA_1', BEST_OLLA[f], 'SALAD_ref']])
    body.sort(key=lambda r: (r[1] != '3.5 GHz', float(r[0].split()[0])))
    T[11] = ('Πίνακας 11: Φασματική απόδοση (bits/s/Hz) και BLER σε διάφορες ταχύτητες του χρήστη (σπόρος 1). '
             'Το καλύτερο βήμα του OLLA είναι 0.5 dB στα 3.5 GHz και 0.25 dB στα 28 GHz.',
             md_table(['Ταχύτητα', 'Συχνότητα', 'ILLA, τέλεια εκτίμηση', 'OLLA, βήμα 1 dB',
                       'OLLA, καλύτερο βήμα', 'SALAD, παράμετροι αναφοράς'], body))

# --- Δεδομένα του Σχήματος 23: σαρώσεις με τον σπόρο 99 ---
if HAVE['sensitivity']:
    body = []
    for f in FREQS:
        d = R[(R.freq == f) & (R.seed == SEL_SEED)]
        for r in d.sort_values('key').itertuples():
            extra = f' (στόχος στο κάτω όριο {r.clipped:.1f}%)' if r.key.startswith('SALAD') else ''
            body.append([FLAB[f], r.key, f'{r.SE:.3f}', f'{r.BLER:.1f}%{extra}'])
    T['sweep'] = ('Δεδομένα: σαρώσεις με τον σπόρο 99 (Σχήμα 23)',
                  md_table(['Συχνότητα', 'Ρύθμιση', 'SE [bits/s/Hz]', 'BLER'], body))


# --- Δεδομένα των Σχημάτων 20 και 22: φασματική απόδοση ανά τμήμα της διαδρομής ---
def seg_means(f, key, seeds=SEG_SEEDS, field='se'):
    """Μέση τιμή του field σε κάθε τμήμα της διαδρομής, μέσος όρος των σπόρων."""
    out = []
    for s in seeds:
        x = hist(one(f, key, s))[field].astype(float).reshape(N_POS, SPP)
        out.append([x[a:b].mean() for a, b, _ in SEGMENTS[f]])
    return np.mean(out, axis=0)


SEED_TXT = ', '.join(str(s) for s in SEG_SEEDS)
SEG_KEYS = {f: ['ILLA_ideal', 'OLLA_1', BEST_OLLA[f], 'SALAD_ref', 'SALAD_AW'] for f in FREQS}
SEG = {f: {k: seg_means(f, k) for k in SEG_KEYS[f]} for f in FREQS}
body = []
for f in FREQS:
    sinr = seg_means(f, 'ILLA_ideal', field='sinr')
    for i, (a, b, lab) in enumerate(SEGMENTS[f]):
        body.append([FLAB[f], f'{ROMAN[i]} {lab}', f'{a * SPP * SLOT:.2f}–{b * SPP * SLOT:.2f}',
                     f'{sinr[i]:.1f}'.replace('-', M)]
                    + [f'{SEG[f][k][i]:.2f}' for k in SEG_KEYS[f]])
T['seg'] = (f'Δεδομένα: φασματική απόδοση ανά τμήμα της διαδρομής, μέσος όρος των σπόρων {SEED_TXT} '
            '(Σχήματα 20 και 22). Στήλες: ILLA τέλεια, OLLA 1 dB, OLLA καλύτερο βήμα, SALAD, SALAD + AW',
            md_table(['Συχνότητα', 'Τμήμα', 't [s]', 'SINR [dB]', 'ILLA', 'OLLA 1', 'OLLA best', 'SALAD',
                      'SALAD+AW'], body))

# --- 3.5 GHz, LOS plateau: πόσο συχνά επιλέγεται κάθε υψηλό MCS και πόσο συχνά αποτυγχάνει ---
a, b, _ = SEGMENTS['3p5GHz'][2]
body = []
for k in ['ILLA_ideal', 'OLLA_1', BEST_OLLA['3p5GHz'], 'SALAD_ref', BEST_SALAD['3p5GHz']]:
    mc, hq = [], []
    for s in SEG_SEEDS:
        h = hist(one('3p5GHz', k, s))
        mc.append(h['mcs'].reshape(N_POS, SPP)[a:b].ravel())
        hq.append(h['harq'].reshape(N_POS, SPP)[a:b].ravel())
    mc, hq = np.concatenate(mc).astype(int), np.concatenate(hq).astype(int)
    cells = []
    for m in [24, 25, 26, 27]:
        sel = mc == m
        cells.append(f'{100 * sel.mean():.1f}%' + (f' ({bler(hq[sel]):.0f}% αποτυχίες)' if sel.sum() > 50 else ''))
    other = 100 * (mc < 24).mean()
    body.append([k] + cells + [f'{other:.1f}%', f'{bler(hq):.1f}%', f'{np.mean(seg_means("3p5GHz", k)[2]):.2f}'])
T['plateau'] = (f'Δεδομένα: 3.5 GHz, LOS plateau (θέσεις {a}–{b - 1}), μερίδιο των χρονοθυρίδων ανά MCS και, σε '
                f'παρένθεση, το ποσοστό αποτυχιών σε αυτό το MCS (σπόροι {SEED_TXT})',
                md_table(['Μηχανισμός', 'MCS 24', 'MCS 25', 'MCS 26', 'MCS 27', 'MCS < 24', 'BLER', 'SE'], body))

# --- Αριθμοί που αναφέρει το κείμενο ---
lines = []
for k in ['ILLA_ideal', 'OLLA_1', BEST_OLLA['28GHz'], 'SALAD_ref', 'SALAD_AW']:
    v = [one('28GHz', k, s)['BLER_after'] for s in SEEDS]
    v = [x for x in v if pd.notna(x)]
    lines.append(f'- 28 GHz, BLER μετά τη διακοπή, {k}: {np.mean(v):.1f}% ({len(v)} σπόροι)')
a, b = SEGMENTS['28GHz'][2][0], N_POS
cells = []
for k in ['ILLA_ideal', 'OLLA_1', BEST_OLLA['28GHz'], 'SALAD_ref', 'SALAD_AW']:
    v = [bler(hist(one('28GHz', k, s))['harq'].reshape(N_POS, SPP)[a:b].ravel()) for s in SEG_SEEDS]
    cells.append(f'{k} {np.mean(v):.1f}%')
lines.append(f'- 28 GHz, BLER στα τμήματα III έως VIII (σπόροι {SEED_TXT}): ' + ', '.join(cells))
for f in FREQS:
    for k in ['SALAD_ref', 'SALAD_AW']:
        h = hist(one(f, k, 42))
        sl = slice(OUTAGE_POS * SPP, None) if f == '28GHz' else slice(None)
        txt = f'- {FLAB[f]}, σπόρος 42, {k}: διάμεσος στιγμιαίος στόχος {np.median(h["tau"][sl].astype(float)):.3f}'
        if f == '3p5GHz':
            a, b, _ = SEGMENTS[f][2]
            e = h['est'].reshape(N_POS, SPP)[a:b].mean()
            s = h['sinr'].astype(float).reshape(N_POS, SPP)[a:b].mean()
            txt += f', στο plateau εκτίμηση SINR του SALAD {e:.1f} dB έναντι ενεργού SINR {s:.1f} dB'
        lines.append(txt)
if HAVE['sensitivity']:
    for f in FREQS:
        d = R[(R.freq == f) & (R.seed == SEL_SEED) & (R.spp == SPP) & (R.y0 == -40)]
        for pre, lab in [('OLLA_', 'OLLA, βήματα 0.25 έως 4 dB'), ('SALAD_lr', 'SALAD, ρυθμοί μάθησης 0.6 έως 3')]:
            dd = d[d.key.str.startswith(pre) | ((d.key == 'SALAD_ref') if pre == 'SALAD_lr' else False)]
            lo, hi = dd.loc[dd.SE.idxmin()], dd.loc[dd.SE.idxmax()]
            lines.append(f'- {FLAB[f]}, σπόρος 99, {lab}: από {lo.SE:.3f} ({lo.key}) έως {hi.SE:.3f} ({hi.key}), '
                         f'η καλύτερη ρύθμιση υπερέχει της χειρότερης κατά {100 * (hi.SE / lo.SE - 1):.1f}%')
        tu = d[d.key == 'SALAD_tuned'].SE.iloc[0]
        lines.append(f'- {FLAB[f]}, σπόρος 99, SALAD με τη ρύθμιση του συμπιεσμένου χρόνου: {tu:.3f} '
                     f'(παράμετροι αναφοράς {d[d.key == "SALAD_ref"].SE.iloc[0]:.3f})')
for f in FREQS:
    for a_, b_ in [('SALAD_ref', 'OLLA_1'), ('SALAD_ref', BEST_OLLA[f]), (BEST_SALAD[f], BEST_OLLA[f]),
                   ('SALAD_AW', 'SALAD_ref'), ('SALAD_ref', 'ILLA_ideal'), ('OLLA_1', 'ILLA_ideal')]:
        p = paired(f, a_, b_)
        lines.append(f"- {FLAB[f]}: {a_} − {b_} = {sgn(p['m'])} [{sgn(p['lo'])}, {sgn(p['hi'])}] "
                     f"({pct(p['rel'])}), νίκες {p['wins']}/5")
if HAVE['routes']:
    for f in FREQS:
        for y0 in ROUTES_Y0:
            s_, o_ = route_val(f, 'SALAD_ref', y0), route_val(f, BEST_OLLA[f], y0)
            i_, a_ = route_val(f, 'ILLA_ideal', y0), route_val(f, 'SALAD_AW', y0)
            txt = f'- {FLAB[f]}, διαδρομή {M}{-y0} m: SALAD έναντι ILLA τέλειας εκτίμησης {pct(100 * (s_ / i_ - 1))}'
            if not np.isnan(a_):
                txt += (f', anti-windup έναντι SALAD {pct(100 * (a_ / s_ - 1))}, '
                        f'anti-windup έναντι καλύτερου OLLA {pct(100 * (a_ / o_ - 1))}')
            if y0 != -40:
                h = hist(one(f, 'SALAD_ref', 42, y0=y0))['harq'].reshape(N_POS, SPP)
                nt = np.where((h == -1).all(axis=1))[0]
                txt += f', θέσεις χωρίς μετάδοση {len(nt)} ({nt.min()}–{nt.max()})'
            lines.append(txt)
    lines.append(f'- Διαδρομές: μέγιστη διαφορά ανάμεσα στους σπόρους 42 και 1: {ROUTE_SPREAD:.2f}%')
T['text'] = ('Δεδομένα: αριθμοί για το κείμενο', '\n'.join(lines))

with open(os.path.join(FIG_DIR, 'tables_k4.md'), 'w', encoding='utf-8') as fh:
    for k, (cap, tab) in T.items():
        fh.write(f'### {cap}\n\n{tab}\n\n')
for k, (cap, tab) in T.items():
    print(cap, '\n', tab, '\n', sep='')

# ============================================================================
# 4. Μορφή των σχημάτων
# ============================================================================
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C_ILLA, C_OLLA, C_SALAD, C_AW, C_NOISY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
C_SINR = '#6f6e69'
STYLE = {'ILLA_ideal': dict(color=C_ILLA, label='ILLA, ideal estimate'),
         'ILLA_noisy': dict(color=C_NOISY, label='ILLA, noisy estimate'),
         'OLLA_1': dict(color=C_OLLA, label='OLLA, 1 dB step'),
         'SALAD_ref': dict(color=C_SALAD, label='SALAD, reference parameters'),
         'SALAD_AW': dict(color=C_AW, label='SALAD + anti-windup')}
SHORT = {'ILLA_ideal': 'ILLA', 'OLLA_1': 'OLLA', 'SALAD_ref': 'SALAD', 'SALAD_AW': 'SALAD + AW'}
plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.titleweight': 'normal',
    'axes.titlecolor': INK, 'axes.labelcolor': INK, 'axes.labelsize': 8.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False, 'legend.fontsize': 8, 'legend.handlelength': 2.2, 'legend.labelcolor': INK,
    'lines.linewidth': 1.4, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})
FIG_W = 6.3        # πλάτος σε ίντσες (16 cm)
MA_SLOTS = 200     # κυλιόμενο παράθυρο 0.1 s


def movavg(x, w):
    """Κεντρικός κυλιόμενος μέσος όρος w slots (NaN στα άκρα)."""
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w // 2: w // 2 + len(x) - w + 1] = (c[w:] - c[:-w]) / w
    return out


def win_bler(harq, w):
    """BLER [%] σε κυλιόμενο παράθυρο w slots (μόνο τα slots με μετάδοση)."""
    h = np.asarray(harq).astype(int)
    v = movavg(h >= 0, w)
    return 100 * movavg(h == 0, w) / np.where(v > 0, v, np.nan)


def seg_lines(ax, f, labels=False):
    """Κάθετες γραμμές στα όρια των τμημάτων και, προαιρετικά, οι λατινικοί αριθμοί τους."""
    for i, (a, b, _) in enumerate(SEGMENTS[f]):
        if a > 0:
            ax.axvline(a * SPP * SLOT, color=AXIS, lw=0.7, zorder=0.5)
        if labels:
            ax.text((a + b) / 2 * SPP * SLOT, 1.02, ROMAN[i], transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7.5, color=INK2)


def top_legend(fig, ax, handles=None, labels=None, ncol=3, y=1.13):
    """Υπόμνημα πάνω από το πλαίσιο ax (y σε μονάδες του πλαισίου)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.8,
              columnspacing=1.6, borderaxespad=0)


def save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p)
    plt.close(fig)
    print('αποθηκεύτηκε:', p)


# ============================================================================
# 5. Σχήματα 19 και 21: χρονοσειρές μιας εκτέλεσης στα 30 km/h (σπόρος 1)
# ============================================================================
TS_KEYS = ['ILLA_ideal', 'OLLA_1', 'SALAD_ref']


def fig_timeseries(f, seed=1):
    H = {k: hist(one(f, k, seed)) for k in TS_KEYS}
    t = np.arange(len(H['OLLA_1']['se'])) * SLOT
    fig, ax = plt.subplots(3, 1, figsize=(FIG_W, 6.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1.25, 1], hspace=0.12))
    ax[0].plot(t, H['ILLA_ideal']['sinr'].astype(float), color=C_SINR, lw=0.5)
    ax[0].set_ylabel('Effective SINR [dB]')
    for k in TS_KEYS:
        ax[1].plot(t, movavg(H[k]['se'], MA_SLOTS), lw=1.1, **STYLE[k])
        ax[2].plot(t, win_bler(H[k]['harq'], MA_SLOTS), lw=1.0, color=STYLE[k]['color'])
    ax[1].set_ylabel('SE [bits/s/Hz]')
    ax[2].axhline(10, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax[2].text(t[-1], 10, 'BLER target 10%', ha='right', va='bottom', fontsize=7.5, color=INK2, zorder=5,
               bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    ax[2].set_ylabel('BLER [%]')
    ax[2].set_ylim(0, 25 if f == '3p5GHz' else 60)
    ax[2].set_xlabel('Time [s]')
    ax[2].set_xlim(0, t[-1])
    for i, a in enumerate(ax):
        seg_lines(a, f, labels=(i == 0))
    h, l = ax[1].get_legend_handles_labels()
    top_legend(fig, ax[0], h, l, ncol=3, y=1.13)
    return fig


save(fig_timeseries('3p5GHz'), 'fig19_timeseries_3p5GHz.png')
save(fig_timeseries('28GHz'), 'fig21_timeseries_28GHz.png')


# ============================================================================
# 6. Σχήματα 20 και 22: φασματική απόδοση ανά τμήμα της διαδρομής
# ============================================================================
def fig_segments(f):
    keys = ['ILLA_ideal', 'OLLA_1', 'SALAD_ref']
    segs = SEGMENTS[f]
    x = np.arange(len(segs))
    w, gap = 0.24, 0.03
    fig, ax = plt.subplots(figsize=(FIG_W, 2.75))
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1) * (w + gap), SEG[f][k], width=w, color=STYLE[k]['color'], label=STYLE[k]['label'],
               zorder=2)
    ax.set_xticks(x, [f'{ROMAN[i]}\n{lab}\n{a * SPP * SLOT:.1f}–{b * SPP * SLOT:.1f} s'
                      for i, (a, b, lab) in enumerate(segs)], fontsize=7.5 if len(segs) < 7 else 7)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('Mean SE [bits/s/Hz]')
    ax.set_xlim(-0.6, len(segs) - 0.4)
    top_legend(fig, ax, ncol=3, y=1.04)
    return fig


save(fig_segments('3p5GHz'), 'fig20_segments_3p5GHz.png')
save(fig_segments('28GHz'), 'fig22_segments_28GHz.png')


# ============================================================================
# 7. Σχήμα 23: ευαισθησία στη ρύθμιση (σπόρος 99)
# ============================================================================
def fig_sweep():
    fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 4.9), sharey='row', gridspec_kw=dict(hspace=0.6, wspace=0.08))
    for i, f in enumerate(FREQS):
        d = R[(R.freq == f) & (R.seed == SEL_SEED) & (R.spp == SPP) & (R.y0 == -40)]
        ol = d[d.key.str.startswith('OLLA_')].copy()
        ol['x'] = ol.key.str[5:].astype(float)
        sa = d[d.key.str.startswith('SALAD_lr') | (d.key == 'SALAD_ref')].copy()
        sa['x'] = [1.2 if k == 'SALAD_ref' else float(k[8:]) for k in sa.key]
        illa = d[d.key == 'ILLA_ideal'].SE.iloc[0]
        for j, (dd, col, xlab, xdef, deflab) in enumerate([
                (ol.sort_values('x'), C_OLLA, 'OLLA step after NACK [dB]', 1.0, 'Sionna default'),
                (sa.sort_values('x'), C_SALAD, 'SALAD learning rate', 1.2, 'reference')]):
            ax = axs[i, j]
            ax.axhline(illa, color=C_ILLA, lw=1.1, ls=(0, (4, 3)), zorder=1)
            ax.plot(dd.x, dd.SE, color=col, marker='o', ms=4.5, mec='white', mew=1.0, zorder=3)
            r0 = dd[np.isclose(dd.x, xdef)].iloc[0]          # η προεπιλεγμένη ρύθμιση, με κύκλο
            ax.plot(r0.x, r0.SE, marker='o', ms=9, mfc='none', mec=INK2, mew=0.9, zorder=4)
            ax.annotate(deflab, (r0.x, r0.SE), xytext=(7, 7), textcoords='offset points', ha='left',
                        va='bottom', fontsize=7.5, color=INK2)
            ax.set_xscale('log')
            ax.set_xticks(dd.x, [f'{v:g}' for v in dd.x])
            ax.minorticks_off()
            ax.set_xlabel(xlab)
            ax.set_title(f'{FLAB[f]}: {"OLLA" if j == 0 else "SALAD"}', loc='left')
            if j == 0:
                ax.text(dd.x.max(), illa, 'ILLA, ideal estimate', ha='right', va='bottom', fontsize=7.5,
                        color=INK2)
        axs[i, 0].set_ylabel('SE [bits/s/Hz]')
        vals = np.r_[ol.SE.values, sa.SE.values, illa]
        pad = 0.12 * (vals.max() - vals.min())
        axs[i, 0].set_ylim(vals.min() - 1.6 * pad, vals.max() + pad)
    return fig


if HAVE['sensitivity']:
    save(fig_sweep(), 'fig23_sensitivity.png')


# ============================================================================
# 8. Σχήμα 24: ο κορεσμός του ελεγκτή του SALAD στα 28 GHz (σπόρος 42)
# ============================================================================
def fig_windup(f='28GHz', seed=42):
    H = {'SALAD_ref': hist(one(f, 'SALAD_ref', seed)), 'SALAD_AW': hist(one(f, 'SALAD_AW', seed))}
    t = np.arange(len(H['SALAD_ref']['se'])) * SLOT
    fig, ax = plt.subplots(4, 1, figsize=(FIG_W, 6.9), sharex=True, gridspec_kw=dict(hspace=0.14))
    for k in ['SALAD_AW', 'SALAD_ref']:
        h, st = H[k], STYLE[k]
        ax[0].plot(t, h['integ'], lw=1.4, **st)                       # αθροιστής του ελεγκτή
        if k == 'SALAD_AW':
            ax[1].plot(t, h['tau'].astype(float), lw=0.35, color=st['color'], alpha=0.9)
        else:
            ax[1].plot(t, h['tau'].astype(float), lw=1.6, color=st['color'])
        ax[2].plot(t, win_bler(h['harq'], MA_SLOTS), lw=1.0, color=st['color'])
        ax[3].plot(t, movavg(h['se'], MA_SLOTS), lw=1.1, color=st['color'])
    ax[0].set_ylabel('Integral error')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Instantaneous\nBLER target')
    ax[2].set_ylabel('BLER [%]')
    ax[2].set_ylim(0, 40)
    ax[3].set_ylabel('SE [bits/s/Hz]')
    for a in (ax[1], ax[2]):
        y = 0.1 if a is ax[1] else 10
        a.axhline(y, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=4)
        a.text(t[-1], y, 'BLER target 10%', ha='right', va='bottom', fontsize=7.5, color=INK2, zorder=5,
               bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85))
    ax[3].set_xlabel('Time [s]')
    ax[3].set_xlim(0, t[-1])
    for i, a in enumerate(ax):
        seg_lines(a, f, labels=(i == 0))
    h, l = ax[0].get_legend_handles_labels()
    top_legend(fig, ax[0], h[::-1], l[::-1], ncol=2, y=1.15)
    return fig


save(fig_windup(), 'fig24_windup_28GHz.png')


# ============================================================================
# 9. Σχήμα 27: φασματική απόδοση και BLER σε διάφορες ταχύτητες (σπόρος 1)
# ============================================================================
def fig_speed():
    fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 4.9), sharex=True, gridspec_kw=dict(hspace=0.14, wspace=0.28))
    vmap = dict(SPEEDS)
    for j, f in enumerate(FREQS):
        series = [('ILLA_ideal', dict(color=C_ILLA, ls='-'), 'ILLA, ideal estimate'),
                  (BEST_OLLA[f], dict(color=C_OLLA, ls='-'), 'OLLA, best step'),
                  ('OLLA_1', dict(color=C_OLLA, ls=(0, (3, 2))), 'OLLA, 1 dB step'),
                  ('SALAD_ref', dict(color=C_SALAD, ls='-'), 'SALAD, reference')]
        for k, st, lab in series:
            d = R[(R.freq == f) & (R.key == k) & (R.seed == SPEED_SEED) & (R.y0 == -40)].copy()
            d['v'] = [vmap[s] for s in d.spp]
            d = d.sort_values('v')
            mk = dict(marker='o', ms=4.5, mec='white', mew=1.0)
            axs[0, j].plot(d.v, d.SE, label=lab, **st, **mk)
            axs[1, j].plot(d.v, d.BLER, **st, **mk)
        axs[0, j].set_title(FLAB[f], loc='left')
        axs[1, j].axhline(10, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=0.8)
        axs[1, j].set_xlabel('User speed [km/h]')
    axs[0, 0].set_ylabel('SE [bits/s/Hz]')
    axs[1, 0].set_ylabel('BLER [%]')
    for a in axs.ravel():
        a.set_xscale('log')
        a.set_xticks([3, 10, 30.4, 60.8, 115.4], ['3', '10', '30', '61', '115'])
        a.minorticks_off()
        a.set_xlim(2.4, 145)
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(h, l, loc='lower left', bbox_to_anchor=(0.0, 1.13), ncol=4, handlelength=2.0,
                     columnspacing=1.2, borderaxespad=0)
    return fig


if HAVE['speed']:
    save(fig_speed(), 'fig27_speed.png')


# ============================================================================
# 10. Σχήμα 28: συμπιεσμένος χρόνος έναντι 30 km/h (πέντε σπόροι)
# ============================================================================
def fig_compression():
    keys = ['ILLA_ideal', 'OLLA_1', 'SALAD_ref', 'SALAD_AW']
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.3), gridspec_kw=dict(wspace=1.05))
    for ax, f in zip(axs, FREQS):
        ends = []
        for k in keys:
            y = [comp(f, k)[0], five(f, k).mean()]
            ax.plot([0, 1], y, marker='o', ms=5, mec='white', mew=1.0, color=STYLE[k]['color'],
                    label=STYLE[k]['label'], lw=1.6, zorder=3 if k != 'SALAD_AW' else 2.5)
            ends.append([y[1], k])
        # Ετικέτες στο δεξί άκρο, απομακρυσμένες όταν πέφτουν η μία πάνω στην άλλη
        lo, hi = ax.get_ylim()
        gap = 0.075 * (hi - lo)
        ends.sort()
        pos = [e[0] for e in ends]
        for _ in range(50):
            for i in range(1, len(pos)):
                if pos[i] - pos[i - 1] < gap:
                    mid = (pos[i] + pos[i - 1]) / 2
                    pos[i - 1], pos[i] = mid - gap / 2, mid + gap / 2
        for (y, k), yl in zip(ends, pos):
            ax.annotate(f'{SHORT[k]}  {y:.3f}', (1, y), xytext=(1.1, yl), textcoords='data', va='center',
                        fontsize=7.5, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.6, shrinkA=0, shrinkB=3))
        ax.set_xlim(-0.15, 1.05)
        ax.set_xticks([0, 1], ['compressed\n(1 slot per position,\n≈1150 km/h)', '30 km/h\n(38 slots\nper position)'])
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='x', visible=False)
        ax.set_title(FLAB[f], loc='left')
        ax.set_ylabel('SE [bits/s/Hz]')
    h, l = axs[0].get_legend_handles_labels()
    axs[0].legend(h, l, loc='lower left', bbox_to_anchor=(0.0, 1.12), ncol=4, handlelength=1.6,
                  columnspacing=1.2, borderaxespad=0)
    return fig


if HAVE['compressed']:
    save(fig_compression(), 'fig28_compression.png')

print('Τα σχήματα και το tables_k4.md είναι στον φάκελο', FIG_DIR)
