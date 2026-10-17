# -*- coding: utf-8 -*-
"""
k4_3_routes.py — Ενότητα 4.3.3: τρεις επιπλέον διαδρομές στον ίδιο δρόμο.

Παράγει τα δεδομένα του Πίνακα 10 και του Σχήματος 27. Οι διαδρομές έχουν το ίδιο
μήκος (80 m) και την ίδια ταχύτητα (30 km/h) με την κύρια, αλλά αρχίζουν στο
y0 = -30, -50 και -60 m.
Οι τιμές της κύριας διαδρομής (y0 = -40 m) προέρχονται από το k4_2_main.py.
Δύο σπόροι (42 και 1) ανά διαδρομή. Μηχανισμοί: ILLA με τέλεια εκτίμηση, OLLA με
βήμα 1 dB και με το καλύτερο βήμα, SALAD με τις παραμέτρους αναφοράς, και στα
28 GHz επιπλέον SALAD με anti-windup.

Οι διαδρομές -30, -50 και -60 m περιέχουν θέσεις όπου δεν φτάνει καμία διαδρομή
διάδοσης (10, 33 και 95 θέσεις). Εκεί δεν γίνεται μετάδοση (HARQ = -1) και τα slots
αυτά δεν μετρούν στο BLER. Το περιτύλιγμα του SALAD (la_common.SaladWrapper) βγάζει
από την ουρά του SALAD το MCS κάθε τέτοιου slot, που δεν στάλθηκε ποτέ.

Χρόνος: 54 εκτελέσεις, περίπου 3 ώρες σε GPU T4.
Έξοδος: results/routes.csv και results/hist/routes_*.npz.
"""
from la_common import SALAD_REF, spec, run_all

ROUTES = [-30, -50, -60]
ROUTE_SEEDS = [42, 1]
BEST_OLLA = {'3p5GHz': 0.5, '28GHz': 0.25}   # [dB], όπως στο k4_2_main.py

specs = []
for f in ['28GHz', '3p5GHz']:
    for y0 in ROUTES:
        for s in ROUTE_SEEDS:
            specs += [spec('routes', f, 'ILLA_ideal', s, y0=y0),
                      spec('routes', f, 'OLLA', s, y0=y0, delta=1.0),
                      spec('routes', f, 'OLLA', s, y0=y0, delta=BEST_OLLA[f]),
                      spec('routes', f, 'SALAD', s, y0=y0, salad=SALAD_REF)]
            if f == '28GHz':
                specs += [spec('routes', f, 'SALAD', s, y0=y0, salad=SALAD_REF, aw=True)]

if __name__ == '__main__':
    df = run_all(specs, 'routes.csv')
    print(df.groupby(['freq', 'y0', 'algorithm'], sort=False)[['SE', 'BLER']].mean().round(3).to_string())
