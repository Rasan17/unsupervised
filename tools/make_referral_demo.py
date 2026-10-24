"""Generates the simulated referral-patterns demonstration (not real patients or referrals).

demo_referrals.csv   8000 referrals to a neurosurgical service with five built-in referral patterns:
                     routine degenerative spine from general practice; emergency and tertiary cranial
                     transfers; suspected tumour on the two-week-wait pathway; repeat referrals for chronic
                     pain; and functional neurosurgery from internal consultants. Mixed variables (source,
                     urgency, pathway, distance, waiting times, previous referrals, imaging), some missing
                     values, the outcome of the referral (to describe the patterns afterwards, not to find
                     them) and the simulated true pattern (keep it as "describe only").

Usage: python3 tools/make_referral_demo.py   (reproducible: fixed seed)
"""
import csv, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
COLS = ['Referral_ID', 'Age', 'Sex', 'Referral_Source', 'Urgency', 'Pathway', 'Distance_km', 'Days_to_Triage',
        'Days_to_Clinic', 'Previous_Referrals_2y', 'Imaging_Before_Referral', 'Referral_Outcome', 'Simulated_Referral_Pattern']
SOURCES = ['General practice', 'Emergency department', 'Other hospital', 'Consultant (internal)', 'Physiotherapy']
URGENCY = ['Routine', 'Urgent', 'Two-week wait']
PATHWAY = ['Spine', 'Cranial', 'Functional', 'Peripheral nerve']
OUTCOME = ['Listed for surgery', 'Clinic follow-up', 'Discharged', 'Did not attend']
PATTERNS = {
    'Routine degenerative spine': dict(share=0.45, age=(66, 10), female=0.5, src=[0.88, 0.02, 0.02, 0.05, 0.03], urg=[0.90, 0.09, 0.01], path=[0.93, 0.02, 0.01, 0.04],
                                       dist=25, triage=10, clinic=95, prior=0.3, img=0.80, out=[0.18, 0.47, 0.32, 0.03]),
    'Emergency and tertiary cranial': dict(share=0.20, age=(58, 18), female=0.45, src=[0.02, 0.55, 0.42, 0.01, 0.00], urg=[0.03, 0.95, 0.02], path=[0.08, 0.90, 0.01, 0.01],
                                           dist=60, triage=0.3, clinic=5, prior=0.2, img=0.99, out=[0.45, 0.40, 0.13, 0.02]),
    'Suspected tumour (two-week wait)': dict(share=0.12, age=(64, 12), female=0.5, src=[0.60, 0.03, 0.02, 0.35, 0.00], urg=[0.01, 0.02, 0.97], path=[0.15, 0.82, 0.00, 0.03],
                                             dist=30, triage=2, clinic=12, prior=0.3, img=0.97, out=[0.35, 0.50, 0.13, 0.02]),
    'Repeat referrals, chronic pain': dict(share=0.18, age=(48, 11), female=0.6, src=[0.50, 0.02, 0.03, 0.05, 0.40], urg=[0.93, 0.06, 0.01], path=[0.60, 0.01, 0.09, 0.30],
                                           dist=15, triage=20, clinic=130, prior=3.0, img=0.30, out=[0.04, 0.22, 0.54, 0.20]),
    'Functional neurosurgery': dict(share=0.05, age=(55, 14), female=0.45, src=[0.12, 0.00, 0.10, 0.75, 0.03], urg=[0.94, 0.06, 0.00], path=[0.02, 0.01, 0.97, 0.00],
                                    dist=90, triage=8, clinic=60, prior=1.0, img=0.70, out=[0.20, 0.70, 0.08, 0.02]),
}


def main(n=8000, seed=2041):
    rng = np.random.default_rng(seed)
    names = list(PATTERNS)
    counts = [int(round(PATTERNS[k]['share'] * n)) for k in names]
    counts[0] += n - sum(counts)
    rows = []
    for name, cnt in zip(names, counts):
        q = PATTERNS[name]
        for _ in range(cnt):
            pick = lambda opts, probs: opts[rng.choice(len(opts), p=np.array(probs) / sum(probs))]
            age = int(np.clip(round(rng.normal(*q['age'])), 16, 95))
            dist = round(float(q['dist'] * np.exp(0.6 * rng.normal())), 1)
            triage = int(round(q['triage'] * np.exp(0.5 * rng.normal()))) if q['triage'] >= 1 else int(rng.random() < q['triage'])
            clinic = int(max(0, round(q['clinic'] * np.exp(0.35 * rng.normal()))))
            rows.append([age, 'Female' if rng.random() < q['female'] else 'Male', pick(SOURCES, q['src']), pick(URGENCY, q['urg']), pick(PATHWAY, q['path']),
                         dist, triage, clinic, int(rng.poisson(q['prior'])), 'Yes' if rng.random() < q['img'] else 'No', pick(OUTCOME, q['out']), name])
    order = rng.permutation(len(rows))
    with open(os.path.join(ROOT, 'demo_referrals.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(COLS)
        for k, i in enumerate(order):
            r = list(rows[i])
            if rng.random() < 0.03: r[5] = ''   # distance not recorded
            if rng.random() < 0.05: r[7] = ''   # not yet seen in clinic
            if rng.random() < 0.04: r[9] = ''   # imaging not recorded
            w.writerow([f'REF-{k + 1:05d}'] + r)
    print(f'written: demo_referrals.csv ({n} referrals; patterns ' + ' / '.join(str(c) for c in counts) + ')')


if __name__ == '__main__':
    main()
