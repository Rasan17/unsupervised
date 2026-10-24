"""Generates the simulated demonstration cohorts (not real patients).

demo_inph_phenotypes.csv   360 patients with three built-in sub-types of idiopathic normal-pressure
                           hydrocephalus: "classic CSF-dynamic" (high Rout, DESH, narrow callosal angle,
                           good tap test), "vascular-comorbid" (older, hypertension, white-matter disease) and
                           "neurodegenerative overlap" (low MMSE, wider callosal angle, poor tap test).
                           Includes outcome columns (shunt response, gait improvement) that are not used to find
                           the sub-types but to describe them afterwards, some missing values, and the simulated true
                           sub-type (to see how well clustering recovers it; keep it as "describe only").
demo_inph_with_errors.csv  The same cohort with 11 planted data problems (decimal and typing errors, a minus sign,
                           spelling variants, a decimal comma, a duplicate, an out-of-scale grade and one patient
                           with an unusual combination of plausible values), described in Simulated_Planted_Problem.
demo_no_subtypes.csv       360 patients with the same variables drawn from one population (correlated
                           variables, no sub-types): any "clusters" found here are artefacts.

Usage: python3 tools/make_demo_cohorts.py   (reproducible: fixed seeds)
"""
import csv, os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
COLS = ['Patient_ID', 'Age', 'Sex', 'Hypertension', 'White_Matter_Fazekas', 'MMSE', 'Gait_Score', 'Evans_Index',
        'Callosal_Angle_Deg', 'DESH_Sign', 'Rout_mmHg_min_ml', 'Tap_Test_Gait_Gain_Pct', 'Shunt_Response', 'Gait_Improvement_12mo', 'Simulated_True_Subtype']
FAZ = ['0', '1', '2', '3']


def draw(rng, kind, n):
    p = {'classic': dict(age=71, htn=0.25, faz=[0.40, 0.40, 0.15, 0.05], mmse=26, gait=2.5, evans=0.38, ca=68, desh=0.90, rout=19.0, tap=32, resp=0.85),
         'vascular': dict(age=79, htn=0.90, faz=[0.03, 0.15, 0.40, 0.42], mmse=23, gait=3.0, evans=0.35, ca=94, desh=0.30, rout=12.0, tap=14, resp=0.55),
         'neuro': dict(age=75, htn=0.40, faz=[0.25, 0.40, 0.25, 0.10], mmse=17, gait=2.3, evans=0.36, ca=110, desh=0.15, rout=9.5, tap=4, resp=0.30),
         'single': None}[kind]
    rows = []
    for _ in range(n):
        if kind == 'single':
            # one population: a continuum, with realistic correlations but no sub-types
            z = rng.normal()
            q = dict(age=75 + 4 * z, htn=0.5, faz=[0.2, 0.3, 0.3, 0.2], mmse=22 - 2 * z, gait=2.6, evans=0.36, ca=90 + 8 * z, desh=0.47, rout=13.5 - 1.5 * z, tap=16 - 4 * z, resp=0.57)
        else:
            q = p
        age = np.clip(rng.normal(q['age'], 6), 55, 92)
        sex = 'Male' if rng.random() < 0.58 else 'Female'
        htn = 'Yes' if rng.random() < q['htn'] else 'No'
        faz = rng.choice(FAZ, p=q['faz'])
        mmse = int(np.clip(round(rng.normal(q['mmse'], 2.6)), 8, 30))
        gait = int(np.clip(round(rng.normal(q['gait'], 0.8)), 0, 4))
        evans = np.clip(rng.normal(q['evans'], 0.025), 0.30, 0.46)
        ca = np.clip(rng.normal(q['ca'], 11), 45, 140)
        desh = 'Present' if rng.random() < q['desh'] else 'Absent'
        rout = np.clip(rng.normal(q['rout'], 3.0), 4, 35)
        tap = rng.normal(q['tap'], 9)
        resp = 'Responder' if rng.random() < q['resp'] else 'Non-responder'
        gimp = int(np.clip(round(rng.normal(1.8 if resp == 'Responder' else 0.2, 0.7)), -1, 4))
        truth = {'classic': 'Classic CSF-dynamic', 'vascular': 'Vascular-comorbid', 'neuro': 'Neurodegenerative overlap', 'single': 'None (one population)'}[kind]
        rows.append([round(float(age)), sex, htn, faz, mmse, gait, f'{evans:.2f}', round(float(ca)), desh, f'{rout:.1f}', f'{tap:.0f}', resp, gimp, truth])
    return rows


def write(name, rows, rng, prefix):
    with open(os.path.join(ROOT, name), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(COLS)
        for i, r in enumerate(rows):
            r = list(r)
            if rng.random() < 0.10: r[COLS.index('Rout_mmHg_min_ml') - 1] = ''   # no infusion study
            if rng.random() < 0.04: r[COLS.index('Tap_Test_Gait_Gain_Pct') - 1] = ''
            if rng.random() < 0.03: r[COLS.index('White_Matter_Fazekas') - 1] = ''
            w.writerow([f'{prefix}-{i + 1:03d}'] + r)


def plant_errors(src, dst):
    """The sub-types cohort with planted data problems, for the data checks: each problem is described in the
    describe-only column Simulated_Planted_Problem (empty for the other patients)."""
    with open(os.path.join(ROOT, src)) as f:
        rows = list(csv.reader(f))
    head, body = rows[0] + ['Simulated_Planted_Problem'], [r + [''] for r in rows[1:]]
    c = {h: i for i, h in enumerate(head)}
    def setv(i, col, value, why):
        body[i][c[col]] = value; body[i][c['Simulated_Planted_Problem']] = why
    setv(10, 'Evans_Index', str(int(round(float(body[10][c['Evans_Index']]) * 100))), 'decimal point missing in Evans index')
    setv(25, 'MMSE', str(int(body[25][c['MMSE']]) * 10), 'MMSE typed with an extra 0')
    setv(40, 'Age', str(int(body[40][c['Age']]) // 10), 'age missing a digit')
    setv(55, 'Callosal_Angle_Deg', '-' + body[55][c['Callosal_Angle_Deg']], 'minus sign in callosal angle')
    setv(70, 'Sex', body[70][c['Sex']].lower(), 'sex in lower case')
    setv(85, 'Hypertension', body[85][c['Hypertension']] + 's', 'misspelt hypertension entry')
    i = next(k for k in range(100, 140) if body[k][c['Rout_mmHg_min_ml']])
    setv(i, 'Rout_mmHg_min_ml', body[i][c['Rout_mmHg_min_ml']].replace('.', ','), 'decimal comma in Rout')
    body[114][-1] = f'entered twice ({body[114][0]} and {body[114][0][:3]}999)'
    body[115] = [body[114][0][:3] + '999'] + body[114][1:]
    for col, v in [('Callosal_Angle_Deg', '124'), ('DESH_Sign', 'Present'), ('Tap_Test_Gait_Gain_Pct', '44'), ('Rout_mmHg_min_ml', '22.0'), ('MMSE', '17'), ('White_Matter_Fazekas', '3')]:
        body[130][c[col]] = v
    body[130][c['Simulated_Planted_Problem']] = 'unusual combination (each value plausible)'
    setv(145, 'Gait_Score', '9', 'gait score 9 on a 0-4 scale')
    setv(160, 'Tap_Test_Gait_Gain_Pct', str(int(body[160][c['Tap_Test_Gait_Gain_Pct']] or 30) * 10), 'tap-test gain typed with an extra 0')
    with open(os.path.join(ROOT, dst), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(head); w.writerows(body)


if __name__ == '__main__':
    rng = np.random.default_rng(2031)
    rows = draw(rng, 'classic', 150) + draw(rng, 'vascular', 120) + draw(rng, 'neuro', 90)
    order = rng.permutation(len(rows)); rows = [rows[i] for i in order]
    write('demo_inph_phenotypes.csv', rows, rng, 'PH')
    rng2 = np.random.default_rng(2032)
    write('demo_no_subtypes.csv', draw(rng2, 'single', 360), rng2, 'NS')
    plant_errors('demo_inph_phenotypes.csv', 'demo_inph_with_errors.csv')
    print('written: demo_inph_phenotypes.csv (sub-types 150 / 120 / 90), demo_no_subtypes.csv (one population, 360), demo_inph_with_errors.csv (11 planted problems)')
