// use-clusters.js: step 6, "Use the clusters". Turns the chosen clusters into a classification clinicians can
// apply: a draft name and description of each class, short bedside rules (a classification tree, with
// cross-validated agreement with the clusters), a calculator for a new patient, the outcomes of each class and
// what is needed before clinical use. Plain script (window.UMEUse); every value from the data is escaped.
(function (root) {
    'use strict';
    const CE = root.ClusterEngine, CH = root.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const LETTERS = 'ABCDEFGHIJ';
    const pretty = name => String(name).replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
    // For names inside a phrase: ordinary capitalised words in lower case; acronyms (MMSE) and units (mmHg) kept
    const lower = name => pretty(name).split(' ').map(w => (/^[A-Z][a-z]+$/.test(w) ? w.toLowerCase() : w)).join(' ');
    const num = v => Number(v.toFixed(Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2));
    const pct = x => `${Math.round(100 * x)}%`;

    let ctx = null;   // { S, P, METHOD, levelsOf, profileTable, onNames }
    const U = { names: [], depth: null, tree: null, cv: null };

    // ---------------------------------------------------------------------------------- data for the rules
    // Values of the clustering variables for each patient as recorded (missing = null)
    function tableOf(P) {
        return P.rows.map(r => P.specs.map(s => {
            const v = r[s.col];
            if (v === null || v === undefined || String(v).trim() === '') return null;
            if (s.type === 'continuous') { const x = Number(v); return Number.isFinite(x) ? x : null; }
            return String(v).trim();
        }));
    }
    const varsOf = P => P.specs.map(s => ({ name: s.col, type: s.type, levels: s.levels }));

    // ---------------------------------------------------------------------------------- describing each class
    // Effect sizes: standardised mean difference for numbers and grades, Cohen's h for categories (each level)
    function features(P, labels, k) {
        const T = tableOf(P), vars = varsOf(P), out = Array.from({ length: k }, () => []);
        vars.forEach((v, j) => {
            const coded = T.map(r => CE.codeValue(r[j], v));
            for (let c = 0; c < k; c++) {
                const inC = [], rest = [];
                coded.forEach((x, i) => { if (x !== null) (labels[i] === c ? inC : rest).push(x); });
                if (inC.length < 3 || rest.length < 3) continue;
                if (v.type === 'nominal') {
                    [...new Set(coded.filter(x => x !== null))].forEach(lv => {
                        const p1 = inC.filter(x => x === lv).length / inC.length, p0 = rest.filter(x => x === lv).length / rest.length;
                        const h = 2 * Math.asin(Math.sqrt(p1)) - 2 * Math.asin(Math.sqrt(p0));
                        if (h > 0) out[c].push({ v, effect: h, short: `${lower(v.name)} ${lv}`, long: `${pretty(v.name)} "${lv}" more often (${pct(p1)} vs ${pct(p0)} in the other patients)` });
                    });
                } else {
                    const m = a => a.reduce((s, x) => s + x, 0) / a.length, sd = a => { const mu = m(a); return Math.sqrt(a.reduce((s, x) => s + (x - mu) ** 2, 0) / Math.max(1, a.length - 1)); };
                    const pooled = Math.sqrt(((inC.length - 1) * sd(inC) ** 2 + (rest.length - 1) * sd(rest) ** 2) / Math.max(1, inC.length + rest.length - 2)) || 1;
                    const d = (m(inC) - m(rest)) / pooled, dir = d > 0 ? 'higher' : 'lower';
                    const med = a => { const s = a.slice().sort((x, y) => x - y), h = (s.length - 1) / 2; return (s[Math.floor(h)] + s[Math.ceil(h)]) / 2; };
                    const lvl = x => (v.levels ? v.levels[Math.round(x)] ?? num(x) : num(x));
                    const detail = v.type === 'ordinal' ? `median ${lvl(med(inC))} vs ${lvl(med(rest))}` : `mean ${num(m(inC))} vs ${num(m(rest))}`;
                    out[c].push({ v, effect: Math.abs(d), short: `${dir} ${lower(v.name)}`, long: `${dir} ${pretty(v.name)} (${detail} in the other patients)` });
                }
            }
        });
        return out.map(list => {
            const byVar = new Map();
            list.sort((a, b) => b.effect - a.effect).forEach(f => { if (!byVar.has(f.v.name)) byVar.set(f.v.name, f); });
            const top = [...byVar.values()], strong = top.filter(f => f.effect >= 0.5);
            return { top: (strong.length >= 2 ? strong : top.slice(0, 2)).slice(0, 4), weak: strong.length < 2 };
        });
    }
    const classLabel = c => `${LETTERS[c]} · ${U.names[c] || `Class ${LETTERS[c]}`}`;

    // ---------------------------------------------------------------------------------- the rules
    function minLeafFor(n) { return Math.max(5, Math.ceil(0.03 * n)); }
    // Stratified 10-fold cross-validation of the tree: agreement of its predictions with the clusters for
    // patients not used to derive the rules
    function crossValidate(T, vars, labels, depth, seed) {
        const n = labels.length, k = Math.max(...labels) + 1, rng = CE.createPRNG(seed), fold = new Array(n);
        for (let c = 0; c < k; c++) {
            const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0);
            for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
            idx.forEach((i, r) => { fold[i] = r % 10; });
        }
        const pred = new Array(n);
        for (let f = 0; f < 10; f++) {
            const tr = [], te = [];
            for (let i = 0; i < n; i++) (fold[i] === f ? te : tr).push(i);
            if (!te.length) continue;
            const tree = CE.classificationTree(tr.map(i => T[i]), vars, tr.map(i => labels[i]), { maxDepth: depth, minLeaf: minLeafFor(tr.length) });
            te.forEach(i => { pred[i] = CE.predictTree(tree, T[i]); });
        }
        const agree = pred.filter((p, i) => p === labels[i]).length / n;
        return { pred, agree, kappa: CE.cohenKappa(labels, pred) };
    }
    function fmtCond(v, c) {
        if (v.type === 'nominal') return c.is ? `${pretty(v.name)} is "${c.is}"` : `${pretty(v.name)} is not ${c.not.map(x => `"${x}"`).join(' or ')}`;
        const show = x => (v.type === 'ordinal' ? v.levels[x] : num(x));
        if (c.lo !== undefined && c.hi !== undefined) return `${pretty(v.name)} ${v.type === 'ordinal' ? `from ${show(c.lo + 1)} to ${show(c.hi)}` : `above ${show(c.lo)} and at most ${show(c.hi)}`}`;
        if (c.hi !== undefined) return `${pretty(v.name)} ${v.type === 'ordinal' ? `${show(c.hi)} or lower` : `≤ ${show(c.hi)}`}`;
        return `${pretty(v.name)} ${v.type === 'ordinal' ? `${show(c.lo + 1)} or higher` : `> ${show(c.lo)}`}`;
    }
    // One readable rule per leaf: conditions on the same variable are combined
    function ruleText(rule, vars) {
        const per = new Map();
        rule.conds.forEach(({ node, left }) => {
            const c = per.get(node.variable) || {};
            if (node.kind === 'is') { if (left) c.is = node.level; else (c.not = c.not || []).push(node.level); } else if (left) c.hi = c.hi === undefined ? node.threshold : Math.min(c.hi, node.threshold); else c.lo = c.lo === undefined ? node.threshold : Math.max(c.lo, node.threshold);
            if (c.is) delete c.not;
            per.set(node.variable, c);
        });
        return [...per.entries()].map(([j, c]) => fmtCond(vars[j], c));
    }
    function fitRules() {
        const { P } = ctx, labels = ctx.S.result.labels, T = tableOf(P), vars = varsOf(P);
        if (!U.depthTable) U.depthTable = [1, 2, 3, 4].map(d => ({ d, ...crossValidate(T, vars, labels, d, P.seed) }));
        if (U.depth === null) {
            const best = Math.max(...U.depthTable.map(r => r.agree));
            U.depth = U.depthTable.find(r => r.agree >= best - 0.02).d;
        }
        U.tree = CE.classificationTree(T, vars, labels, { maxDepth: U.depth, minLeaf: minLeafFor(labels.length) });
        U.cv = U.depthTable.find(r => r.d === U.depth);
        U.apparent = T.filter((x, i) => CE.predictTree(U.tree, x) === labels[i]).length / labels.length;
        U.vars = vars;
        U.hasMissing = vars.map((_, j) => T.some(r => r[j] === null));
    }

    // ---------------------------------------------------------------------------------- full-clustering assignment
    function references() {
        const { P } = ctx, labels = ctx.S.result.labels, k = ctx.S.result.k, n = labels.length;
        if (P.method === 'lca' && ctx.S.result.model) return { kind: 'model', model: ctx.S.result.model };
        if (P.method === 'pam-gower' || P.method === 'lca') {
            const D = P.gower.D, medoids = [];
            for (let c = 0; c < k; c++) {
                const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0);
                let best = idx[0], bestS = Infinity;
                idx.forEach(i => { let s = 0; idx.forEach(j => { s += D[i * n + j]; }); if (s < bestS) { bestS = s; best = i; } });
                medoids.push(best);
            }
            return { kind: 'medoid', medoids };
        }
        if (!P.Z) return null;
        const p = P.Z[0].length, C = Array.from({ length: k }, () => new Array(p).fill(0)), cnt = new Array(k).fill(0);
        P.Z.forEach((z, i) => { cnt[labels[i]]++; z.forEach((v, j) => { C[labels[i]][j] += v; }); });
        C.forEach((row, c) => row.forEach((_, j) => { row[j] /= cnt[c]; }));
        return { kind: 'centroid', C };
    }
    // Numerical coding of a new patient, as for the clustering (grades as index, categories as indicators)
    function encodeNew(P, x) {
        const out = [];
        P.specs.forEach((s, j) => {
            const v = x[j];
            if (s.type === 'continuous') out.push(v === null ? null : v);
            else if (s.type === 'ordinal') out.push(v === null ? null : s.levels.indexOf(v));
            else if (s.levels.length === 2) out.push(v === null ? null : v === s.levels[1] ? 1 : 0);
            else s.levels.forEach(l => out.push(v === null ? null : v === l ? 1 : 0));
        });
        return out;
    }
    function assignFull(x) {
        const { P } = ctx, ref = U.ref, k = ctx.S.result.k;
        if (!ref) return null;
        let d;
        if (ref.kind === 'model') {
            const prob = CE.lcaPosterior(ref.model, x), order = prob.map((v, c) => c).sort((a, b) => prob[b] - prob[a]);
            return { prob, order, borderline: prob[order[0]] < 0.7, kind: 'model' };
        }
        if (ref.kind === 'medoid') {
            const vars = varsOf(P), dist = CE.gowerDistances(x, ref.medoids.map(i => P.raw[i]), vars, P.gower.weights);
            d = dist;
        } else {
            const e = encodeNew(P, x), z = e.map((v, j) => (v === null ? null : (v - P.zstat.mean[j]) / P.zstat.sd[j]));
            const used = z.filter(v => v !== null).length;
            if (!used) return null;
            d = ref.C.map(cen => Math.sqrt(z.reduce((s, v, j) => (v === null ? s : s + (v - cen[j]) ** 2), 0) * (z.length / used)));
        }
        const order = d.map((v, c) => c).sort((a, b) => d[a] - d[b]);
        return { d, order, borderline: k > 1 && d[order[1]] > 0 && d[order[0]] / d[order[1]] > 0.85, kind: ref.kind };
    }

    // ---------------------------------------------------------------------------------- rendering
    function render(c) {
        ctx = c;
        const { S, P } = ctx, { k, labels } = S.result;
        if (!U.key || U.key !== S.result) { Object.assign(U, { key: S.result, names: [], depth: null, depthTable: null }); }
        const feats = features(P, labels, k);
        if (!U.names.length) U.names = feats.map(f => f.top.slice(0, 2).map(t => t.short).join(', ').replace(/^./, ch => ch.toUpperCase()));
        U.feats = feats;
        fitRules();
        U.ref = references();
        draw();
    }
    function draw() {
        const { S, P, METHOD } = ctx, { k, labels, stab, test } = S.result, n = labels.length;
        const sizes = new Array(k).fill(0); labels.forEach(l => { sizes[l]++; });
        const weakest = Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0));
        const trustworthy = weakest >= 0.75 && (!test || test.pValue <= 0.05);
        const rules = CE.treeRules(U.tree);
        const byClass = Array.from({ length: k }, () => []);
        rules.forEach(r => byClass[r.leaf.predict].push(r));
        const cvWord = a => (a >= 0.9 ? 'excellent' : a >= 0.8 ? 'good' : a >= 0.7 ? 'moderate' : 'poor');
        // Confusion (cross-validated)
        const conf = Array.from({ length: k }, () => new Array(k).fill(0));
        labels.forEach((l, i) => { conf[l][U.cv.pred[i]]++; });
        const nameInputs = U.feats.map((f, c) => `
            <div class="classcard"><div class="classhead"><i class="sw s${c}"></i><strong>${LETTERS[c]}</strong><span class="muted">${sizes[c]} patients (${pct(sizes[c] / n)})</span></div>
                <label class="small" for="name-${c}">Name</label><input id="name-${c}" class="name-input" type="text" maxlength="80" value="${esc(U.names[c])}">
                <p class="small">Compared with the other patients: ${esc(f.top.map(t => t.long).join('; '))}.${f.weak ? ' <span class="muted">(Only small differences: this class is not sharply distinct.)</span>' : ''}</p></div>`).join('');
        const rulesHtml = byClass.map((rs, c) => `<div class="rulegroup"><h4><i class="sw s${c}"></i>${esc(classLabel(c))}</h4>${rs.length ? `<ol>${rs.map(r => {
            const conds = ruleText(r, U.vars);
            return `<li>${conds.length ? `If ${esc(conds.join(' and '))}` : 'All patients'} <span class="muted">(${r.leaf.n} patients; ${pct(r.leaf.counts[r.leaf.predict] / r.leaf.n)} of them are in this cluster)</span></li>`;
        }).join('')}</ol>` : '<p class="muted small">No rule leads to this class at this depth: the rules cannot recognise it (try deeper rules, or use the full clustering in the calculator).</p>'}</div>`).join('');
        const depthRows = U.depthTable.map(r => `<tr${r.d === U.depth ? ' class="pick"' : ''}><td>${r.d}</td><td>${pct(r.agree)}</td><td>${r.kappa.toFixed(2)}</td></tr>`).join('');
        const confTable = `<table><thead><tr><th>Cluster \\ rules</th>${Array.from({ length: k }, (_, c) => `<th>${LETTERS[c]}</th>`).join('')}<th>Recognised</th></tr></thead><tbody>${conf.map((row, c) => `<tr><td><i class="sw s${c}"></i>${LETTERS[c]}</td>${row.map((v, j) => `<td${j === c ? ' class="diag"' : ''}>${v}</td>`).join('')}<td>${pct(row[c] / Math.max(1, sizes[c]))}</td></tr>`).join('')}</tbody></table>`;
        const form = P.specs.map((s, j) => {
            const id = `np-${j}`;
            if (s.type === 'continuous') {
                const xs = P.rows.map(r => Number(r[s.col])).filter(Number.isFinite);
                return `<div class="field"><label for="${id}">${esc(pretty(s.col))}</label><input id="${id}" type="number" step="any" placeholder="${xs.length ? `${num(Math.min(...xs))} to ${num(Math.max(...xs))}` : ''}"></div>`;
            }
            return `<div class="field"><label for="${id}">${esc(pretty(s.col))}</label><select id="${id}"><option value="">Not recorded</option>${s.levels.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('')}</select></div>`;
        }).join('');
        $('use-out').innerHTML = `
            <div class="verdict ${trustworthy ? 'good' : 'bad'}">${trustworthy
                ? `A classification with ${k} classes from ${esc(METHOD[P.method].short)} clustering. The clusters reproduced in bootstrap resamples${test ? ' and showed more separation than chance' : ''}, so a classification is reasonable to propose. It is a hypothesis from these data: it needs replication in another cohort before clinical use.`
                : 'Caution: the clusters chosen in step 5 did not pass the checks (stability and the test of no real sub-groups). A classification built on them is likely to describe noise. It is shown so that you can see how it would work, not for use.'}</div>
            <h3>1. The classes</h3>
            <p class="muted small">Each class is described by the variables that differ most from the other patients (standardised differences for numbers and grades, Cohen's h for categories). The names are drafts made from these differences: replace them with clinically meaningful names (for example "classic CSF-dynamic"). A name is a hypothesis, not a diagnosis.</p>
            <div class="classgrid">${nameInputs}</div>
            <h3>2. Bedside rules</h3>
            <p class="muted small">Short rules that place a patient in a class from a few variables (a classification tree fitted to the clusters; each rule covers at least ${minLeafFor(n)} patients). If a value is not recorded, the rules follow the branch most patients took.</p>
            <div class="row"><label for="rule-depth">Most conditions per rule</label><select id="rule-depth">${[1, 2, 3, 4].map(d => `<option value="${d}"${d === U.depth ? ' selected' : ''}>${d}</option>`).join('')}</select></div>
            <div class="rules">${rulesHtml}</div>
            <p class="lead small">The rules give the same class as the full clustering for <strong>${pct(U.cv.agree)}</strong> of patients not used to derive them (10-fold cross-validation; Cohen's kappa ${U.cv.kappa.toFixed(2)}; ${cvWord(U.cv.agree)}). On the patients used to derive them: ${pct(U.apparent)}.${U.cv.agree < 0.8 ? ' The rules are only an approximation of the clusters: for an individual patient, use the calculator below, which also applies the full clustering.' : ''}</p>
            <div class="grid2"><div><div class="table-wrap">${confTable}</div><p class="muted small">Cross-validated: rows are the clusters, columns the class given by the rules.</p></div>
                <div><div class="table-wrap"><table><thead><tr><th>Conditions per rule</th><th>Agreement (cross-validated)</th><th>Kappa</th></tr></thead><tbody>${depthRows}</tbody></table></div><p class="muted small">The suggested depth is the simplest within 2 percentage points of the best agreement: longer rules are harder to apply and to replicate.</p></div></div>
            <h3>3. Classify a new patient</h3>
            <p class="muted small">Enter the values of the clustering variables (leave any that are not available). The patient is classified by the rules and by the full clustering (${U.ref ? (U.ref.kind === 'model' ? 'probabilities from the latent class model' : U.ref.kind === 'medoid' ? 'nearest representative patient by Gower dissimilarity' : 'nearest cluster centre on the standardised variables') : 'not available for these data'}). Nothing is stored or sent.</p>
            <div class="newpatient">${form}<div class="row"><button id="np-go" class="btn primary" type="button">Classify this patient</button><button id="np-clear" class="btn" type="button">Clear</button></div></div>
            <div id="np-out" aria-live="polite"></div>
            <h3>4. Do the classes differ in outcomes?</h3>
            ${P.describe.length ? ctx.profileTable({ only: 'describe', names: c => classLabel(c) }) : '<p class="muted">No variables were kept as "describe only". To see whether the classes matter clinically, keep outcomes (for example shunt response) as "Describe clusters only" in step 2.</p>'}
            <h3>5. Before clinical use</h3>
            <ol class="small">
                <li><strong>Replicate</strong> in an independent cohort: repeat the clustering there and check that similar classes appear, and apply these rules to see how many patients they classify the same way.</li>
                <li><strong>Show that the classes matter</strong>: pre-specify the outcomes (for example shunt response at 12 months) and test whether the classes differ in them, in the new cohort.</li>
                <li><strong>Check the rules with clinicians</strong>: thresholds should make clinical sense; a rule that relies on a variable recorded unreliably will not travel.</li>
                <li><strong>Report fully</strong>: the variables, method, number of clusters and how it was chosen, missing data handling, stability, and these rules, so others can reproduce the classification.</li>
                <li><strong>Do not use it for treatment decisions</strong> for individual patients until it has been validated.</li>
            </ol>
            <div class="row"><button id="sheet-btn" class="btn" type="button">Download the classification sheet (.txt)</button><button id="to-tree" class="btn primary" type="button">Next: the decision tree</button></div>`;
        for (let c = 0; c < k; c++) $(`name-${c}`).addEventListener('change', e => { U.names[c] = e.target.value.trim(); draw(); });
        $('rule-depth').addEventListener('change', e => { U.depth = Number(e.target.value); fitRules(); draw(); });
        $('np-go').addEventListener('click', classifyNew);
        $('np-clear').addEventListener('click', () => { P.specs.forEach((_, j) => { $(`np-${j}`).value = ''; }); $('np-out').innerHTML = ''; });
        $('sheet-btn').addEventListener('click', downloadSheet);
    }
    function classifyNew() {
        const { P } = ctx;
        const x = P.specs.map((s, j) => { const v = $(`np-${j}`).value.trim(); if (v === '') return null; if (s.type === 'continuous') { const a = Number(v); return Number.isFinite(a) ? a : null; } return v; });
        const recorded = x.filter(v => v !== null).length;
        if (!recorded) { $('np-out').innerHTML = '<p class="status bad">Enter at least one value.</p>'; return; }
        const warn = [];
        P.specs.forEach((s, j) => {
            if (s.type !== 'continuous' || x[j] === null) return;
            const xs = P.rows.map(r => Number(r[s.col])).filter(Number.isFinite), lo = Math.min(...xs), hi = Math.max(...xs);
            if (x[j] < lo || x[j] > hi) warn.push(`${pretty(s.col)} ${num(x[j])} is outside the range in these data (${num(lo)} to ${num(hi)})`);
        });
        const path = CE.treePath(U.tree, x), byRules = path.leaf.predict;
        const steps = path.steps.map(({ node, left }) => {
            const v = U.vars[node.variable], val = x[node.variable], cond = ruleText({ conds: [{ node, left }] }, U.vars)[0];
            return `${esc(cond)}${val === null ? ' <span class="muted">(not recorded: followed the larger branch)</span>' : ` <span class="muted">(${esc(v.type === 'continuous' ? num(val) : val)})</span>`}`;
        });
        const full = assignFull(x);
        const missingShare = 1 - recorded / x.length;
        let html = `<div class="verdict ${full && full.order[0] !== byRules ? 'mixed' : 'good'}">`;
        html += `<p><strong>By the rules: ${esc(classLabel(byRules))}</strong>${steps.length ? ` because ${steps.join(', and ')}.` : '.'}</p>`;
        if (full) {
            const a = full.order[0], b = full.order[1], unit = full.kind === 'medoid' ? 'Gower dissimilarity' : 'distance';
            if (full.kind === 'model') html += `<p><strong>By the latent class model: ${esc(classLabel(a))}</strong> (probability ${pct(full.prob[a])}${b !== undefined ? `; ${esc(classLabel(b))} ${pct(full.prob[b])}` : ''}).${full.borderline ? ' The model is not confident: the patient could belong to either class.' : ''}</p>`;
            else html += `<p><strong>By the full clustering: ${esc(classLabel(a))}</strong> (${unit} ${full.d[a].toFixed(2)}${b !== undefined ? `; next ${esc(classLabel(b))}, ${full.d[b].toFixed(2)}` : ''}).${full.borderline ? ' The patient lies near the boundary between these two classes.' : ''}</p>`;
            html += a === byRules ? '<p>Both agree.</p>' : '<p>The rules and the full clustering disagree: the patient is near a boundary, so the class is uncertain.</p>';
        }
        if (missingShare > 0) html += `<p class="muted small">${x.length - recorded} of ${x.length} variables not recorded: the classification is less certain.</p>`;
        if (warn.length) html += `<p class="small"><strong>Check:</strong> ${esc(warn.join('; '))}. The classification may not apply to this patient.</p>`;
        $('np-out').innerHTML = html + '</div>';
    }
    function downloadSheet() {
        const { S, P, METHOD } = ctx, { k, labels, stab, test } = S.result, n = labels.length;
        const sizes = new Array(k).fill(0); labels.forEach(l => { sizes[l]++; });
        const L = [];
        L.push('Classification from cluster analysis (Unsupervised Medical Explorer)', '');
        L.push(`Data: ${S.data.name}; ${n} patients. Created ${new Date().toISOString().slice(0, 10)}.`);
        L.push(`Clustering variables: ${P.cols.map(pretty).join(', ')}.`);
        L.push(`Method: ${METHOD[P.method].name}; ${k} clusters; missing values: ${P.mice ? `multiple imputation (${P.mice.m} datasets, consensus)` : P.missing === 'fill' ? 'filled with median or most frequent value' : P.missing === 'complete' ? 'patients with missing values left out' : 'recorded values of each pair only'}; random seed ${P.seed}.`);
        L.push(`Checks: lowest cluster stability (mean bootstrap Jaccard) ${Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0)).toFixed(2)}${test ? `; test of no real sub-groups p = ${test.pValue.toFixed(2)}` : ''}.`, '');
        L.push('CLASSES');
        U.feats.forEach((f, c) => { L.push(`${classLabel(c)}: ${sizes[c]} patients (${pct(sizes[c] / n)}).`, `   ${f.top.map(t => t.long).join('; ')}.`); });
        L.push('', `BEDSIDE RULES (at most ${U.depth} condition(s) each)`);
        CE.treeRules(U.tree).forEach(r => { const conds = ruleText(r, U.vars); L.push(`- ${conds.length ? `If ${conds.join(' and ')}` : 'All patients'} -> ${classLabel(r.leaf.predict)} (${r.leaf.n} patients, ${pct(r.leaf.counts[r.leaf.predict] / r.leaf.n)} in this cluster)`); });
        L.push('  If a value is not recorded, follow the branch taken by most patients.');
        L.push(`Agreement of the rules with the clusters: ${pct(U.cv.agree)} (10-fold cross-validation), kappa ${U.cv.kappa.toFixed(2)}.`, '');
        L.push('BEFORE CLINICAL USE: replicate in an independent cohort; show that the classes differ in pre-specified outcomes; review the thresholds with clinicians; do not use for individual treatment decisions until validated.');
        L.push('', 'For research and education only; not a medical device.');
        const url = URL.createObjectURL(new Blob([L.join('\r\n')], { type: 'text/plain' }));
        const a = document.createElement('a'); a.href = url; a.download = `classification_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}.txt`;
        document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }
    // For the patient export: class names and the class given by the rules
    function exportColumns() {
        if (!U.tree || !ctx || U.key !== ctx.S.result) return null;
        const T = tableOf(ctx.P);
        return { names: ctx.S.result.labels.map(l => classLabel(l)), rules: T.map(x => classLabel(CE.predictTree(U.tree, x))) };
    }

    // ---------------------------------------------------------------------------------- step 8: the decision tree
    // The same tree as the bedside rules, drawn as a diagram. Colours are passed as attributes (not CSS), so the
    // downloaded file looks the same outside the page.
    const LIGHT = { bg: '#ffffff', ink: '#14212a', muted: '#50626c', line: '#b8c6cc', node: '#f6f8f9', c: ['#0b6c84', '#c0582a', '#5b8f2a', '#8a4fa8', '#c4972b', '#2a6fc0', '#a8466e', '#4f6a74'] };
    function pagePalette() {
        const cs = getComputedStyle(document.documentElement), v = k => cs.getPropertyValue(k).trim();
        const ok = x => (/^#[0-9a-f]{3,8}$/i.test(x) ? x : null);
        return { bg: null, ink: ok(v('--ink')) || LIGHT.ink, muted: ok(v('--muted')) || LIGHT.muted, line: ok(v('--line')) || LIGHT.line, node: ok(v('--soft')) || ok(v('--surface')) || LIGHT.node, c: LIGHT.c.map((d, i) => ok(v(`--c${i}`)) || d) };
    }
    function question(node) {
        const v = U.vars[node.variable];
        if (node.kind === 'is') return `${pretty(v.name)} is "${node.level}"?`;
        if (v.type === 'ordinal') return `${pretty(v.name)} ${v.levels[node.threshold]} or lower?`;
        return `${pretty(v.name)} ≤ ${num(node.threshold)}?`;
    }
    function wrap(text, max) {
        const words = String(text).split(' '), lines = [''];
        words.forEach(w => { const cur = lines[lines.length - 1]; if ((cur + ' ' + w).trim().length > max && cur) lines.push(w); else lines[lines.length - 1] = (cur + ' ' + w).trim(); });
        return lines.slice(0, 2).map((l, i, a) => (i === 1 && a.length === 2 && words.join(' ').length > a.join(' ').length ? l.slice(0, max - 1) + '…' : l));
    }
    function treeSvg(pal) {
        const W = 190, H = 92, GX = 16, GY = 58, k = ctx.S.result.k;
        // Layout: leaves in order from left to right, parents centred over their children
        const nodes = [];
        let slot = 0;
        (function place(node, depth) {
            const rec = { node, depth };
            nodes.push(rec);
            if (node.left) { const l = place(node.left, depth + 1), r = place(node.right, depth + 1); rec.x = (l.x + r.x) / 2; rec.kids = [l, r]; } else rec.x = slot++ * (W + GX);
            return rec;
        })(U.tree.root, 0);
        const depth = Math.max(...nodes.map(r => r.depth)), width = Math.max(W, slot * (W + GX) - GX), height = (depth + 1) * (H + GY) - GY + 8;
        const t = (x, y, str, { size = 12, fill = pal.ink, weight = 400, anchor = 'middle' } = {}) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">${esc(str)}</text>`;
        const bar = (x, y, w, counts) => { const n = counts.reduce((a, b) => a + b, 0); let cx = x, out = ''; counts.forEach((c, i) => { if (!c) return; const bw = (w * c) / n; out += `<rect x="${cx.toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="7" fill="${pal.c[i % 8]}"><title>${esc(classLabel(i))}: ${c}</title></rect>`; cx += bw; }); return out; };
        let g = `<svg class="tree" xmlns="http://www.w3.org/2000/svg" viewBox="-4 -4 ${width + 8} ${height + 8}" width="${width + 8}" height="${height + 8}" role="img" aria-label="Decision tree that assigns patients to the ${k} classes">`;
        if (pal.bg) g += `<rect x="-4" y="-4" width="${width + 8}" height="${height + 8}" fill="${pal.bg}"/>`;
        nodes.forEach(r => {   // edges first
            if (!r.kids) return;
            const y0 = r.depth * (H + GY) + H, xc = r.x + W / 2;
            r.kids.forEach((ch, side) => {
                const x1 = ch.x + W / 2, y1 = ch.depth * (H + GY), ym = (y0 + y1) / 2;
                g += `<path d="M${xc},${y0} C${xc},${ym} ${x1},${ym} ${x1},${y1}" fill="none" stroke="${pal.line}" stroke-width="1.5"/>`;
                g += t((xc + x1) / 2 + (side ? 10 : -10), ym - 2, side ? 'no' : 'yes', { size: 11, fill: pal.muted, weight: 600 });
            });
        });
        nodes.forEach(r => {
            const x = r.x, y = r.depth * (H + GY), nd = r.node, n = nd.n;
            if (r.kids) {
                g += `<rect x="${x}" y="${y}" width="${W}" height="${H}" rx="8" fill="${pal.node}" stroke="${pal.line}"/>`;
                wrap(question(nd), 26).forEach((ln, i) => { g += t(x + W / 2, y + 20 + i * 15, ln, { weight: 600 }); });
                g += t(x + W / 2, y + 60, `${n} patients${nd.missLeft !== undefined && U.hasMissing[nd.variable] ? ` · not recorded: ${nd.missLeft ? 'yes' : 'no'}` : ''}`, { size: 10.5, fill: pal.muted });
                g += bar(x + 10, y + H - 17, W - 20, nd.counts);
            } else {
                const c = nd.predict, pc = nd.counts[c] / n;
                g += `<rect x="${x}" y="${y}" width="${W}" height="${H}" rx="8" fill="${pal.node}" stroke="${pal.c[c % 8]}" stroke-width="2.5"/>`;
                g += `<rect x="${x + 10}" y="${y + 11}" width="12" height="12" rx="2" fill="${pal.c[c % 8]}"/>`;
                const name = classLabel(c), lines = wrap(name, 23);
                lines.forEach((ln, i) => { g += `<text x="${x + 28}" y="${y + 22 + i * 15}" font-size="12.5" font-weight="700" fill="${pal.ink}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">${esc(ln)}<title>${esc(name)}</title></text>`; });
                g += t(x + W / 2, y + 52, `${n} patients`, { size: 11, fill: pal.muted });
                g += t(x + W / 2, y + 66, `${Math.round(100 * pc)}% of them in class ${LETTERS[c]}`, { size: 11, fill: pal.muted });
                g += bar(x + 10, y + H - 15, W - 20, nd.counts);
            }
        });
        return g + '</svg>';
    }
    function drawTree() {
        const el = $('tree-out'); if (!el) return;
        const { S } = ctx, { k, stab, test } = S.result, weakest = Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0)), trustworthy = weakest >= 0.75 && (!test || test.pValue <= 0.05);
        const leaves = CE.treeRules(U.tree).length;
        el.innerHTML = `
            ${trustworthy ? '' : '<div class="verdict bad">Caution: the clusters chosen did not pass the checks (stability and the test of no real sub-groups). The tree is shown to illustrate how it would work, not for use.</div>'}
            <p>Start at the top and answer each question for the patient: <strong>yes</strong> goes down to the left, <strong>no</strong> to the right, until you reach a coloured box, which gives the class. If a value is not recorded, follow the branch shown in the box ("not recorded: yes" or "no"), the one most patients took.</p>
            <div class="row"><label for="tree-depth">Most questions on a path</label><select id="tree-depth">${[1, 2, 3, 4].map(d => `<option value="${d}"${d === U.depth ? ' selected' : ''}>${d}</option>`).join('')}</select></div>
            <div class="tree-wrap">${treeSvg(pagePalette())}</div>
            ${CH.legend(Array.from({ length: k }, (_, c) => classLabel(c)))}
            <p class="lead small">${leaves} end points. The tree gives the same class as the full clustering for <strong>${pct(U.cv.agree)}</strong> of patients not used to build it (10-fold cross-validation; Cohen's kappa ${U.cv.kappa.toFixed(2)}). The bar in each box shows how that box's patients divide between the classes (colours as in the legend).</p>
            <p class="muted small">The tree is a classification tree (CART, Gini impurity) fitted to the clusters, with at least ${minLeafFor(S.result.labels.length)} patients in each end point; the depth suggested is the simplest within 2 percentage points of the best cross-validated agreement. Class names and the depth are shared with step 7. Like the clusters themselves, the tree needs validation in another cohort before clinical use.</p>`;
        $('tree-depth').addEventListener('change', e => { U.depth = Number(e.target.value); fitRules(); draw(); drawTree(); });
    }
    function renderTree(c) {
        if (!U.key || U.key !== c.S.result || !U.tree) render(c); else ctx = c;
        drawTree();
    }
    // The diagram in light colours on white, as a stand-alone SVG file (used by the image download)
    const treeSvgLight = () => (U.tree && ctx ? '<?xml version="1.0" encoding="UTF-8"?>\n' + treeSvg(LIGHT).replace('<svg class="tree" ', '<svg ') : null);
    root.UMEUse = { render, renderTree, exportColumns, treeSvgLight, _state: U };
})(typeof window !== 'undefined' ? window : globalThis);
