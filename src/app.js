// app.js: the interface of the Unsupervised Medical Explorer (plain script; uses DataEngine, ClusterEngine,
// UMECharts, UMEUse). Six steps: data, variables, approach, number of clusters, clusters, use of the clusters. All computation stays in the
// browser; every value from the data is escaped before display.
(function () {
    'use strict';
    const DE = window.DataEngine, CE = window.ClusterEngine, NN = window.NNEngine, AE = window.AnomalyEngine, CH = window.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const yieldUI = () => new Promise(r => setTimeout(r, 0));
    const S = { data: null, vars: {}, prep: null, explore: null, result: null };
    // Demographic columns start as "describe only": phenotypes are usually built from clinical features and then
    // described by age and sex; yes/no variables such as sex can otherwise dominate the clusters (especially with Gower)
    const DEMOGRAPHIC_RE = /^(age|sex|gender|ethnicity|ethnic[\s_-]?group|race)([\s_-]|$)/i;
    const TARGET_NAME_RE = /^(target|outcome|response|shunt_response|label|class|diagnos|result|improvement)/i;
    const OUTCOME_RE = /(response|responder|outcome|improve|mortality|death|died|surviv|revision|complication|readmission|recurrence|discharge)/i;
    const METHOD = {
        'pam-gower': { name: 'k-medoids (PAM) with Gower dissimilarity', short: 'PAM / Gower', note: 'Recommended for clinical data that mix numbers, grades and categories, and it tolerates missing values. Each cluster is represented by a real patient (its medoid). With balanced weighting (the default) every variable contributes equally on average; with equal weights (classic Gower), yes/no variables count for more than measurements and can split patients into artificial groups.' },
        'kmeans': { name: 'k-means on standardised variables', short: 'k-means', note: 'For numerical variables; categories are coded 0/1. Assumes roughly round clusters of similar spread.' },
        'ward': { name: 'Hierarchical clustering (Ward) on standardised variables', short: 'Hierarchical (Ward)', note: 'Builds a tree of nested groupings; cutting it gives the clusters. Similar to k-means; needs no random start.' },
        'lca': { name: 'Latent class analysis (model-based; numbers, grades and categories)', short: 'Latent class', note: 'A statistical model: each class has its own distribution of every variable (averages and spreads for numbers, proportions for grades and categories). Gives each patient a probability of belonging to each class and BIC to compare numbers of classes; uses the recorded values directly when some are missing. Widely used for clinical phenotypes. Assumes the variables are unrelated within a class, so strongly related variables can create extra classes.' },
        'neural': { name: 'Neural network: self-organising map (at least 1,000 records)', short: 'Neural network (SOM)', note: 'A self-organising map (Kohonen) is a neural network that arranges the records on a grid of units, similar records on the same or neighbouring units, so it also gives a map of the patterns (for example of referral patterns). The units are then grouped with k-means weighted by the records they hold (Vesanto & Alhoniemi 2000). Numbers, grades and categories are used; missing values are left out of the distances. Suited to large administrative datasets: it is not offered below 1,000 records and works best with 5,000 or more. The map is trained once on all records; the stability check regroups its units in bootstrap resamples.' },
        'gmm': { name: 'Gaussian mixture model (numerical variables only)', short: 'Gaussian mixture', note: 'Allows clusters of different size and spread, with BIC to compare numbers of clusters. Only meaningful for numerical variables (yes/no and graded variables make BIC unreliable).' }
    };

    const MICE_M = 5;   // number of imputed datasets
    // Large datasets: dissimilarities between every pair of patients (silhouettes, maps, PAM, Ward) need n x n
    // memory, so above SAMPLE_MAX they are computed on a random sample of SAMPLE_N patients
    const SAMPLE_MAX = 3000, SAMPLE_N = 2000;
    // Neural-network clustering: not offered below NEURAL_MIN records; suggested from NEURAL_GOOD
    const NEURAL_MIN = 1000, NEURAL_GOOD = 5000;

    // ------------------------------------------------------------------------------------- step navigation
    function goTo(step) {
        document.querySelectorAll('.step').forEach(s => { s.hidden = s.dataset.step !== String(step); });
        document.querySelectorAll('.tab').forEach(t => { const on = t.dataset.step === String(step); t.classList.toggle('active', on); if (on) t.setAttribute('aria-current', 'step'); else t.removeAttribute('aria-current'); });
        window.scrollTo(0, 0);
        if (String(step) === '7' && S.result && !$('use-out').innerHTML) renderUse();
        if (String(step) === '3' && S.data) renderChecks();
        if (String(step) === '8' && S.result) { try { window.UMEUse.renderTree({ S, P: S.explore.P, METHOD, profileTable }); } catch (e) { $('tree-out').innerHTML = `<p class="status bad">The tree could not be drawn: ${esc(e.message)}</p>`; } }
    }
    // Exclusions from the data checks invalidate later results
    function renderChecks() { window.UMEChecks.render({ S, levelsOf, onChange: () => { S.explore = null; S.result = null; enableTabs(4); } }); }
    function renderUse() {
        try { window.UMEUse.render({ S, P: S.explore.P, METHOD, profileTable }); } catch (e) { $('use-out').innerHTML = `<p class="status bad">The classification could not be built: ${esc(e.message)}</p>`; }
    }
    function enableTabs(upTo) { document.querySelectorAll('.tab').forEach(t => { t.disabled = Number(t.dataset.step) > upTo; }); }
    function status(id, text, kind = '') { const el = $(id); el.textContent = text; el.className = `status ${kind}`; }

    // ------------------------------------------------------------------------------------- step 1: data
    function loadText(text, name, demo = false) {
        let parsed;
        try { parsed = DE.parseCSV(text); } catch (e) { status('data-status', `The file could not be read: ${e.message}`, 'bad'); return; }
        if (!parsed.rows.length) { status('data-status', 'The file has no data rows.', 'bad'); return; }
        // Decimal commas (European exports, "67,5"): columns written consistently that way are read as numbers. In
        // comma-separated files a comma followed by exactly three digits may be a thousands separator ("1,000"), so
        // such columns are converted only in semicolon- or tab-separated files
        const decimalComma = [];
        parsed.headers.forEach(h => {
            const vals = parsed.rows.map(r => r[h]).filter(v => !DE.isMissing(v)).map(v => String(v).trim());
            if (!vals.length || vals.some(v => v.includes('.'))) return;
            const withComma = vals.filter(v => /^[-+]?\d+,\d+$/.test(v));
            if (!withComma.length || vals.filter(v => /^[-+]?\d+(,\d+)?$/.test(v)).length < 0.95 * vals.length) return;
            if (parsed.delimiter === ',' && withComma.every(v => /,\d{3}$/.test(v))) return;
            parsed.rows.forEach(r => { const v = String(r[h] ?? '').trim(); if (/^[-+]?\d+,\d+$/.test(v)) r[h] = v.replace(',', '.'); });
            decimalComma.push(h);
        });
        S.data = { ...parsed, name, analysis: DE.analyzeColumns(parsed.headers, parsed.rows) };
        const ban = $('data-banner'); ban.hidden = false;
        ban.textContent = `Data: ${name}${demo ? ' (simulated demonstration, not real patients)' : ` (${parsed.rows.length.toLocaleString('en-GB')} rows, your file)`}`;
        S.data.demo = demo;
        S.vars = {}; S.prep = null; S.explore = null; S.result = null; S.methodTouched = false; S.excluded = new Set(); S.markedGenuine = new Set();
        if (window.UMEChecks) $('check-out').innerHTML = '';
        parsed.headers.forEach(h => {
            const a = S.data.analysis[h];
            // Outcome columns are recognised by name only (the data engine also treats the last column as the outcome, a
            // convention of prediction models that does not apply here); columns with no values or one value are ignored
            const distinct = new Set(parsed.rows.map(r => r[h]).filter(v => !DE.isMissing(v)).map(v => String(v).trim()));
            const role = a.role === 'meta' || distinct.size <= 1 ? 'ignore' : (OUTCOME_RE.test(h) || TARGET_NAME_RE.test(h) || /simulated|true.?sub.?type/i.test(h) || DEMOGRAPHIC_RE.test(h)) ? 'describe' : 'cluster';
            let type = a.type === 'continuous' && a.isNumeric ? 'continuous' : a.type === 'ordinal' ? 'ordinal' : 'nominal';
            // A column that is numbers apart from a few entry errors (e.g. "12,5") stays numerical: the odd entries are
            // treated as missing in the analysis and reported by the data checks
            if (!a.isNumeric) {
                const rec = parsed.rows.map(r => r[h]).filter(v => !DE.isMissing(v)), num = rec.filter(v => Number.isFinite(Number(String(v).trim())));
                if (rec.length >= 20 && num.length >= 0.95 * rec.length) type = new Set(num.map(Number)).size > 10 ? 'continuous' : 'ordinal';
            }
            S.vars[h] = { role, type };
        });
        const issues = DE.parseIssues ? DE.parseIssues(parsed) : [];
        if (decimalComma.length) issues.push(`Decimal commas were read as decimal points in ${decimalComma.length} column(s): ${decimalComma.join(', ')}.`);
        const miss = parsed.headers.reduce((a, h) => a + (S.data.analysis[h].missingCount || 0), 0);
        $('data-summary').innerHTML = `<p><strong>${esc(name)}</strong>: ${parsed.rows.length} patients, ${parsed.headers.length} columns, ${miss} missing value(s).</p>
            ${issues.length ? `<p class="warn">${issues.map(esc).join(' ')}</p>` : ''}
            <div class="table-wrap"><table><thead><tr>${parsed.headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
            <tbody>${parsed.rows.slice(0, 6).map(r => `<tr>${parsed.headers.map(h => `<td>${esc(r[h] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
        status('data-status', '');
        enableTabs(2); renderVariables(); goTo(2);
    }
    async function loadDemo(file) {
        try { const r = await fetch(file); if (!r.ok) throw new Error(String(r.status)); loadText(await r.text(), file, true); }
        catch (e) { status('data-status', `The demo file could not be loaded (${e.message}). If you opened the page from your disk, load the CSV file with "Choose a CSV file" instead.`, 'bad'); }
    }

    // Patients included in the analysis (all, minus those left out after the data checks)
    const includedRows = () => (S.excluded && S.excluded.size ? S.data.rows.filter((_, i) => !S.excluded.has(i)) : S.data.rows.slice());

    // ------------------------------------------------------------------------------------- step 2: variables
    // Levels of a grade or category. Grades: numerically, by conventional clinical words (none < mild < moderate <
    // severe, low < medium < high) or the order the user set in step 2; text otherwise sorts naturally
    function levelsOf(col) {
        const vals = [...new Set(includedRows().map(r => r[col]).filter(v => !DE.isMissing(v)).map(v => String(v).trim()))];
        if (vals.every(v => Number.isFinite(Number(v)))) return vals.sort((a, b) => Number(a) - Number(b));
        if (S.vars[col] && S.vars[col].type === 'ordinal') {
            const order = S.vars[col].order;
            if (order && order.length) return [...order.filter(o => vals.includes(o)), ...DE.sortOrdinalLevels(vals.filter(v => !order.includes(v)))];
            return DE.sortOrdinalLevels(vals);
        }
        return vals.sort();
    }
    // Whether a text grade's order was recognised from clinical words (or set by the user), rather than guessed
    function orderKnown(col) {
        const lv = levelsOf(col);
        if (lv.every(v => Number.isFinite(Number(v))) || (S.vars[col].order && S.vars[col].order.length)) return true;
        const words = ['none', 'absent', 'never', 'normal', 'very low', 'low', 'minimal', 'mild', 'slight', 'moderate', 'medium', 'intermediate', 'average', 'high', 'marked', 'severe', 'very high', 'very severe', 'critical', 'extreme'];
        return lv.every(v => words.some(w => { const s = v.toLowerCase(); return s === w || s.startsWith(w + ' ') || s.endsWith(' ' + w); }));
    }
    function renderVariables() {
        const rows = S.data.headers.map((h, i) => {
            const v = S.vars[h], a = S.data.analysis[h];
            return `<tr><td><strong>${esc(h)}</strong><div class="muted">${a.isNumeric ? `numbers, ${a.distinctCount} distinct` : `${a.distinctCount} categories`}${a.missingCount ? `, ${a.missingCount} missing` : ''}</div></td>
                <td><select id="role-${i}" data-col="${esc(h)}" class="role" aria-label="Use of ${esc(h)}">
                    <option value="cluster"${v.role === 'cluster' ? ' selected' : ''}>Use to find clusters</option>
                    <option value="describe"${v.role === 'describe' ? ' selected' : ''}>Describe clusters only</option>
                    <option value="ignore"${v.role === 'ignore' ? ' selected' : ''}>Ignore</option></select></td>
                <td><select id="type-${i}" data-col="${esc(h)}" class="vtype" aria-label="Type of ${esc(h)}">
                    <option value="continuous"${v.type === 'continuous' ? ' selected' : ''}${a.isNumeric ? '' : ' disabled'}>Number</option>
                    <option value="ordinal"${v.type === 'ordinal' ? ' selected' : ''}>Ordered grade</option>
                    <option value="nominal"${v.type === 'nominal' ? ' selected' : ''}>Category</option></select>${gradeOrder(h, i)}</td></tr>`;
        }).join('');
        $('var-table').innerHTML = `<table><thead><tr><th>Column</th><th>Use</th><th>Type</th></tr></thead><tbody>${rows}</tbody></table>`;
        document.querySelectorAll('.role').forEach(s => s.addEventListener('change', e => { S.vars[e.target.dataset.col].role = e.target.value; updateVarSummary(); }));
        document.querySelectorAll('.vtype').forEach(s => s.addEventListener('change', e => { S.vars[e.target.dataset.col].type = e.target.value; renderVariables(); }));
        document.querySelectorAll('.grade-order').forEach(inp => inp.addEventListener('change', e => {
            const col = e.target.dataset.col, want = e.target.value.split('<').map(x => x.trim()).filter(Boolean), have = levelsOf(col);
            S.vars[col].order = want.every(w => have.includes(w)) && have.every(h2 => want.includes(h2)) ? want : null;
            renderVariables();
        }));
        const pc = $('patient-col');
        pc.innerHTML = '<option value="">Each row is a different patient</option>' + S.data.headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join('');
        updateVarSummary();
    }
    // The order of a text grade, editable ("Mild < Moderate < Severe"), with a caution when it could not be recognised
    function gradeOrder(h, i) {
        if (S.vars[h].type !== 'ordinal') return '';
        const lv = levelsOf(h);
        if (lv.every(v => Number.isFinite(Number(v)))) return '';
        return `<div class="grade-order-box small"><label for="order-${i}">Order (lowest first)</label><input id="order-${i}" class="grade-order" data-col="${esc(h)}" type="text" value="${esc(lv.join(' < '))}">${orderKnown(h) ? '' : '<div class="warn">Order not recognised: check it, and correct it if needed (levels separated by &lt;).</div>'}</div>`;
    }
    function updateVarSummary() {
        const used = Object.keys(S.vars).filter(h => S.vars[h].role === 'cluster'), desc = Object.keys(S.vars).filter(h => S.vars[h].role === 'describe');
        const types = { continuous: 0, ordinal: 0, nominal: 0 }; used.forEach(h => { types[S.vars[h].type]++; });
        $('var-summary').innerHTML = `${used.length} variable(s) used to find clusters (${types.continuous} numbers, ${types.ordinal} ordered grades, ${types.nominal} categories); ${desc.length} used only to describe them.`
            + (used.length < 2 ? ' <span class="warn">Choose at least two variables to find clusters.</span>' : '')
            + (Object.keys(S.vars).some(h => DEMOGRAPHIC_RE.test(h) && S.vars[h].role === 'describe') ? ' <span class="muted">Age and sex start as "describe only", so that the clusters reflect clinical features rather than demographics (yes/no variables such as sex can otherwise dominate them); change this if demographics are part of the phenotype you are looking for.</span>' : '')
            + (desc.length ? '' : ' <span class="muted">Tip: outcomes (for example shunt response) are best kept as "Describe clusters only": they then show whether the clusters differ in outcome, without having shaped them.</span>');
        $('to-step3').disabled = used.length < 2; $('to-check').disabled = used.length < 2;
        renderAdvice(used);
        renderVarWarnings(used);
        const onlyNumbers = used.every(h => S.vars[h].type === 'continuous');
        $('method-gmm').disabled = !onlyNumbers;
        const nRec = includedRows().length;
        $('method-neural').disabled = nRec < NEURAL_MIN; $('method-pam').disabled = nRec > SAMPLE_MAX; $('method-ward').disabled = nRec > SAMPLE_MAX;
        ['neural', 'pam-gower', 'ward'].forEach(m => { if ($('method').value === m && $('method' + (m === 'pam-gower' ? '-pam' : '-' + m)).disabled) $('method').value = nRec > SAMPLE_MAX ? 'lca' : 'pam-gower'; });
        if (!onlyNumbers && $('method').value === 'gmm') $('method').value = 'pam-gower';
        $('gower-weights').disabled = !['pam-gower', 'lca'].includes($('method').value);
        $('method-hint').textContent = METHOD[$('method').value].note + (onlyNumbers ? '' : ' (Gaussian mixtures need numerical variables only.)');
    }

    // Warnings about the data before the analysis: unusable columns, dates, codes for "not recorded", strongly related
    // clustering variables (one construct counted twice) and too many variables for the number of patients
    function renderVarWarnings(used) {
        const rows = includedRows(), n = rows.length, out = [];
        const num = v => (DE.isMissing(v) ? NaN : Number(String(v).trim()));
        S.data.headers.filter(h => S.vars[h].role !== 'ignore').forEach(h => {
            const vals = rows.map(r => r[h]), rec = vals.filter(v => !DE.isMissing(v)).map(v => String(v).trim());
            if (!rec.length) { out.push(`<strong>${esc(h)}</strong>: no recorded values; set it to "Ignore".`); return; }
            if (new Set(rec).size === 1) { if (S.vars[h].role === 'cluster') out.push(`<strong>${esc(h)}</strong>: the same value for every patient; it cannot separate patients (set it to "Ignore").`); return; }
            if (rec.length < 0.5 * n) out.push(`<strong>${esc(h)}</strong>: ${Math.round(100 * (1 - rec.length / n))}% of values missing; results that depend on it are uncertain.`);
            if (rec.filter(v => /^\d{4}-\d{1,2}-\d{1,2}([ T].*)?$|^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$/.test(v)).length >= 0.8 * rec.length || (rec.every(v => /^(19|20)\d{6}$/.test(v)))) out.push(`<strong>${esc(h)}</strong>: looks like dates; as a category or a plain number it is not meaningful. Convert it to an interval (for example days from referral to clinic) or set it to "Ignore".`);
            if (S.vars[h].type !== 'nominal') { const codes = AE.missingCodes(vals.map(num)); if (codes.length) out.push(`<strong>${esc(h)}</strong>: ${codes.map(c => `${c.value} (${c.count} patients)`).join(', ')} may be a code for "not recorded" (outside the range of the other values); if so, leave these cells empty, or they will be treated as real values.`); }
        });
        // Rare categories of clustering variables: too few patients to shape or describe a cluster reliably
        const rare = used.filter(h => S.vars[h].type !== 'continuous').map(h => { const cnt = new Map(); rows.forEach(r => { if (!DE.isMissing(r[h])) { const v = String(r[h]).trim(); cnt.set(v, (cnt.get(v) || 0) + 1); } }); const few = [...cnt].filter(([, c]) => c < 5); return few.length && cnt.size > 1 ? `${esc(h)}: ${few.map(([v, c]) => `"${esc(v)}" (${c})`).join(', ')}` : null; }).filter(Boolean);
        if (rare.length) out.push(`<strong>Rare categories</strong> (fewer than 5 patients): ${rare.join('; ')}. They can create tiny clusters or unstable latent classes; consider merging them with a related category.`);
        // Strongly related clustering variables
        const codeOf = h => { const t = S.vars[h].type; if (t === 'nominal') return rows.map(r => (DE.isMissing(r[h]) ? null : String(r[h]).trim())); const lv = t === 'ordinal' ? levelsOf(h) : null; return rows.map(r => { if (DE.isMissing(r[h])) return null; if (lv) { const k = lv.indexOf(String(r[h]).trim()); return k < 0 ? null : k; } const x = Number(r[h]); return Number.isFinite(x) ? x : null; }); };
        const rank = xs => { const idx = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), rk = new Array(xs.length); let i = 0; while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let q = i; q <= j; q++) rk[idx[q][1]] = (i + j) / 2; i = j + 1; } return rk; };
        const pearson = (a, b) => { const m = a.length, ma = a.reduce((x, y) => x + y, 0) / m, mb = b.reduce((x, y) => x + y, 0) / m; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < m; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; } return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0; };
        const assoc = (h1, h2) => {
            const a = codeOf(h1), b = codeOf(h2), ok = a.map((v, i) => (v !== null && b[i] !== null ? i : -1)).filter(i => i >= 0);
            if (ok.length < 20) return null;
            const n1 = S.vars[h1].type === 'nominal', n2 = S.vars[h2].type === 'nominal';
            if (!n1 && !n2) return { v: Math.abs(pearson(rank(ok.map(i => a[i])), rank(ok.map(i => b[i])))), what: 'rank correlation' };
            if (n1 && n2) { const t = CE.chiSquareTest(ok.map(i => a[i]), ok.map(i => b[i])); return t.statistic === null ? null : { v: CE.cramersV(t.statistic, ok.length, new Set(ok.map(i => a[i])).size, new Set(ok.map(i => b[i])).size), what: "Cramér's V" }; }
            // number and category: correlation ratio (eta)
            const x = n1 ? b : a, g = n1 ? a : b, m = ok.reduce((s2, i) => s2 + x[i], 0) / ok.length, tot = ok.reduce((s2, i) => s2 + (x[i] - m) ** 2, 0), byG = new Map();
            ok.forEach(i => { const e = byG.get(g[i]) || [0, 0]; e[0] += x[i]; e[1]++; byG.set(g[i], e); });
            let between = 0; byG.forEach(([sx, c]) => { between += c * (sx / c - m) ** 2; });
            return tot > 0 ? { v: Math.sqrt(between / tot), what: 'correlation ratio' } : null;
        };
        const strong = [];
        for (let i = 0; i < used.length; i++) for (let j = i + 1; j < used.length; j++) { const r = assoc(used[i], used[j]); if (r && r.v >= 0.8) strong.push(`${esc(used[i])} and ${esc(used[j])} (${r.what} ${r.v.toFixed(2)})`); }
        if (strong.length) out.push(`<strong>Strongly related clustering variables</strong>: ${strong.join('; ')}. Each pair measures much the same thing, so that construct counts about twice in the clustering. Consider keeping one of each pair (the other as "Describe only"), unless the double weight is intended.`);
        // Variables relative to patients
        const coded = used.reduce((a, h) => a + (S.vars[h].type === 'nominal' ? Math.max(1, levelsOf(h).length - 1) : 1), 0);
        if (used.length && n < 10 * coded) out.push(`<strong>Many variables for the number of patients</strong>: ${coded} coded variable(s) for ${n} patients (fewer than 10 patients per variable). Clusters, stability and maps become unreliable when variables are many relative to patients; choose the variables most relevant to the question.`);
        $('warn-reminder').innerHTML = out.length ? `<p class="warnbox small">${out.length} warning(s) about the data were shown in step 2 (variables). <button type="button" class="btn small" id="warn-goto">Review them</button></p>` : '';
        if (out.length) $('warn-goto').addEventListener('click', () => goTo(2));
        $('var-warnings').innerHTML = out.length ? `<div class="warnbox small"><strong>Before the analysis:</strong><ul>${out.map(w => `<li>${w}</li>`).join('')}</ul></div>` : '';
    }
    // Which method suits these data: from the mix of variable types, the share of missing values and the size
    function advise(used) {
        const n = S.data.rows.length, nCont = used.filter(h => S.vars[h].type === 'continuous').length, nCat = used.length - nCont;
        const miss = used.reduce((a, h) => a + S.data.rows.filter(r => DE.isMissing(r[h]) || (S.vars[h].type === 'continuous' && !Number.isFinite(Number(r[h])))).length, 0) / Math.max(1, n * used.length);
        const missText = miss > 0.05 ? ` About ${Math.round(100 * miss)}% of the values are missing.` : '';
        if (!used.length) return null;
        if (n >= NEURAL_GOOD) return { method: 'neural', why: `A large dataset (${n.toLocaleString('en-GB')} records): a neural network (self-organising map) arranges the records on a map of similar patterns and groups its units, which suits large administrative data with variables of mixed types. Confirm the groups with k-means or latent class analysis: patterns that appear with both are more convincing.${missText}` };
        if (!nCat && miss <= 0.05) return { method: 'kmeans', why: `All ${used.length} clustering variables are numbers and few values are missing: k-means on standardised variables is simple and efficient. Latent class analysis (latent profiles) or a Gaussian mixture also allow clusters of different spread and give membership probabilities.` };
        if (!nCat) return { method: 'lca', why: `All clustering variables are numbers, but some values are missing.${missText} Latent class analysis (latent profiles) uses each patient's recorded values directly, without filling them in.` };
        if (nCat >= nCont) return { method: 'lca', why: `Most clustering variables are categories or grades (${nCat} of ${used.length}): latent class analysis is designed for these, gives each patient's probability of belonging to each class, and uses recorded values directly when some are missing.${missText}` };
        if (n > 3000) return { method: 'lca', why: `A mix of numbers and categories with ${n} patients: PAM needs every pair of patients' dissimilarity, which becomes slow above a few thousand patients; latent class analysis handles mixed data at this size.` };
        return { method: 'pam-gower', why: `A mix of numbers (${nCont}) and categories or grades (${nCat}): PAM with balanced Gower dissimilarity treats each type appropriately and tolerates missing values.${missText} Latent class analysis is a model-based alternative; k-means also works well when most variables are numbers.` };
    }
    function renderAdvice(used) {
        const a = advise(used), box = $('method-advice');
        if (!a) { box.innerHTML = ''; return; }
        if (!S.methodTouched && $('method').value !== a.method && !(a.method === 'gmm' && $('method-gmm').disabled)) { $('method').value = a.method; }
        const n = includedRows().length, nTxt = n.toLocaleString('en-GB');
        const neuralNote = n < NEURAL_MIN ? `Neural-network clustering needs at least ${NEURAL_MIN.toLocaleString('en-GB')} records, so it is not offered for these data (${nTxt}): with fewer records a network learns the noise of these particular patients and its groups change from run to run.`
            : n < NEURAL_GOOD ? `Neural-network clustering is available (at least ${NEURAL_MIN.toLocaleString('en-GB')} records; these data have ${nTxt}), but below about ${NEURAL_GOOD.toLocaleString('en-GB')} records simpler methods are usually as good and more stable.`
                : `Neural-network clustering is suitable at this size (${nTxt} records; it needs at least ${NEURAL_MIN.toLocaleString('en-GB')} and works best from ${NEURAL_GOOD.toLocaleString('en-GB')}).`;
        const sizeNote = n > SAMPLE_MAX ? ` With more than ${SAMPLE_MAX.toLocaleString('en-GB')} records, PAM and Ward are not offered (they need the dissimilarity of every pair of records), and silhouettes and maps use a random sample of ${SAMPLE_N.toLocaleString('en-GB')} records.` : '';
        box.innerHTML = `<p><strong>Suggested for these data: ${esc(METHOD[a.method].name)}.</strong> ${esc(a.why)}${$('method').value === a.method ? '' : ` <button id="use-advice" class="btn small" type="button">Use it</button>`}</p><p class="small">${esc(neuralNote + sizeNote)}</p><p class="muted small">Tip: repeat the analysis with a second method. Sub-groups that appear with both are more convincing.</p>`;
        const b = $('use-advice'); if (b) b.addEventListener('click', () => { $('method').value = a.method; S.methodTouched = true; updateVarSummary(); });
    }

    // ------------------------------------------------------------------------------------- data preparation
    // o: optional overrides (method, missing, gowerWeights, cols) for the sensitivity analysis
    function prepare(o = {}) {
        const method = o.method ?? $('method').value, missingChoice = o.missing ?? $('missing').value, missing = method === 'neural' && missingChoice === 'mice' ? 'gower' : missingChoice, gowerWeights = o.gowerWeights ?? $('gower-weights').value, cols = o.cols ?? S.data.headers.filter(h => S.vars[h].role === 'cluster');
        const describeCols = S.data.headers.filter(h => S.vars[h].role === 'describe');
        const specs = cols.map(h => ({ col: h, type: S.vars[h].type, levels: S.vars[h].type === 'continuous' ? null : levelsOf(h) }));
        let rows = includedRows();
        // Original row numbers (to compare patients across specifications)
        const rowNo = new Map(S.data.rows.map((r, i) => [r, i]));
        rows.forEach(r => { if (!r.__row && r.__row !== 0) Object.defineProperty(r, '__row', { value: rowNo.get(r), enumerable: false, writable: true }); });
        const isMiss = (r, h) => DE.isMissing(r[h]) || (S.vars[h].type === 'continuous' && !Number.isFinite(Number(r[h])));
        const nAll = rows.length;
        if (missing === 'complete' || (missing === 'gower' && method !== 'pam-gower' && method !== 'lca' && method !== 'neural')) rows = rows.filter(r => cols.every(h => !isMiss(r, h)));
        let filled = 0;
        const fillVal = {};
        if (missing === 'fill') specs.forEach(s => {
            const vals = rows.filter(r => !isMiss(r, s.col)).map(r => r[s.col]);
            if (s.type === 'continuous') { const x = vals.map(Number).sort((a, b) => a - b); fillVal[s.col] = x.length ? (x.length % 2 ? x[(x.length - 1) / 2] : (x[x.length / 2 - 1] + x[x.length / 2]) / 2) : 0; }
            else { const cnt = {}; vals.forEach(v => { const k = String(v).trim(); cnt[k] = (cnt[k] || 0) + 1; }); fillVal[s.col] = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0]; }
        });
        if (missing === 'fill') rows = rows.map(r => { const c = { ...r }; Object.defineProperty(c, '__row', { value: r.__row, enumerable: false }); cols.forEach(h => { if (isMiss(r, h)) { c[h] = fillVal[h]; filled++; } }); return c; });
        if (rows.length < 10) throw new Error(`Only ${rows.length} patients remain after handling missing values; at least 10 are needed.`);
        cols.forEach(h => { if (rows.every(r => isMiss(r, h))) throw new Error(`${h} has no recorded values in the patients analysed; set it to "Ignore" in step 2.`); });
        const seed = Number($('seed').value) || 2024, pcol = $('patient-col').value;
        if ((method === 'pam-gower' || method === 'ward') && rows.length > SAMPLE_MAX) throw new Error(`${METHOD[method].short} needs the dissimilarity of every pair of records and is limited to ${SAMPLE_MAX.toLocaleString('en-GB')} records; these data have ${rows.length.toLocaleString('en-GB')}. Choose latent class analysis, k-means or the neural network.`);
        if (method === 'neural' && rows.length < NEURAL_MIN) throw new Error(`The neural network needs at least ${NEURAL_MIN.toLocaleString('en-GB')} records; these data have ${rows.length}.`);
        // Random sample for the pairwise dissimilarities of large datasets (the same for every imputed dataset)
        const sampleIdx = rows.length > SAMPLE_MAX ? (() => { const rng = CE.createPRNG(seed + 77), a = rows.map((_, i) => i); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, SAMPLE_N).sort((x, y) => x - y); })() : null;
        // Records with a clustering value that was not recorded: imputed (filled or multiply imputed) or, with methods
        // that use recorded values only, left out of that record's distances
        const incomplete = rows.map(r => cols.some(h => isMiss(S.data.rows[r.__row], h)));
        const incompleteKind = missing === 'fill' || missing === 'mice' ? 'imputed' : missing === 'complete' ? null : 'missing';
        // Rows as recorded (before any filling): used to describe patients, never the filled values
        const recorded = rows.map(r => S.data.rows[r.__row]);
        const base = { recorded, fillVal, incomplete, incompleteKind, rowIds: rows.map(r => r.__row), sampleIdx, neuralMissingNote: method === 'neural' && missingChoice === 'mice', manualExcluded: S.excluded ? S.excluded.size : 0, method, missing, gowerWeights, cols, describe: describeCols, specs, rows, excluded: nAll - rows.length, filled, seed,
            groups: pcol ? rows.map(r => String(r[pcol] ?? '')) : null, patientCol: pcol, allContinuous: specs.every(s => s.type === 'continuous') };
        if (missing !== 'mice') return { ...base, ...encode(rows, specs, method, isMiss, gowerWeights, sampleIdx) };
        // Multiple imputation by chained equations (predictive mean matching), from the clustering variables only:
        // outcomes kept for description must not shape the clusters. Each imputed dataset is clustered and the
        // results are combined into a consensus (Basagana et al. 2013, Am J Epidemiol).
        const clean = rows.map(r => { const c = { ...r }; cols.forEach(h => { if (isMiss(r, h)) c[h] = ''; }); return c; });
        const cfg = {}; cols.forEach(h => { cfg[h] = { role: 'feature', type: S.vars[h].type === 'nominal' ? 'nominal' : 'numeric' }; });
        const missingValues = clean.reduce((a, r) => a + cols.filter(h => DE.isMissing(r[h])).length, 0);
        const missingPatients = clean.filter(r => cols.some(h => DE.isMissing(r[h]))).length;
        const mi = DE.miceImpute(clean, cfg, { m: MICE_M, iterations: 10, seed });
        const imps = mi.datasets.map(d => encode(d, specs, method, isMiss, gowerWeights, sampleIdx));
        const P = { ...base, ...imps[0], mice: { m: MICE_M, iterations: 10, missingValues, missingPatients } };
        P.impP = imps.map(e => ({ ...P, ...e }));
        // Map: principal components of the average of the imputed (standardised) datasets
        if (imps[0].Z) P.Zmap = imps[0].Z.map((r, i) => r.map((_, j) => imps.reduce((a, e) => a + e.Z[i][j], 0) / imps.length));
        return P;
    }
    // Raw values (for Gower and the mixed-data test) and a numerical coding (for k-means, Ward, mixtures, PCA)
    function encode(rows, specs, method, isMiss, gowerWeights, sampleIdx = null) {
        const raw = rows.map(r => specs.map(s => (isMiss(r, s.col) ? null : (s.type === 'continuous' ? Number(r[s.col]) : String(r[s.col]).trim()))));
        const complete = raw.every(r => r.every(v => v !== null));
        let X = null, Z = null, zstat = null;
        const colNames = [];
        if (complete) {
            X = raw.map(r => { const out = []; specs.forEach((s, j) => { const v = r[j]; if (s.type === 'continuous') out.push(v); else if (s.type === 'ordinal') out.push(s.levels.indexOf(v)); else if (s.levels.length === 2) out.push(v === s.levels[1] ? 1 : 0); else s.levels.forEach(l => out.push(v === l ? 1 : 0)); }); return out; });
            specs.forEach(s => { if (s.type === 'nominal' && s.levels.length > 2) s.levels.forEach(l => colNames.push(`${s.col} = ${l}`)); else colNames.push(s.type === 'nominal' ? `${s.col} = ${s.levels[1]}` : s.col); });
            const st = CE.standardize(X); Z = st.Z; zstat = { mean: st.mean, sd: st.sd };
        }
        // Gower is also computed for latent classes: silhouettes and the new-patient calculator use it
        const take = A => (sampleIdx ? sampleIdx.map(i => A[i]) : A);
        const gower = method === 'pam-gower' || method === 'lca' || method === 'neural' ? CE.gowerMatrix(take(raw), specs.map(s => ({ type: s.type, levels: s.levels })), { weights: gowerWeights === 'balanced' ? 'balanced' : null }) : null;
        const euclid = Z && !gower ? CE.euclideanMatrix(take(Z)) : null;
        return { raw, X, Z, zstat, colNames, gower, euclid, D: gower || euclid };
    }
    const lcaVars = P => P.specs.map(s => ({ type: s.type, levels: s.levels }));
    const modelBased = P => P.method === 'gmm' || P.method === 'lca';
    // Silhouette on the dissimilarities (on the random sample for large datasets)
    const sampleLabels = (P, labels) => (P.sampleIdx ? P.sampleIdx.map(i => labels[i]) : labels);
    const silOf = (Q, labels) => CE.silhouette(Q.D, sampleLabels(Q, labels));
    // Neural network (self-organising map): trained once on all records (missing values left out of the distances);
    // its units are grouped by weighted k-means for the full fit and for every bootstrap resample
    async function trainNeural(P) {
        const vars = P.specs.map(s => ({ name: s.col, type: s.type, levels: s.levels })), coded = NN.codeData(P.raw, vars);
        status('explore-status', `Training the self-organising map on ${P.rows.length.toLocaleString('en-GB')} records…`); await yieldUI();
        const map = await NN.somAsync(coded.X, coded.M, {}, (d, E) => status('explore-status', `Training the self-organising map on ${P.rows.length.toLocaleString('en-GB')} records: epoch ${d} of ${E}…`));
        P.neural = { map, coded, grid: `${map.rows} x ${map.cols}`, units: map.rows * map.cols, epochs: map.epochs, qe: map.qe, te: map.te };
    }
    // Clustering of a subset of the patients (indices into P.rows), for the main fit and for bootstrap resamples
    function clusterIdx(P, k, idx, seed) {
        const sub = M => { const m = idx.length, D = new Float64Array(m * m); for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) D[a * m + b] = M.D[idx[a] * M.n + idx[b]]; return { n: m, D }; };
        if (P.method === 'pam-gower') return CE.pam(sub(P.gower), k).labels;
        if (P.method === 'kmeans') return CE.kmeans(idx.map(i => P.Z[i]), k, { seed, nInit: 5 }).labels;
        if (P.method === 'neural') { const m = P.neural.map, w = new Array(m.W.length).fill(0); idx.forEach(i => { w[m.bmu[i]]++; }); const g = NN.weightedKmeans(m.W, w, k, { seed, nInit: 5 }); return idx.map(i => g.labels[m.bmu[i]]); }
        if (P.method === 'ward') return CE.cutTree(CE.hierarchical(sub(P.euclid), 'ward'), idx.length, k);
        // Bootstrap refits: fewer starts and a looser convergence criterion (the classes, not the last decimals, matter)
        if (P.method === 'lca') return CE.lca(idx.map(i => P.raw[i]), lcaVars(P), k, { nInit: 3, seed, tol: 1e-6, maxIter: 300 }).labels;
        return CE.gmm(idx.map(i => P.Z[i]), k, { covType: 'diag', nInit: 3, seed }).labels;
    }
    async function fitOne(P, k) {
        const all = P.rows.map((_, i) => i);
        if (P.method === 'gmm') { const g = CE.gmm(P.Z, k, { covType: 'diag', nInit: 5, seed: P.seed }); return { labels: g.labels, bic: g.bic }; }
        if (P.method === 'neural') { const m = P.neural.map, g = NN.weightedKmeans(m.W, m.hits, k, { seed: P.seed, nInit: 10 }); return { labels: m.bmu.map(u => g.labels[u]), centroids: g.centres, unitLabels: g.labels }; }
        if (P.method === 'lca') { const m = await CE.lcaAsync(P.raw, lcaVars(P), k, { nInit: 10, seed: P.seed, tol: 1e-8, maxIter: 1000 }); return { labels: m.labels, bic: m.bic, entropy: m.entropy, posterior: m.posterior, model: m.model, lcaFit: { startsAtBest: m.startsAtBest, starts: m.starts, converged: m.converged } }; }
        return { labels: clusterIdx(P, k, all, P.seed) };
    }
    // With multiple imputation: cluster every imputed dataset, then the consensus partition is k-medoids on the
    // proportion of imputed datasets in which two patients are NOT in the same cluster.
    async function fullFit(P, k) {
        if (!P.impP) return fitOne(P, k);
        const fits = [];
        for (const Q of P.impP) fits.push(await fitOne(Q, k));
        const n = P.rows.length, m = fits.length;
        if (n > SAMPLE_MAX) {
            // Large datasets: align each imputed dataset's clusters to the first (greatest overlap) and take the majority
            const ref = fits[0].labels, aligned = fits.map(f => { const tab = Array.from({ length: k }, () => new Array(k).fill(0)); f.labels.forEach((l, i) => { tab[l][ref[i]]++; }); const map = new Array(k).fill(-1), used = new Set(); tab.flatMap((row, a) => row.map((v, b) => [v, a, b])).sort((x, y) => y[0] - x[0]).forEach(([, a, b]) => { if (map[a] < 0 && !used.has(b)) { map[a] = b; used.add(b); } }); return f.labels.map(l => map[l]); });
            const voted = ref.map((_, i) => { const c = new Array(k).fill(0); aligned.forEach(a => { c[a[i]]++; }); return c.indexOf(Math.max(...c)); });
            const labels = CE.relabelBySize({ labels: voted }).labels;
            return { labels, bic: modelBased(P) ? fits.reduce((a, f) => a + f.bic, 0) / m : undefined, entropy: P.method === 'lca' ? fits.reduce((a, f) => a + f.entropy, 0) / m : undefined, agreement: fits.reduce((a, f) => a + CE.adjustedRand(f.labels, labels), 0) / m };
        }
        const D = new Float64Array(n * n);
        for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) { let apart = 0; fits.forEach(f => { if (f.labels[a] !== f.labels[b]) apart++; }); D[a * n + b] = D[b * n + a] = apart / m; }
        const labels = CE.pam({ n, D }, k).labels;
        return { labels, bic: modelBased(P) ? fits.reduce((a, f) => a + f.bic, 0) / m : undefined, entropy: P.method === 'lca' ? fits.reduce((a, f) => a + f.entropy, 0) / m : undefined,
            agreement: fits.reduce((a, f) => a + CE.adjustedRand(f.labels, labels), 0) / m };
    }
    // Statistic averaged over the imputed datasets (or computed once without imputation)
    const pooled = (P, fn) => { const Qs = P.impP || [P]; return Qs.reduce((a, Q) => a + fn(Q), 0) / Qs.length; };
    // Clustering of a bootstrap resample: with imputation, successive resamples cycle through the imputed datasets,
    // so the stability also reflects the uncertainty of the imputed values.
    function resampleFn(P, k, salt) { let b = 0; return idx => { const Q = P.impP ? P.impP[b++ % P.impP.length] : P; return clusterIdx(Q, k, idx, P.seed + idx.length * salt); }; }



    // ------------------------------------------------------------------------------------- analysis record
    const BUILD = (() => { const el = document.querySelector('script[src*="src/app.js"]'); const m = el && /[?&]v=(\d+)/.exec(el.getAttribute('src')); return m ? `build ${m[1]}` : 'unknown build'; })();
    const DISCLAIMER = 'For research and education only; not a medical device. Not for diagnosis or for decisions about individual patients. The methods are checked against established software and in simulations; the tool has not yet been validated on real cohorts or reviewed independently by a statistician.';
    const PURPOSE = { exploratory: 'Exploratory: looking for possible sub-groups or patterns', confirmatory: 'Confirmatory: testing sub-groups and outcomes specified in advance', quality: 'Data quality: looking for errors and unusual records', reduction: 'Dimension reduction: summarising many variables', groundwork: 'Groundwork for a future prediction model' };
    function analysisRecord() {
        const { P, res, gap } = S.explore, R = S.result, sug = suggestion(), ids = window.UMEChecks.label ? i => window.UMEChecks.label(S, i) : i => `Row ${i + 1}`;
        const r3 = v => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);
        return {
            software: { name: 'Unsupervised Medical Explorer', build: BUILD, created: new Date().toISOString(), copyright: '© 2026 Dr G Narenthiran. All rights reserved.' },
            disclaimer: DISCLAIMER,
            purpose: PURPOSE[$('purpose').value] || $('purpose').value, clinicalQuestion: $('question').value.trim() || null,
            data: { file: S.data.name, simulatedDemo: !!S.data.demo, recordsInFile: S.data.rows.length, recordsAnalysed: P.rows.length, leftOutMissing: P.excluded, valuesFilled: P.filled || 0,
                leftOutAfterDataChecks: [...(S.excluded || [])].sort((a, b) => a - b).map(i => ({ record: ids(i), reasons: window.UMEChecks.reasonsFor ? window.UMEChecks.reasonsFor(i) : [] })), flagsMarkedGenuine: window.UMEChecks.markedGenuine ? window.UMEChecks.markedGenuine() : [] },
            variables: Object.fromEntries(S.data.headers.map(h => [h, { role: S.vars[h].role, type: S.vars[h].type }])),
            warnings: [...document.querySelectorAll('#var-warnings li')].map(li => li.textContent),
            settings: { method: METHOD[P.method].name, missingValues: P.missing, multipleImputation: P.mice ? { datasets: P.mice.m, iterations: P.mice.iterations } : null, gowerWeights: ['pam-gower', 'lca'].includes(P.method) ? P.gowerWeights : null,
                clustersCompared: [res[0].k, res[res.length - 1].k], randomSeed: P.seed, patientIdentifierColumn: P.patientCol || null,
                sampleForPairwiseDissimilarities: P.sampleIdx ? P.sampleIdx.length : null, neuralNetwork: P.neural ? { grid: P.neural.grid, epochs: P.neural.epochs, topographicError: r3(P.neural.te) } : null },
            candidates: res.map(r => ({ clusters: r.k, averageSilhouette: r3(r.silhouette), lowestStability: r3(r.minJaccard), lowestStability95: r.minCI ? r.minCI.map(r3) : null, calinskiHarabasz: r3(r.ch), daviesBouldin: r3(r.db), bic: r3(r.bic), entropy: r3(r.entropy), agreementAcrossImputations: r3(r.agreement), smallestCluster: r.smallest, gap: gap ? r3(gap.table[r.k - 1].gap) : null })),
            selectionRule: 'Stable solutions only (every cluster mean bootstrap Jaccard >= 0.75, 25 resamples); model-based methods: lowest BIC among them; otherwise the smallest number with average silhouette within 0.02 of the best. A rule of thumb, not proof.',
            suggestion: sug.text,
            chosen: R ? {
                clusters: R.k, sizes: Array.from({ length: R.k }, (_, c) => R.labels.filter(l => l === c).length), averageSilhouette: r3(R.sil.mean),
                stability: { resamples: R.stab.resamples, perCluster: R.stab.perCluster.map(c => ({ cluster: c.cluster + 1, meanJaccard: r3(c.meanJaccard), interval95: c.ciMean ? c.ciMean.map(r3) : null, shareAcrossResamples: c.shareRange ? c.shareRange.map(r3) : null, dissolved: r3(c.dissolved) })),
                    patientsStableInAtLeast90pc: (() => { const ps = R.stab.patientStability.filter(v => v !== null); return ps.length ? r3(ps.filter(v => v >= 0.9).length / ps.length) : null; })() },
                testOfNoRealSubgroups: R.test ? { reference: R.test.kind, pValue: r3(R.test.pValue), resamples: R.test.B, observedR2: r3(R.test.observed), correlationFitError: Number.isFinite(R.test.fitError) ? r3(R.test.fitError) : null, pooledOverImputations: R.test.pooled || null } : null,
                verdict: (document.querySelector('#clusters-out .verdict') || {}).textContent?.replace(/\s+/g, ' ').trim() || null
            } : null,
            sensitivity: R && R.sens ? R.sens.results.map(r => ({ specification: `${r.group}: ${r.label}`, patientsCompared: r.n ?? null, adjustedRand: r3(r.ari), sameCluster: r3(r.same), error: r.error || null })) : null,
            classification: window.UMEUse.summary ? window.UMEUse.summary() : null
        };
    }
    // R script repeating the chosen clustering with established packages, to compare with the app (src/r-script.js)
    function downloadR() {
        const { P } = S.explore, text = window.UMERScript.build({ S, P, R: S.result, build: BUILD, seed: P.seed });
        const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' })), a = document.createElement('a');
        a.href = url; a.download = `repeat_in_R_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}.R`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }
    function recordText(A) {
        const L = [], j = v => (v === null || v === undefined ? '–' : v);
        L.push('ANALYSIS RECORD: Unsupervised Medical Explorer', `${A.software.build} · ${A.software.created.slice(0, 10)} · ${A.software.copyright}`, A.disclaimer, '');
        L.push(`Purpose: ${A.purpose}`, `Clinical question: ${j(A.clinicalQuestion)}`, '');
        L.push(`Data: ${A.data.file}${A.data.simulatedDemo ? ' (simulated demonstration data, not real patients)' : ''}; ${A.data.recordsInFile} records in the file, ${A.data.recordsAnalysed} analysed${A.data.leftOutMissing ? `, ${A.data.leftOutMissing} left out for missing values` : ''}${A.data.valuesFilled ? `, ${A.data.valuesFilled} values filled` : ''}.`);
        if (A.data.leftOutAfterDataChecks.length) { L.push(`Left out after the data checks (${A.data.leftOutAfterDataChecks.length}):`); A.data.leftOutAfterDataChecks.forEach(e => L.push(`  - ${e.record}: ${e.reasons.join('; ') || 'left out by the user'}`)); }
        if (A.data.flagsMarkedGenuine.length) L.push(`Data-check flags marked as genuine by the user: ${A.data.flagsMarkedGenuine.map(g => `${g.flag} in ${g.column} (${g.values})`).join('; ')}`);
        L.push('', 'Variables:'); Object.entries(A.variables).forEach(([h, v]) => L.push(`  - ${h}: ${v.role === 'cluster' ? 'used to find clusters' : v.role === 'describe' ? 'describe only' : 'ignored'} (${v.type})`));
        if (A.warnings.length) { L.push('', 'Warnings shown before the analysis:'); A.warnings.forEach(w => L.push(`  - ${w}`)); }
        const st = A.settings;
        L.push('', `Method: ${st.method}. Missing values: ${st.missingValues}${st.multipleImputation ? ` (${st.multipleImputation.datasets} imputed datasets)` : ''}. Gower weights: ${j(st.gowerWeights)}. Clusters compared: ${st.clustersCompared.join(' to ')}. Random seed: ${st.randomSeed}.${st.sampleForPairwiseDissimilarities ? ` Silhouettes and maps on a random sample of ${st.sampleForPairwiseDissimilarities}.` : ''}`);
        L.push('', 'Candidate numbers of clusters (silhouette; lowest stability [95% interval]; smallest cluster):');
        A.candidates.forEach(c => L.push(`  ${c.clusters}: ${j(c.averageSilhouette)}; ${j(c.lowestStability)}${c.lowestStability95 ? ` [${c.lowestStability95.join('–')}]` : ''}; ${c.smallestCluster}${c.bic !== null ? `; BIC ${c.bic}` : ''}${c.gap !== null ? `; gap ${c.gap}` : ''}`));
        L.push(`Selection rule: ${A.selectionRule}`, `Suggestion: ${A.suggestion}`);
        if (A.chosen) {
            const c = A.chosen;
            L.push('', `Chosen: ${c.clusters} clusters, sizes ${c.sizes.join(' / ')}; average silhouette ${c.averageSilhouette}.`);
            c.stability.perCluster.forEach(p => L.push(`  Cluster ${p.cluster}: stability ${p.meanJaccard} [${(p.interval95 || []).join('–')}], dissolved in ${Math.round(100 * p.dissolved)}% of ${c.stability.resamples} resamples`));
            L.push(`  Patients in their cluster in at least 90% of resamples: ${c.stability.patientsStableInAtLeast90pc !== null ? Math.round(100 * c.stability.patientsStableInAtLeast90pc) + '%' : '–'}`);
            if (c.testOfNoRealSubgroups) L.push(`  Test of no real sub-groups (${c.testOfNoRealSubgroups.reference} reference, ${c.testOfNoRealSubgroups.resamples} datasets): p = ${c.testOfNoRealSubgroups.pValue}`);
            L.push(`  Verdict: ${j(c.verdict)}`);
        }
        if (A.sensitivity) { L.push('', 'Sensitivity to modelling choices (adjusted Rand; same cluster):'); A.sensitivity.forEach(r => L.push(`  - ${r.specification}: ${r.error ? 'not possible (' + r.error + ')' : `${r.adjustedRand}; ${Math.round(100 * r.sameCluster)}%`}`)); }
        if (A.classification) { const k2 = A.classification; L.push('', `Classification: ${k2.classes.map(c => `${c.class} ${c.name}`).join('; ')}`, `Bedside rules (at most ${k2.maxConditions} conditions; cross-validated agreement with the clusters ${Math.round(100 * k2.crossValidatedAgreement)}%, kappa ${k2.kappa.toFixed(2)}; ${k2.note})`); k2.rules.forEach(r => L.push(`  - If ${r.conditions.join(' and ') || 'all'} -> ${r.class} (${r.patients} patients, ${Math.round(100 * r.purity)}%)`)); }
        L.push('', 'For research and education only; not a medical device. Clusters found in one dataset are hypotheses: replicate them in another cohort.');
        return L.join('\r\n');
    }
    function downloadRecord(kind) {
        const A = analysisRecord(), base = `analysis_record_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}`;
        const blob = kind === 'json' ? new Blob([JSON.stringify(A, null, 2)], { type: 'application/json' }) : new Blob([recordText(A)], { type: 'text/plain' });
        const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = `${base}.${kind === 'json' ? 'json' : 'txt'}`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }
    // ------------------------------------------------------------------------------------- sensitivity analysis
    // The chosen number of clusters refitted under reasonable alternative specifications (each clustering variable
    // left out, other suitable methods, other handling of missing values, other Gower weights); each is compared with
    // the main solution on the patients both include: adjusted Rand index and share of patients in the same cluster
    // (after matching the clusters by greatest overlap), and every patient's share of alternatives that agree.
    async function sensitivity() {
        const P = S.explore.P, k = S.result.k, main = new Map(P.rowIds.map((id, i) => [id, S.result.labels[i]]));
        const n = includedRows().length, anyMissing = P.cols.some(h => includedRows().some(r => DE.isMissing(r[h]) || (S.vars[h].type === 'continuous' && !Number.isFinite(Number(r[h])))));
        const allCont = P.cols.every(h => S.vars[h].type === 'continuous');
        const allowed = m => (m === 'pam-gower' || m === 'ward' ? n <= SAMPLE_MAX : m === 'neural' ? n >= NEURAL_MIN : m === 'gmm' ? allCont : true);
        const specs = [];
        if (P.cols.length >= 3) P.cols.forEach(c => specs.push({ group: 'Variables', label: `Without ${c}`, o: { cols: P.cols.filter(x => x !== c) } }));
        ['pam-gower', 'kmeans', 'lca', 'neural', 'ward', 'gmm'].filter(m => m !== P.method && allowed(m)).forEach(m => specs.push({ group: 'Method', label: METHOD[m].short, o: { method: m } }));
        if (anyMissing) ['fill', 'complete', 'mice', 'gower'].filter(m => m !== P.missing && (m !== 'mice' || n <= SAMPLE_MAX) && (m !== 'gower' || ['pam-gower', 'lca', 'neural'].includes(P.method))).forEach(m => specs.push({ group: 'Missing values', label: { fill: 'Filled (median / most frequent)', complete: 'Complete cases only', mice: 'Multiple imputation (MICE)', gower: 'Recorded values only' }[m], o: { missing: m } }));
        if (P.method === 'pam-gower') specs.push({ group: 'Weights', label: P.gowerWeights === 'balanced' ? 'Equal Gower weights' : 'Balanced Gower weights', o: { gowerWeights: P.gowerWeights === 'balanced' ? 'equal' : 'balanced' } });
        const results = [], agree = new Map(), seen = new Map();
        const st = $('sens-status'); $('sens-btn').disabled = true;
        for (const [t, sp] of specs.entries()) {
            st.textContent = `Sensitivity analysis ${t + 1} of ${specs.length}: ${sp.label}…`; await yieldUI();
            try {
                const Q = prepare({ method: P.method, missing: P.missing, gowerWeights: P.gowerWeights, cols: P.cols, ...sp.o });
                if (Q.method === 'neural') await trainNeural(Q);
                const f = await fullFit(Q, k), ids = Q.rowIds.filter(id => main.has(id)), lab = new Map(Q.rowIds.map((id, i) => [id, f.labels[i]]));
                // Match the alternative's clusters to the main ones by greatest overlap
                const kq = Math.max(...f.labels) + 1, tab = Array.from({ length: kq }, () => new Array(k).fill(0));
                ids.forEach(id => { tab[lab.get(id)][main.get(id)]++; });
                const map = new Array(kq).fill(-1), used = new Set();
                tab.flatMap((row, a) => row.map((v, b) => [v, a, b])).sort((x, y) => y[0] - x[0]).forEach(([, a, b]) => { if (map[a] < 0 && !used.has(b)) { map[a] = b; used.add(b); } });
                let same = 0;
                ids.forEach(id => { const ok = map[lab.get(id)] === main.get(id); if (ok) same++; seen.set(id, (seen.get(id) || 0) + 1); if (ok) agree.set(id, (agree.get(id) || 0) + 1); });
                results.push({ ...sp, n: ids.length, ari: CE.adjustedRand(ids.map(id => main.get(id)), ids.map(id => lab.get(id))), same: ids.length ? same / ids.length : null });
            } catch (e) { results.push({ ...sp, error: e.message }); }
        }
        const share = new Map([...main.keys()].map(id => [id, seen.get(id) ? (agree.get(id) || 0) / seen.get(id) : null]));
        S.result.sens = { results, share };
        st.textContent = ''; $('sens-btn').disabled = false;
        renderSensitivity();
    }
    function sensEvidenceCells() {
        const sens = S.result && S.result.sens;
        if (!sens) return '<td>Sensitivity to modelling choices</td><td>not run</td><td>run it below ("Run the sensitivity analysis")</td>';
        const ok = sens.results.filter(r => !r.error).map(r => r.ari).sort((x, y) => x - y);
        if (!ok.length) return '<td>Sensitivity to modelling choices</td><td>–</td><td>no alternative could be fitted</td>';
        const med = ok[Math.floor((ok.length - 1) / 2)], low = ok[0];
        return `<td>Sensitivity to modelling choices (adjusted Rand, median and lowest of ${ok.length})</td><td>${med.toFixed(2)}; ${low.toFixed(2)}</td><td class="${med >= 0.6 ? 'ok' : 'warn'}">${med >= 0.8 ? 'robust to most choices' : med >= 0.6 ? 'mostly robust' : 'depends on the choices made'}${low < 0.4 ? ' (at least one alternative gives different clusters)' : ''}</td>`;
    }
    function renderSensitivity() {
        const el = $('sens-out'); if (!el || !S.result || !S.result.sens) return;
        const evs = $('ev-sens'); if (evs) evs.innerHTML = sensEvidenceCells();
        const { results, share } = S.result.sens, f2 = v => (Number.isFinite(v) ? v.toFixed(2) : '–');
        const vals = [...share.values()].filter(v => v !== null), all = vals.filter(v => v === 1).length, half = vals.filter(v => v < 0.5).length;
        const word = a => (a >= 0.8 ? 'very similar' : a >= 0.6 ? 'similar' : a >= 0.4 ? 'partly similar' : 'different');
        el.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Alternative specification</th><th>Patients compared</th><th>Agreement (adjusted Rand)</th><th>Same cluster</th></tr></thead><tbody>${results.map(r => `<tr><td><span class="muted">${esc(r.group)}:</span> ${esc(r.label)}</td>${r.error ? `<td colspan="3" class="muted">not possible: ${esc(r.error)}</td>` : `<td>${r.n.toLocaleString('en-GB')}</td><td>${f2(r.ari)} <span class="muted">${word(r.ari)}</span></td><td>${Math.round(100 * r.same)}%</td>`}</tr>`).join('')}</tbody></table></div>
            <p class="small"><strong>Patients:</strong> ${Math.round(100 * all / Math.max(1, vals.length))}% stay in the same cluster under every alternative; ${Math.round(100 * half / Math.max(1, vals.length))}% change cluster under at least half of them. The download lists each patient's share ("robustness to modelling choices").</p>
            <p class="muted small">Each alternative keeps the chosen number of clusters and every other setting; clusters are matched to the main solution by greatest overlap. Clusters that persist when a variable is left out, with another method and with other handling of missing values are more convincing; clusters that depend on one variable or one method should be described as such.</p>`;
    }
    // ------------------------------------------------------------------------------------- step 4: how many clusters?
    async function explore() {
        let P;
        try { P = prepare(); } catch (e) { status('explore-status', e.message, 'bad'); return; }
        S.prep = P; S.result = null;
        if (P.method === 'neural') { $('explore-btn').disabled = true; try { await trainNeural(P); } catch (e) { status('explore-status', `The neural network could not be trained: ${e.message}`, 'bad'); $('explore-btn').disabled = false; return; } }
        const kMax = Math.min(Number($('kmax').value) || 8, P.rows.length - 1), ks = []; for (let k = 2; k <= kMax; k++) ks.push(k);
        $('explore-out').innerHTML = ''; $('explore-btn').disabled = true;
        const res = [];
        try {
            for (const k of ks) {
                status('explore-status', `Fitting ${k} clusters, with a quick stability check (${METHOD[P.method].short})…`); await yieldUI();
                const fit = await fullFit(P, k);
                const sil = pooled(P, Q => silOf(Q, fit.labels).mean);
                const stab = await CE.bootstrapStabilityAsync(P.rows.length, fit.labels, resampleFn(P, k, 1), { B: 25, seed: P.seed, groups: P.groups }, (d, B) => status('explore-status', `${k} clusters (${METHOD[P.method].short}): stability check, resample ${d} of ${B}…`));
                const sizes = new Array(k).fill(0); fit.labels.forEach(l => { sizes[l]++; });
                const weakC = stab.perCluster.reduce((a, c) => ((c.meanJaccard ?? 0) < (a.meanJaccard ?? 0) ? c : a));
                res.push({ k, labels: fit.labels, silhouette: sil, minCI: weakC.ciMean, minJaccard: Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0)), meanJaccard: stab.perCluster.reduce((a, c) => a + (c.meanJaccard ?? 0), 0) / k,
                    ch: P.Z ? pooled(P, Q => CE.calinskiHarabasz(Q.Z, fit.labels)) : null, db: P.Z ? pooled(P, Q => CE.daviesBouldin(Q.Z, fit.labels)) : null, bic: fit.bic ?? null, centroids: fit.centroids ?? null, unitLabels: fit.unitLabels ?? null, entropy: fit.entropy ?? null, posterior: fit.posterior ?? null, model: fit.model ?? null, smallest: Math.min(...sizes), agreement: fit.agreement ?? null, lcaFit: fit.lcaFit ?? null });
            }
            let gap = null;
            if (P.Z && (P.method === 'kmeans' || P.method === 'ward')) { status('explore-status', 'Gap statistic (comparison with reference data without clusters)…'); await yieldUI(); gap = CE.gapStatistic(P.Z, kMax, { B: 30, seed: P.seed, nInit: 3 }); }
            S.explore = { P, res, gap };
            renderExplore();
            status('explore-status', '');
        } catch (e) { status('explore-status', `The analysis stopped: ${e.message}`, 'bad'); }
        $('explore-btn').disabled = false;
    }
    // Suggested number of clusters: only stable solutions count (every cluster's mean bootstrap Jaccard >= 0.75);
    // among them the smallest k whose average silhouette is within 0.02 of the best (simpler is preferred when
    // the evidence is similar). The gap statistic (k-means, Ward) and BIC (mixtures) are shown alongside.
    function suggestion() {
        const { res, gap, P } = S.explore;
        if (gap && gap.kBest === 1) return { k: null, text: 'The gap statistic finds no more structure than in reference data without clusters (it chooses 1 cluster). Any clusters below may not be real.' };
        const stable = res.filter(r => r.minJaccard >= 0.75);
        if (!stable.length) return { k: null, text: 'No number of clusters gives stable clusters (every cluster reproducing in bootstrap resamples). The data may not contain distinct sub-groups; any clusters below should be treated as exploratory at best.' };
        // Model-based methods: the stable solution with the lowest BIC; otherwise the simplest stable solution with a
        // near-best silhouette
        const best = Math.max(...stable.map(r => r.silhouette));
        const pick = modelBased(P) ? stable.reduce((a, r) => (r.bic < a.bic ? r : a)) : stable.filter(r => r.silhouette >= best - 0.02).reduce((a, r) => (r.k < a.k ? r : a));
        const bicBest = modelBased(P) ? res.reduce((a, r) => (r.bic < a.bic ? r : a)) : null;
        const extra = modelBased(P) ? (bicBest.k !== pick.k ? ` Lowest BIC overall: ${bicBest.k} clusters, but that solution is not stable.` : '') + (P.method === 'lca' && pick.entropy < 0.6 ? ` Entropy is low (${pick.entropy.toFixed(2)}): the classes overlap heavily, and with related variables latent class analysis can create classes that are not real sub-groups.` : '') : gap ? ` Gap statistic: ${gap.kBest} clusters.` : '';
        const others = stable.filter(r => r.k !== pick.k).map(r => r.k);
        const also = others.length ? ` Also stable: ${others.join(', ')} clusters.${modelBased(P) ? '' : ' Average silhouette tends to favour 2 clusters,'} ${modelBased(P) ? 'Look' : 'so look'} at these too and choose the number that makes clinical sense.` : '';
        return { k: pick.k, text: `Suggested: ${pick.k} clusters, ${modelBased(P) ? 'the stable solution with the lowest BIC' : 'the simplest stable solution with a near-best average silhouette'}.${extra}${also}` };
    }
    function renderExplore() {
        const { res, gap, P } = S.explore, sug = suggestion(), f3 = v => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(3));
        const stabWord = v => (v >= 0.75 ? 'stable' : v >= 0.6 ? 'indicates a pattern' : v >= 0.5 ? 'weak' : 'not reproducible');
        const rows = res.map(r => `<tr${r.k === sug.k ? ' class="pick"' : ''}><td>${r.k}${r.k === sug.k ? ' (suggested)' : ''}</td><td>${f3(r.silhouette)}</td><td>${f3(r.minJaccard)}${r.minCI ? ` <span class="muted">(${f3(r.minCI[0])}–${f3(r.minCI[1])})</span>` : ''} <span class="muted">${stabWord(r.minJaccard)}${r.minCI && r.minCI[0] < 0.75 && r.minCI[1] >= 0.75 ? ', borderline' : ''}</span></td>
            ${P.Z ? `<td>${f3(r.ch)}</td><td>${f3(r.db)}</td>` : ''}${gap ? `<td>${f3(gap.table[r.k - 1].gap)} ± ${f3(gap.table[r.k - 1].se)}</td>` : ''}${modelBased(P) ? `<td>${r.bic.toFixed(0)}</td>` : ''}${P.method === 'lca' ? `<td>${f3(r.entropy)}</td>` : ''}${P.mice ? `<td>${f3(r.agreement)}</td>` : ''}<td>${r.smallest}</td></tr>`).join('');
        const charts = CH.lineChart({ series: [{ name: 'Average silhouette', cls: 0, points: res.map(r => ({ x: r.k, y: r.silhouette })) }, { name: 'Stability (lowest cluster Jaccard)', cls: 1, points: res.map(r => ({ x: r.k, y: r.minJaccard })) }], xLabel: 'Number of clusters', yLabel: 'Value', aria: 'Silhouette and stability by number of clusters' })
            + CH.legend(['Average silhouette', 'Stability (lowest cluster Jaccard)']);
        $('explore-out').innerHTML = `
            <p class="lead">${esc(sug.text)} ${P.manualExcluded ? `${P.manualExcluded} patient(s) were left out after the data checks. ` : ''}${P.excluded ? `${P.excluded} patient(s) with missing values were left out.` : ''}${P.filled ? ` ${P.filled} missing value(s) were filled with the median or most frequent value.` : ''}${P.mice ? ` Missing values (${P.mice.missingValues}, in ${P.mice.missingPatients} patients) were multiply imputed: ${P.mice.m} imputed datasets by chained equations with predictive mean matching, from the clustering variables only; each dataset was clustered and the results combined into a consensus.${gap ? ' The gap statistic uses the first imputed dataset.' : ''}` : ''}${P.neural ? ` Self-organising map of ${P.neural.grid} units (${(P.rows.length / P.neural.units).toFixed(0)} records per unit on average), trained for ${P.neural.epochs} epochs; topographic error ${(100 * P.neural.te).toFixed(1)}% (records whose two closest units are not neighbours on the map; low is good). Its units are grouped by k-means weighted by the records they hold.${P.neuralMissingNote ? ' (The map handles missing values itself, so the recorded values were used instead of multiple imputation.)' : ''}` : ''}${P.sampleIdx ? ` Silhouettes use a random sample of ${P.sampleIdx.length.toLocaleString('en-GB')} of the ${P.rows.length.toLocaleString('en-GB')} records.` : ''}</p>
            <div class="grid2"><div>${charts}</div><div class="muted small">
                <p><strong>Average silhouette</strong>: how clearly patients sit in their own cluster rather than the nearest other one (Kaufman and Rousseeuw: above 0.70 strong, 0.51–0.70 reasonable, 0.26–0.50 weak and possibly artificial, 0.25 or below no substantial structure).</p>
                <p><strong>Stability</strong>: the clustering is repeated on 25 bootstrap resamples; the lowest mean Jaccard similarity of any cluster to its counterpart (0.75 or more: stable; below 0.5: that cluster does not reproduce), with the 95% interval of that mean. With 25 resamples the interval is wide: "borderline" solutions (interval including 0.75) are checked with 100 resamples in the next step. The suggestion is a transparent rule of thumb, not proof that a number of clusters is correct.</p>
                ${modelBased(P) ? '<p><strong>BIC</strong>: fit of the model penalised for its complexity; lower is better. A difference of 10 or more is very strong evidence for the lower value, 2 to 6 positive evidence, under 2 negligible (Raftery 1995).</p>' : ''}
                ${P.method === 'lca' ? '<p><strong>Entropy</strong>: how clearly patients are assigned to one class (1: every patient certainly in one class; above 0.8 good; below 0.6 classes overlap a lot).</p>' : ''}
                ${P.mice ? '<p><strong>Agreement across imputations</strong>: average adjusted Rand index between the clustering of each imputed dataset and the consensus (1: the imputed values make no difference; below about 0.65: the clusters depend on how the missing values are filled).</p>' : ''}
                ${gap ? '<p><strong>Gap</strong>: how much more compact the clusters are than in reference data without clusters (± its standard error). Tibshirani\'s rule (the smallest number whose gap is at least the next gap minus its standard error) is used only to warn when it chooses 1 cluster.</p>' : ''}
                <p>No single number decides: choose the smallest number of clusters that is stable and clinically interpretable.</p></div></div>
            <div class="table-wrap"><table><thead><tr><th>Clusters</th><th>Average silhouette</th><th>Stability</th>${P.Z ? '<th>Calinski–Harabasz ↑</th><th>Davies–Bouldin ↓</th>' : ''}${gap ? '<th>Gap ± SE</th>' : ''}${modelBased(P) ? '<th>BIC ↓</th>' : ''}${P.method === 'lca' ? '<th>Entropy ↑</th>' : ''}${P.mice ? '<th>Agreement across imputations</th>' : ''}<th>Smallest cluster</th></tr></thead><tbody>${rows}</tbody></table></div>
            <div class="row"><label for="k-choice">Describe</label><select id="k-choice">${res.map(r => `<option value="${r.k}"${r.k === (sug.k || 2) ? ' selected' : ''}>${r.k} clusters</option>`).join('')}</select><button id="describe-btn" class="btn primary">Describe these clusters</button></div>`;
        $('describe-btn').addEventListener('click', () => describe(Number($('k-choice').value)));
        enableTabs(5);
    }

    // ------------------------------------------------------------------------------------- step 5: the clusters
    async function describe(k) {
        const P = S.explore.P, ex = S.explore.res.find(r => r.k === k);
        goTo(6); enableTabs(6);
        $('use-out').innerHTML = '';
        $('clusters-out').innerHTML = '';
        try {
            status('clusters-status', `Checking stability on 100 bootstrap resamples…`); await yieldUI();
            const labels = ex.labels;
            const stab = await CE.bootstrapStabilityAsync(P.rows.length, labels, resampleFn(P, k, 7), { B: 100, seed: P.seed + 1, groups: P.groups }, (d, B) => status('clusters-status', `Checking stability: bootstrap resample ${d} of ${B}…`));
            status('clusters-status', 'Testing whether the data have more structure than a single population…'); await yieldUI();
            let test = null;
            // Mixed data: categories with more than two levels enter the test with one reference level left out (their
            // indicators always add up to 1, which no correlation model can reproduce exactly)
            const keep = [], groups = []; let col = 0;
            P.specs.forEach(sp => { const w = sp.type === 'nominal' && sp.levels.length > 2 ? sp.levels.length : 1; const g = []; for (let q = 0; q < w; q++) { if (!(w > 1 && q === 0)) { if (w > 1) g.push(keep.length); keep.push(col); } col++; } if (g.length) groups.push(g); });
            const refX = X => X.map(r => keep.map(c => r[c]));
            // The Gaussian reference is exact for (near-)normal numbers but gives false positives when measurements are
            // skewed (43% in simulations), so it is used only when no clustering variable is clearly skewed; otherwise
            // the copula reference, which keeps each variable's own distribution
            const skewedVar = P.allContinuous && P.X ? P.specs.some((sp, j) => { const xs = P.X.map(r => r[j]).slice().sort((a2, b2) => a2 - b2), q = f => xs[Math.floor(f * (xs.length - 1))], lo = q(0.1), md = q(0.5), hi = q(0.9); return hi > lo && Math.abs((hi + lo - 2 * md) / (hi - lo)) > 0.2; }) : false;
            const useGaussian = P.allContinuous && !skewedVar;
            const testOne = async Q => (useGaussian ? { ...await CE.nullClusterTestAsync(Q.Z, k, { B: 99, seed: P.seed }), kind: 'gaussian' } : { ...await CE.copulaNullTestAsync(refX(Q.X), k, { B: 99, seed: P.seed, groups }), kind: 'copula', skewedNumbers: skewedVar });
            if (P.X && !P.impP) test = await testOne(P);
            if (P.X && P.impP) {
                // One test per imputed dataset, combined by the median p-value (Eekhout et al. 2017)
                const tests = [];
                for (const Q of P.impP) tests.push(await testOne(Q));
                tests.sort((a, b) => a.pValue - b.pValue); const mid = tests[Math.floor(tests.length / 2)];
                test = { ...mid, pooled: { m: tests.length, min: tests[0].pValue, max: tests[tests.length - 1].pValue } };
            }
            const sils = (P.impP || [P]).map(Q => silOf(Q, labels)), sLab = sampleLabels(P, labels);
            const sil = { mean: sils.reduce((a, x) => a + x.mean, 0) / sils.length, values: sLab.map((_, i) => sils.reduce((a, x) => a + x.values[i], 0) / sils.length), labels: sLab };
            const pca = P.Z ? CE.pca(P.Zmap || P.Z, 2) : null;
            S.result = { k, labels, stab, test, sil, pca, posterior: ex.posterior, model: ex.model, entropy: ex.entropy, lcaFit: ex.lcaFit, neural: P.neural ? { ...P.neural, centroids: ex.centroids, unitLabels: ex.unitLabels } : null };
            renderClusters();
            status('clusters-status', '');
            enableTabs(8);
        } catch (e) { status('clusters-status', `The analysis stopped: ${e.message}`, 'bad'); }
    }
    // opts.only = 'describe': outcome variables only; opts.names(c): column heading for cluster c
    function profileTable(opts = {}) {
        const { P } = S.explore, { labels, k } = S.result, n = labels.length, head = opts.names || (c => `Cluster ${c + 1}`);
        const groups = Array.from({ length: k }, (_, c) => labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0));
        const fmtNum = xs => { if (!xs.length) return '–'; const m = xs.reduce((a, b) => a + b, 0) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, xs.length - 1)); return `${m.toFixed(Math.abs(m) < 10 ? 2 : 1)} (${sd.toFixed(Math.abs(m) < 10 ? 2 : 1)})`; };
        const q = (xs, f) => { const s = xs.slice().sort((a, b) => a - b); const h = (s.length - 1) * f, lo = Math.floor(h); return s[lo] + (s[Math.ceil(h)] - s[lo]) * (h - lo); };
        const fmtOrd = xs => (xs.length ? `${q(xs, 0.5)} [${q(xs, 0.25)}–${q(xs, 0.75)}]` : '–');
        const pf = p => (p === null || p === undefined ? '–' : p < 0.001 ? '< 0.001' : p.toFixed(3));
        const rowsOut = [], tests = [];
        // Describe-only variables: p-value, Benjamini-Hochberg q-value (false discovery rate over these variables) and
        // effect size (Cramer's V for categories, epsilon squared for numbers and grades); filled in after the loop
        const slot = t => { tests.push(t); return `@@T${tests.length - 1}@@`; };
        [...(opts.only === 'describe' ? [] : P.cols.map(c => [c, true])), ...P.describe.map(c => [c, false])].forEach(([col, used]) => {
            const t = S.vars[col].type, vals = P.recorded.map(r => r[col]), ok = vals.map(v => !DE.isMissing(v) && (t !== 'continuous' || Number.isFinite(Number(v))));
            const missingN = ok.filter(x => !x).length;
            if (t === 'nominal' || (t === 'ordinal' && !vals.filter((v, i) => ok[i]).every(v => Number.isFinite(Number(v))))) {
                const levels = levelsOf(col), sv = vals.map(v => (DE.isMissing(v) ? null : String(v).trim()));
                const test = used ? null : CE.chiSquareTest(sv.filter(v => v !== null), labels.filter((_, i) => sv[i] !== null));
                if (test) { const okI = sv.map((v, i) => (v !== null ? i : -1)).filter(i => i >= 0); test.effect = CE.cramersV(test.statistic, okI.length, new Set(okI.map(i => sv[i])).size, new Set(okI.map(i => labels[i])).size); test.effectKind = 'V'; test.mark = slot(test); }
                levels.forEach((lv, li) => {
                    const cell = idx => { const okIdx = idx.filter(i => sv[i] !== null), c = okIdx.filter(i => sv[i] === lv).length; return okIdx.length ? `${c} (${(100 * c / okIdx.length).toFixed(0)}%)` : '–'; };
                    rowsOut.push(`<tr${used ? '' : ' class="describe"'}><td>${li === 0 ? `<strong>${esc(col)}</strong>${missingN ? ` <span class="muted">(${missingN} missing)</span>` : ''}` : ''}<div class="lvl">${esc(lv)}, n (%)</div></td>${groups.map(g => `<td>${cell(g)}</td>`).join('')}<td>${cell(labels.map((_, i) => i))}</td><td>${li === 0 ? (used ? '<span class="muted">used to form clusters</span>' : `${test.mark}${test.minExpected < 5 ? ' <span class="muted">(small counts)</span>' : ''}`) : ''}</td></tr>`);
                });
            } else {
                const xs = idx => idx.filter(i => ok[i]).map(i => Number(vals[i]));
                const fmt = t === 'ordinal' ? fmtOrd : fmtNum;
                const okI = labels.map((_, i) => i).filter(i => ok[i]);
                const test = used ? null : CE.kruskalWallis(okI.map(i => Number(vals[i])), okI.map(i => labels[i]));
                if (test) { test.effect = test.statistic !== null ? CE.epsilonSquared(test.statistic, okI.length) : null; test.effectKind = 'eps'; test.mark = slot(test); }
                rowsOut.push(`<tr${used ? '' : ' class="describe"'}><td><strong>${esc(col)}</strong>${missingN ? ` <span class="muted">(${missingN} missing)</span>` : ''}<div class="lvl">${t === 'ordinal' ? 'median [IQR]' : 'mean (SD)'}</div></td>${groups.map(g => `<td>${fmt(xs(g))}</td>`).join('')}<td>${fmt(xs(labels.map((_, i) => i)))}</td><td>${used ? '<span class="muted">used to form clusters</span>' : test.mark}</td></tr>`);
            }
        });
        const qs = CE.benjaminiHochberg(tests.map(t => t.p));
        const size = t => { if (t.effect === null || t.effect === undefined) return ''; const e = t.effect; const w = t.effectKind === 'V' ? (e >= 0.5 ? 'large' : e >= 0.3 ? 'medium' : e >= 0.1 ? 'small' : 'negligible') : (e >= 0.26 ? 'large' : e >= 0.08 ? 'medium' : e >= 0.01 ? 'small' : 'negligible'); return `<div class="muted">${t.effectKind === 'V' ? "Cramér's V" : 'ε²'} ${e.toFixed(2)} (${w})</div>`; };
        const cellFor = (t, qv) => `p ${pf(t.p)}${tests.length > 1 ? ` · q ${pf(qv)}` : ''}${size(t)}`;
        const body = rowsOut.join('').replace(/@@T(\d+)@@/g, (_, i) => cellFor(tests[Number(i)], qs[Number(i)]));
        return `<div class="table-wrap"><table class="profile"><thead><tr><th>Variable</th>${groups.map((g, c) => `<th><i class="sw s${c}"></i>${esc(head(c))}<div class="muted">n = ${g.length}</div></th>`).join('')}<th>All<div class="muted">n = ${n}</div></th><th>Describe-only variables: p, q, effect size</th></tr></thead><tbody>${body}</tbody></table></div>
            <p class="muted small">${opts.only === 'describe' ? `P-values: Kruskal–Wallis test (numbers, grades) or chi-square test (categories). ${tests.length > 1 ? `q: Benjamini–Hochberg adjustment for the ${tests.length} variables tested (false discovery rate); judge by q when looking across several outcomes. ` : ''}Effect size: Cramér's V (categories) or ε² (numbers, grades), with conventional labels. These comparisons are exploratory unless the outcomes were specified in advance; q-values adjust for testing several variables, not for the clusters having been found in these same patients, so confirm differences in an independent cohort. Differences in outcome support, but do not prove, that the classes are clinically meaningful.</p>` : ''}${opts.only === 'describe' ? '<p hidden>' : '<p class="muted small">'}Variables used to form the clusters differ between clusters by construction, so no p-values are shown for them. For variables used only to describe the clusters (shaded), p-values come from the Kruskal–Wallis test (numbers, grades) or the chi-square test (categories)${tests.length > 1 ? `; q is the Benjamini–Hochberg adjustment for the ${tests.length} variables tested (false discovery rate)` : ''}; effect size: Cramér's V or ε². With several variables some small p-values occur by chance; these comparisons are exploratory unless the outcomes were specified in advance, and q-values do not correct for the clusters having been found in these same patients.</p>`;
    }
    function renderClusters() {
        const { P } = S.explore, { k, labels, stab, test, sil, posterior } = S.result;
        // Share of each cluster's patients with imputed (or, with recorded-values-only methods, missing) clustering values
        const anyInc = P.incomplete && P.incompleteKind && P.incomplete.some(Boolean);
        const impCol = anyInc ? (P.incompleteKind === 'imputed' ? 'Patients with imputed values' : 'Patients with missing values') : null;
        // Missing values by cluster: if missingness differs between clusters, the clusters may partly reflect why values are
        // missing (sicker patients, tests not ordered), which no imputation can correct
        const missRows = P.specs.map(sp => { const miss = P.rowIds.map(r => DE.isMissing(S.data.rows[r][sp.col]) || (sp.type === 'continuous' && !Number.isFinite(Number(S.data.rows[r][sp.col])))); return { col: sp.col, miss }; }).filter(m => m.miss.some(Boolean));
        const missHtml = missRows.length ? (() => {
            const pm = p => (p < 0.001 ? '< 0.001' : p.toFixed(3));
            const tests = missRows.map(m => CE.chiSquareTest(m.miss.map(v => (v ? 'missing' : 'recorded')), labels)), ps = tests.map(t => t.p), qs = CE.benjaminiHochberg(ps.map(p => (p === null ? 1 : p)));
            const rowsH = missRows.map((m, t) => `<tr><td>${esc(m.col)}</td><td>${Math.round(100 * m.miss.filter(Boolean).length / m.miss.length)}%</td>${Array.from({ length: k }, (_, c) => { const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0); return `<td>${Math.round(100 * idx.filter(i => m.miss[i]).length / Math.max(1, idx.length))}%</td>`; }).join('')}<td>${ps[t] === null ? '–' : `${pm(ps[t])}${missRows.length > 1 ? ` · q ${pm(qs[t])}` : ''}${tests[t].minExpected < 5 ? ' <span class="muted">(small counts)</span>' : ''}`}</td></tr>`).join('');
            const diff = ps.some((p, t) => p !== null && qs[t] <= 0.05);
            return `<details${diff ? ' open' : ''}><summary class="small">Missing values by cluster${diff ? ' <strong>(differ between clusters)</strong>' : ''}</summary><div class="table-wrap"><table><thead><tr><th>Clustering variable</th><th>Missing</th>${Array.from({ length: k }, (_, c) => `<th>Cluster ${c + 1}</th>`).join('')}<th>p · q</th></tr></thead><tbody>${rowsH}</tbody></table></div>
                <p class="muted small">${diff ? '<strong>Missingness differs between clusters</strong>: the clusters may partly reflect why values are missing (for example sicker patients, or tests ordered only for some), and imputation cannot correct that. ' : ''}Multiple imputation and the other options assume that values are missing at random given the recorded data; combining imputations reflects the uncertainty of the imputed values but does not remove bias when missingness depends on the missing values themselves. The sensitivity analysis shows how much the clusters change with other ways of handling missing values.</p></details>`;
        })() : '';
        const impShare = c => { const m = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0), x = m.filter(i => P.incomplete[i]).length; return `${Math.round(100 * x / Math.max(1, m.length))}%`; };
        const certainty = posterior ? Array.from({ length: k }, (_, c) => { const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0); return idx.reduce((a, i) => a + posterior[i][c], 0) / Math.max(1, idx.length); }) : null;
        const sizes = new Array(k).fill(0); labels.forEach(l => { sizes[l]++; });
        const stabWord = v => (v >= 0.75 ? 'stable' : v >= 0.6 ? 'indicates a pattern' : v >= 0.5 ? 'weak' : 'not reproducible');
        const f3 = v => (Number.isFinite(v) ? v.toFixed(3) : '–'), pf = p => (p <= 0.01 ? '≤ 0.01' : p.toFixed(2));
        const weakest = Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0));
        const evidence = [];
        if (test) evidence.push(test.pValue <= 0.05 ? `more separation than ${test.kind === 'gaussian' ? 'a single Gaussian population with the same covariance' : 'a single population with the same variable distributions and correlations'} gives by chance (p ${pf(test.pValue)})` : `no more separation than ${test.kind === 'gaussian' ? 'a single Gaussian population' : 'a single population with the same distributions and correlations'} gives by chance (p = ${pf(test.pValue)})`);
        evidence.push(weakest >= 0.75 ? 'every cluster reproduces in bootstrap resamples' : weakest >= 0.5 ? 'at least one cluster reproduces only weakly in bootstrap resamples' : 'at least one cluster does not reproduce in bootstrap resamples');
        // A non-significant copula test is weak evidence (the test is conservative: in simulations it detected moderately
        // overlapping sub-groups only about two times in three), so with stable clusters the verdict is "mixed", not "bad"
        const conservativeMiss = test && test.pValue > 0.05 && test.kind === 'copula' && weakest >= 0.75;
        // Without the test (it needs complete data) stability alone is weaker evidence: at best "mixed"
        const untested = !test && weakest >= 0.75;
        const verdict = test && test.pValue <= 0.05 && weakest >= 0.75 ? 'good' : conservativeMiss || untested ? 'mixed' : (test && test.pValue > 0.05) || weakest < 0.5 ? 'bad' : 'mixed';
        // Evidence summary: statistical evidence for structure, separately from clinical usefulness (which the app cannot judge)
        const sils = sil.mean, minC = stab.perCluster.reduce((m, c) => (c.size < m.size ? c : m)), weakC = stab.perCluster.reduce((m, c) => ((c.meanJaccard ?? 0) < (m.meanJaccard ?? 0) ? c : m));
        const evRow = (what, val, judge, cls = '') => `<tr><td>${what}</td><td>${val}</td><td class="${cls}">${judge}</td></tr>`;
        const lf = S.result.lcaFit;
        const evRows = [
            evRow('Stability (weakest cluster, mean Jaccard)', `${f3(weakC.meanJaccard)}${weakC.ciMean ? ` (${f3(weakC.ciMean[0])}–${f3(weakC.ciMean[1])})` : ''}`, weakest >= 0.75 ? (weakC.ciMean && weakC.ciMean[0] < 0.75 ? 'stable, borderline' : 'stable') : stabWord(weakest), weakest >= 0.75 ? 'ok' : 'warn'),
            evRow('Separation (average silhouette)', f3(sils), sils > 0.7 ? 'strong' : sils > 0.5 ? 'reasonable' : sils > 0.25 ? 'weak, possibly artificial' : 'no substantial structure', sils > 0.5 ? 'ok' : 'warn'),
            test ? evRow('Test against a single population', `p ${pf(test.pValue)} (${test.kind === 'gaussian' ? 'Gaussian' : 'copula'})`, test.pValue <= 0.05 ? 'more structure than a single population' : test.kind === 'copula' ? 'not shown (conservative test)' : 'not shown', test.pValue <= 0.05 ? 'ok' : 'warn') : evRow('Test against a single population', 'not run', 'needs complete data (fill or leave out missing values)', 'warn'),
            evRow('Smallest cluster', `${minC.size} patients (${(100 * minC.size / labels.length).toFixed(0)}%)`, minC.size < 10 || minC.size < 0.05 * labels.length ? 'small: its profile and stability are uncertain' : 'adequate', minC.size < 10 || minC.size < 0.05 * labels.length ? 'warn' : 'ok'),
            ...(lf ? [evRow('Latent class fit', `${lf.startsAtBest} of ${lf.starts} random starts reached the best solution${lf.converged ? '' : '; not converged'}`, lf.converged && lf.startsAtBest >= 2 ? 'reproducible optimum' : 'may be a chance local optimum: rerun with another seed', lf.converged && lf.startsAtBest >= 2 ? 'ok' : 'warn')] : []),
            ...(S.result.posterior ? [evRow('Membership certainty (relative entropy)', f3(S.result.entropy), S.result.entropy >= 0.8 ? 'clear-cut classes' : S.result.entropy >= 0.6 ? 'moderate' : 'many patients between classes', S.result.entropy >= 0.6 ? 'ok' : 'warn')] : []),
            `<tr id="ev-sens">${sensEvidenceCells()}</tr>`
        ].join('');
        const evidenceHtml = `<details class="evidence" open><summary><strong>Evidence summary</strong></summary><div class="table-wrap"><table><thead><tr><th>Statistical evidence for structure</th><th>Value</th><th>Reading</th></tr></thead><tbody>${evRows}</tbody></table></div>
            <p class="small"><strong>Not tested by the app (clinical usefulness):</strong> whether clinicians can recognise and name the clusters; whether they differ in outcomes specified before the analysis; whether they reappear in an independent cohort (step 7, section 5, and the validation protocol); and whether they would change understanding or management. Statistical structure is necessary but not sufficient: a stable, well-separated partition can still be clinically uninteresting, and the number of clusters suggested by the statistics need not be the clinically right number of phenotypes.</p></details>`;
        const somOn = !!(S.result.neural && S.result.neural.unitLabels);
        $('clusters-out').innerHTML = `
            <div class="verdict ${verdict}"><strong>${k} clusters (${esc(METHOD[P.method].short)}).</strong> Evidence: ${esc(evidence.join('; '))}.
                ${verdict === 'good' ? `These clusters are consistent with real sub-groups in these data (stable, and more separated than a single population); they still need replication in another cohort before clinical interpretation.${test ? ' Note: a single population with a curved or branching (non-linear) pattern can also give this result; check the map of the patients for gaps between the clusters' + (P.method === 'neural' ? ' (and the self-organising map for grey, empty units between them)' : '') + ', and the dimensions panel for a continuum.' : ''}` : verdict === 'bad' ? 'These clusters may be artefacts of the method: clustering algorithms always return clusters, even in data without sub-groups. Do not interpret them as patient sub-types.' : (conservativeMiss ? 'The clusters reproduce in resamples, but the (conservative) test did not find more separation than a single population: they may reflect modest structure or a continuum rather than distinct sub-groups. Treat them as exploratory.' : untested ? 'The clusters reproduce in resamples, but the test against a single population could not be run (it needs complete data: in step 4, fill or leave out missing values to run it). Stable clusters can also arise from a continuum, so treat them as exploratory until the test is run.' : 'The evidence is mixed: treat the clusters as exploratory.')}</div>
            ${evidenceHtml}
            ${{ confirmatory: '<p class="small note">Confirmatory purpose: the sub-groups and the outcomes compared must have been specified before the analysis; report the q-values and effect sizes, and the sensitivity analysis.</p>', groundwork: '<p class="small note">Groundwork for a prediction model: clusters are not predictors of outcome by themselves; a prediction model needs its own development and validation.</p>' }[$('purpose').value] || ''}
            <div class="grid2">
                <div><h3>Clusters</h3><div class="table-wrap"><table><thead><tr><th>Cluster</th><th>Patients</th><th>Stability (mean Jaccard, 95% interval)</th><th>Share of patients across resamples</th><th>Dissolved in</th>${impCol ? `<th>${impCol}</th>` : ''}${certainty ? '<th>Average membership probability</th>' : ''}</tr></thead><tbody>
                    ${stab.perCluster.map(c => `<tr><td><i class="sw s${c.cluster}"></i>Cluster ${c.cluster + 1}</td><td>${c.size} (${(100 * c.size / labels.length).toFixed(0)}%)</td><td>${f3(c.meanJaccard)}${c.ciMean ? ` <span class="muted">(${f3(c.ciMean[0])}–${f3(c.ciMean[1])})</span>` : ''} <span class="muted">${stabWord(c.meanJaccard)}${c.ciMean && c.ciMean[0] < 0.75 && c.ciMean[1] >= 0.75 ? ', borderline' : ''}</span></td><td>${c.shareRange ? `${Math.round(100 * c.shareRange[0])}–${Math.round(100 * c.shareRange[1])}%` : '–'} <span class="muted">(${(100 * c.size / labels.length).toFixed(0)}% in all data)</span></td><td>${(100 * c.dissolved).toFixed(0)}% of resamples</td>${impCol ? `<td>${impShare(c.cluster)}</td>` : ''}${certainty ? `<td>${f3(certainty[c.cluster])}</td>` : ''}</tr>`).join('')}</tbody></table></div>
                    ${certainty ? `<p class="muted small">Average membership probability: how certain the model is, on average, that the patients of a class belong to it (above 0.8: well separated). Entropy ${f3(S.result.entropy)}.</p>` : ''}
                    ${missHtml}
                    ${impCol ? `<p class="small">${P.incompleteKind === 'imputed' ? `${impCol}: patients with at least one clustering value that was not recorded and was ${P.missing === 'mice' ? 'multiply imputed' : 'filled in'}.` : `${impCol}: patients with at least one clustering value not recorded (left out of their dissimilarities).`} A cluster with a much larger share than the others may be shaped by ${P.incompleteKind === 'imputed' ? 'the imputed values' : 'the missing values'} rather than by recorded data: treat it with caution. The maps can mark these patients.</p>` : ''}
                    <p class="muted small">Stability: ${stab.resamples} bootstrap resamples${P.groups ? ` (whole patients by "${esc(P.patientCol)}")` : ''}; mean Jaccard 0.75 or more: stable; 0.6–0.75: indicates a pattern; below 0.5: not reproducible (Hennig 2007). Average agreement with the full-data clustering: adjusted Rand ${f3(stab.meanAdjustedRand)}. The 95% interval shows how precisely the ${stab.resamples} resamples estimate the mean; "borderline" when it includes 0.75. The 0.75 limit is a convention (Hennig 2007), not proof that a cluster is real.${(() => { const ps = (stab.patientStability || []).filter(v => v !== null); if (!ps.length) return ''; const hi = ps.filter(v => v >= 0.9).length, lo = ps.filter(v => v < 0.7).length; return ` <strong>Patients:</strong> ${Math.round(100 * hi / ps.length)}% stay in their cluster in at least 90% of the resamples that include them; ${Math.round(100 * lo / ps.length)}% in fewer than 70% (assignments that depend on the sample; the download lists each patient's stability).`; })()}${P.mice ? ` The resamples cycle through the ${P.mice.m} imputed datasets. Agreement of each imputed dataset's clustering with the consensus: adjusted Rand ${f3(S.explore.res.find(r => r.k === k).agreement)}.` : ''}</p>
                    ${test ? `<p class="muted small">Test of "no real sub-groups": how well ${k} groups can be separated in these data (R² = ${f3(test.observed)}: the share of the variation between groups when k-means divides the same variables into ${k}; the test asks whether the data have structure, whichever method formed the clusters) against ${test.B} reference datasets from ${test.kind === 'gaussian' ? 'a single Gaussian population with exactly the data\'s covariance (exact for Gaussian data)' : `a single population with each variable's own distribution and the observed correlations (a Gaussian copula; conservative: a small p is strong evidence, a large p does not prove there are no sub-groups${test.skewedNumbers ? '; used instead of the Gaussian reference because at least one measurement is skewed, which makes the Gaussian reference give false positives' : ''})`}: reference mean ${f3(test.nullMean)}, 95th centile ${f3(test.nullQuantile95)}.${Number.isFinite(test.fitError) ? ` The reference data reproduce the observed correlations to within ${test.fitError.toFixed(3)} (largest difference)${test.fitError > 0.05 ? '<strong>: some correlations could only be approximated, so the reference is approximate and the p-value should be read with caution</strong>' : ''}.` : ''}${test.pooled ? ` With multiple imputation the test was run on each of the ${test.pooled.m} imputed datasets and the median p-value is reported (range ${pf(test.pooled.min)} to ${pf(test.pooled.max)}); the figures above are from the median dataset.` : ''}</p>` : '<p class="muted small">The test of "no real sub-groups" needs complete data (choose to fill or leave out missing values).</p>'}
                </div>
                ${somOn ? '' : '<div><h3>Map of the patients</h3><div id="map-panel"></div></div>'}
            </div>
            ${somOn ? '<div class="grid2 maps-pair"><div><h3>Map of the patients</h3><div id="map-panel"></div></div><div><h3>The self-organising map</h3><div id="som-view"></div></div></div><div id="som-table"></div>' : ''}
            <h3>Dimensions of the data: what drives the differences between patients</h3>
            <div id="dims-panel"></div>
            <h3>How well each patient fits (silhouette)</h3>
            <div class="sil">${CH.silhouettePlot({ values: sil.values, labels: sil.labels || labels, aria: 'Silhouette width of each patient by cluster' })}</div>
            <p class="muted small">Average silhouette ${f3(sil.mean)}${P.mice ? ' (averaged over the imputed datasets)' : ''}${P.sampleIdx ? ` (random sample of ${P.sampleIdx.length.toLocaleString('en-GB')} records)` : ''}. Silhouette width: 1 = clearly in its own cluster, 0 = between two clusters, below 0 = closer to another cluster than to its own.</p>
            <h3>Sensitivity to modelling choices</h3>
            <p class="small">Do the same patients group together if a variable is left out, another method is used or missing values are handled differently? <button id="sens-btn" class="btn small" type="button">Run the sensitivity analysis</button></p>
            <p id="sens-status" class="status small" role="status"></p><div id="sens-out"></div>
            <h3>Clinical profile of the clusters</h3>
            ${profileTable()}
            <div class="row"><button id="to-use" class="btn primary" type="button">Next: use the clusters</button><button id="export-btn" class="btn" type="button">Download patients with their cluster (.CSV)</button><button id="record-txt" class="btn" type="button">Download the analysis report (.txt)</button><button id="record-json" class="btn" type="button">Download the analysis record (.json)</button><button id="record-r" class="btn" type="button">Download an R script to repeat it (.R)</button></div>
            <p class="muted small">The report and record list the purpose, data, exclusions with their reasons, variable roles, warnings, every setting (including the random seed), all candidate solutions, the chosen one with its uncertainty, the sensitivity analysis and the classification: everything needed to report and repeat the analysis. They contain no patient-level data.</p>`;
        $('export-btn').addEventListener('click', exportCsv);
        $('sens-btn').addEventListener('click', () => sensitivity());
        $('record-txt').addEventListener('click', () => downloadRecord('txt'));
        $('record-json').addEventListener('click', () => downloadRecord('json'));
        $('record-r').addEventListener('click', downloadR);
        renderSensitivity();
        $('to-use').addEventListener('click', () => goTo(7));
        $('tree-out').innerHTML = '';
        window.UMEMaps.highlight(null);
        window.UMEMaps.render({ S, P });
        if (S.result.neural && S.result.neural.unitLabels) window.UMESom.render({ S, P });
    }
    function exportCsv() {
        if (!confirm('The file will contain your patients\' data (every column of the loaded file) with the cluster of each patient. Save it only where your data-governance rules allow. Continue?')) return;
        const { P } = S.explore, labels = S.result.labels;
        const extra = window.UMEUse.exportColumns();
        const ps = S.result.stab && S.result.stab.patientStability;
        const sens = S.result.sens ? S.result.sens.share : null;
        const head = [...S.data.headers, 'Row in file', 'Cluster', 'Assignment stability (share of resamples)', ...(P.incompleteKind ? [P.incompleteKind === 'imputed' ? 'Imputed clustering values' : 'Missing clustering values'] : []), ...(sens ? ['Robustness to modelling choices (share of alternatives)'] : []), ...(extra ? ['Class', 'Class by bedside rules'] : [])];
        const cell = v => { let s = String(v ?? ''); if (/^[=+\-@\t\r]/.test(s) && !Number.isFinite(Number(s))) s = `'${s}`; return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
        const lines = [head.map(cell).join(','), ...P.rows.map((r, i) => [...S.data.headers.map(h => S.data.rows[P.rowIds[i]][h]), P.rowIds[i] + 1, labels[i] + 1, ps && ps[i] !== null ? ps[i].toFixed(2) : '', ...(P.incompleteKind ? [P.incomplete[i] ? 'yes' : ''] : []), ...(sens ? [sens.get(P.rowIds[i]) !== null && sens.get(P.rowIds[i]) !== undefined ? sens.get(P.rowIds[i]).toFixed(2) : ''] : []), ...(extra ? [extra.names[i], extra.rules[i]] : [])].map(cell).join(','))];
        const url = URL.createObjectURL(new Blob([lines.join('\r\n')], { type: 'text/csv' }));
        const a = document.createElement('a'); a.href = url; a.download = `clusters_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}.csv`;
        document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    // ------------------------------------------------------------------------------------- wiring
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => goTo(t.dataset.step)));
        $('file').addEventListener('change', e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => loadText(String(r.result), f.name); r.readAsText(f); });
        $('demo-phenotypes').addEventListener('click', () => loadDemo('demo_inph_phenotypes.csv'));
        $('demo-none').addEventListener('click', () => loadDemo('demo_no_subtypes.csv'));
        $('demo-errors').addEventListener('click', () => loadDemo('demo_inph_with_errors.csv'));
        $('demo-referrals').addEventListener('click', () => loadDemo('demo_referrals.csv'));
        $('to-step3').addEventListener('click', () => { enableTabs(4); goTo(4); });
        $('to-check').addEventListener('click', () => { enableTabs(4); goTo(3); renderChecks(); });
        $('check-btn').addEventListener('click', () => { renderChecks(); window.UMEChecks.start(); });
        $('check-next').addEventListener('click', () => { enableTabs(4); goTo(4); });
        $('method').addEventListener('change', () => { S.methodTouched = true; updateVarSummary(); });
        $('explore-btn').addEventListener('click', () => { goTo(5); explore(); });
        document.addEventListener('click', e => { if (e.target && e.target.id === 'to-tree') goTo(8); });
        $('theme').addEventListener('click', () => { const d = document.documentElement, dark = d.getAttribute('data-theme') !== 'dark' && !(d.getAttribute('data-theme') !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches); d.setAttribute('data-theme', dark ? 'dark' : 'light');
            // The tree diagram takes its colours from the theme
            if (S.result && !document.querySelector('section.step[data-step="8"]').hidden) window.UMEUse.renderTree({ S, P: S.explore.P, METHOD, profileTable });
        });
        enableTabs(1); goTo(1);
    });
    window.UMEApp = { state: S, prepare, loadText };
})();
