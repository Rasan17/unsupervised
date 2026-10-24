// som-explorer.js: the self-organising map of step 6 as an interactive view. Drag a rectangle over the map (or click
// a unit; Shift / Cmd / Ctrl adds to the selection, or use the cluster buttons) to select units; the table below then
// lists only the records on those units, with a summary of how they differ from all records, sorting, paging and a
// CSV download. Plain script (window.UMESom); pointer events (mouse, pen, touch); every value is escaped.
(function (root) {
    'use strict';
    const CH = root.UMECharts, esc = CH.esc, DE = root.DataEngine;
    const $ = id => document.getElementById(id);
    const PAGE = 50;
    let V = null;   // { S, P, nn, labels, k, sel: Set of units, sortCol, sortDir, page }

    function cellSize(m) { return Math.max(12, Math.min(30, Math.floor(440 / Math.max(m.rows, m.cols)))); }
    function chart() {
        const m = V.nn.map, cell = cellSize(m), W = m.cols * cell, H = m.rows * cell, maxHits = Math.max(...m.hits), any = V.sel.size > 0, part = partial();
        let g = `<svg class="chart som" id="som-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Self-organising map of the records coloured by cluster">`;
        m.pos.forEach(([r, c], u) => {
            const hits = m.hits[u], op = hits ? (0.25 + 0.75 * Math.sqrt(hits / maxHits)).toFixed(2) : 1;
            const cls = `${hits ? `bar s${V.nn.unitLabels[u] % 8}` : 'pt na'}${V.sel.has(u) ? (part.has(u) ? ' sel part' : ' sel') : any ? ' dim' : ''}${V.picked && V.picked.u === u ? ' picked' : ''}`;
            g += `<rect class="${cls}" data-unit="${u}" x="${c * cell + 1}" y="${r * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="2" fill-opacity="${op}"><title>Unit ${u + 1}: ${hits} record(s)${hits ? `, cluster ${V.nn.unitLabels[u] + 1}` : ''}</title></rect>`;
        });
        return g + '<rect id="som-band" class="band" x="0" y="0" width="0" height="0" display="none"/></svg>';
    }
    const recordLabel = i => { const id = V.S.data.headers.find(h => V.S.vars[h].role === 'ignore' && /(^|[\s_-])(id|mrn|nhs|number|record)$/i.test(h)); const v = id ? String(V.P.rows[i][id] ?? '').trim() : ''; return v || `row ${i + 1}`; };
    // Selected records: those chosen on the map of the patients (V.rec), otherwise all records on the selected units
    function selectedRecords() { return V.rec ? [...V.rec].sort((a, b) => a - b) : V.P.rows.map((_, i) => i).filter(i => V.sel.has(V.nn.map.bmu[i])); }
    // Units holding some but not all of the records chosen on the map of the patients
    function partial() { if (!V.rec) return new Set(); const n = new Map(); V.rec.forEach(i => { const u = V.nn.map.bmu[i]; n.set(u, (n.get(u) || 0) + 1); }); return new Set([...n].filter(([u, c]) => c < V.nn.map.hits[u]).map(([u]) => u)); }
    function draw() {
        const el = $('som-view'); if (!el) return;
        const k = V.k;
        el.innerHTML = `
            <div class="grid2"><div>${chart()}${CH.legend(Array.from({ length: k }, (_, c) => `Cluster ${c + 1}`))}</div>
            <div class="small"><p class="muted">Each square is a unit of the map: records with similar values fall on the same or neighbouring units. Colour: the cluster of the unit; the stronger the colour, the more records it holds (hover for the number). Units without records are grey.</p>
                <p><strong>Select a region:</strong> drag a rectangle over this map, or click a unit. Hold Shift while dragging to add, Cmd (Ctrl on Windows) to remove; Shift- or Cmd-click adds or removes one unit. The records on the selected units are listed below. You can also select patients on the map of the patients above: click a point, drag a rectangle, or hold Option (Alt) and drag for a freehand outline.</p>
                <p><strong>Deselect:</strong> "Clear selection", the Escape key, or a click on an empty part of either map.</p>
                <div class="row" role="group" aria-label="Select the units of a cluster">${Array.from({ length: k }, (_, c) => `<button type="button" class="btn small" data-selc="${c}"><i class="sw s${c % 8}"></i>Cluster ${c + 1}</button>`).join('')}<button type="button" class="btn small" id="som-clear"${V.sel.size ? '' : ' disabled'}>Clear selection</button></div></div></div>
            ${V.rec ? `<p class="small note">Selected on the map of the patients: ${V.rec.size.toLocaleString('en-GB')} record(s), on the ${V.sel.size} unit(s) outlined here (dashed: only some of the unit's records are selected). The table lists exactly these records.</p>` : ''}
            ${V.picked && V.sel.has(V.picked.u) ? `<p class="small note">Selected from the map of the patients: record ${esc(recordLabel(V.picked.i))} is on unit ${V.picked.u + 1} (outlined), with ${V.nn.map.hits[V.picked.u]} record(s).</p>` : ''}
            <div id="som-table"></div>`;
        wire();
        table();
        // The same records highlighted on the map of the patients (FAMD, t-SNE, UMAP or autoencoder) above
        if (root.UMEMaps && root.UMEMaps.highlight) root.UMEMaps.highlight(V.sel.size ? new Set(selectedRecords()) : null);
    }
    function wire() {
        const svg = $('som-svg'), band = $('som-band');
        const toUnit = e => { const t = e.target; return t && t.dataset && t.dataset.unit !== undefined ? Number(t.dataset.unit) : null; };
        const pt = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse()); return [q.x, q.y]; };
        let start = null, add = false, remove = false, far = 0;
        svg.addEventListener('pointerdown', e => { start = pt(e); far = 0; remove = e.metaKey || e.ctrlKey; add = e.shiftKey && !remove; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
        svg.addEventListener('pointermove', e => {
            if (!start) return;
            const [x, y] = pt(e), x0 = Math.min(x, start[0]), y0 = Math.min(y, start[1]);
            far = Math.max(far, Math.hypot(x - start[0], y - start[1]));
            band.setAttribute('x', x0); band.setAttribute('y', y0); band.setAttribute('width', Math.abs(x - start[0])); band.setAttribute('height', Math.abs(y - start[1])); band.removeAttribute('display');
        });
        svg.addEventListener('pointerup', e => {
            if (!start) return;
            const [x, y] = pt(e), m = V.nn.map, cell = cellSize(m), moved = Math.max(far, Math.hypot(x - start[0], y - start[1])) > 3;
            if (moved) {
                const x0 = Math.min(x, start[0]), x1 = Math.max(x, start[0]), y0 = Math.min(y, start[1]), y1 = Math.max(y, start[1]);
                const hit = m.pos.map(([r, c], u) => [u, c * cell + cell / 2, r * cell + cell / 2]).filter(([, cx, cy]) => cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1).map(([u]) => u);
                if (remove) {
                    // Remove: these units, or (after a selection on the map of the patients) their records
                    if (V.rec) { hit.forEach(u => { V.rec.forEach(i => { if (m.bmu[i] === u) V.rec.delete(i); }); }); if (!V.rec.size) V.rec = null; V.sel = new Set(V.rec ? [...V.rec].map(i => m.bmu[i]) : []); } else hit.forEach(u => V.sel.delete(u));
                } else {
                    if (!add) { V.sel.clear(); V.rec = null; } else if (V.rec) { V.rec = null; }
                    hit.forEach(u => V.sel.add(u));
                }
            } else {
                const c0 = Math.floor(start[0] / cell), r0 = Math.floor(start[1] / cell), u = toUnit(e) ?? (r0 >= 0 && r0 < m.rows && c0 >= 0 && c0 < m.cols ? r0 * m.cols + c0 : null);
                // A click on an empty unit or outside the grid clears the selection
                if (u === null || !m.hits[u]) { V.sel.clear(); V.rec = null; }
                else if (add || remove) { V.rec = null; if (V.sel.has(u)) V.sel.delete(u); else V.sel.add(u); }
                else if (V.sel.size === 1 && V.sel.has(u)) { V.sel.clear(); V.rec = null; }
                else { V.sel.clear(); V.rec = null; V.sel.add(u); }
            }
            start = null; V.page = 0; V.picked = null; draw();
        });
        svg.addEventListener('pointercancel', () => { start = null; band.setAttribute('display', 'none'); });
        document.querySelectorAll('#som-view [data-selc]').forEach(b => b.addEventListener('click', e => {
            const c = Number(b.dataset.selc), units = V.nn.unitLabels.map((l, u) => (l === c && V.nn.map.hits[u] ? u : -1)).filter(u => u >= 0);
            if (!(e.shiftKey || e.metaKey || e.ctrlKey)) V.sel.clear();
            units.forEach(u => V.sel.add(u)); V.page = 0; V.picked = null; V.rec = null; draw();
        }));
        $('som-clear').addEventListener('click', () => { V.sel.clear(); V.page = 0; V.picked = null; V.rec = null; draw(); });
    }
    // Summary of the selection against all records, then the records themselves
    function summary(idx) {
        const { S, P, labels, k } = V, all = P.rows.map((_, i) => i), cols = [...P.cols, ...P.describe];
        const mix = Array.from({ length: k }, (_, c) => idx.filter(i => labels[i] === c).length);
        const miss = v => DE.isMissing(v);
        const rowsOut = cols.map(col => {
            const t = S.vars[col].type, vals = i => P.rows[i][col];
            if (t === 'continuous') {
                const mean = ix => { const xs = ix.map(vals).filter(v => !miss(v) && Number.isFinite(Number(v))).map(Number); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN; };
                const a = mean(idx), b = mean(all), f = v => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)) : '–');
                return `<tr><td>${esc(col)}</td><td>mean ${f(a)}</td><td>mean ${f(b)}</td></tr>`;
            }
            // Category (or grade): the level most over-represented in the selection
            const share = (ix, l) => { const ok = ix.filter(i => !miss(vals(i))); return ok.length ? ok.filter(i => String(vals(i)).trim() === l).length / ok.length : 0; };
            const levels = [...new Set(all.map(vals).filter(v => !miss(v)).map(v => String(v).trim()))];
            const best = levels.map(l => ({ l, a: share(idx, l), b: share(all, l) })).sort((x, y) => (y.a - y.b) - (x.a - x.b))[0];
            return best ? `<tr><td>${esc(col)}</td><td>"${esc(best.l)}" ${Math.round(100 * best.a)}%</td><td>${Math.round(100 * best.b)}%</td></tr>` : '';
        }).join('');
        return `<p><strong>${idx.length.toLocaleString('en-GB')} record(s) selected</strong> ${V.rec ? 'on the map of the patients, on' : 'on'} ${V.sel.size} unit(s)${idx.length ? ': ' + mix.map((v, c) => (v ? `cluster ${c + 1} ${v / idx.length < 0.005 ? '<1' : Math.round(100 * v / idx.length)}%` : '')).filter(Boolean).join(', ') : ''}.</p>
            <details${idx.length ? ' open' : ''}><summary class="small">How the selection differs from all ${all.length.toLocaleString('en-GB')} records</summary><div class="table-wrap"><table><thead><tr><th>Variable</th><th>Selected</th><th>All records</th></tr></thead><tbody>${rowsOut}</tbody></table></div><p class="muted small">Numbers: mean. Categories and grades: the category most over-represented in the selection, and its share among all records.</p></details>`;
    }
    function table() {
        const el = $('som-table'); if (!el) return;
        if (!V.sel.size) { el.innerHTML = '<p class="muted">No units selected: drag over the map or click a unit to see its records.</p>'; return; }
        const { S, P, labels } = V, idx = selectedRecords(), heads = [...S.data.headers, 'Cluster', 'Map unit'];
        const value = (i, h) => (h === 'Cluster' ? labels[i] + 1 : h === 'Map unit' ? V.nn.map.bmu[i] + 1 : P.rows[i][h] ?? '');
        if (V.sortCol !== null) {
            const h = heads[V.sortCol], num = idx.every(i => { const v = value(i, h); return v === '' || Number.isFinite(Number(v)); });
            idx.sort((a, b) => { const x = value(a, h), y = value(b, h); const c = num ? (x === '' ? Infinity : Number(x)) - (y === '' ? Infinity : Number(y)) : String(x).localeCompare(String(y)); return V.sortDir * (c || a - b); });
        }
        const pages = Math.max(1, Math.ceil(idx.length / PAGE)); V.page = Math.min(V.page, pages - 1);
        const shown = idx.slice(V.page * PAGE, V.page * PAGE + PAGE);
        el.innerHTML = `${summary(idx)}
            <div class="row"><button type="button" class="btn small" id="som-prev"${V.page ? '' : ' disabled'}>Previous</button><span class="small">Records ${(V.page * PAGE + 1).toLocaleString('en-GB')}–${Math.min(idx.length, (V.page + 1) * PAGE).toLocaleString('en-GB')} of ${idx.length.toLocaleString('en-GB')}</span><button type="button" class="btn small" id="som-next"${V.page < pages - 1 ? '' : ' disabled'}>Next</button>
                <button type="button" class="btn small" id="som-csv">Download the selected records (.CSV)</button></div>
            <div class="table-wrap"><table class="records"><thead><tr>${heads.map((h, c) => `<th><button type="button" class="sort" data-col="${c}" aria-label="Sort by ${esc(h)}">${esc(h)}${V.sortCol === c ? (V.sortDir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr></thead>
            <tbody>${shown.map(i => `<tr>${heads.map(h => `<td>${esc(value(i, h))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
        el.querySelectorAll('button.sort').forEach(b => b.addEventListener('click', () => { const c = Number(b.dataset.col); if (V.sortCol === c) V.sortDir = -V.sortDir; else { V.sortCol = c; V.sortDir = 1; } table(); }));
        $('som-prev').addEventListener('click', () => { V.page--; table(); });
        $('som-next').addEventListener('click', () => { V.page++; table(); });
        $('som-csv').addEventListener('click', () => {
            if (!confirm('The file will contain the selected records (every column of the loaded file) with their cluster and map unit. Save it only where your data-governance rules allow. Continue?')) return;
            const cell = v => { let s2 = String(v ?? ''); if (/^[=+\-@\t\r]/.test(s2) && !Number.isFinite(Number(s2))) s2 = `'${s2}`; return /[",\r\n]/.test(s2) ? `"${s2.replace(/"/g, '""')}"` : s2; };
            const lines = [heads.map(cell).join(','), ...idx.map(i => heads.map(h => cell(value(i, h))).join(','))];
            const url = URL.createObjectURL(new Blob([lines.join('\r\n')], { type: 'text/csv' })), a = document.createElement('a');
            a.href = url; a.download = `selected_records_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}.csv`; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
        });
    }
    // From the map of the patients: select the unit of a clicked record (toggle; add with Shift / Cmd / Ctrl)
    // mode: 'replace', 'add' or 'remove' (records chosen on the map of the patients)
    function selectRecords(rows, mode = 'replace') {
        if (!V || !V.nn) return;
        if (mode === true) mode = 'add';
        const rec = mode === 'replace' ? new Set() : new Set(V.rec ? V.rec : selectedRecords());
        rows.forEach(i => { if (mode === 'remove') rec.delete(i); else rec.add(i); });
        V.rec = rec.size ? rec : null; V.sel = new Set([...rec].map(i => V.nn.map.bmu[i])); V.picked = null; V.page = 0;
        draw();
    }
    function selectRecord(i, add) {
        if (!V || !V.nn) return;
        const u = V.nn.map.bmu[i];
        if (add) { if (V.sel.has(u)) V.sel.delete(u); else V.sel.add(u); } else if (V.sel.size === 1 && V.sel.has(u)) V.sel.clear(); else { V.sel.clear(); V.sel.add(u); }
        V.rec = null; V.picked = V.sel.has(u) ? { i, u } : null; V.page = 0;
        draw();
    }
    function clear() { if (!V) return; V.sel.clear(); V.rec = null; V.picked = null; V.page = 0; draw(); }
    function hasSelection() { return !!(V && V.sel && V.sel.size); }
    // Escape clears the selection (not while typing in a text or number box; a drop-down keeping the focus after a
    // choice does not stop it, as clicking a map does not move the focus)
    const typing = t => !!t && (t.tagName === 'TEXTAREA' || t.isContentEditable || (t.tagName === 'INPUT' && !/^(checkbox|radio|button|submit|range|color|file)$/i.test(t.type || '')));
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && hasSelection() && !typing(e.target)) clear(); });
    function render(ctx) {
        if (!V || V.key !== ctx.S.result) V = { key: ctx.S.result, sel: new Set(), sortCol: null, sortDir: 1, page: 0 };
        Object.assign(V, { S: ctx.S, P: ctx.P, nn: ctx.S.result.neural, labels: ctx.S.result.labels, k: ctx.S.result.k });
        draw();
    }
    root.UMESom = { render, selectRecord, selectRecords, clear, hasSelection, _state: () => V };
})(typeof window !== 'undefined' ? window : globalThis);
