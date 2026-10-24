// charts.js: small SVG charts drawn from data (no library): line charts, scatter maps and silhouette bars.
// Colours come from CSS classes (theme-aware); every label is escaped. Loaded as a plain script (window.UMECharts).
(function (root) {
    'use strict';
    const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)).replace(/^-(0(\.0+)?)$/, '$1');
    function niceTicks(lo, hi, n = 5) {
        if (!(hi > lo)) { hi = lo + 1; }
        const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))), err = (hi - lo) / n / step;
        const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
        const out = []; for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9 * s; v += s) out.push(Number(v.toPrecision(12)));
        return out;
    }
    function frame(W, H, m, xs, ys, xLabel, yLabel, aria) {
        const x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
        const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
        const xt = niceTicks(xMin, xMax), yt = niceTicks(yMin, yMax);
        const xl = Math.min(xMin, xt[0]), xh = Math.max(xMax, xt[xt.length - 1]), yl = Math.min(yMin, yt[0]), yh = Math.max(yMax, yt[yt.length - 1]);
        const sx = v => x0 + (xh > xl ? (v - xl) / (xh - xl) : 0.5) * (x1 - x0), sy = v => y0 - (yh > yl ? (v - yl) / (yh - yl) : 0.5) * (y0 - y1);
        let g = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria)}">`;
        yt.forEach(v => { g += `<line class="grid" x1="${x0}" x2="${x1}" y1="${sy(v)}" y2="${sy(v)}"/><text class="tick" x="${x0 - 6}" y="${sy(v) + 4}" text-anchor="end">${esc(fmt(v))}</text>`; });
        xt.forEach(v => { g += `<text class="tick" x="${sx(v)}" y="${y0 + 16}" text-anchor="middle">${esc(fmt(v))}</text>`; });
        g += `<line class="axis" x1="${x0}" x2="${x1}" y1="${y0}" y2="${y0}"/><line class="axis" x1="${x0}" x2="${x0}" y1="${y0}" y2="${y1}"/>`;
        g += `<text class="label" x="${(x0 + x1) / 2}" y="${H - 4}" text-anchor="middle">${esc(xLabel)}</text>`;
        g += `<text class="label" transform="translate(13 ${(y0 + y1) / 2}) rotate(-90)" text-anchor="middle">${esc(yLabel)}</text>`;
        return { g, sx, sy };
    }
    /** Line chart: series [{ name, cls, points: [{x, y}] }], one marker per point */
    function lineChart({ series, xLabel, yLabel, aria, W = 420, H = 260, integerX = true }) {
        const pts = series.flatMap(s => s.points).filter(p => Number.isFinite(p.y));
        if (!pts.length) return '';
        const f = frame(W, H, { l: 52, r: 14, t: 12, b: 40 }, pts.map(p => p.x), pts.map(p => p.y), xLabel, yLabel, aria);
        let g = f.g;
        series.forEach((s, i) => {
            const ok = s.points.filter(p => Number.isFinite(p.y));
            g += `<polyline class="line s${s.cls ?? i}" points="${ok.map(p => `${f.sx(p.x)},${f.sy(p.y)}`).join(' ')}"/>`;
            ok.forEach(p => { g += `<circle class="dot s${s.cls ?? i}" cx="${f.sx(p.x)}" cy="${f.sy(p.y)}" r="3.5"><title>${esc(s.name)}: ${esc(integerX ? p.x : fmt(p.x))} → ${esc(fmt(p.y))}</title></circle>`; });
        });
        return g + '</svg>';
    }
    /** Scatter map: points [{x, y, group, tip}] coloured by group, or by colour (a hex value, for a colour scale);
     *  group null and no colour: drawn grey (value not recorded). noAxes: for t-SNE / UMAP, whose axes have no units. */
    function scatter({ points, xLabel, yLabel, aria, W = 440, H = 340, noAxes = false }) {
        if (!points.length) return '';
        const f = frame(W, H, { l: 46, r: 14, t: 12, b: 40 }, points.map(p => p.x), points.map(p => p.y), xLabel, yLabel, aria);
        let g = noAxes ? f.g.replace(/<text class="tick"[^>]*>[^<]*<\/text>/g, '') : f.g;
        const hex = /^#[0-9a-f]{6}$/i;
        points.forEach(p => {
            const cls = p.color && hex.test(p.color) ? 'pt' : p.group === null || p.group === undefined ? 'pt na' : `pt s${p.group % 8}`;
            g += `<circle class="${cls}"${p.color && hex.test(p.color) ? ` fill="${p.color}"` : ''} cx="${f.sx(p.x).toFixed(1)}" cy="${f.sy(p.y).toFixed(1)}" r="3"><title>${esc(p.tip)}</title></circle>`;
        });
        return g + '</svg>';
    }
    /** Colour-scale legend (low to high) for numbers */
    function scaleLegend(colours, lo, hi, label) {
        const w = 18, bars = colours.map((c, i) => `<rect x="${i * w}" y="0" width="${w}" height="10" fill="${/^#[0-9a-f]{6}$/i.test(c) ? c : '#888888'}"/>`).join('');
        return `<div class="legend"><span>${esc(label)}: ${esc(lo)}</span><svg class="scale" viewBox="0 0 ${colours.length * w} 10" width="${colours.length * w}" height="10" aria-hidden="true">${bars}</svg><span>${esc(hi)}</span><span><i class="sw na"></i>not recorded</span></div>`;
    }
    /** Silhouette plot: one horizontal bar per patient, sorted within each cluster */
    function silhouettePlot({ values, labels, aria, W = 440 }) {
        const k = Math.max(...labels) + 1, order = [];
        for (let c = 0; c < k; c++) values.map((v, i) => i).filter(i => labels[i] === c).sort((a, b) => values[b] - values[a]).forEach(i => order.push(i));
        const n = order.length, gapPx = 6, barH = Math.max(0.6, Math.min(3, 300 / n)), H = n * barH + k * gapPx + 46;
        const x0 = 60, x1 = W - 14, lo = Math.min(-0.2, ...values), sx = v => x0 + (v - lo) / (1 - lo) * (x1 - x0);
        let g = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria)}">`, y = 8, prev = -1;
        order.forEach(i => { if (labels[i] !== prev) { if (prev >= 0) y += gapPx; g += `<text class="tick" x="${x0 - 8}" y="${y + 10}" text-anchor="end">Cluster ${labels[i] + 1}</text>`; prev = labels[i]; } const v = values[i]; g += `<rect class="bar s${labels[i] % 8}" x="${Math.min(sx(0), sx(v))}" y="${y}" width="${Math.abs(sx(v) - sx(0))}" height="${barH}"/>`; y += barH; });
        [lo, 0, 0.25, 0.5, 0.75, 1].filter((v, i, a) => a.indexOf(v) === i).forEach(v => { g += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="4" y2="${y + 2}"/><text class="tick" x="${sx(v)}" y="${y + 16}" text-anchor="middle">${esc(fmt(v))}</text>`; });
        g += `<text class="label" x="${(x0 + x1) / 2}" y="${y + 34}" text-anchor="middle">Silhouette width (1 = clearly in its cluster, 0 = between clusters, below 0 = closer to another cluster)</text>`;
        return g.replace(`viewBox="0 0 ${W} ${H}"`, `viewBox="0 0 ${W} ${y + 42}"`) + '</svg>';
    }
    function legend(names) { return `<div class="legend">${names.map((n, i) => `<span><i class="sw s${i % 8}"></i>${esc(n)}</span>`).join('')}</div>`; }
    root.UMECharts = { lineChart, scatter, silhouettePlot, legend, scaleLegend, esc };
})(typeof window !== 'undefined' ? window : globalThis);
