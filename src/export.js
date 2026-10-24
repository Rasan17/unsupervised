// export.js: every chart and diagram of the app can be downloaded as an image, SVG (for papers, scales without
// loss) or PNG at double resolution (for slides). A small download bar is added above each chart automatically,
// including charts drawn later. The exported image uses the light colours on a white background and includes the
// chart's legend. Colours are copied from the page's style sheet into attributes, so the file stands alone.
// Plain script (window.UMEExport); works under the page's strict security policy (no inline styles, data: images).
(function (root) {
    'use strict';
    const NS = 'http://www.w3.org/2000/svg';
    const PROPS = ['fill', 'stroke', 'stroke-width', 'stroke-opacity', 'fill-opacity', 'opacity', 'font-size', 'font-family', 'font-weight', 'stroke-dasharray', 'stroke-linecap'];
    const slug = s => String(s || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'chart';
    const dataName = () => { try { const n = root.UMEApp && root.UMEApp.state.data && root.UMEApp.state.data.name; return n ? '_' + slug(String(n).replace(/\.[^.]+$/, '')) : ''; } catch (e) { return ''; } };
    // Evaluate styles with the light theme, then restore the viewer's theme
    function withLight(fn) {
        const d = document.documentElement, prev = d.getAttribute('data-theme');
        d.setAttribute('data-theme', 'light');
        try { return fn(); } finally { if (prev === null) d.removeAttribute('data-theme'); else d.setAttribute('data-theme', prev); }
    }
    function inlineStyles(src, dst) {
        const a = [src, ...src.querySelectorAll('*')], b = [dst, ...dst.querySelectorAll('*')];
        a.forEach((el, k) => {
            const cs = getComputedStyle(el), t = b[k];
            if (cs.display === 'none') { t.setAttribute('display', 'none'); return; }
            PROPS.forEach(p => { const v = cs.getPropertyValue(p); if (v && v !== 'normal' && !(p === 'stroke-dasharray' && v === 'none')) t.setAttribute(p, v); });
            t.removeAttribute('class');
        });
    }
    // The legend shown below a chart (an HTML element) redrawn in SVG under the chart
    function legendFor(svg) {
        let el = svg.nextElementSibling;
        while (el && !(el.classList && el.classList.contains('legend'))) { if (el.tagName && el.tagName.toLowerCase() === 'svg') return null; el = el.nextElementSibling; }
        if (!el) return null;
        const items = [];
        el.childNodes.forEach(node => {
            if (node.nodeType !== 1) return;
            if (node.tagName.toLowerCase() === 'svg') { items.push({ scale: [...node.querySelectorAll('rect')].map(r => r.getAttribute('fill')) }); return; }
            const sw = node.querySelector('i.sw'), text = node.textContent.trim();
            items.push({ colour: sw ? getComputedStyle(sw).backgroundColor : null, text });
        });
        return items;
    }
    function buildSvg(svg) {
        return withLight(() => {
            const vb = (svg.getAttribute('viewBox') || `0 0 ${svg.clientWidth} ${svg.clientHeight}`).split(/[\s,]+/).map(Number);
            const [x0, y0, w, h] = vb, clone = svg.cloneNode(true);
            inlineStyles(svg, clone);
            const legend = legendFor(svg) || [];
            const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim() || '#14212a';
            // Lay the legend out in rows under the chart
            const rows = [], fs = 12, rowH = 18, pad = 8;
            let cx = 0, row = [];
            legend.forEach(it => {
                const width = it.scale ? it.scale.length * 14 + 6 : (it.colour ? 16 : 0) + it.text.length * 6.6 + 14;
                if (cx + width > w && row.length) { rows.push(row); row = []; cx = 0; }
                row.push({ ...it, x: cx }); cx += width;
            });
            if (row.length) rows.push(row);
            const legendH = rows.length ? rows.length * rowH + pad : 0, H = h + legendH;
            const out = document.createElementNS(NS, 'svg');
            out.setAttribute('xmlns', NS); out.setAttribute('viewBox', `${x0} ${y0} ${w} ${H}`); out.setAttribute('width', String(w)); out.setAttribute('height', String(H));
            const bg = document.createElementNS(NS, 'rect'); bg.setAttribute('x', String(x0)); bg.setAttribute('y', String(y0)); bg.setAttribute('width', String(w)); bg.setAttribute('height', String(H)); bg.setAttribute('fill', '#ffffff'); out.appendChild(bg);
            const g = document.createElementNS(NS, 'g'); while (clone.firstChild) g.appendChild(clone.firstChild); out.appendChild(g);
            rows.forEach((r, ri) => r.forEach(it => {
                const y = y0 + h + pad + ri * rowH;
                if (it.scale) { it.scale.forEach((c, q) => { const s = document.createElementNS(NS, 'rect'); s.setAttribute('x', String(x0 + it.x + q * 14)); s.setAttribute('y', String(y)); s.setAttribute('width', '14'); s.setAttribute('height', '10'); s.setAttribute('fill', c || '#888888'); out.appendChild(s); }); return; }
                let tx = x0 + it.x;
                if (it.colour) { const s = document.createElementNS(NS, 'rect'); s.setAttribute('x', String(tx)); s.setAttribute('y', String(y)); s.setAttribute('width', '10'); s.setAttribute('height', '10'); s.setAttribute('rx', '2'); s.setAttribute('fill', it.colour); out.appendChild(s); tx += 16; }
                const t = document.createElementNS(NS, 'text'); t.setAttribute('x', String(tx)); t.setAttribute('y', String(y + 9)); t.setAttribute('font-size', String(fs)); t.setAttribute('font-family', 'system-ui, -apple-system, Segoe UI, sans-serif'); t.setAttribute('fill', ink); t.textContent = it.text; out.appendChild(t);
            }));
            return { text: '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(out), width: w, height: H };
        });
    }
    function save(blob, name) {
        const url = URL.createObjectURL(blob), a = document.createElement('a');
        a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    function exportSvg(svg) {
        const tree = svg.classList.contains('tree') && root.UMEUse && root.UMEUse.treeSvgLight ? root.UMEUse.treeSvgLight() : null;
        const name = slug(svg.getAttribute('aria-label')) + dataName();
        if (tree) { save(new Blob([tree], { type: 'image/svg+xml' }), name + '.svg'); return; }
        save(new Blob([buildSvg(svg).text], { type: 'image/svg+xml' }), name + '.svg');
    }
    function exportPng(svg) {
        const tree = svg.classList.contains('tree') && root.UMEUse && root.UMEUse.treeSvgLight ? root.UMEUse.treeSvgLight() : null;
        const built = tree ? { text: tree, width: Number((tree.match(/width="([\d.]+)"/) || [])[1]) || 800, height: Number((tree.match(/height="([\d.]+)"/) || [])[1]) || 600 } : buildSvg(svg);
        const img = new Image(), scale = 2;
        img.onload = () => {
            const c = document.createElement('canvas'); c.width = Math.round(built.width * scale); c.height = Math.round(built.height * scale);
            const ctx = c.getContext('2d'); ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, c.width, c.height); ctx.drawImage(img, 0, 0, c.width, c.height);
            c.toBlob(b => { if (b) save(b, slug(svg.getAttribute('aria-label')) + dataName() + '.png'); }, 'image/png');
        };
        // A data: address (allowed by the page's security policy for images)
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(built.text)));
    }
    // Add a download bar above every chart, now and whenever charts are drawn later
    function enhance(scope) {
        (scope || document).querySelectorAll('svg.chart, svg.tree').forEach(svg => {
            if (svg.dataset.exportReady) return;
            svg.dataset.exportReady = '1';
            const bar = document.createElement('div'); bar.className = 'export-bar';
            bar.innerHTML = '<span>Download image:</span><button type="button" class="btn small" data-fmt="svg">SVG</button><button type="button" class="btn small" data-fmt="png">PNG</button>';
            bar.querySelectorAll('button').forEach(b => { b.setAttribute('aria-label', `Download ${svg.getAttribute('aria-label') || 'chart'} as ${b.dataset.fmt.toUpperCase()}`); b.addEventListener('click', () => (b.dataset.fmt === 'svg' ? exportSvg(svg) : exportPng(svg))); });
            svg.parentNode.insertBefore(bar, svg);
        });
    }
    document.addEventListener('DOMContentLoaded', () => {
        enhance(document);
        new MutationObserver(() => enhance(document)).observe(document.body, { childList: true, subtree: true });
    });
    root.UMEExport = { enhance, buildSvg, exportSvg, exportPng };
})(typeof window !== 'undefined' ? window : globalThis);
