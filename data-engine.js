/**
 * NPH ML/Neuralnet Predictor - Data Engine
 * Handles CSV parsing, variable role & type deduction, missing value detection,
 * multi-strategy data imputation, feature encoding, scaling, and train/test splitting.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    const TARGET_NAME_RE = /^(target|outcome|response|shunt_response|label|class|diagnos|result|improvement)/i;

    // ISO (2024-01-31, with optional time) and day/month/year or month/day/year dates (31/01/2024, 1-31-24)
    const DATE_RE = /^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$/;

    // Hard limits on the one-hot encoded design matrix: beyond these the browser tab can run out of memory
    const MAX_ENCODED_COLUMNS = 2000;
    const MAX_MATRIX_CELLS = 60e6;
    const MAX_VIF_COLUMNS = 500;

    // Conventional clinical ordering words (lowest rank first) used to order text ordinal levels
    const ORDINAL_WORD_ORDER = [
        'none', 'absent', 'never', 'normal', 'very low', 'low', 'minimal', 'mild', 'slight',
        'moderate', 'medium', 'intermediate', 'average', 'high', 'marked', 'severe', 'very high', 'very severe', 'critical', 'extreme'
    ];

    function ordinalWordRank(value) {
        const v = String(value).trim().toLowerCase();
        let best = -1;
        let bestLen = 0;
        ORDINAL_WORD_ORDER.forEach((w, idx) => {
            // Longest matching word wins (so "very high" beats "high")
            if ((v === w || v.startsWith(w + ' ') || v.endsWith(' ' + w) || v.includes(' ' + w + ' ')) && w.length > bestLen) {
                best = idx;
                bestLen = w.length;
            }
        });
        return best;
    }

    function arrayMin(arr) {
        let m = Infinity;
        for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
        return m;
    }

    function arrayMax(arr) {
        let m = -Infinity;
        for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
        return m;
    }

    const DataEngine = {
        /**
         * Orders ordinal levels: numerically when all numeric; by conventional clinical words
         * (none < mild < moderate < severe, low < medium < high) when every level is recognised;
         * otherwise by natural (numeric-aware) text order, so "Grade 2" < "Grade 10".
         */
        sortOrdinalLevels(values) {
            const levels = Array.from(new Set(values.map(v => (typeof v === 'number' ? v : String(v)))));
            if (levels.every(v => !isNaN(Number(v)) && String(v).trim() !== '')) {
                return levels.sort((a, b) => Number(a) - Number(b));
            }
            const wordRanks = levels.map(ordinalWordRank);
            const rankSet = new Set(wordRanks);
            if (wordRanks.every(r => r >= 0) && rankSet.size === levels.length) {
                return levels
                    .map((v, i) => ({ v, r: wordRanks[i] }))
                    .sort((a, b) => a.r - b.r)
                    .map(o => o.v);
            }
            return levels.sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }));
        },

        /**
         * Returns a function mapping a cell to a number for statistical tests: ordinal columns with text
         * levels (or a user-defined level order) map to their ordered rank; everything else to Number().
         * Missing / non-numeric cells map to NaN.
         */
        numericValueFn(rows, col, conf = {}) {
            const vals = rows.map(r => r[col]).filter(v => !this.isMissing(v));
            const hasCustomOrder = Array.isArray(conf.ordinalOrder) && conf.ordinalOrder.length > 0;
            const isTextOrdinal = conf.type === 'ordinal' && vals.some(v => isNaN(Number(v)));
            if (conf.type === 'ordinal' && (isTextOrdinal || hasCustomOrder)) {
                const { rankMap } = this.buildOrdinalRankMap(vals, conf.ordinalOrder);
                return (v) => (this.isMissing(v) || rankMap[String(v).trim()] === undefined ? NaN : rankMap[String(v).trim()]);
            }
            return (v) => (this.isMissing(v) ? NaN : Number(v));
        },

        /**
         * Standardises an ordinal rank with its training mean/SD (metadata from older preps without
         * mean/std is used unscaled)
         */
        scaleOrdinalRank(rank, meta) {
            if (typeof meta.mean !== 'number' || typeof meta.std !== 'number') return rank;
            return Number(((rank - meta.mean) / meta.std).toFixed(5));
        },

        /**
         * Builds a value -> rank lookup for an ordinal column
         */
        buildOrdinalRankMap(values, customOrder = null) {
            let ranks = this.sortOrdinalLevels(values);
            if (Array.isArray(customOrder) && customOrder.length > 0) {
                // User-defined clinical order first; any observed levels not listed keep their default order after it
                const norm = v => String(v).trim().toLowerCase();
                const ordered = [];
                customOrder.forEach(level => {
                    const match = ranks.find(r => norm(r) === norm(level));
                    if (match !== undefined && !ordered.includes(match)) ordered.push(match);
                });
                ranks = ordered.concat(ranks.filter(r => !ordered.includes(r)));
            }
            const rankMap = {};
            ranks.forEach((val, rk) => { rankMap[String(val)] = rk; });
            return { ranks, rankMap };
        },

        /**
         * RFC 4180 compliant CSV Parser with auto delimiter detection
         */
        parseCSV(csvText) {
            if (!csvText || typeof csvText !== 'string') {
                throw new Error("Empty or invalid CSV file content.");
            }
            // Strip a UTF-8 byte-order mark (added by Excel's "CSV UTF-8" export); otherwise the first
            // header silently becomes "\uFEFFName" and no longer matches its column name
            csvText = csvText.replace(/^\uFEFF/, '');

            // Detect delimiter: check first 3 lines
            const sampleLines = csvText.split(/\r\n|\n|\r/).filter(l => l.trim().length > 0).slice(0, 3);
            let delimiter = ',';
            if (sampleLines.length > 0) {
                const commaCount = (sampleLines[0].match(/,/g) || []).length;
                const semiCount = (sampleLines[0].match(/;/g) || []).length;
                const tabCount = (sampleLines[0].match(/\t/g) || []).length;
                if (semiCount > commaCount && semiCount >= tabCount) delimiter = ';';
                else if (tabCount > commaCount && tabCount >= semiCount) delimiter = '\t';
            }

            const rows = [];
            let currentRow = [];
            let currentField = '';
            let fieldQuoted = false;   // field content came from a quoted section (kept verbatim)
            let afterQuote = '';       // characters between a closing quote and the next delimiter
            let inQuotes = false;
            let i = 0;
            const len = csvText.length;

            // Unquoted fields are trimmed; quoted fields keep their exact content (RFC 4180), with only
            // whitespace outside the quotes discarded
            let malformedQuoteCount = 0; // text between a closing quote and the next delimiter (kept, reported)
            const finishField = () => {
                if (fieldQuoted && afterQuote.trim()) malformedQuoteCount++;
                const value = fieldQuoted ? currentField + afterQuote.trim() : currentField.trim();
                currentRow.push(value);
                currentField = '';
                afterQuote = '';
                fieldQuoted = false;
            };

            while (i < len) {
                const char = csvText[i];
                const nextChar = csvText[i + 1];

                if (inQuotes) {
                    if (char === '"') {
                        if (nextChar === '"') {
                            currentField += '"';
                            i += 2;
                        } else {
                            inQuotes = false;
                            i++;
                        }
                    } else {
                        currentField += char;
                        i++;
                    }
                    continue;
                }

                if (char === '"' && !fieldQuoted && currentField.trim() === '') {
                    // Opening quote (any whitespace before it is not part of the value)
                    inQuotes = true;
                    fieldQuoted = true;
                    currentField = '';
                    i++;
                } else if (char === delimiter) {
                    finishField();
                    i++;
                } else if (char === '\r' || char === '\n') {
                    finishField();
                    i += (char === '\r' && nextChar === '\n') ? 2 : 1;
                    if (currentRow.some(val => val.length > 0)) {
                        rows.push(currentRow);
                    }
                    currentRow = [];
                } else {
                    if (fieldQuoted) afterQuote += char;
                    else currentField += char;
                    i++;
                }
            }

            if (inQuotes) {
                throw new Error('The file has a quotation mark (") that is never closed, so the rest of the file would be read as one value. Please check the quoting (e.g. re-save the file as CSV from the spreadsheet).');
            }
            if (currentField.length > 0 || fieldQuoted || currentRow.length > 0) {
                finishField();
                if (currentRow.some(val => val.length > 0)) {
                    rows.push(currentRow);
                }
            }

            if (rows.length < 2) {
                throw new Error("CSV must contain at least a header row and one data row.");
            }

            const rawHeaders = rows[0];
            // Deduplicate headers if any (renames are reported to the user via renamedHeaders)
            const seenHeaders = {};
            const renamedHeaders = [];
            // A new name never collides with any other header in the file (e.g. "a, a, a_2" gives a, a_3, a_2),
            // and names that would alter the row objects themselves are renamed
            const reserved = new Set(rawHeaders.map((h, idx) => (h || `Column_${idx + 1}`).trim()));
            const used = new Set();
            const UNSAFE = new Set(['__proto__', 'constructor', 'prototype']);
            const headers = rawHeaders.map((h, idx) => {
                const original = (h || `Column_${idx + 1}`).trim();
                let name = original;
                if (used.has(name) || UNSAFE.has(name)) {
                    let k = seenHeaders[original] = (seenHeaders[original] || 1) + 1;
                    while (used.has(`${original}_${k}`) || reserved.has(`${original}_${k}`)) k++;
                    seenHeaders[original] = k;
                    name = `${original}_${k}`;
                    renamedHeaders.push({ column: idx + 1, from: original, to: name });
                }
                used.add(name);
                return name;
            });

            const dataRows = [];
            let raggedRowCount = 0; // rows with more or fewer cells than the header (padded / truncated)
            for (let r = 1; r < rows.length; r++) {
                const row = rows[r];
                if (row.length !== headers.length) raggedRowCount++;
                const obj = {};
                for (let c = 0; c < headers.length; c++) {
                    const rawVal = c < row.length ? row[c] : '';
                    obj[headers[c]] = rawVal;
                }
                dataRows.push(obj);
            }

            // Cell values that are names of JavaScript's built-in object properties ("constructor", "__proto__",
            // "toString", …) would break the lookups the analysis keys by category value: refuse them clearly
            const RESERVED = new Set(Object.getOwnPropertyNames(Object.prototype));
            for (let r = 0; r < dataRows.length; r++) {
                for (const h of headers) {
                    const v = dataRows[r][h];
                    if (v && RESERVED.has(String(v).trim())) {
                        throw new Error(`Row ${r + 2}, column "${h}" contains "${String(v).trim()}", a reserved word that the app cannot use as a value. Please rename it (e.g. "${String(v).trim()}_value") and load the file again.`);
                    }
                }
            }

            return {
                headers,
                rows: dataRows,
                delimiter,
                totalRows: dataRows.length,
                totalCols: headers.length,
                raggedRowCount,
                renamedHeaders,
                malformedQuoteCount
            };
        },

        /**
         * How typical a patient's combination of predictor values is, compared with the training patients:
         * squared Mahalanobis distance from the training mean (encoded, scaled predictors; covariance with a
         * small ridge so one-hot columns are allowed), and the 99th / 99.9th percentiles of the training patients'
         * own distances. Only aggregates are kept (no patient values). null when too few patients or too many
         * encoded columns.
         */
        fitTypicality(X) {
            if (!Array.isArray(X) || X.length < 30 || !X[0] || X[0].length < 2 || X[0].length > 300) return null;
            const rng = this.createPRNG(97);
            const rows = X.length > 3000 ? Array.from({ length: 3000 }, () => X[Math.floor(rng() * X.length)]) : X;
            const n = rows.length, p = rows[0].length;
            const mean = new Array(p).fill(0);
            rows.forEach(r => { for (let j = 0; j < p; j++) mean[j] += r[j] / n; });
            const C = Array.from({ length: p }, () => new Float64Array(p));
            rows.forEach(r => {
                for (let a = 0; a < p; a++) { const da = r[a] - mean[a]; if (da === 0) continue; for (let b = a; b < p; b++) C[a][b] += da * (r[b] - mean[b]); }
            });
            let trace = 0;
            for (let a = 0; a < p; a++) { for (let b = a; b < p; b++) { C[a][b] /= (n - 1); C[b][a] = C[a][b]; } trace += C[a][a]; }
            const lam = 1e-3 * (trace / p) + 1e-9;
            // Inverse by Gauss-Jordan elimination with partial pivoting
            const M = C.map((r, i) => { const row = new Float64Array(2 * p); row.set(r); row[i] += lam; row[p + i] = 1; return row; });
            for (let c = 0; c < p; c++) {
                let piv = c;
                for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
                if (Math.abs(M[piv][c]) < 1e-12) return null;
                [M[c], M[piv]] = [M[piv], M[c]];
                const d = M[c][c];
                for (let k = 0; k < 2 * p; k++) M[c][k] /= d;
                for (let r = 0; r < p; r++) { if (r === c) continue; const f = M[r][c]; if (f) for (let k = 0; k < 2 * p; k++) M[r][k] -= f * M[c][k]; }
            }
            const inv = M.map(r => Array.from(r.subarray(p)));
            const t = { mean, inv, q99: 0, q999: 0, n };
            const d2 = rows.map(r => this.typicalityDistance(r, t)).sort((a, b) => a - b);
            t.q99 = d2[Math.min(n - 1, Math.floor(0.99 * (n - 1)))];
            t.q999 = d2[Math.min(n - 1, Math.floor(0.999 * (n - 1)))];
            return t;
        },

        /** Squared Mahalanobis distance of an encoded patient from the training mean (see fitTypicality) */
        typicalityDistance(x, t) {
            const p = t.mean.length, d = new Array(p);
            for (let j = 0; j < p; j++) d[j] = x[j] - t.mean[j];
            let s = 0;
            for (let a = 0; a < p; a++) { let r = 0; const ia = t.inv[a]; for (let b = 0; b < p; b++) r += ia[b] * d[b]; s += d[a] * r; }
            return s;
        },

        /** 'typical', 'unusual' (beyond 99% of training patients) or 'very unusual' (beyond 99.9%); null if unknown */
        typicality(x, t) {
            if (!t || !x || x.length !== t.mean.length) return null;
            const d2 = this.typicalityDistance(x, t);
            return { d2, level: d2 > t.q999 ? 'very unusual' : d2 > t.q99 ? 'unusual' : 'typical' };
        },

        /** Problems found while reading a CSV file, as messages for the user (same checks for every upload) */
        parseIssues(parsed) {
            const out = [];
            if (parsed.renamedHeaders && parsed.renamedHeaders.length) out.push(`Duplicate column name(s) were renamed so every column is unique:\n${parsed.renamedHeaders.map(r => `• column ${r.column}: "${r.from}" → "${r.to}"`).join('\n')}`);
            if (parsed.malformedQuoteCount > 0) out.push(`${parsed.malformedQuoteCount} cell(s) have text after a closing quotation mark (e.g. "abc"def). The text was kept as part of the value; please check the file's quoting.`);
            if (parsed.raggedRowCount > 0) out.push(`${parsed.raggedRowCount} row(s) have a different number of cells from the header row. Short rows were padded with blanks (treated as missing) and extra cells were ignored. Please check the file's delimiters and quoting.`);
            return out;
        },

        /**
         * Checks if a cell value is missing / blank / NaN / NA
         */
        isMissing(val) {
            if (val === null || val === undefined) return true;
            const str = String(val).trim();
            if (str === '') return true;
            const lower = str.toLowerCase();
            // "None" is not treated as missing: in clinical data it is often a genuine answer (no symptoms, no comorbidity)
            if (lower === 'na' || lower === 'nan' || lower === 'null' || lower === '?' || lower === '.' || lower === 'n/a') return true;
            // Infinite numbers ("Infinity", "-inf", "1e999") cannot be used in any calculation: treated as missing
            if (/^[+-]?inf(inity)?$/.test(lower)) return true;
            const n = Number(str);
            return typeof val === 'number' ? !isFinite(val) : (!isNaN(n) && !isFinite(n));
        },

        /**
         * Analyzes dataset columns: missing counts, data types, distinct values, and heuristics
         */
        analyzeColumns(headers, rows) {
            const analysis = {};

            headers.forEach((col, colIdx) => {
                let missingCount = 0;
                const distinctMap = new Map();
                let numericCount = 0;
                let validCount = 0;
                const validNumbers = [];

                rows.forEach((row, rowIdx) => {
                    const val = row[col];
                    if (this.isMissing(val)) {
                        missingCount++;
                    } else {
                        validCount++;
                        const strVal = String(val).trim();
                        distinctMap.set(strVal, (distinctMap.get(strVal) || 0) + 1);

                        const num = Number(strVal);
                        if (!isNaN(num) && strVal !== '') {
                            numericCount++;
                            validNumbers.push(num);
                        }
                    }
                });

                const isNumeric = validCount > 0 && numericCount === validCount;
                const distinctCount = distinctMap.size;
                const distinctValues = Array.from(distinctMap.keys());

                // Heuristic Role determination
                let role = 'feature';
                const lowerCol = col.toLowerCase();

                if (lowerCol.match(/^(patient_id|id|mrn|record_id|patient|subjid|subject|hash)$/i) || (colIdx === 0 && distinctCount === validCount && !isNumeric)) {
                    role = 'meta';
                } else if (colIdx === headers.length - 1 || TARGET_NAME_RE.test(lowerCol)) {
                    role = 'target';
                }

                // Heuristic Type determination
                let type = 'nominal';
                if (isNumeric) {
                    const allIntegers = validNumbers.every(n => Number.isInteger(n));
                    if (distinctCount <= 7 && allIntegers && distinctCount >= 2) {
                        type = 'ordinal';
                    } else if (distinctCount > 7 || !allIntegers) {
                        type = 'continuous';
                    } else {
                        type = 'nominal';
                    }
                } else {
                    // Check if string labels resemble ordinal grades (e.g. grade 1, grade 2 or mild, moderate, severe)
                    const lowerValues = distinctValues.map(v => v.toLowerCase());
                    const looksOrdinal = lowerValues.some(v => v.includes('mild') || v.includes('mod') || v.includes('sev') || v.includes('stage') || v.includes('grade') || v.match(/^[0-9]+$/));
                    if (looksOrdinal && distinctCount <= 7) {
                        type = 'ordinal';
                    } else {
                        type = 'nominal';
                    }
                }

                // Calculate summary stats for numbers
                let min = null, max = null, mean = null, median = null, mode = null;
                if (validNumbers.length > 0) {
                    validNumbers.sort((a, b) => a - b);
                    min = validNumbers[0];
                    max = validNumbers[validNumbers.length - 1];
                    const sum = validNumbers.reduce((acc, v) => acc + v, 0);
                    mean = Number((sum / validNumbers.length).toPrecision(6)); // display only
                    const mid = Math.floor(validNumbers.length / 2);
                    median = validNumbers.length % 2 === 0 ? Number(((validNumbers[mid - 1] + validNumbers[mid]) / 2).toPrecision(6)) : validNumbers[mid]; // display only
                }

                // Calculate mode for any type
                let bestFreq = 0;
                distinctMap.forEach((freq, val) => {
                    if (freq > bestFreq) {
                        bestFreq = freq;
                        mode = isNumeric ? Number(val) : val;
                    }
                });

                analysis[col] = {
                    name: col,
                    index: colIdx,
                    total: rows.length,
                    validCount,
                    missingCount,
                    missingPercent: Number(((missingCount / rows.length) * 100).toFixed(1)),
                    distinctCount,
                    distinctValues,
                    isNumeric,
                    role, // 'meta', 'target', 'feature'
                    type, // 'continuous', 'ordinal', 'nominal'
                    stats: {
                        min,
                        max,
                        mean,
                        median,
                        mode,
                        distribution: Object.fromEntries(distinctMap)
                    },
                    // Default imputation strategy suggestion
                    suggestedImputation: type === 'continuous' ? 'median' : 'mode'
                };

                // Text columns that cannot sensibly be one-hot encoded default to Meta / ID (the user can
                // still change this in Step 2): identifier-like free text (names, doctors, hospitals...)
                // would add one column per distinct value (tens of thousands on large files, exhausting
                // browser memory), and calendar dates are not meaningful as unordered categories.
                if (role === 'feature' && !isNumeric && validCount > 0) {
                    const sample = distinctValues.slice(0, 200);
                    const dateLike = sample.filter(v => DATE_RE.test(v.trim())).length >= 0.9 * sample.length;
                    if (dateLike) {
                        analysis[col].role = 'meta';
                        analysis[col].autoMetaReason = 'Looks like a calendar date: excluded by default (dates are not meaningful as categories; derive e.g. a duration or year if needed).';
                    } else if (distinctCount > 20 && distinctCount / validCount > 0.5) {
                        analysis[col].role = 'meta';
                        analysis[col].autoMetaReason = `Identifier-like text (${distinctCount.toLocaleString('en-GB')} distinct values in ${validCount.toLocaleString('en-GB')} rows): excluded by default because one-hot encoding would add a column per value.`;
                    }
                }
            });

            // Only one target may be suggested: prefer an outcome-like column name, else the last column
            const targetCols = headers.filter(h => analysis[h].role === 'target');
            if (targetCols.length > 1) {
                const keep = targetCols.find(h => TARGET_NAME_RE.test(h.toLowerCase())) || targetCols[targetCols.length - 1];
                targetCols.forEach(h => {
                    if (h !== keep) analysis[h].role = 'feature';
                });
            }

            return analysis;
        },

        /**
         * Imputes missing predictor values according to each column's configured strategy.
         *
         * Fit/apply semantics (to keep test data out of model development):
         *  - Called WITHOUT `fittedPlan`: fill values are computed from `rows` themselves and the
         *    returned `plan` records them (use this on the TRAINING rows).
         *  - Called WITH `fittedPlan`: the stored training-derived fill values are applied unchanged,
         *    and KNN neighbours are drawn only from the training reference rows (use this on TEST rows).
         * Row deletion ('drop' strategy, missing outcome) is decided per row from that row's own data.
         */
        imputeDataset(rows, columnConfigs, columnAnalysis, fittedPlan = null) {
            this.assertNoOutcomeAsPredictor(columnConfigs);
            let workingRows = rows.map(r => ({ ...r }));
            const audit = [];
            const rowsToDrop = new Set();
            const targetMissingRows = new Set();

            // First pass: identify rows that must be dropped.
            // Rows with a missing outcome (target) are always excluded: an outcome must never be imputed.
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return; // skip meta columns

                if (config.role === 'target') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) targetMissingRows.add(rIdx);
                    });
                } else if (config.imputationStrategy === 'drop') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) {
                            rowsToDrop.add(rIdx);
                        }
                    });
                }
            });

            if (targetMissingRows.size > 0) {
                audit.push({
                    type: 'drop_rows',
                    description: `Excluded ${targetMissingRows.size} row(s) with a missing outcome (target values are never imputed).`
                });
            }
            const featureOnlyDrops = [...rowsToDrop].filter(idx => !targetMissingRows.has(idx)).length;
            if (featureOnlyDrops > 0) {
                audit.push({
                    type: 'drop_rows',
                    description: `Dropped ${featureOnlyDrops} row(s) with missing values in columns marked for row deletion.`
                });
            }
            if (targetMissingRows.size > 0 || rowsToDrop.size > 0) {
                workingRows = workingRows.filter((_, idx) => !rowsToDrop.has(idx) && !targetMissingRows.has(idx));
            }

            // Pristine (pre-imputation) copy: the KNN neighbour pool when fitting, and the stored reference
            const snapshotRows = workingRows.map(r => ({ ...r }));
            const plan = fittedPlan || { columns: {}, referenceRows: snapshotRows };
            const knnPool = plan.referenceRows;
            const poolIsSelf = !fittedPlan;

            // Second pass: compute (or reuse) fill values and impute
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return;
                if (config.role === 'target') return;
                if (config.imputationStrategy === 'drop') return;

                const colType = config.type;
                let colPlan = fittedPlan ? fittedPlan.columns[col] : null;

                if (!colPlan) {
                    let strategy = config.imputationStrategy;
                    const validValues = [];
                    const validNumbers = [];
                    snapshotRows.forEach(r => {
                        const val = r[col];
                        if (!this.isMissing(val)) {
                            validValues.push(val);
                            const n = Number(val);
                            if (!isNaN(n)) validNumbers.push(n);
                        }
                    });

                    // Mean/median/KNN need numbers: text-valued columns fall back to the most frequent value
                    if ((strategy === 'mean' || strategy === 'median' || strategy === 'knn') && validNumbers.length < validValues.length) {
                        strategy = 'mode';
                    }

                    const meanValue = validNumbers.length > 0 ? validNumbers.reduce((a, b) => a + b, 0) / validNumbers.length : 0;
                    let fillValue = null;
                    if (strategy === 'mean') {
                        fillValue = meanValue; // full precision (fixed-decimal rounding is scale-dependent)
                    } else if (strategy === 'median') {
                        if (validNumbers.length > 0) {
                            const sorted = [...validNumbers].sort((a, b) => a - b);
                            const mid = Math.floor(sorted.length / 2);
                            if (colType === 'ordinal') {
                                // Lower median keeps the fill value on an observed ordinal level
                                fillValue = sorted[Math.floor((sorted.length - 1) / 2)];
                            } else {
                                fillValue = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
                            }
                        } else {
                            fillValue = 0;
                        }
                    } else if (strategy === 'mode') {
                        const freqMap = new Map();
                        validValues.forEach(v => freqMap.set(String(v), (freqMap.get(String(v)) || 0) + 1));
                        let topCount = 0;
                        let topVal = null;
                        freqMap.forEach((count, v) => {
                            if (count > topCount) {
                                topCount = count;
                                topVal = v;
                            }
                        });
                        if ((colType === 'continuous' || colType === 'ordinal') && topVal !== null && !isNaN(Number(topVal))) {
                            fillValue = Number(topVal);
                        } else {
                            fillValue = topVal !== null ? topVal : (colType === 'nominal' ? 'Unknown' : 0);
                        }
                    } else if (strategy === 'constant') {
                        if (colType === 'continuous' || colType === 'ordinal') {
                            fillValue = Number(config.customImputeVal !== undefined && config.customImputeVal !== '' ? config.customImputeVal : 0);
                        } else {
                            fillValue = String(config.customImputeVal || 'Missing');
                        }
                    } else if (strategy === 'knn') {
                        fillValue = 'knn_dynamic'; // filled cell-by-cell below
                    }

                    // Scales for KNN distances (from the fitting rows only)
                    const refScale = {};
                    // Distance features: continuous and numeric-valued ordinal predictors (text ordinals excluded)
                    const numericCols = Object.keys(columnConfigs).filter(c => {
                        const cc = columnConfigs[c];
                        if (cc.role !== 'feature' || c === col) return false;
                        if (cc.type === 'continuous') return true;
                        if (cc.type !== 'ordinal') return false;
                        const vals = snapshotRows.map(r => r[c]).filter(v => !this.isMissing(v));
                        return vals.length > 0 && vals.every(v => !isNaN(Number(v)));
                    });
                    if (strategy === 'knn') {
                        numericCols.forEach(nCol => {
                            const vals = snapshotRows.map(r => r[nCol]).filter(v => !this.isMissing(v)).map(Number).filter(n => !isNaN(n));
                            const m = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
                            const sd = vals.length ? Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length) : 0;
                            refScale[nCol] = sd > 0 ? sd : 1;
                        });
                    }

                    colPlan = { strategy, fillValue, meanValue, numericCols, refScale };
                    plan.columns[col] = colPlan;
                }

                const { strategy, fillValue } = colPlan;

                let imputedCount = 0;
                if (strategy === 'knn') {
                    const k = 3;
                    workingRows.forEach((row, rowIdx) => {
                        if (!this.isMissing(row[col])) {
                            const n = Number(row[col]);
                            if (!isNaN(n)) row[col] = n;
                            return;
                        }
                        // Neighbours come from the (pre-imputation) fitting rows only
                        const distances = [];
                        knnPool.forEach((otherRow, otherIdx) => {
                            if ((poolIsSelf && otherIdx === rowIdx) || this.isMissing(otherRow[col]) || isNaN(Number(otherRow[col]))) return;
                            let distSq = 0;
                            let matchedFeatures = 0;
                            colPlan.numericCols.forEach(nCol => {
                                if (!this.isMissing(row[nCol]) && !this.isMissing(otherRow[nCol])) {
                                    const d = (Number(row[nCol]) - Number(otherRow[nCol])) / colPlan.refScale[nCol];
                                    if (!isNaN(d)) {
                                        distSq += d * d;
                                        matchedFeatures++;
                                    }
                                }
                            });
                            if (matchedFeatures > 0) {
                                // Mean squared distance over shared features, so rows sharing fewer features are not favoured
                                distances.push({ val: Number(otherRow[col]), dist: Math.sqrt(distSq / matchedFeatures) });
                            }
                        });

                        if (distances.length > 0) {
                            distances.sort((a, b) => a.dist - b.dist);
                            const neighbors = distances.slice(0, Math.min(k, distances.length));
                            const avg = neighbors.reduce((acc, item) => acc + item.val, 0) / neighbors.length;
                            row[col] = avg;
                        } else {
                            row[col] = colPlan.meanValue; // fallback: training mean
                        }
                        imputedCount++;
                    });
                } else {
                    workingRows.forEach(row => {
                        if (this.isMissing(row[col])) {
                            row[col] = fillValue;
                            imputedCount++;
                        } else if (colType === 'continuous' || colType === 'ordinal') {
                            const n = Number(row[col]);
                            if (!isNaN(n)) row[col] = n;
                        }
                    });
                }

                if (imputedCount > 0) {
                    audit.push({
                        type: 'impute_column',
                        column: col,
                        strategy,
                        imputedCount,
                        filledValue: fillValue === 'knn_dynamic' ? 'KNN: mean of 3 nearest training neighbours (standardised distance)' : fillValue
                    });
                }
            });

            return {
                imputedRows: workingRows,
                audit,
                totalRowsAfter: workingRows.length,
                plan
            };
        },

        /**
         * Preprocess features and target with optional explicit featureCols and targetCol
         */
        prepareDataset(rows, columnConfigs, featureCols = null, targetCol = null) {
            const configs = {};
            Object.keys(columnConfigs).forEach(c => {
                configs[c] = { ...columnConfigs[c] };
            });

            if (targetCol) {
                Object.keys(configs).forEach(c => {
                    if (configs[c].role === 'target') configs[c].role = 'meta';
                });
                if (configs[targetCol]) configs[targetCol].role = 'target';
                else configs[targetCol] = { role: 'target', type: 'nominal' };
            }

            if (featureCols && Array.isArray(featureCols)) {
                Object.keys(configs).forEach(c => {
                    if (c !== targetCol) configs[c].role = 'meta';
                });
                featureCols.forEach(c => {
                    if (configs[c]) configs[c].role = 'feature';
                    else configs[c] = { role: 'feature', type: 'continuous' };
                });
            }

            return this.prepareFeaturesAndTarget(rows, configs);
        },

        /**
         * Preprocess features and target into numeric arrays for machine learning models
         */
        /**
         * Safeguard, checked wherever predictors are encoded or imputed: no part of the outcome may be a predictor.
         * Throws if a predictor is marked as the follow-up time or event of a time-to-event outcome, or if a
         * column is both predictor and outcome. Applies whatever route led there (interface, settings file, code).
         */
        assertNoOutcomeAsPredictor(columnConfigs) {
            Object.keys(columnConfigs || {}).forEach(col => {
                const c = columnConfigs[col];
                if (c && c.role === 'feature' && (c.survivalTime || c.survivalEvent)) {
                    throw new Error(`Safeguard: "${col}" is part of the time-to-event outcome (${c.survivalTime ? 'follow-up time' : 'event status'}) and cannot be used as a predictor. The analysis was stopped.`);
                }
            });
        },

        /**
         * Safeguard: a training/testing split must not share a patient. Same record object in both, or (when
         * records are grouped by patient) the same patient in both, throws.
         */
        assertDisjointSplit(trainRows, testRows, groupOf = null) {
            const seen = new Set(trainRows);
            if (testRows.some(r => seen.has(r))) throw new Error('Safeguard: the same record is in both the training and the testing data. The analysis was stopped.');
            if (groupOf) {
                const g = new Set(trainRows.map(groupOf).map(String));
                const shared = testRows.map(groupOf).map(String).find(k => g.has(k));
                if (shared !== undefined) throw new Error(`Safeguard: patient "${shared}" has records in both the training and the testing data. The analysis was stopped.`);
            }
        },

        /**
         * Standardised value of a continuous predictor, used by training (prepareFeaturesAndTarget) and by every
         * prediction (transformSingleRow), so both encode identically (missing -> 0, the training mean)
         */
        scaleContinuous(num, meta) {
            const scaled = (num - meta.mean) / meta.std;
            return isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
        },

        prepareFeaturesAndTarget(rows, columnConfigs) {
            this.assertNoOutcomeAsPredictor(columnConfigs);
            const featureCols = [];
            let targetCol = null;

            Object.keys(columnConfigs).forEach(col => {
                const conf = columnConfigs[col];
                if (conf.role === 'feature') {
                    featureCols.push(col);
                } else if (conf.role === 'target') {
                    targetCol = col;
                }
            });

            if (!targetCol) {
                throw new Error("Please select exactly one Target variable to predict.");
            }
            if (featureCols.length === 0) {
                throw new Error("Please select at least one Feature (predictor) variable.");
            }

            const targetConf = columnConfigs[targetCol];
            const isClassification = targetConf.type === 'nominal' || targetConf.type === 'ordinal';

            // Rows without a usable outcome cannot be used for training (outcomes are never imputed)
            const droppedMissingTarget = rows.filter(r => this.isMissing(r[targetCol]) ||
                (!isClassification && isNaN(Number(r[targetCol])))).length;
            if (droppedMissingTarget > 0) {
                rows = rows.filter(r => !this.isMissing(r[targetCol]) &&
                    (isClassification || !isNaN(Number(r[targetCol]))));
            }
            if (rows.length === 0) {
                throw new Error(`No rows with a valid value for the target '${targetCol}'.`);
            }

            // 1. Build Target Encoder
            let targetClasses = [];
            let targetClassToIdx = {};
            let targetIdxToClass = {};
            let targetStats = null;

            if (isClassification) {
                // Collect distinct target classes
                const classSet = new Set();
                rows.forEach(r => {
                    const val = String(r[targetCol]).trim(); // trimmed as in prepareTestingDataset, so " Yes" and "Yes" match
                    classSet.add(val);
                });
                targetClasses = Array.from(classSet);
                // Sort classes naturally
                targetClasses.sort();
                // Binary outcome: the user-chosen positive (event) class is placed at index 1, which every
                // binary metric (AUC, sensitivity, PPV, TP/FN) treats as the positive class
                const chosenPositive = targetConf.positiveClass !== undefined && targetConf.positiveClass !== null
                    ? String(targetConf.positiveClass).trim() : null;
                if (targetClasses.length === 2 && chosenPositive !== null && targetClasses.includes(chosenPositive)) {
                    targetClasses = [targetClasses.find(c => c !== chosenPositive), chosenPositive];
                }
                targetClasses.forEach((cls, idx) => {
                    targetClassToIdx[cls] = idx;
                    targetIdxToClass[idx] = cls;
                });
            } else {
                // Regression target: collect numbers
                const nums = rows.map(r => Number(r[targetCol])).filter(n => !isNaN(n));
                const sum = nums.reduce((a, b) => a + b, 0);
                const mean = sum / nums.length;
                const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length;
                targetStats = {
                    mean,
                    std: Math.sqrt(variance) || 1,
                    min: arrayMin(nums),
                    max: arrayMax(nums)
                };
            }

            // Guard: estimate the encoded width BEFORE allocating anything, and stop with an explanation
            // (instead of crashing the tab) if high-cardinality categorical predictors would make it too large
            {
                const widths = featureCols.map(col => {
                    const conf = columnConfigs[col];
                    if (conf.type !== 'nominal') return { col, w: 1 };
                    const set = new Set();
                    for (let i = 0; i < rows.length; i++) set.add(String(rows[i][col]));
                    return { col, w: set.size };
                });
                const totalWidth = widths.reduce((a, b) => a + b.w, 0);
                if (totalWidth > MAX_ENCODED_COLUMNS || totalWidth * rows.length > MAX_MATRIX_CELLS) {
                    const worst = widths.filter(x => x.w > 50).sort((a, b) => b.w - a.w)
                        .map(x => `"${x.col}" (${x.w.toLocaleString('en-GB')} categories)`);
                    throw new Error(`The predictors would expand to ${totalWidth.toLocaleString('en-GB')} encoded columns × ` +
                        `${rows.length.toLocaleString('en-GB')} rows, which exceeds what a browser can hold in memory. ` +
                        (worst.length ? `High-cardinality categorical predictor(s): ${worst.join(', ')}. ` : '') +
                        `Set identifier-like columns (names, IDs, doctors, hospitals, dates) to "Meta / ID" in Step 2, or group rare categories.`);
                }
            }

            // 2. Build Feature Encoders
            // For Nominal: One-Hot Encoding map
            // For Ordinal: Natural / sorted rank mapping
            // For Continuous: Standardization (mean, std)
            const featureMetadata = {};
            let totalEncodedFeatures = 0;
            const encodedFeatureNames = [];

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf.type === 'nominal') {
                    const catSet = new Set();
                    const catFreq = {};
                    rows.forEach(r => {
                        const c = String(r[col]);
                        catSet.add(c);
                        catFreq[c] = (catFreq[c] || 0) + 1;
                    });
                    const categories = Array.from(catSet).sort();
                    let modeCat = categories[0];
                    categories.forEach(c => { if (catFreq[c] > catFreq[modeCat]) modeCat = c; });
                    featureMetadata[col] = {
                        type: 'nominal',
                        categories,
                        mode: modeCat,
                        startIndex: totalEncodedFeatures,
                        length: categories.length
                    };
                    categories.forEach(cat => {
                        encodedFeatureNames.push(`${col}_${cat}`);
                    });
                    totalEncodedFeatures += categories.length;
                } else if (conf.type === 'ordinal') {
                    // Ordinal mapping: map ordered distinct levels to integers 0, 1, 2...
                    const { ranks, rankMap } = this.buildOrdinalRankMap(rows.map(r => r[col]), conf.ordinalOrder);
                    // Median rank of the training rows: fallback for missing / unseen levels at prediction time
                    const rankVals = rows.map(r => rankMap[String(r[col])]).filter(v => v !== undefined).sort((a, b) => a - b);
                    const defaultRank = rankVals.length ? rankVals[Math.floor((rankVals.length - 1) / 2)] : 0;
                    // Ranks are z-scored with the training mean/SD (same convention as continuous features),
                    // so a many-level scale (e.g. GCS, mRS) does not dominate k-NN distances or L2 penalties
                    const rankMean = rankVals.length ? rankVals.reduce((a, b) => a + b, 0) / rankVals.length : 0;
                    const rankStd = rankVals.length
                        ? Math.sqrt(rankVals.reduce((a, b) => a + (b - rankMean) * (b - rankMean), 0) / rankVals.length) || 1
                        : 1;
                    featureMetadata[col] = {
                        type: 'ordinal',
                        ranks,
                        rankMap,
                        defaultRank,
                        mean: rankMean,
                        std: rankStd,
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                } else {
                    // Continuous: compute mean and standard deviation
                    const nums = rows.map(r => Number(r[col])).filter(n => !isNaN(n));
                    const sum = nums.reduce((a, b) => a + b, 0);
                    const mean = sum / (nums.length || 1);
                    const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (nums.length || 1);
                    // A (near-)constant predictor: no scaling (avoids dividing by a tiny spread from rounding noise)
                    const sdRaw = Math.sqrt(variance);
                    const std = sdRaw > 1e-8 * Math.max(1, Math.abs(mean)) ? sdRaw : 1;
                    featureMetadata[col] = {
                        type: 'continuous',
                        mean,
                        std,
                        min: nums.length ? arrayMin(nums) : 0,
                        max: nums.length ? arrayMax(nums) : 0,
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                }
            });

            // 3. Transform Rows into Matrix X and Vector y
            const X = [];
            const y = [];

            rows.forEach(row => {
                const xRow = new Array(totalEncodedFeatures).fill(0);
                featureCols.forEach(col => {
                    const meta = featureMetadata[col];
                    const val = row[col];

                    if (meta.type === 'nominal') {
                        const catIdx = meta.categories.indexOf(String(val));
                        if (catIdx !== -1) {
                            xRow[meta.startIndex + catIdx] = 1;
                        }
                    } else if (meta.type === 'ordinal') {
                        const rankVal = meta.rankMap[String(val)] !== undefined ? meta.rankMap[String(val)] : meta.defaultRank;
                        xRow[meta.startIndex] = this.scaleOrdinalRank(rankVal, meta);
                    } else {
                        // Continuous: standardized (missing -> training mean, i.e. 0 after scaling)
                        const num = this.isMissing(val) ? NaN : Number(val);
                        xRow[meta.startIndex] = this.scaleContinuous(num, meta);
                    }
                });

                X.push(xRow);

                if (isClassification) {
                    const targetClass = String(row[targetCol]).trim();
                    y.push(targetClassToIdx[targetClass] !== undefined ? targetClassToIdx[targetClass] : 0);
                } else {
                    y.push(Number(row[targetCol]));
                }
            });

            return {
                X,
                y,
                featureCols,
                targetCol,
                targetConf,
                isClassification,
                targetClasses,
                targetClassToIdx,
                targetIdxToClass,
                targetStats,
                featureMetadata,
                encodedFeatureNames,
                totalDimensions: totalEncodedFeatures,
                droppedMissingTarget,
                // Derived "was missing" indicator columns: indicator -> source predictor (re-derived for new patients)
                derivedIndicators: Object.fromEntries(featureCols.filter(c => columnConfigs[c] && columnConfigs[c].derivedFrom).map(c => [c, columnConfigs[c].derivedFrom])),
                // Binary metrics (AUC, sensitivity, TP/FN) treat class index 1 as the positive class
                positiveClass: isClassification && targetClasses.length === 2 ? targetClasses[1] : null
            };
        },

        /** Name of the "was missing" indicator column derived from a predictor */
        missingIndicatorName(col) { return `${col} (was missing)`; },

        /**
         * Transform a single arbitrary row (for single-case or batch inference).
         * Missing or unrecognised values are replaced with training-derived values
         * (continuous: training mean; ordinal: median training rank; nominal: most frequent
         * training category). If an `issues` array is supplied, each substitution is recorded
         * so the caller can warn the user.
         */
        // ranges (optional array): numeric inputs outside the training range are listed in it
        // ({ column, value, min, max }): the prediction is then an extrapolation (or the value is a typing error)
        transformSingleRow(inputRow, prepMeta, issues = null, ranges = null) {
            const { featureCols, featureMetadata, totalDimensions } = prepMeta;
            const xRow = new Array(totalDimensions).fill(0);
            const note = (col, kind, rawVal) => {
                if (issues) issues.push({ column: col, kind, value: rawVal });
            };

            // "Was missing" indicators are always re-derived from their source predictor (blank -> Yes)
            const derived = prepMeta.derivedIndicators || {};
            if (Object.keys(derived).length) {
                inputRow = { ...inputRow };
                Object.keys(derived).forEach(ind => { inputRow[ind] = this.isMissing(inputRow[derived[ind]]) ? 'Yes' : 'No'; });
            }
            // Blank predictors are filled exactly as in training: with the fitted imputation plan (standard
            // workflow: median / mean / most frequent / constant / k-NN as chosen in Step 3) or the fitted
            // fill values (7-step pipeline). Only without either does the generic fallback below apply.
            let row = inputRow;
            const prefilled = new Set();
            const missingCols = featureCols.filter(c => this.isMissing(inputRow[c]));
            // Multiple imputation: the patient is imputed once per imputation chain and encoded each time; the
            // first version is returned, carrying all of them (x.__mi) for the pooled model to average over
            const mice = prepMeta.imputation && prepMeta.imputation.mice;
            if (mice && missingCols.length) {
                const plain = { ...prepMeta, imputation: null, fillValues: null, derivedIndicators: {} };
                const versions = mice.chains.map((_, l) => this.transformSingleRow(this.miceApplyRow(inputRow, mice, l), plain, null, l === 0 ? ranges : null));
                missingCols.forEach(c => note(c, 'missing', inputRow[c]));
                const first = versions[0].slice();
                Object.defineProperty(first, '__mi', { value: versions, enumerable: false });
                return first;
            }
            if (missingCols.length) {
                let filled = null;
                if (prepMeta.imputation && prepMeta.imputation.plan) {
                    try {
                        filled = this.imputeDataset([{ ...inputRow }], prepMeta.imputation.configs, null, prepMeta.imputation.plan).imputedRows[0] || null;
                    } catch (e) { filled = null; }
                } else if (prepMeta.fillValues) {
                    filled = { ...inputRow };
                    missingCols.forEach(c => { if (prepMeta.fillValues[c] !== undefined && prepMeta.fillValues[c] !== null) filled[c] = prepMeta.fillValues[c]; });
                }
                if (filled) {
                    row = filled;
                    missingCols.forEach(c => {
                        if (!this.isMissing(filled[c])) { prefilled.add(c); note(c, 'missing', inputRow[c]); }
                    });
                }
            }

            featureCols.forEach(col => {
                const meta = featureMetadata[col];
                const rawVal = row[col];
                const missing = this.isMissing(rawVal);
                const noteOnce = (c, kind, v) => { if (!prefilled.has(c)) note(c, kind, v); };

                if (meta.type === 'nominal') {
                    let catIdx = missing ? -1 : meta.categories.indexOf(String(rawVal).trim());
                    if (catIdx === -1) {
                        noteOnce(col, missing ? 'missing' : 'unseen', rawVal);
                        catIdx = meta.categories.indexOf(meta.mode !== undefined ? meta.mode : meta.categories[0]);
                    }
                    if (catIdx !== -1) {
                        xRow[meta.startIndex + catIdx] = 1;
                    }
                } else if (meta.type === 'ordinal') {
                    let rankVal = missing ? undefined : meta.rankMap[String(rawVal).trim()];
                    if (rankVal === undefined && !missing && !isNaN(Number(rawVal))) {
                        rankVal = meta.rankMap[String(Number(rawVal))];
                    }
                    if (rankVal === undefined) {
                        noteOnce(col, missing ? 'missing' : 'unseen', rawVal);
                        rankVal = meta.defaultRank !== undefined ? meta.defaultRank : 0;
                    }
                    xRow[meta.startIndex] = this.scaleOrdinalRank(rankVal, meta);
                } else {
                    // Note: Number('') === 0, so blanks must be detected explicitly
                    const num = missing ? NaN : Number(rawVal);
                    if (isNaN(num)) {
                        noteOnce(col, missing ? 'missing' : 'non-numeric', rawVal);
                        xRow[meta.startIndex] = 0; // training mean (0 after standardisation)
                    } else {
                        xRow[meta.startIndex] = this.scaleContinuous(num, meta);
                        if (ranges && !prefilled.has(col) && typeof meta.min === 'number' && typeof meta.max === 'number' && (num < meta.min || num > meta.max)) {
                            ranges.push({ column: col, value: num, min: meta.min, max: meta.max });
                        }
                    }
                }
            });

            return xRow;
        },

        /**
         * Pseudo-Random Number Generator (Mulberry32) for reproducible data splits & shuffling
         */
        createPRNG(seed) {
            if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
                return Math.random;
            }
            let s = Math.floor(Math.abs(Number(seed))) >>> 0;
            if (s === 0) s = 1;
            return function() {
                s |= 0; s = (s + 0x6D2B79F5) | 0;
                let t = Math.imul(s ^ (s >>> 15), 1 | s);
                t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
                return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            };
        },

        /**
         * Patient-level split: every record of a group (e.g. one patient's repeated visits) goes to the same
         * side, so no patient appears in both the Training and the Testing set. Groups are shuffled with the
         * seed and added to the Testing set until it holds about (1 − trainRatio) of the records.
         */
        trainTestSplitGrouped(rows, y, groupKeys, trainRatio = 0.8, seed = null) {
            const rng = this.createPRNG(seed);
            const byGroup = new Map();
            groupKeys.forEach((g, i) => {
                const key = (g === null || g === undefined || String(g).trim() === '') ? `__row${i}` : String(g).trim();
                if (!byGroup.has(key)) byGroup.set(key, []);
                byGroup.get(key).push(i);
            });
            const keys = [...byGroup.keys()].sort();
            for (let i = keys.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [keys[i], keys[j]] = [keys[j], keys[i]];
            }
            const targetTest = Math.max(1, Math.round(rows.length * (1 - trainRatio)));
            const testIndices = [];
            const trainIndices = [];
            let nTestGroups = 0;
            keys.forEach(k => {
                const idx = byGroup.get(k);
                if (testIndices.length < targetTest && keys.length - nTestGroups > 1) { testIndices.push(...idx); nTestGroups++; }
                else trainIndices.push(...idx);
            });
            return {
                trainX: trainIndices.map(i => rows[i]), trainY: trainIndices.map(i => y[i]),
                testX: testIndices.map(i => rows[i]), testY: testIndices.map(i => y[i]),
                trainIndices, testIndices,
                info: { method: 'group', groups: keys.length, trainGroups: keys.length - nTestGroups, testGroups: nTestGroups }
            };
        },

        /**
         * Multiple imputation by chained equations (MICE) with predictive mean matching (PMM).
         * For each predictor with missing values in turn, a ridge regression of that predictor on all the
         * others (and on the outcome, as recommended) is fitted to a bootstrap sample of the observed rows
         * (proper imputation); each missing value is replaced by the observed value of one of the 5 observed rows
         * whose predicted values are closest. Categorical predictors are predicted as one-hot vectors.
         * Returns m completed copies of the rows (imputed values are always values actually observed).
         */
        miceImpute(rows, columnConfigs, opts = {}) {
            const g = this.miceImputeSteps(rows, columnConfigs, opts);
            for (;;) { const r = g.next(); if (r.done) return r.value; }
        },
        /** As miceImpute, returning a promise: hands control back to the page about every 100 ms and reports progress
         *  (onProgress(imputation, m, iteration, iterations)); identical results */
        async miceImputeAsync(rows, columnConfigs, opts = {}, onProgress = null) {
            const g = this.miceImputeSteps(rows, columnConfigs, opts);
            let last = Date.now();
            for (;;) {
                const r = g.next(); if (r.done) return r.value;
                if (Date.now() - last > 100) { if (onProgress && r.value) onProgress(...r.value); await new Promise(res => setTimeout(res, 0)); last = Date.now(); }
            }
        },
        // Multiple imputation as a generator that yields after each variable is updated (see miceImpute)
        *miceImputeSteps(rows, columnConfigs, { m = 5, iterations = 10, seed = 2024, donors = 5 } = {}) {
            const cols = Object.keys(columnConfigs).filter(c => columnConfigs[c].role === 'feature' || columnConfigs[c].role === 'target');
            const isNum = c => columnConfigs[c].type !== 'nominal' && rows.every(r => this.isMissing(r[c]) || !isNaN(Number(r[c])));
            const numeric = {}; cols.forEach(c => { numeric[c] = isNum(c); });
            const levels = {};
            cols.forEach(c => { if (!numeric[c]) levels[c] = [...new Set(rows.filter(r => !this.isMissing(r[c])).map(r => String(r[c]).trim()))].sort(); });
            const missingIdx = {};
            cols.forEach(c => { missingIdx[c] = rows.map((r, i) => (this.isMissing(r[c]) ? i : -1)).filter(i => i >= 0); });
            const targets = cols.filter(c => columnConfigs[c].role === 'feature' && missingIdx[c].length > 0 && missingIdx[c].length < rows.length);
            const n = rows.length;

            // Ridge regression (normal equations); returns coefficient matrix p x q
            const ridge = (X, Y, lam) => {
                const p = X[0].length, q = Y[0].length;
                const A = Array.from({ length: p }, () => new Array(p).fill(0));
                const Bm = Array.from({ length: p }, () => new Array(q).fill(0));
                for (let i = 0; i < X.length; i++) {
                    const xi = X[i];
                    for (let a = 0; a < p; a++) {
                        if (xi[a] === 0) continue;
                        for (let b = a; b < p; b++) A[a][b] += xi[a] * xi[b];
                        for (let k = 0; k < q; k++) Bm[a][k] += xi[a] * Y[i][k];
                    }
                }
                for (let a = 0; a < p; a++) { for (let b = 0; b < a; b++) A[a][b] = A[b][a]; if (a > 0) A[a][a] += lam; }
                // Gaussian elimination on [A | Bm]
                const M = A.map((r, i) => [...r, ...Bm[i]]);
                for (let c = 0; c < p; c++) {
                    let piv = c;
                    for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
                    if (Math.abs(M[piv][c]) < 1e-12) { M[piv][c] = 1e-12; }
                    [M[c], M[piv]] = [M[piv], M[c]];
                    for (let r = 0; r < p; r++) {
                        if (r === c) continue;
                        const f = M[r][c] / M[c][c];
                        if (f) for (let k = c; k < p + q; k++) M[r][k] -= f * M[c][k];
                    }
                }
                return M.map((r, i) => r.slice(p).map(v => v / r[i]));
            };

            const datasets = [];
            for (let l = 0; l < m; l++) {
                const rng = this.createPRNG(seed + 1009 * l);
                const data = rows.map(r => ({ ...r }));
                // Start: random draws from the observed values
                targets.forEach(c => {
                    const obs = rows.filter(r => !this.isMissing(r[c])).map(r => r[c]);
                    missingIdx[c].forEach(i => { data[i][c] = obs[Math.floor(rng() * obs.length)]; });
                });
                for (let it = 0; it < iterations; it++) {
                    for (const c of targets) {
                        // Design matrix from all other variables (current completed values), standardised / one-hot
                        const others = cols.filter(o => o !== c && missingIdx[o].length < rows.length && !(columnConfigs[o].role === 'target' && missingIdx[o].length));
                        const stats = {};
                        others.forEach(o => {
                            if (numeric[o]) {
                                const v = data.map(r => Number(r[o]));
                                const mu = v.reduce((a, b) => a + b, 0) / n;
                                const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / n) || 1;
                                stats[o] = { mu, sd };
                            }
                        });
                        const design = r => {
                            const x = [1];
                            others.forEach(o => {
                                if (numeric[o]) x.push((Number(r[o]) - stats[o].mu) / stats[o].sd);
                                else levels[o].slice(1).forEach(lv => x.push(String(r[o]).trim() === lv ? 1 : 0));
                            });
                            return x;
                        };
                        const resp = r => (numeric[c] ? [Number(r[c])] : levels[c].map(lv => (String(r[c]).trim() === lv ? 1 : 0)));
                        const misSet = new Set(missingIdx[c]);
                        const obsIdx = [];
                        for (let i = 0; i < n; i++) if (!misSet.has(i)) obsIdx.push(i);
                        const Xall = data.map(design);
                        const lam = 1e-3 * obsIdx.length;
                        const betaHat = ridge(obsIdx.map(i => Xall[i]), obsIdx.map(i => resp(rows[i])), lam);
                        const boot = obsIdx.map(() => obsIdx[Math.floor(rng() * obsIdx.length)]);
                        const betaStar = ridge(boot.map(i => Xall[i]), boot.map(i => resp(rows[i])), lam);
                        const pred = (x, B) => B[0].map((_, k) => x.reduce((a, xv, j) => a + xv * B[j][k], 0));
                        const obsPred = obsIdx.map(i => pred(Xall[i], betaHat));
                        // Donor pool (capped for very large data) and nearest predicted values
                        const pool = obsIdx.length > 3000 ? Array.from({ length: 3000 }, () => Math.floor(rng() * obsIdx.length)) : obsIdx.map((_, k) => k);
                        missingIdx[c].forEach(i => {
                            const target = pred(Xall[i], betaStar);
                            const best = [];
                            pool.forEach(k => {
                                const d = obsPred[k].reduce((a, v, j) => a + (v - target[j]) ** 2, 0);
                                if (best.length < donors) { best.push({ k, d }); best.sort((a, b) => a.d - b.d); }
                                else if (d < best[donors - 1].d) { best[donors - 1] = { k, d }; best.sort((a, b) => a.d - b.d); }
                            });
                            const pick = best[Math.floor(rng() * best.length)];
                            data[i][c] = rows[obsIdx[pick.k]][c];
                        });
                        yield [l + 1, m, it + 1, iterations];
                    }
                }
                datasets.push(data);
            }
            return { datasets, imputedColumns: targets, m, iterations };
        },

        /**
         * Multiple imputation for PREDICTION models (MICE with predictive mean matching), fitted on the training
         * patients only. Unlike miceImpute (used for the explanatory odds ratios), the outcome is NOT used in
         * the imputation models, so that test patients and new patients, whose outcome is unknown, are imputed
         * in exactly the same way as the training patients. Returns the m completed training datasets and a
         * fitted "imputer" (one set of imputation models per completed dataset, for every predictor) that
         * miceApplyRow uses to impute any new patient.
         * `chains` (optional) computes only those imputations (0-based): each is seeded on its own, so imputations
         * computed separately (e.g. in parallel workers) and merged in order equal a single run (mergeMiceParts).
         */
        miceFit(rows, columnConfigs, { m = 5, iterations = 10, seed = 2024, donors = 5, maxDonors = 500, maxFitRows = 2000, maxSearch = 1000, onProgress = null, chains: chainList = null } = {}) {
            this.assertNoOutcomeAsPredictor(columnConfigs); // the outcome must never inform the imputation
            const cols = Object.keys(columnConfigs).filter(c => columnConfigs[c].role === 'feature');
            const numeric = {}, levels = {}, start = {};
            cols.forEach(c => {
                const obs = rows.filter(r => !this.isMissing(r[c]));
                numeric[c] = columnConfigs[c].type !== 'nominal' && obs.every(r => !isNaN(Number(r[c])));
                if (!numeric[c]) levels[c] = [...new Set(obs.map(r => String(r[c]).trim()))].sort();
                if (numeric[c]) { const v = obs.map(r => Number(r[c])).sort((a, b) => a - b); start[c] = v.length ? v[Math.floor(v.length / 2)] : 0; }
                else { const cnt = {}; obs.forEach(r => { const k = String(r[c]).trim(); cnt[k] = (cnt[k] || 0) + 1; }); start[c] = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a] || (a < b ? -1 : 1))[0] ?? ''; }
            });
            const n = rows.length;
            const missingIdx = {};
            cols.forEach(c => { missingIdx[c] = rows.map((r, i) => (this.isMissing(r[c]) ? i : -1)).filter(i => i >= 0); });
            const usable = cols.filter(c => missingIdx[c].length < n);            // has observed values
            const incomplete = usable.filter(c => missingIdx[c].length > 0);
            const ridge = this._ridgeSolve;
            // Each column read from text once: numbers, or the index of the category (-1 = missing)
            const coded = {};
            usable.forEach(c => {
                const a = new Float64Array(n);
                rows.forEach((r, i) => {
                    if (this.isMissing(r[c])) a[i] = NaN;
                    else a[i] = numeric[c] ? Number(r[c]) : levels[c].indexOf(String(r[c]).trim());
                });
                coded[c] = a;
            });
            const obsOf = {};
            usable.forEach(c => { const mis = new Set(missingIdx[c]); obsOf[c] = []; for (let i = 0; i < n; i++) if (!mis.has(i)) obsOf[c].push(i); });
            const respOf = c => (numeric[c] ? (i => [coded[c][i]]) : (i => levels[c].map((_, k) => (coded[c][i] === k ? 1 : 0))));
            // Design matrices are stored flat (row i at i * p) in typed arrays: the same arithmetic in the same
            // order as row arrays, so the results are unchanged, with far less memory traffic
            const fitModel = (cur, c, rng, obsIdxAll) => {
                // Imputation model for c on all other usable predictors (current completed values)
                const others = usable.filter(o => o !== c);
                const stats = {};
                others.forEach(o => {
                    if (numeric[o]) {
                        const v = cur[o];
                        let mu = 0; for (let i = 0; i < n; i++) mu += v[i];
                        mu /= n;
                        let ss = 0; for (let i = 0; i < n; i++) ss += (v[i] - mu) ** 2;
                        stats[o] = { mu, sd: Math.sqrt(ss / n) || 1 };
                    }
                });
                let p = 1;
                others.forEach(o => { p += numeric[o] ? 1 : levels[o].length - 1; });
                const Xall = new Float64Array(n * p);
                for (let i = 0; i < n; i++) Xall[i * p] = 1;
                let col = 1;
                others.forEach(o => {
                    const v = cur[o];
                    if (numeric[o]) {
                        const { mu, sd } = stats[o];
                        for (let i = 0; i < n; i++) Xall[i * p + col] = (v[i] - mu) / sd;
                        col++;
                    } else {
                        for (let k = 1; k < levels[o].length; k++, col++) for (let i = 0; i < n; i++) Xall[i * p + col] = v[i] === k ? 1 : 0;
                    }
                });
                // Large cohorts: each imputation model is fitted on a random subset of the observed rows
                // (results are practically unchanged; cohorts below maxFitRows use every row)
                let obsIdx = obsIdxAll;
                if (obsIdx.length > maxFitRows) {
                    const pick = obsIdx.slice();
                    for (let k = 0; k < maxFitRows; k++) { const j = k + Math.floor(rng() * (pick.length - k)); [pick[k], pick[j]] = [pick[j], pick[k]]; }
                    obsIdx = pick.slice(0, maxFitRows);
                }
                const resp = respOf(c);
                const lam = 1e-3 * obsIdx.length;
                const betaHat = ridge(Xall, p, obsIdx, resp, lam);
                const boot = obsIdx.map(() => obsIdx[Math.floor(rng() * obsIdx.length)]);
                const betaStar = ridge(Xall, p, boot, resp, lam);
                return { spec: { c, others, stats }, Xall, p, betaHat, betaStar, obsIdx };
            };
            // Prediction for row i of a flat design matrix
            const pred = (f, i, B) => {
                const q = B[0].length, out = new Array(q).fill(0), X = f.Xall, base = i * f.p;
                for (let j = 0; j < f.p; j++) { const xv = X[base + j]; if (xv === 0) continue; const Bj = B[j]; for (let k = 0; k < q; k++) out[k] += xv * Bj[k]; }
                return out;
            };
            // The `donors` closest predictions (ties: earlier first), one chosen at random. Kept sorted by
            // insertion, which orders them exactly as sorting after every change did
            const bestK = new Int32Array(donors), bestD = new Float64Array(donors);
            const nearest = (target, preds, rng) => {
                let len = 0;
                const q = target.length;
                for (let k = 0; k < preds.length; k++) {
                    const pv = preds[k];
                    let d = 0;
                    for (let j = 0; j < q; j++) d += (pv[j] - target[j]) ** 2;
                    if (len === donors && !(d < bestD[donors - 1])) continue;
                    let pos = len < donors ? len++ : donors - 1;
                    while (pos > 0 && bestD[pos - 1] > d) { bestD[pos] = bestD[pos - 1]; bestK[pos] = bestK[pos - 1]; pos--; }
                    bestD[pos] = d; bestK[pos] = k;
                }
                return bestK[Math.floor(rng() * len)];
            };
            // Same result as nearest() for one-dimensional predictions (numeric predictors), but by binary search
            // in the sorted predictions: the closest `donors` (ties by position, as above), one chosen at random
            const nearestSorted = preds => {
                const order = preds.map((_, k) => k).sort((a, b) => preds[a][0] - preds[b][0] || a - b);
                const vals = order.map(k => preds[k][0]);
                return (target, rng) => {
                    const t = target[0];
                    let lo = 0, hi = vals.length;
                    while (lo < hi) { const mid = (lo + hi) >> 1; if (vals[mid] < t) lo = mid + 1; else hi = mid; }
                    let L = lo - 1, R = lo;
                    const cand = [];
                    while (cand.length < donors && (L >= 0 || R < vals.length)) {
                        if (R >= vals.length || (L >= 0 && t - vals[L] <= vals[R] - t)) cand.push(order[L--]); else cand.push(order[R++]);
                    }
                    const dist = k => (preds[k][0] - t) ** 2;
                    const worst = Math.max(...cand.map(dist));
                    while (L >= 0 && dist(order[L]) <= worst) cand.push(order[L--]);
                    while (R < vals.length && dist(order[R]) <= worst) cand.push(order[R++]);
                    cand.sort((a, b) => dist(a) - dist(b) || a - b);
                    const best = cand.slice(0, donors);
                    return best[Math.floor(rng() * best.length)];
                };
            };

            const datasets = [], chains = [], trace = []; // trace: per imputation, each numeric column's mean imputed value after every round
            for (const l of (chainList || Array.from({ length: m }, (_, k) => k))) {
                const rng = this.createPRNG(seed + 1009 * l);
                const cur = {}, src = {}; // current values, and the row each imputed value was taken from
                usable.forEach(c => { cur[c] = Float64Array.from(coded[c]); src[c] = {}; });
                incomplete.forEach(c => {
                    const obs = obsOf[c];
                    missingIdx[c].forEach(i => { const d = obs[Math.floor(rng() * obs.length)]; cur[c][i] = coded[c][d]; src[c][i] = d; });
                });
                const traceL = {};
                incomplete.forEach(c => { if (numeric[c]) traceL[c] = []; });
                for (let it = 0; it < iterations; it++) {
                    incomplete.forEach(c => {
                        const f = fitModel(cur, c, rng, obsOf[c]);
                        // Donors for predictive mean matching (a random subset of maxSearch in large cohorts)
                        const pool = f.obsIdx.length > maxSearch ? Array.from({ length: maxSearch }, () => f.obsIdx[Math.floor(rng() * f.obsIdx.length)]) : f.obsIdx;
                        const poolPred = pool.map(i => pred(f, i, f.betaHat));
                        const find = numeric[c] ? nearestSorted(poolPred) : (target => nearest(target, poolPred, rng));
                        missingIdx[c].forEach(i => { const d = pool[find(pred(f, i, f.betaStar), rng)]; cur[c][i] = coded[c][d]; src[c][i] = d; });
                    });
                    // Convergence diagnostic: mean of the imputed values of each numeric column after this round
                    Object.keys(traceL).forEach(c => { const idx = missingIdx[c]; let sm = 0; for (const i of idx) sm += cur[c][i]; traceL[c].push(Number((sm / idx.length).toPrecision(6))); });
                    if (onProgress) onProgress(l + 1, m, it + 1, iterations);
                }
                trace.push(traceL);
                // The completed dataset: imputed cells take the donor's original text value
                const data = rows.map(r => ({ ...r }));
                incomplete.forEach(c => missingIdx[c].forEach(i => { data[i][c] = rows[src[c][i]][c]; }));
                cols.filter(c => missingIdx[c].length === n).forEach(c => missingIdx[c].forEach(i => { data[i][c] = start[c]; }));
                datasets.push(data);
                // Imputation models kept for new patients: every usable predictor, fitted on the completed data
                const models = {};
                usable.forEach(c => {
                    const f = fitModel(cur, c, rng, obsOf[c]);
                    const take = f.obsIdx.length > maxDonors ? Array.from({ length: maxDonors }, () => Math.floor(rng() * f.obsIdx.length)) : f.obsIdx.map((_, k) => k);
                    const r4 = v => Number(v.toPrecision(6));
                    models[c] = {
                        others: f.spec.others, stats: f.spec.stats, beta: f.betaStar.map(row => row.map(r4)),
                        donorPred: take.map(k => pred(f, f.obsIdx[k], f.betaHat).map(r4)),
                        donorVal: take.map(k => rows[f.obsIdx[k]][c])
                    };
                });
                chains.push(models);
            }
            // Observed mean and SD of each traced column, to judge the size of any drift across rounds
            const traceScale = {};
            incomplete.filter(c => numeric[c]).forEach(c => {
                const v = obsOf[c].map(i => coded[c][i]), mu = v.reduce((a, b) => a + b, 0) / v.length;
                traceScale[c] = { mean: Number(mu.toPrecision(6)), sd: Number(Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(1, v.length - 1)).toPrecision(6)) };
            });
            return {
                datasets, imputedColumns: incomplete, trace, traceScale,
                imputer: { kind: 'mice', m, iterations: Math.min(iterations, 5), donors, columns: usable, numeric, levels, start, chains, seed }
            };
        },

        /** Joins miceFit results computed for separate imputations (in order) into one result */
        mergeMiceParts(parts) {
            const first = parts[0];
            return {
                datasets: [].concat(...parts.map(r => r.datasets)), imputedColumns: first.imputedColumns,
                trace: [].concat(...parts.map(r => r.trace || [])), traceScale: first.traceScale,
                imputer: { ...first.imputer, chains: [].concat(...parts.map(r => r.imputer.chains)) }
            };
        },

        /** Design row of an imputation model: intercept, standardised numbers, one-hot categories (first level as reference) */
        _miceDesign(r, spec, numeric, levels) {
            const x = [1];
            spec.others.forEach(o => {
                if (numeric[o]) x.push((Number(r[o]) - spec.stats[o].mu) / spec.stats[o].sd);
                else levels[o].slice(1).forEach(lv => x.push(String(r[o]).trim() === lv ? 1 : 0));
            });
            return x;
        },

        /**
         * Imputes one patient's missing predictors with imputation chain l of a fitted imputer (miceFit). The
         * choice among the closest donors is seeded from the patient's own values, so the same patient always
         * gets the same imputations. Returns a completed copy of the row (unchanged if nothing is missing).
         */
        miceApplyRow(row, imputer, l) {
            const miss = imputer.columns.filter(c => this.isMissing(row[c]));
            const out = { ...row };
            if (!miss.length) return out;
            let h = 2166136261 ^ (l + 1);
            const key = imputer.columns.map(c => String(row[c] ?? '')).join('|');
            for (let i = 0; i < key.length; i++) { h ^= key.charCodeAt(i); h = Math.imul(h, 16777619); }
            const rng = this.createPRNG(((h >>> 0) + (imputer.seed || 0)) >>> 0);
            miss.forEach(c => { out[c] = imputer.start[c]; });
            const models = imputer.chains[l];
            for (let it = 0; it < (imputer.iterations || 5); it++) {
                miss.forEach(c => {
                    const md = models[c];
                    if (!md || !md.donorVal.length) return;
                    const x = this._miceDesign(out, md, imputer.numeric, imputer.levels);
                    const target = md.beta[0].map((_, k) => x.reduce((a, xv, j) => a + xv * md.beta[j][k], 0));
                    const best = [];
                    md.donorPred.forEach((pv, k) => {
                        const d = pv.reduce((a, v, j) => a + (v - target[j]) ** 2, 0);
                        if (best.length < imputer.donors) { best.push({ k, d }); best.sort((a, b) => a.d - b.d); }
                        else if (d < best[imputer.donors - 1].d) { best[imputer.donors - 1] = { k, d }; best.sort((a, b) => a.d - b.d); }
                    });
                    out[c] = md.donorVal[best[Math.floor(rng() * best.length)].k];
                });
            }
            return out;
        },

        /**
         * Ridge regression by the normal equations (no penalty on the intercept); coefficients p x q. X is a flat
         * design matrix (row i at i * p), rows lists the rows to use (repeats allowed) and resp(i) gives row i's
         * response vector. With lam > 0 on every non-intercept term and at least one row, the matrix is positive
         * definite, so the tiny-pivot guard below cannot be reached in practice; it only prevents a division by zero.
         */
        _ridgeSolve(X, p, rows, resp, lam) {
            const q = resp(rows[0]).length;
            const A = new Float64Array(p * p), Bm = new Float64Array(p * q);
            for (let r = 0; r < rows.length; r++) {
                const i = rows[r], base = i * p, yi = resp(i);
                for (let a = 0; a < p; a++) {
                    const xa = X[base + a];
                    if (xa === 0) continue;
                    const row = a * p;
                    for (let b = a; b < p; b++) A[row + b] += xa * X[base + b];
                    for (let k = 0; k < q; k++) Bm[a * q + k] += xa * yi[k];
                }
            }
            for (let a = 0; a < p; a++) { for (let b = 0; b < a; b++) A[a * p + b] = A[b * p + a]; if (a > 0) A[a * p + a] += lam; }
            const M = Array.from({ length: p }, (_, i) => [...A.subarray(i * p, i * p + p), ...Bm.subarray(i * q, i * q + q)]);
            for (let c = 0; c < p; c++) {
                let piv = c;
                for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
                if (Math.abs(M[piv][c]) < 1e-12) { M[piv][c] = 1e-12; }
                [M[c], M[piv]] = [M[piv], M[c]];
                for (let r = 0; r < p; r++) {
                    if (r === c) continue;
                    const f = M[r][c] / M[c][c];
                    if (f) for (let k = c; k < p + q; k++) M[r][k] -= f * M[c][k];
                }
            }
            return M.map((r, i) => r.slice(p).map(v => v / r[i]));
        },

        /** Parses common date formats (ISO yyyy-mm-dd, dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, "15 Oct 2026"); null if not a date */
        parseDateValue(v) {
            if (v === null || v === undefined) return null;
            const s = String(v).trim();
            if (!s) return null;
            let m = s.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})/);
            if (m) { const t = Date.UTC(+m[1], +m[2] - 1, +m[3]); return isNaN(t) ? null : t; }
            m = s.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})/);
            if (m) { const t = Date.UTC(+m[3], +m[2] - 1, +m[1]); return isNaN(t) ? null : t; } // day first (UK)
            const t = Date.parse(s);
            return isNaN(t) ? null : t;
        },

        /**
         * Temporal split: the earliest records (by the date column) form the Training set and the most recent
         * ones the Testing set, mimicking use on future patients. Records sharing the cut-off date stay together;
         * records without a usable date are kept in the Training set and counted.
         */
        trainTestSplitTemporal(rows, y, dateValues, trainRatio = 0.8) {
            const dated = [];
            const undated = [];
            dateValues.forEach((v, i) => {
                const t = this.parseDateValue(v);
                if (t === null) undated.push(i); else dated.push({ i, t });
            });
            dated.sort((a, b) => a.t - b.t || a.i - b.i);
            let nTrain = Math.round(dated.length * trainRatio);
            nTrain = Math.min(Math.max(nTrain, 1), Math.max(dated.length - 1, 1));
            // Do not cut through a single date
            while (nTrain < dated.length && nTrain > 0 && dated[nTrain].t === dated[nTrain - 1].t) nTrain++;
            const trainIndices = [...dated.slice(0, nTrain).map(d => d.i), ...undated];
            const testIndices = dated.slice(nTrain).map(d => d.i);
            const iso = t => new Date(t).toISOString().slice(0, 10);
            return {
                trainX: trainIndices.map(i => rows[i]), trainY: trainIndices.map(i => y[i]),
                testX: testIndices.map(i => rows[i]), testY: testIndices.map(i => y[i]),
                trainIndices, testIndices,
                info: {
                    method: 'time', undated: undated.length,
                    trainUntil: nTrain > 0 ? iso(dated[nTrain - 1].t) : null,
                    testFrom: testIndices.length ? iso(dated[nTrain].t) : null,
                    testUntil: testIndices.length ? iso(dated[dated.length - 1].t) : null
                }
            };
        },

        /**
         * Train / Test Split with deterministic random seed support
         */
        trainTestSplit(X, y, trainRatio = 0.8, isClassification = true, seed = null) {
            const total = X.length;
            const indices = Array.from({ length: total }, (_, i) => i);
            const rng = this.createPRNG(seed);

            // Deterministic seeded Fisher-Yates shuffle
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }

            let trainIndices = [];
            let testIndices = [];

            if (isClassification) {
                // Stratified split: partition by class
                const classIndices = {};
                indices.forEach(idx => {
                    const cls = y[idx];
                    if (!classIndices[cls]) classIndices[cls] = [];
                    classIndices[cls].push(idx);
                });

                Object.keys(classIndices).forEach(cls => {
                    const cIdxs = classIndices[cls];
                    let trainCount = Math.max(1, Math.round(cIdxs.length * trainRatio));
                    // Any class with >= 2 cases keeps at least one case in the test set
                    if (cIdxs.length >= 2 && trainCount >= cIdxs.length) trainCount = cIdxs.length - 1;
                    trainIndices.push(...cIdxs.slice(0, trainCount));
                    testIndices.push(...cIdxs.slice(trainCount));
                });
            } else {
                const trainCount = Math.round(total * trainRatio);
                trainIndices = indices.slice(0, trainCount);
                testIndices = indices.slice(trainCount);
            }

            // Ensure test has at least 1 sample if total >= 4
            if (testIndices.length === 0 && total >= 4) {
                testIndices.push(trainIndices.pop());
            }

            return {
                trainX: trainIndices.map(i => X[i]),
                trainY: trainIndices.map(i => y[i]),
                testX: testIndices.map(i => X[i]),
                testY: testIndices.map(i => y[i]),
                trainIndices,
                testIndices
            };
        },

        /**
         * Transforms a separate test dataset using training metadata,
         * extracting X_test and y_test strictly without data leakage.
         */
        prepareTestingDataset(testRows, prepMeta) {
            const { targetCol, isClassification, targetClassToIdx } = prepMeta;
            const X_test = [];
            const y_test = [];
            const validRows = [];
            let missingTargetCount = 0;
            let unknownClassCount = 0;
            const featureIssues = [];

            testRows.forEach(row => {
                const rawTarget = row[targetCol];
                if (this.isMissing(rawTarget)) {
                    missingTargetCount++;
                    return;
                }

                let targetVal;
                if (isClassification) {
                    const strTarget = String(rawTarget).trim();
                    if (targetClassToIdx[strTarget] === undefined) {
                        unknownClassCount++;
                        return;
                    }
                    targetVal = targetClassToIdx[strTarget];
                } else {
                    targetVal = Number(rawTarget);
                    if (isNaN(targetVal)) {
                        missingTargetCount++;
                        return;
                    }
                }

                const xRow = this.transformSingleRow(row, prepMeta, featureIssues);
                X_test.push(xRow);
                y_test.push(targetVal);
                validRows.push(row);
            });

            return {
                X: X_test,
                y: y_test,
                validRows,
                totalEvaluated: X_test.length,
                totalProvided: testRows.length,
                missingTargetCount,
                unknownClassCount,
                // Predictor cells that were missing/unrecognised and filled from training statistics
                imputedFeatureCells: featureIssues.length
            };
        },

        /**
         * =========================================================================
         * 7-STEP PIPELINE EXTENSIONS
         * Strictly ordered, leak-free statistical & machine learning pipeline
         * Conceived & designed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
         * =========================================================================
         */

        /**
         * Fits an imputer strictly on the training set only (zero data leakage)
         */
        fitStrictImputer(trainRows, featureCols, targetCol, columnConfigs, strategy = 'median') {
            const fillValues = {};
            const columnStats = {};

            // Predictors only: the outcome (targetCol) is never imputed. Rows with a missing
            // outcome must be excluded before calling this (see executePipelineStep1).
            const allCols = featureCols.filter(c => c !== targetCol);

            allCols.forEach(col => {
                const conf = columnConfigs[col] || { type: 'continuous' };
                const validVals = [];
                const validNums = [];

                trainRows.forEach(r => {
                    const val = r[col];
                    if (!this.isMissing(val)) {
                        validVals.push(val);
                        const n = Number(val);
                        if (!isNaN(n)) validNums.push(n);
                    }
                });

                let fillVal = 0;
                const isTextOrdinal = conf.type === 'ordinal' && validVals.length > 0 && validNums.length < validVals.length;
                if (conf.type === 'nominal' || isTextOrdinal) {
                    // Mode for nominal
                    const freq = {};
                    let maxCount = 0;
                    let topVal = 'Unknown';
                    validVals.forEach(v => {
                        const s = String(v).trim();
                        freq[s] = (freq[s] || 0) + 1;
                        if (freq[s] > maxCount) {
                            maxCount = freq[s];
                            topVal = s;
                        }
                    });
                    fillVal = topVal;
                    columnStats[col] = { mode: topVal, distinctCount: Object.keys(freq).length };
                } else {
                    // Continuous / numeric Ordinal: Median or Mean (ordinal always uses the median, rounded to a valid level)
                    if (validNums.length > 0) {
                        const sum = validNums.reduce((a, b) => a + b, 0);
                        const mean = sum / validNums.length;
                        const sorted = [...validNums].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
                        
                        if (conf.type === 'ordinal') {
                            // Lower median keeps the fill value on an observed ordinal level
                            fillVal = sorted[Math.floor((sorted.length - 1) / 2)];
                        } else {
                            fillVal = (strategy === 'mean') ? mean : median; // full precision
                        }
                        // Descriptive SD shown to the user: sample SD (n − 1), as in the Step 2 table
                        const variance = validNums.length > 1
                            ? validNums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (validNums.length - 1)
                            : 0;
                        columnStats[col] = {
                            // Display statistics: 6 significant figures (scale-free)
                            mean: Number(mean.toPrecision(6)),
                            median: Number(median.toPrecision(6)),
                            std: Number(Math.sqrt(variance).toPrecision(6)),
                            min: sorted[0],
                            max: sorted[sorted.length - 1],
                            validCount: validNums.length
                        };
                    } else {
                        fillVal = 0;
                        columnStats[col] = { mean: 0, median: 0, std: 1, min: 0, max: 0, validCount: 0 };
                    }
                }

                fillValues[col] = fillVal;
            });

            return {
                fillValues,
                columnStats,
                strategy,
                fittedSampleCount: trainRows.length
            };
        },

        /**
         * Transforms dataset rows using a pre-fitted strict imputer
         */
        transformWithStrictImputer(rows, imputer, columnConfigs) {
            let missingFilledCount = 0;
            const imputedRows = rows.map(r => {
                const newRow = { ...r };
                Object.keys(imputer.fillValues).forEach(col => {
                    if (this.isMissing(newRow[col])) {
                        newRow[col] = imputer.fillValues[col];
                        missingFilledCount++;
                    } else {
                        const conf = columnConfigs[col];
                        if (conf && (conf.type === 'continuous' || conf.type === 'ordinal')) {
                            const n = Number(newRow[col]);
                            if (!isNaN(n)) newRow[col] = n;
                        }
                    }
                });
                return newRow;
            });

            return {
                rows: imputedRows,
                missingFilledCount
            };
        },

        /**
         * Computes sample skewness to test for normality
         * g1 = [n / ((n-1)(n-2))] * sum(((x - mean)/s)^3)
         */
        calculateSkewness(arr) {
            const nums = arr.map(Number).filter(n => !isNaN(n));
            const n = nums.length;
            if (n < 3) return { skewness: 0, isNormal: true, mean: 0, std: 1 };

            const mean = nums.reduce((a, b) => a + b, 0) / n;
            const s2 = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
            const s = Math.sqrt(s2);
            if (s === 0) return { skewness: 0, isNormal: true, mean, std: 0 };

            const m3 = nums.reduce((a, b) => a + Math.pow(b - mean, 3), 0);
            const g1 = (n * m3) / ((n - 1) * (n - 2) * Math.pow(s, 3));
            const isNormal = Math.abs(g1) <= 1.0;

            return {
                skewness: Number(g1.toFixed(3)),
                isNormal,
                // Display values: significant figures, so small-unit measurements stay readable
                mean: Number(mean.toPrecision(4)),
                std: Number(s.toPrecision(4))
            };
        },

        testNormality(arr) {
            return this.calculateSkewness(arr);
        },

        /**
         * Pearson Product-Moment Correlation Coefficient
         */
        calculatePearson(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;
            const xMean = x.reduce((a, b) => a + b, 0) / n;
            const yMean = y.reduce((a, b) => a + b, 0) / n;

            let num = 0;
            let denX = 0;
            let denY = 0;

            for (let i = 0; i < n; i++) {
                const dx = x[i] - xMean;
                const dy = y[i] - yMean;
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
            }

            const den = Math.sqrt(denX * denY);
            if (den === 0) return 0;
            return num / den; // full precision (feeds t-tests and the |r| threshold); round only for display
        },

        /**
         * Spearman Rank-Order Correlation Coefficient
         */
        calculateSpearman(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;

            function getRanks(arr) {
                const indexed = arr.map((val, idx) => ({ val, idx }));
                indexed.sort((a, b) => a.val - b.val);
                const ranks = new Array(n);
                let i = 0;
                while (i < n) {
                    let j = i;
                    while (j < n - 1 && indexed[j + 1].val === indexed[j].val) {
                        j++;
                    }
                    const avgRank = (i + j + 2) / 2; // 1-based average rank
                    for (let k = i; k <= j; k++) {
                        ranks[indexed[k].idx] = avgRank;
                    }
                    i = j + 1;
                }
                return ranks;
            }

            const xRanks = getRanks(x);
            const yRanks = getRanks(y);
            return this.calculatePearson(xRanks, yRanks);
        },

        /**
         * Checks pairwise continuous variable correlation using Training set
         * Applies Pearson when both are normal; Spearman when one/both are skewed
         */
        checkPairwiseCorrelations(trainRows, continuousCols, threshold = 0.8, columnConfigs = {}) {
            const normalityMap = {};
            const toNum = {};
            continuousCols.forEach(col => {
                toNum[col] = this.numericValueFn(trainRows, col, columnConfigs[col] || {});
                const vals = trainRows.map(r => toNum[col](r[col])).filter(n => !isNaN(n));
                normalityMap[col] = this.calculateSkewness(vals);
            });

            const pairs = [];
            for (let i = 0; i < continuousCols.length; i++) {
                for (let j = i + 1; j < continuousCols.length; j++) {
                    const cA = continuousCols[i];
                    const cB = continuousCols[j];
                    // Pairwise-complete observations only
                    const valsA = [];
                    const valsB = [];
                    trainRows.forEach(r => {
                        const a = toNum[cA](r[cA]);
                        const b = toNum[cB](r[cB]);
                        if (!isNaN(a) && !isNaN(b)) {
                            valsA.push(a);
                            valsB.push(b);
                        }
                    });

                    const normA = normalityMap[cA];
                    const normB = normalityMap[cB];
                    const bothNormal = normA.isNormal && normB.isNormal;
                    const method = bothNormal ? 'Pearson (r)' : 'Spearman (ρ)';
                    const coeff = bothNormal ? this.calculatePearson(valsA, valsB) : this.calculateSpearman(valsA, valsB);
                    const absCoeff = Math.abs(coeff);

                    pairs.push({
                        var1: cA,
                        var2: cB,
                        method,
                        coeff,
                        absCoeff,
                        norm1: normA,
                        norm2: normB,
                        bothNormal,
                        isHigh: absCoeff >= threshold
                    });
                }
            }

            // Sort descending by absolute correlation
            pairs.sort((a, b) => b.absCoeff - a.absCoeff);

            return {
                normalityMap,
                allPairs: pairs,
                highPairs: pairs.filter(p => p.isHigh)
            };
        },

        /**
         * Matrix Inversion using Gauss-Jordan elimination with partial pivoting
         */
        invertMatrix(A) {
            const n = A.length;
            const M = A.map((row, i) => {
                const r = [...row];
                const id = new Array(n).fill(0);
                id[i] = 1;
                return r.concat(id);
            });

            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
                }
                [M[i], M[maxRow]] = [M[maxRow], M[i]];

                const pivot = M[i][i];
                if (Math.abs(pivot) < 1e-12) return null; // Singular matrix
                for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;

                for (let k = 0; k < n; k++) {
                    if (k === i) continue;
                    const factor = M[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        M[k][j] -= factor * M[i][j];
                    }
                }
            }
            return M.map(row => row.slice(n));
        },

        /**
         * Computes Variance Inflation Factor (VIF) on Training set features
         * One-hot encodes categoricals (dropping reference column) and computes R^2
         */
        calculateVIF(trainRows, featureCols, columnConfigs) {
            // Build design matrix with reference category dropped for nominal features
            const encodedCols = [];

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf && conf.type === 'nominal') {
                    const distinct = Array.from(new Set(trainRows.map(r => String(r[col])))).sort();
                    // Drop reference column (first category) to avoid dummy trap
                    const nonRef = distinct.slice(1);
                    nonRef.forEach(cat => {
                        const dummyName = `${col}_is_${cat}`;
                        encodedCols.push({
                            name: dummyName,
                            originalCol: col,
                            type: 'dummy',
                            extract: (r) => String(r[col]) === cat ? 1 : 0
                        });
                    });
                } else {
                    // Continuous or Ordinal (text ordinal levels -> ordered ranks)
                    const toNum = this.numericValueFn(trainRows, col, conf || {});
                    encodedCols.push({
                        name: col,
                        originalCol: col,
                        type: 'numeric',
                        extract: (r) => {
                            const n = toNum(r[col]);
                            return isNaN(n) ? 0 : n;
                        }
                    });
                }
            });

            if (encodedCols.length < 2) {
                return featureCols.map(col => ({
                    feature: col,
                    vif: 1.0,
                    rSquared: 0,
                    highVIF: false,
                    dummyCount: 1
                }));
            }

            const N = trainRows.length;
            const P = encodedCols.length;
            // The correlation matrix is P×P: refuse sizes that would freeze or exhaust the browser
            if (P > MAX_VIF_COLUMNS) {
                const counts = {};
                encodedCols.forEach(ec => { counts[ec.originalCol] = (counts[ec.originalCol] || 0) + 1; });
                const worst = Object.entries(counts).filter(([, n]) => n > 50).sort((a, b) => b[1] - a[1])
                    .map(([c, n]) => `"${c}" (${(n + 1).toLocaleString('en-GB')} categories)`);
                throw new Error(`VIF would need a ${P.toLocaleString('en-GB')} × ${P.toLocaleString('en-GB')} correlation matrix (limit ${MAX_VIF_COLUMNS}). ` +
                    (worst.length ? `High-cardinality categorical feature(s): ${worst.join(', ')}. ` : '') +
                    'Set identifier-like columns to "Meta / ID" in Step 2, or exclude them in this step.');
            }

            // Extract numeric values and standardize columns
            const X_raw = Array.from({ length: N }, () => new Array(P).fill(0));
            for (let j = 0; j < P; j++) {
                const extractor = encodedCols[j].extract;
                let sum = 0;
                for (let i = 0; i < N; i++) {
                    const val = extractor(trainRows[i]);
                    X_raw[i][j] = val;
                    sum += val;
                }
                const mean = sum / N;
                let varSum = 0;
                for (let i = 0; i < N; i++) varSum += Math.pow(X_raw[i][j] - mean, 2);
                const std = Math.sqrt(varSum / N) || 1;
                for (let i = 0; i < N; i++) {
                    X_raw[i][j] = (X_raw[i][j] - mean) / std;
                }
            }

            // Generalised VIF (Fox & Monette, 1992) for each ORIGINAL feature, computed from the
            // correlation matrix R of the encoded columns (standardised, so the intercept is implicit and
            // unpenalised): GVIF = det(R_SS)·det(R_OO) / det(R), where S = the feature's columns (one, or
            // its k−1 dummies) and O = all other columns. For a single column this equals the classical VIF
            // 1/(1 − R²). For multi-level nominal features GVIF measures their JOINT collinearity (the old
            // "largest single-dummy VIF" understated it); it is reported on the VIF scale as GVIF^(1/df),
            // i.e. (GVIF^(1/(2·df)))², so the same > 5 threshold applies.
            const R = Array.from({ length: P }, () => new Array(P).fill(0));
            for (let i = 0; i < N; i++) {
                const row = X_raw[i];
                for (let a2 = 0; a2 < P; a2++) {
                    const va = row[a2];
                    if (va === 0) continue;
                    for (let b2 = a2; b2 < P; b2++) R[a2][b2] += va * row[b2];
                }
            }
            for (let a2 = 0; a2 < P; a2++) {
                for (let b2 = a2; b2 < P; b2++) {
                    R[a2][b2] /= N;
                    R[b2][a2] = R[a2][b2];
                }
                // Tiny ridge keeps determinants finite under exact collinearity (scale-free: R has unit diagonal)
                R[a2][a2] += 1e-6;
            }

            const logDet = (M) => {
                const n = M.length;
                if (n === 0) return 0;
                const A = M.map(r => r.slice());
                let ld = 0;
                for (let c = 0; c < n; c++) {
                    let piv = c;
                    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
                    if (Math.abs(A[piv][c]) < 1e-300) return -Infinity;
                    if (piv !== c) [A[piv], A[c]] = [A[c], A[piv]];
                    ld += Math.log(Math.abs(A[c][c]));
                    for (let r = c + 1; r < n; r++) {
                        const f = A[r][c] / A[c][c];
                        if (f === 0) continue;
                        for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
                    }
                }
                return ld;
            };
            const sub = (M, idx) => idx.map(i => idx.map(j => M[i][j]));
            // det(R_OO)/det(R) = det((R⁻¹)_SS), so GVIF = det(R_SS)·det((R⁻¹)_SS): one P×P inversion instead
            // of a determinant of the (P−df)×(P−df) block for every feature (identical result, O(P³) total)
            const Rinv = this.invertMatrix(R);
            const logDetAll = Rinv ? null : logDet(R);

            const byFeature = {};
            encodedCols.forEach((ec, j) => {
                (byFeature[ec.originalCol] = byFeature[ec.originalCol] || []).push(j);
            });

            const results = Object.keys(byFeature).map(col => {
                const S = byFeature[col];
                const df = S.length;
                let logGVIF;
                if (Rinv) {
                    logGVIF = logDet(sub(R, S)) + logDet(sub(Rinv, S));
                } else {
                    const O = [];
                    for (let j = 0; j < P; j++) if (!S.includes(j)) O.push(j);
                    logGVIF = logDet(sub(R, S)) + logDet(sub(R, O)) - logDetAll;
                }
                let gvif = Math.exp(logGVIF);
                if (!isFinite(gvif) || gvif < 1) gvif = isFinite(gvif) ? 1 : 999;
                let vifScale = Math.pow(gvif, 1 / df);
                if (!isFinite(vifScale) || vifScale > 999) vifScale = 999;
                return {
                    feature: col,
                    vif: Number(vifScale.toFixed(2)),               // VIF scale: GVIF^(1/df)
                    gvif: Number(Math.min(gvif, 1e6).toFixed(2)),
                    adjustedGVIF: Number(Math.sqrt(vifScale).toFixed(3)), // GVIF^(1/(2·df))
                    rSquared: Number((1 - 1 / vifScale).toFixed(4)),     // equivalent R² on the VIF scale
                    type: columnConfigs[col]?.type || 'continuous',
                    dummyCount: df,
                    highVIF: vifScale > 5.0
                };
            });
            results.sort((a, b) => b.vif - a.vif);
            return results;
        },

        /**
         * Accurate Lanczos approximation for log(Gamma(x))
         */
        logGamma(x) {
            const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
            let y = x;
            let tmp = x + 5.5;
            tmp -= (x + 0.5) * Math.log(tmp);
            let ser = 1.000000000190015;
            for (let j = 0; j <= 5; j++) ser += cof[j] / ++y;
            return -tmp + Math.log(2.5066282746310005 * ser / x);
        },

        /**
         * Incomplete Gamma function P(a, x) for Chi-Square p-values
         */
        gammp(a, x) {
            if (x <= 0 || a <= 0) return 0;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            if (x < a + 1) {
                // Series representation
                let sum = 1 / a, del = sum, ap = a;
                for (let n = 1; n <= ITMAX; n++) {
                    ap += 1;
                    del *= x / ap;
                    sum += del;
                    if (Math.abs(del) < Math.abs(sum) * EPS) {
                        return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
                    }
                }
                return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
            } else {
                // Continued fraction representation
                let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
                for (let i = 1; i <= ITMAX; i++) {
                    const an = -i * (i - a);
                    b += 2;
                    d = an * d + b;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = b + an / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return 1 - (Math.exp(-x + a * Math.log(x) - this.logGamma(a)) * h);
            }
        },

        chiSquarePValue(chiSq, df) {
            if (chiSq <= 0 || df <= 0) return 1.0;
            const p = 1 - this.gammp(df / 2, chiSq / 2);
            return Math.max(0, Math.min(1, p));
        },

        /**
         * Incomplete Beta function I_x(a, b) for ANOVA / F-test p-values
         */
        betai(a, b, x) {
            if (x <= 0) return 0;
            if (x >= 1) return 1;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            const bt = Math.exp(this.logGamma(a + b) - this.logGamma(a) - this.logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));

            function betacf(aa, bb, xx) {
                const qab = aa + bb, qap = aa + 1, qam = aa - 1;
                let c = 1, d = 1 - qab * xx / qap;
                if (Math.abs(d) < FPMIN) d = FPMIN;
                d = 1 / d;
                let h = d;
                for (let m = 1; m <= ITMAX; m++) {
                    const m2 = 2 * m;
                    let num = m * (bb - m) * xx / ((qam + m2) * (aa + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    h *= d * c;
                    num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return h;
            }

            if (x < (a + 1) / (a + b + 2)) {
                return (bt * betacf(a, b, x)) / a;
            } else {
                return 1 - ((bt * betacf(b, a, 1 - x)) / b);
            }
        },

        fTestPValue(fStat, df1, df2) {
            if (fStat <= 0 || df1 <= 0 || df2 <= 0) return 1.0;
            const x = df2 / (df2 + df1 * fStat);
            return Math.max(0, Math.min(1, this.betai(df2 / 2, df1 / 2, x)));
        },

        /**
         * Univariate statistical baseline tests (Step 4)
         * - ANOVA F-test for continuous features vs categorical target
         * - Chi-Square test for categorical features vs categorical target
         * Returns summary table sorted by p-value (lowest to highest)
         */
        runUnivariateTesting(rows, survivingCols, targetCol, columnConfigs) {
            const isTargetContinuous = columnConfigs[targetCol]?.type === 'continuous';
            const results = [];
            // Exclude rows with a missing outcome from every test
            rows = rows.filter(r => !this.isMissing(r[targetCol]) && (!isTargetContinuous || !isNaN(Number(r[targetCol]))));

            // One-way ANOVA on numeric values grouped by a categorical key
            const oneWayAnova = (pairs) => {
                const groups = {};
                pairs.forEach(({ g, v }) => {
                    if (!groups[g]) groups[g] = [];
                    groups[g].push(v);
                });
                const groupKeys = Object.keys(groups).filter(k => groups[k].length > 0);
                const K = groupKeys.length;
                const N = pairs.length;
                const grandMean = N > 0 ? pairs.reduce((a, p) => a + p.v, 0) / N : 0;
                let ssBetween = 0;
                let ssWithin = 0;
                groupKeys.forEach(k => {
                    const grp = groups[k];
                    const grpMean = grp.reduce((a, b) => a + b, 0) / grp.length;
                    ssBetween += grp.length * Math.pow(grpMean - grandMean, 2);
                    grp.forEach(v => { ssWithin += Math.pow(v - grpMean, 2); });
                });
                const df1 = K - 1;
                const df2 = N - K;
                if (df1 < 1 || df2 < 1) {
                    return { fStat: 0, pVal: 1, df1: Math.max(0, df1), df2: Math.max(0, df2) };
                }
                const msBetween = ssBetween / df1;
                const msWithin = ssWithin / df2;
                const fStat = msWithin > 0 ? (msBetween / msWithin) : (msBetween > 0 ? Infinity : 0);
                const pVal = fStat === Infinity ? 0 : this.fTestPValue(fStat, df1, df2);
                return { fStat, pVal, df1, df2 };
            };

            // Numeric value of a feature cell (ordinal text levels are converted to their ordered rank)
            const numericValueFn = (col) => this.numericValueFn(rows, col, columnConfigs[col] || {});

            const fmtStat = (v) => (v === Infinity ? Infinity : Number(v.toFixed(3)));

            survivingCols.forEach(col => {
                const isFeatureContinuous = columnConfigs[col]?.type === 'continuous' || columnConfigs[col]?.type === 'ordinal';

                if (!isTargetContinuous) {
                    // Categorical Target (e.g. Binary Classification)
                    const targetClasses = Array.from(new Set(rows.map(r => String(r[targetCol])))).sort();
                    const K = targetClasses.length;
                    const classIndex = new Map(targetClasses.map((c, i) => [c, i]));

                    if (isFeatureContinuous) {
                        // One-Way ANOVA F-test (equivalent to the two-sample t-test when K = 2: F = t²)
                        const toNum = numericValueFn(col);
                        const pairs = [];
                        rows.forEach(r => {
                            const v = toNum(r[col]);
                            if (!isNaN(v)) pairs.push({ g: String(r[targetCol]), v });
                        });
                        const a = oneWayAnova(pairs);

                        results.push({
                            feature: col,
                            scale: columnConfigs[col]?.type === 'ordinal' ? 'Ordinal (ranks treated as equally spaced numbers)' : 'Continuous',
                            testName: K === 2 ? "Two-Sample t-test (as ANOVA F = t²)" : "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: fmtStat(a.fStat),
                            df: `${a.df1}, ${a.df2}`,
                            pValue: Number(a.pVal.toPrecision(4)),
                            significant: a.pVal < 0.05
                        });
                    } else {
                        // Categorical vs Categorical: Chi-Square Test
                        const validRows = rows.filter(r => !this.isMissing(r[col]));
                        const categories = Array.from(new Set(validRows.map(r => String(r[col])))).sort();
                        const catIndex = new Map(categories.map((c, i) => [c, i]));
                        const R = categories.length;

                        // Contingency Table
                        const obs = Array.from({ length: R }, () => new Array(K).fill(0));
                        const rowTotals = new Array(R).fill(0);
                        const colTotals = new Array(K).fill(0);
                        let totalN = 0;

                        validRows.forEach(r => {
                            const rIdx = catIndex.has(String(r[col])) ? catIndex.get(String(r[col])) : -1;
                            const cIdx = classIndex.has(String(r[targetCol])) ? classIndex.get(String(r[targetCol])) : -1;
                            if (rIdx !== -1 && cIdx !== -1) {
                                obs[rIdx][cIdx]++;
                                rowTotals[rIdx]++;
                                colTotals[cIdx]++;
                                totalN++;
                            }
                        });

                        let chiSq = 0;
                        let lowExpected = 0;
                        for (let i = 0; i < R; i++) {
                            for (let j = 0; j < K; j++) {
                                const expected = (rowTotals[i] * colTotals[j]) / (totalN || 1);
                                if (expected < 5) lowExpected++;
                                if (expected > 0) {
                                    chiSq += Math.pow(obs[i][j] - expected, 2) / expected;
                                }
                            }
                        }

                        const df = (R - 1) * (K - 1);
                        const pVal = df >= 1 ? this.chiSquarePValue(chiSq, df) : 1;
                        const cells = R * K;

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: "Pearson Chi-Square (χ²)" + (lowExpected > 0 && cells > 0 && lowExpected / cells > 0.2
                                ? ' ⚠️ >20% of expected counts < 5: approximation unreliable' : ''),
                            statisticLabel: 'χ² stat',
                            statisticValue: Number(chiSq.toFixed(3)),
                            df: `${df}`,
                            pValue: Number(pVal.toPrecision(4)),
                            significant: pVal < 0.05
                        });
                    }
                } else {
                    // Continuous Target (Regression)
                    if (isFeatureContinuous) {
                        const toNum = numericValueFn(col);
                        const valsX = [];
                        const valsY = [];
                        rows.forEach(r => {
                            const x = toNum(r[col]);
                            const y = Number(r[targetCol]);
                            if (!isNaN(x) && !isNaN(y)) {
                                valsX.push(x);
                                valsY.push(y);
                            }
                        });
                        const n = valsX.length;
                        const rCoeff = this.calculatePearson(valsX, valsY);
                        let pVal = 1;
                        let tStat = 0;
                        if (n > 2) {
                            if (Math.abs(rCoeff) >= 1) {
                                tStat = rCoeff > 0 ? Infinity : -Infinity;
                                pVal = 0;
                            } else {
                                tStat = rCoeff * Math.sqrt((n - 2) / (1 - rCoeff * rCoeff));
                                pVal = this.fTestPValue(tStat * tStat, 1, n - 2);
                            }
                        }

                        results.push({
                            feature: col,
                            scale: columnConfigs[col]?.type === 'ordinal' ? 'Ordinal (ranks treated as equally spaced numbers)' : 'Continuous',
                            testName: "Pearson Linear Correlation (t-test)",
                            statisticLabel: `r (t = ${Number.isFinite(tStat) ? tStat.toFixed(2) : '∞'})`,
                            statisticValue: Number(rCoeff.toFixed(3)),
                            df: `${Math.max(0, n - 2)}`,
                            pValue: Number(pVal.toPrecision(4)),
                            significant: pVal < 0.05
                        });
                    } else {
                        // Categorical feature vs Continuous Target: One-Way ANOVA
                        const pairs = [];
                        rows.forEach(r => {
                            if (!this.isMissing(r[col])) pairs.push({ g: String(r[col]), v: Number(r[targetCol]) });
                        });
                        const a = oneWayAnova(pairs);

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: a.df1 === 1 ? "Two-Sample t-test (as ANOVA F = t²)" : "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: fmtStat(a.fStat),
                            df: `${a.df1}, ${a.df2}`,
                            pValue: Number(a.pVal.toPrecision(4)),
                            significant: a.pVal < 0.05
                        });
                    }
                }
            });

            // Sort ascending by p-value (lowest to highest) as specified in Step 4
            results.sort((a, b) => a.pValue - b.pValue);
            return results;
        },

        /**
         * Step 5: Imbalance Handling using SMOTENC
         * Applied strictly to the Training set ONLY when the cohort is imbalanced:
         * the smallest class is < 30% of the training set for a binary outcome, or, for
         * multi-class outcomes, smaller than 30/70 (≈ 43%) of the largest class (the same
         * ratio as the binary 30% rule). Every under-represented class is oversampled up to
         * the size of the largest class.
         */
        applySMOTENC(trainRows, featureCols, targetCol, columnConfigs, seed = 42) {
            const isClassification = columnConfigs[targetCol]?.type !== 'continuous';
            if (!isClassification) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    reason: 'Target is continuous (Regression); SMOTENC applies to classification.'
                };
            }

            const classCounts = {};
            trainRows.forEach(r => {
                const c = String(r[targetCol]);
                classCounts[c] = (classCounts[c] || 0) + 1;
            });

            const classes = Object.keys(classCounts);
            if (classes.length < 2) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    classCounts,
                    reason: 'Single class present in training data.'
                };
            }

            const total = trainRows.length;
            let minorityClass = classes[0];
            let majorityClass = classes[0];
            let minCount = Infinity;
            let maxCount = -Infinity;

            classes.forEach(c => {
                if (classCounts[c] < minCount) {
                    minCount = classCounts[c];
                    minorityClass = c;
                }
                if (classCounts[c] > maxCount) {
                    maxCount = classCounts[c];
                    majorityClass = c;
                }
            });

            const minorityPct = minCount / total;
            const isBinary = classes.length === 2;
            const ratioThreshold = 0.30 / 0.70;
            const imbalanced = isBinary ? minorityPct < 0.30 : (minCount / maxCount) < ratioThreshold;
            const ruleText = isBinary
                ? '30% minority-share threshold'
                : `minority/majority ratio threshold of ${(ratioThreshold * 100).toFixed(0)}%`;

            if (!imbalanced) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    minorityClass,
                    minorityPct: Number((minorityPct * 100).toFixed(1)),
                    classCounts,
                    reason: isBinary
                        ? `Minority class is ${(minorityPct * 100).toFixed(1)}% (≥ 30% threshold). No synthetic resampling applied.`
                        : `Smallest class (${minCount}) is ${((minCount / maxCount) * 100).toFixed(0)}% of the largest (${maxCount}), within the ${ruleText}. No synthetic resampling applied.`
                };
            }

            // Continuous features and numeric ordinals are interpolated (ordinals then snapped to an observed level);
            // nominal features and text ordinals are treated as categorical.
            const isNumericCol = (c) => trainRows.every(r => !isNaN(Number(r[c])) && String(r[c]).trim() !== '');
            const contCols = featureCols.filter(c => columnConfigs[c]?.type === 'continuous' ||
                (columnConfigs[c]?.type === 'ordinal' && isNumericCol(c)));
            const catCols = featureCols.filter(c => !contCols.includes(c));
            const ordinalLevels = {};
            contCols.forEach(c => {
                if (columnConfigs[c]?.type === 'ordinal') {
                    ordinalLevels[c] = Array.from(new Set(trainRows.map(r => Number(r[c])))).sort((a, b) => a - b);
                }
            });

            const rng = this.createPRNG(seed);
            const syntheticRows = [];
            const generatedPerClass = {};
            const skippedClasses = [];

            classes.forEach(cls => {
                const needed = maxCount - classCounts[cls];
                if (needed <= 0) return;
                const minorityRows = trainRows.filter(r => String(r[targetCol]) === cls);
                const m = minorityRows.length;
                if (m < 2) {
                    skippedClasses.push(cls);
                    return;
                }

                // Standard deviations of continuous features within this class; distances use standardised units
                const stdMap = {};
                contCols.forEach(col => {
                    const vals = minorityRows.map(r => Number(r[col])).filter(n => !isNaN(n));
                    const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
                    const v = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                    const s = Math.sqrt(v);
                    stdMap[col] = s > 0 ? s : 1.0;
                });
                // SMOTENC penalises a categorical mismatch by the squared median std of the continuous features.
                // Because continuous differences are already divided by their std, that median std is 1 on this scale.
                const catPenalty = 1.0;

                const k = Math.min(5, m - 1);

                // Pre-converted cell values (avoids Number()/String() inside the O(m²) distance loop)
                const contRaw = minorityRows.map(r => contCols.map(col => Number(r[col])));
                const contStd = contCols.map(col => stdMap[col]);
                const catRaw = minorityRows.map(r => catCols.map(col => String(r[col])));

                // k nearest minority neighbours of case i, computed once per case and cached (the previous
                // version recomputed and fully sorted all m distances for every synthetic sample: O(n·m²)).
                // Top-k insertion keeps exactly the order of a stable sort by distance (ties: lower index first).
                const neighbourCache = new Map();
                const nearestOf = (i) => {
                    if (neighbourCache.has(i)) return neighbourCache.get(i);
                    const best = []; // ascending by dist, length <= k
                    const ci = contRaw[i];
                    const ki = catRaw[i];
                    for (let j = 0; j < m; j++) {
                        if (j === i) continue;
                        let distSq = 0;
                        const cj = contRaw[j];
                        for (let c = 0; c < ci.length; c++) {
                            const diff = (ci[c] - cj[c]) / contStd[c];
                            if (!isNaN(diff)) distSq += diff * diff;
                        }
                        const kj = catRaw[j];
                        for (let c = 0; c < ki.length; c++) {
                            if (ki[c] !== kj[c]) distSq += catPenalty;
                        }
                        const dist = Math.sqrt(distSq);
                        if (best.length === k && dist >= best[k - 1].dist) continue;
                        let pos = best.length;
                        while (pos > 0 && best[pos - 1].dist > dist) pos--;
                        best.splice(pos, 0, { row: minorityRows[j], dist });
                        if (best.length > k) best.pop();
                    }
                    neighbourCache.set(i, best);
                    return best;
                };

                for (let sIdx = 0; sIdx < needed; sIdx++) {
                    const i = Math.floor(rng() * m);
                    const baseRow = minorityRows[i];
                    const neighbors = nearestOf(i);
                    const chosenNeighbor = neighbors[Math.floor(rng() * neighbors.length)].row;

                    const newRow = {};
                    const lambda = rng();
                    contCols.forEach(col => {
                        const v1 = Number(baseRow[col]);
                        const v2 = Number(chosenNeighbor[col]);
                        let v = v1 + lambda * (v2 - v1);
                        if (ordinalLevels[col]) {
                            // Snap to the nearest observed ordinal level
                            v = ordinalLevels[col].reduce((best, lvl) => (Math.abs(lvl - v) < Math.abs(best - v) ? lvl : best), ordinalLevels[col][0]);
                            newRow[col] = v;
                        } else {
                            newRow[col] = v; // full precision
                        }
                    });

                    // Categorical features: mode among the k neighbours (ties -> the base sample's value)
                    catCols.forEach(col => {
                        const votes = {};
                        neighbors.forEach(nbr => {
                            const v = String(nbr.row[col]);
                            votes[v] = (votes[v] || 0) + 1;
                        });
                        let bestCat = String(baseRow[col]);
                        let topVotes = votes[bestCat] || 0;
                        Object.keys(votes).forEach(v => {
                            if (votes[v] > topVotes) {
                                topVotes = votes[v];
                                bestCat = v;
                            }
                        });
                        newRow[col] = bestCat;
                    });

                    newRow[targetCol] = cls;
                    syntheticRows.push(newRow);
                }
                generatedPerClass[cls] = needed;
            });

            const syntheticCount = syntheticRows.length;
            if (syntheticCount === 0) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    minorityClass,
                    minorityPct: Number((minorityPct * 100).toFixed(1)),
                    classCounts,
                    reason: `Imbalance detected (${ruleText}) but too few minority samples (< 2) for SMOTENC neighbour search. No resampling applied.`
                };
            }

            const balancedRows = [...trainRows, ...syntheticRows];
            const newCounts = { ...classCounts };
            Object.keys(generatedPerClass).forEach(c => { newCounts[c] += generatedPerClass[c]; });

            const classList = Object.keys(generatedPerClass).map(c => `${c} (+${generatedPerClass[c]})`).join(', ');
            return {
                balancedRows,
                applied: true,
                minorityClass,
                minorityPctBefore: Number((minorityPct * 100).toFixed(1)),
                minorityPctAfter: Number((newCounts[minorityClass] / balancedRows.length * 100).toFixed(1)),
                originalCounts: classCounts,
                balancedCounts: newCounts,
                classCounts,
                syntheticCountGenerated: syntheticCount,
                skippedClasses,
                reason: `Smallest class was ${(minorityPct * 100).toFixed(1)}% of the training set (${ruleText} breached). SMOTENC synthesised ${syntheticCount} training instance(s): ${classList}.` +
                    (skippedClasses.length ? ` Classes with < 2 samples could not be oversampled: ${skippedClasses.join(', ')}.` : '')
            };
        }
    };

    window.DataEngine = DataEngine;
})(window);
