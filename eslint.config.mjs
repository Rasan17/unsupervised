// Lint check (run in CI): names used but never defined (e.g. a helper deleted while still in use), plus rules that
// catch real mistakes rather than style:
// unused variables and imports, loose equality, unreachable code, duplicate keys, fall-through, shadowed names.
const browser = ['document', 'window', 'alert', 'confirm', 'prompt', 'console', 'URL', 'Blob', 'FileReader', 'fetch',
    'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'localStorage', 'sessionStorage', 'Event', 'navigator',
    'location', 'Worker', 'requestAnimationFrame', 'getComputedStyle', 'performance', 'self', 'matchMedia'];
const app = ['DataEngine', 'ClusterEngine', 'EmbedEngine', 'AnomalyEngine', 'NNEngine', 'UMEChecks', 'UMECharts', 'UMEUse', 'UMEMaps', 'globalThis'];
export default [{
    files: ['src/**/*.js', 'cluster-engine.js', 'embed-engine.js', 'anomaly-engine.js', 'nn-engine.js'],
    languageOptions: { ecmaVersion: 2023, sourceType: 'script', globals: Object.fromEntries([...browser, ...app].map(g => [g, 'readonly'])) },
    rules: {
        'no-undef': 'error',
        'no-unused-vars': ['error', { args: 'none', caughtErrors: 'none', ignoreRestSiblings: true }],
        eqeqeq: ['error', 'smart'],
        'no-unreachable': 'error', 'no-dupe-keys': 'error', 'no-dupe-else-if': 'error', 'no-self-assign': 'error',
        'no-fallthrough': 'error', 'no-const-assign': 'error', 'no-unsafe-finally': 'error', 'no-cond-assign': 'error',
        'no-redeclare': ['error', { builtinGlobals: false }], 'no-shadow': 'error',
        'no-use-before-define': ['error', { functions: false, classes: false, variables: true }]
    }
}];
