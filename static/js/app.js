/* ══════════════════════════════════════════════════════════════════
   State & constants
══════════════════════════════════════════════════════════════════ */
const state = {
  models: [],
  embTagged: new Set(),
  selected: new Set(),
  results: {},           // model → result object (filled progressively)
  charts: [],
  pcaChart: null,
  gaugeChart: null,
  activeTab: 'overview',
};

const PALETTE = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#39d353'];
const TESTS = [
  { key:'speed',          label:'Vitesse',         icon:'🚀' },
  { key:'sts',            label:'STS',              icon:'🎯' },
  { key:'retrieval',      label:'Retrieval',        icon:'🔎' },
  { key:'classification', label:'Classification',   icon:'📦' },
  { key:'robustness',     label:'Robustesse',       icon:'🛡'  },
  { key:'multilingual',   label:'Multilingue',      icon:'🌍' },
];

/* ══════════════════════════════════════════════════════════════════
   DOM refs
══════════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const ollamaStatus  = $('ollama-status');
const modelList     = $('model-list');
const refreshBtn    = $('refresh-models');
const runBtn        = $('run-btn');
const hero          = $('hero');
const progressView  = $('progress-view');
const progressModel = $('progress-model');
const progressTest  = $('progress-test');
const progressSteps = $('progress-steps');
const resultsView   = $('results-view');
const tooltip       = $('tooltip');

/* ══════════════════════════════════════════════════════════════════
   Init
══════════════════════════════════════════════════════════════════ */
(async () => {
  await checkHealth();
  await loadModels();
  initNav();
  initTabs();
  initTooltips();
  initExplorer();
  initHistory();
  initViz();
})();

/* ══════════════════════════════════════════════════════════════════
   Health
══════════════════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const d = await (await fetch('/api/health')).json();
    if (d.ollama === 'ok') {
      ollamaStatus.className = 'badge badge-ok';
      ollamaStatus.innerHTML = '<span class="dot"></span><span class="badge-label">Ollama connecté</span>';
    } else throw new Error();
  } catch {
    ollamaStatus.className = 'badge badge-error';
    ollamaStatus.innerHTML = '<span class="dot"></span><span class="badge-label">Ollama hors ligne</span>';
  }
}

/* ══════════════════════════════════════════════════════════════════
   Models
══════════════════════════════════════════════════════════════════ */
async function loadModels() {
  modelList.innerHTML = '<p class="muted-sm">Chargement…</p>';
  try {
    const d = await (await fetch('/api/models')).json();
    state.models = d.models;
    state.embTagged = new Set(d.embedding_tagged);
    renderModels();
    populateModelSelects();
  } catch (e) {
    modelList.innerHTML = `<p class="muted-sm" style="color:var(--red)">Erreur : ${e.message}</p>`;
  }
}

function renderModels() {
  modelList.innerHTML = '';
  if (!state.models.length) {
    modelList.innerHTML = '<p class="muted-sm">Aucun modèle Ollama trouvé.</p>';
    return;
  }
  state.models.forEach(name => {
    const label = document.createElement('label');
    label.className = 'model-item' + (state.selected.has(name) ? ' selected' : '');
    label.innerHTML = `
      <input type="checkbox" ${state.selected.has(name) ? 'checked' : ''}/>
      <span class="chk">${state.selected.has(name) ? '✓' : ''}</span>
      <span class="model-name">${name}</span>
      ${state.embTagged.has(name) ? '<span class="embed-tag">embed</span>' : ''}`;
    label.addEventListener('change', () => {
      if (state.selected.has(name)) { state.selected.delete(name); label.classList.remove('selected'); label.querySelector('.chk').textContent = ''; }
      else { state.selected.add(name); label.classList.add('selected'); label.querySelector('.chk').textContent = '✓'; }
      runBtn.disabled = state.selected.size === 0;
    });
    modelList.appendChild(label);
  });
}

function populateModelSelects() {
  const selects = ['exp-model', 'viz-model-select'];
  selects.forEach(id => {
    const el = $(id);
    if (!el) return;
    const prev = el.value;
    el.innerHTML = '<option value="">— sélectionner —</option>';
    state.models.forEach(m => el.insertAdjacentHTML('beforeend', `<option ${m===prev?'selected':''} value="${m}">${m}</option>`));
    $('exp-run') && ($('exp-run').disabled = !$('exp-model').value);
  });
}

refreshBtn.addEventListener('click', loadModels);

/* ══════════════════════════════════════════════════════════════════
   Navigation
══════════════════════════════════════════════════════════════════ */
function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;
      ['panel-benchmark','panel-explorer','panel-history'].forEach(p => $(p).classList.add('hidden'));
      $(`panel-${view}`).classList.remove('hidden');
    });
  });
}

/* ══════════════════════════════════════════════════════════════════
   Tabs
══════════════════════════════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
      tab.classList.add('active');
      state.activeTab = tab.dataset.tab;
      $(`tab-${tab.dataset.tab}`).classList.remove('hidden');
    });
  });
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
  $(`tab-${name}`)?.classList.remove('hidden');
  state.activeTab = name;
}

/* ══════════════════════════════════════════════════════════════════
   Tooltips
══════════════════════════════════════════════════════════════════ */
function initTooltips() {
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) { tooltip.classList.add('hidden'); return; }
    tooltip.textContent = el.dataset.tip;
    tooltip.classList.remove('hidden');
  });
  document.addEventListener('mousemove', e => {
    tooltip.style.left = (e.clientX + 14) + 'px';
    tooltip.style.top  = (e.clientY + 14) + 'px';
  });
  document.addEventListener('mouseout', e => {
    if (!e.target.closest('[data-tip]')) tooltip.classList.add('hidden');
  });
}

/* ══════════════════════════════════════════════════════════════════
   Run benchmark (SSE streaming)
══════════════════════════════════════════════════════════════════ */
runBtn.addEventListener('click', runBenchmark);

async function runBenchmark() {
  const models = [...state.selected];
  if (!models.length) return;

  const cfg = {
    models,
    run_speed:          $('chk-speed').checked,
    run_sts:            $('chk-sts').checked,
    run_retrieval:      $('chk-retrieval').checked,
    run_classification: $('chk-cls').checked,
    run_robustness:     $('chk-rob').checked,
    run_multilingual:   $('chk-multi').checked,
    custom_texts: (() => {
      const v = $('custom-texts').value.trim();
      return v ? v.split('\n').map(l => l.trim()).filter(Boolean) : null;
    })(),
  };

  if (!cfg.run_speed && !cfg.run_sts && !cfg.run_retrieval &&
      !cfg.run_classification && !cfg.run_robustness && !cfg.run_multilingual) {
    return alert('Sélectionnez au moins un test.');
  }

  // Reset
  state.results = {};
  state.charts.forEach(c => c.destroy());
  state.charts = [];
  hero.classList.add('hidden');
  resultsView.classList.add('hidden');
  progressView.classList.remove('hidden');
  runBtn.disabled = true;

  // Build step tracker
  const stepState = {};
  progressSteps.innerHTML = '';
  models.forEach(m => {
    stepState[m] = {};
    const modelDiv = document.createElement('div');
    modelDiv.className = 'progress-step';
    modelDiv.innerHTML = `<span class="step-icon">📦</span><span class="step-label"><strong>${m}</strong></span>`;
    progressSteps.appendChild(modelDiv);
    TESTS.forEach(t => {
      if (!cfg[`run_${t.key}`]) return;
      const div = document.createElement('div');
      div.className = 'progress-step';
      div.id = `step-${m}-${t.key}`;
      div.innerHTML = `<span class="step-icon" style="margin-left:16px">${t.icon}</span><span class="step-label step-pending">${t.label}</span><span class="step-status"></span>`;
      progressSteps.appendChild(div);
      stepState[m][t.key] = 'pending';
    });
  });

  function updateStep(model, test, status, text='') {
    const el = $(`step-${model}-${test}`);
    if (!el) return;
    const lbl = el.querySelector('.step-label');
    const sta = el.querySelector('.step-status');
    lbl.className = `step-label step-${status}`;
    sta.textContent = text;
  }

  try {
    const resp = await fetch('/api/benchmark/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const ev = JSON.parse(line.slice(6));
        handleEvent(ev);
      }
    }
  } catch (e) {
    progressTest.textContent = '❌ Erreur réseau : ' + e.message;
  } finally {
    runBtn.disabled = false;
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case 'model_start':
        progressModel.textContent = ev.model;
        progressTest.textContent  = 'Démarrage…';
        break;
      case 'test_start':
        progressTest.textContent = TESTS.find(t => t.key === ev.test)?.label + '…';
        updateStep(ev.model, ev.test, 'running', '⏳');
        break;
      case 'test_done':
        updateStep(ev.model, ev.test, 'done', '✓');
        if (!state.results[ev.model]) state.results[ev.model] = {};
        state.results[ev.model][ev.test] = ev.data;
        break;
      case 'test_error':
        updateStep(ev.model, ev.test, 'error', '✗');
        if (!state.results[ev.model]) state.results[ev.model] = {};
        state.results[ev.model][ev.test] = { error: ev.error };
        break;
      case 'model_done':
        state.results[ev.model] = ev.result;
        break;
      case 'done':
        progressView.classList.add('hidden');
        resultsView.classList.remove('hidden');
        renderResults();
        break;
    }
  }
}

/* ══════════════════════════════════════════════════════════════════
   Render all results
══════════════════════════════════════════════════════════════════ */
function renderResults() {
  const models = Object.keys(state.results);
  renderOverview(models);
  renderSpeedTab(models);
  renderSTSTab(models);
  renderRetrievalTab(models);
  renderClassificationTab(models);
  renderRobustnessTab(models);
  renderMultilingualTab(models);
  initVizForResults(models);
  switchTab('overview');
}

/* ── Chart helpers ───────────────────────────────────────────────── */
function chartDefaults() {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#8b949e', font: { size: 11 } } },
      tooltip: { backgroundColor: '#1c2128', titleColor: '#e6edf3', bodyColor: '#8b949e', borderColor: '#30363d', borderWidth: 1 },
    },
    scales: {
      x: { ticks: { color: '#6e7681', font: { size: 11 } }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#6e7681', font: { size: 11 } }, grid: { color: '#21262d' } },
    },
  };
}

function makeChartCard(parentId, title, sub) {
  const card   = document.createElement('div');
  card.className = 'chart-card';
  const wrap   = document.createElement('div');
  wrap.className = 'chart-wrap';
  wrap.style.height = '220px';
  const canvas = document.createElement('canvas');
  wrap.appendChild(canvas);
  card.innerHTML = `<div class="chart-title">${title}</div><div class="chart-sub">${sub}</div>`;
  card.appendChild(wrap);
  $(parentId).appendChild(card);
  return canvas;
}

function bar(canvas, labels, datasets, opts = {}) {
  const c = new Chart(canvas, {
    type: 'bar',
    data: { labels: labels.map(s), datasets },
    options: { ...chartDefaults(), ...opts },
  });
  state.charts.push(c);
  return c;
}

function s(str) { return str.length > 18 ? str.slice(0, 16) + '…' : str; }
function color(i, a = 1) {
  const hex = PALETTE[i % PALETTE.length];
  if (a === 1) return hex + 'cc';
  return hex;
}

/* ── Overview tab ────────────────────────────────────────────────── */
function renderOverview(models) {
  // Leaderboard
  const lb = $('leaderboard');
  lb.innerHTML = '';
  const sorted = [...models].sort((a, b) =>
    (state.results[b]?.overall_score ?? 0) - (state.results[a]?.overall_score ?? 0));
  const max = state.results[sorted[0]]?.overall_score ?? 100;

  sorted.forEach((m, i) => {
    const r = state.results[m];
    const sc = r?.overall_score ?? null;
    const row = document.createElement('div');
    row.className = 'lb-row';
    const rankEmoji = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i+1}.`;
    const pills = TESTS
      .filter(t => r?.[t.key] && !r[t.key].error)
      .map(t => `<span class="lb-pill">${t.icon}</span>`).join('');
    row.innerHTML = `
      <div class="lb-rank lb-rank-${i+1}">${rankEmoji}</div>
      <div class="lb-name">${m}</div>
      <div class="lb-bar-wrap"><div class="lb-bar" style="width:${sc?sc/max*100:0}%;background:${PALETTE[i%PALETTE.length]}"></div></div>
      <div class="lb-pills">${pills}</div>
      <div class="lb-score">${sc !== null ? sc : '—'}<small style="font-size:.6rem;color:var(--muted)">/100</small></div>`;
    lb.appendChild(row);
  });

  // Radar chart
  const radarCanvas = $('radar-chart');
  if (radarCanvas._chart) { radarCanvas._chart.destroy(); }
  const radarLabels = ['STS', 'Retrieval', 'Classification', 'Robustesse', 'Multilingue', 'Vitesse'];
  const radarDatasets = models.map((m, i) => {
    const r = state.results[m] || {};
    return {
      label: m,
      data: [
        r.sts?.pearson_r != null ? Math.max(0, r.sts.pearson_r) : null,
        r.retrieval?.ndcg_at_5 ?? null,
        r.classification?.nearest_centroid_accuracy ?? null,
        r.robustness?.discrimination_ratio != null ? Math.min(1, Math.max(0, r.robustness.discrimination_ratio - 1)) : null,
        r.multilingual?.alignment_score != null ? Math.min(1, Math.max(0, r.multilingual.alignment_score + 0.1)) : null,
        r.speed?.latency_mean_ms != null ? Math.min(1, 1/(1+r.speed.latency_mean_ms/80)) : null,
      ],
      borderColor: PALETTE[i % PALETTE.length],
      backgroundColor: PALETTE[i % PALETTE.length] + '22',
      pointBackgroundColor: PALETTE[i % PALETTE.length],
      borderWidth: 2,
    };
  });
  const rc = new Chart(radarCanvas, {
    type: 'radar',
    data: { labels: radarLabels, datasets: radarDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { r: { min: 0, max: 1, ticks: { display: false, stepSize: 0.2 }, grid: { color: '#30363d' }, pointLabels: { color: '#8b949e', font: { size: 11 } } } },
      plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } }, tooltip: { backgroundColor: '#1c2128', titleColor: '#e6edf3', bodyColor: '#8b949e' } },
    },
  });
  radarCanvas._chart = rc;
  state.charts.push(rc);

  // Overall bar
  const ovCanvas = $('overall-chart');
  if (ovCanvas._chart) ovCanvas._chart.destroy();
  const oc = bar(ovCanvas, sorted,
    [{ label: 'Score /100', data: sorted.map(m => state.results[m]?.overall_score ?? 0), backgroundColor: sorted.map((_, i) => PALETTE[i % PALETTE.length] + 'cc'), borderColor: sorted.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 6 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 100 } } }
  );
  ovCanvas._chart = oc;
}

/* ── Speed tab ───────────────────────────────────────────────────── */
function renderSpeedTab(models) {
  const c = $('speed-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.speed?.latency_mean_ms != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données de vitesse.</p>'; return; }

  bar(makeChartCard('speed-charts', '⏱ Latence moyenne', 'ms par texte — moins = mieux'), good,
    [{ label: 'Latence moy. (ms)', data: good.map(m => state.results[m].speed.latency_mean_ms), backgroundColor: good.map((_, i) => color(i)), borderColor: good.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 5 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } });

  // Distribution
  const distCanvas = makeChartCard('speed-charts', '📊 Distribution de latence', 'Min / P50 / P95 / Max');
  const distC = new Chart(distCanvas, {
    type: 'bar',
    data: {
      labels: good.map(s),
      datasets: [
        { label: 'Min',  data: good.map(m => state.results[m].speed.latency_min_ms), backgroundColor: '#3fb95088', borderColor: '#3fb950', borderWidth: 2, borderRadius: 5 },
        { label: 'P50',  data: good.map(m => state.results[m].speed.latency_p50_ms), backgroundColor: '#58a6ff88', borderColor: '#58a6ff', borderWidth: 2, borderRadius: 5 },
        { label: 'P95',  data: good.map(m => state.results[m].speed.latency_p95_ms), backgroundColor: '#d2992288', borderColor: '#d29922', borderWidth: 2, borderRadius: 5 },
        { label: 'Max',  data: good.map(m => state.results[m].speed.latency_max_ms), backgroundColor: '#f8514988', borderColor: '#f85149', borderWidth: 2, borderRadius: 5 },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } },
  });
  state.charts.push(distC);

  bar(makeChartCard('speed-charts', '⚡ Throughput', 'Textes par seconde — plus = mieux'), good,
    [{ label: 'Throughput (t/s)', data: good.map(m => state.results[m].speed.throughput_per_sec), backgroundColor: good.map((_, i) => color(i)), borderColor: good.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 5 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } });

  // Table
  const t = $('speed-table');
  const bestTP = Math.max(...good.map(m => state.results[m].speed.throughput_per_sec));
  t.innerHTML = `<div class="table-title">🚀 Résultats de vitesse</div>
  <table><thead><tr><th>Modèle</th><th>Latence moy.</th><th>P50</th><th>P95</th><th>Max</th><th>Throughput</th><th>Dim.</th></tr></thead>
  <tbody>${good.map(m => {
    const sp = state.results[m].speed;
    const best = sp.throughput_per_sec === bestTP;
    return `<tr><td class="mono">${m}</td><td class="num">${sp.latency_mean_ms} ms</td><td class="num">${sp.latency_p50_ms} ms</td><td class="num">${sp.latency_p95_ms} ms</td><td class="num">${sp.latency_max_ms} ms</td><td class="num ${best?'good':''}">${sp.throughput_per_sec} t/s</td><td class="num">${sp.embedding_dim ?? '—'}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── STS tab ─────────────────────────────────────────────────────── */
function renderSTSTab(models) {
  const c = $('sts-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.sts?.pearson_r != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données STS.</p>'; return; }

  bar(makeChartCard('sts-charts', '📈 Corrélation de Pearson', 'Corrélation avec les scores humains · plus proche de 1 = mieux'), good,
    [{ label: 'Pearson r', data: good.map(m => state.results[m].sts.pearson_r), backgroundColor: good.map((_, i) => color(i)), borderColor: good.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 5 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, min: -1, max: 1 } } });

  // Scatter for first model
  const fm = good[0];
  const sts = state.results[fm].sts;
  const scCanvas = makeChartCard('sts-charts', `🔬 Prédictions vs réalité — ${s(fm)}`, 'Similarité cosinus prédite vs score annoté humain');
  const sc = new Chart(scCanvas, {
    type: 'scatter',
    data: { datasets: [{ label: fm, data: sts.ground_truth.map((gt, i) => ({ x: gt, y: sts.predicted[i] })), backgroundColor: '#58a6ffcc', pointRadius: 5 }] },
    options: { ...chartDefaults(), scales: { x: { ...chartDefaults().scales.x, title: { display: true, text: 'Score humain', color: '#6e7681' }, min: 0, max: 1 }, y: { ...chartDefaults().scales.y, title: { display: true, text: 'Cosine sim.', color: '#6e7681' }, min: 0, max: 1 } } },
  });
  state.charts.push(sc);

  $('sts-table').innerHTML = `<div class="table-title">🎯 Résultats STS</div>
  <table><thead><tr><th>Modèle</th><th>Pearson r</th><th>Paires</th><th>Interprétation</th></tr></thead>
  <tbody>${good.map(m => {
    const pr = state.results[m].sts.pearson_r;
    const cls = pr > 0.7 ? 'good' : pr > 0.5 ? 'ok' : 'bad';
    const lbl = pr > 0.7 ? 'Excellent' : pr > 0.5 ? 'Bon' : pr > 0.3 ? 'Modéré' : 'Faible';
    return `<tr><td class="mono">${m}</td><td class="num ${cls}">${pr.toFixed(4)}</td><td class="num">${state.results[m].sts.num_pairs}</td><td class="${cls}">${lbl}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Retrieval tab ───────────────────────────────────────────────── */
function renderRetrievalTab(models) {
  const c = $('retrieval-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.retrieval?.ndcg_at_5 != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données de retrieval.</p>'; return; }

  const metrics = [
    { key: 'ndcg_at_5', label: 'NDCG@5', color: '#58a6ff' },
    { key: 'recall_at_5', label: 'Recall@5', color: '#3fb950' },
    { key: 'mrr', label: 'MRR', color: '#d29922' },
  ];
  const canvas = makeChartCard('retrieval-charts', '🔎 Métriques de retrieval', 'NDCG@5 · Recall@5 · MRR — plus = mieux');
  const rc = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: good.map(s),
      datasets: metrics.map(({ key, label, color: col }) => ({
        label, data: good.map(m => state.results[m].retrieval[key]),
        backgroundColor: col + '88', borderColor: col, borderWidth: 2, borderRadius: 5,
      })),
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 1 } } },
  });
  state.charts.push(rc);

  // Per-query NDCG for first model
  const fm = good[0];
  const pq = state.results[fm].retrieval.per_query;
  if (pq?.length) {
    const qc = makeChartCard('retrieval-charts', `📋 NDCG@5 par requête — ${s(fm)}`, 'Performance par requête individuelle');
    bar(qc, pq.map(q => q.query.slice(0, 28) + '…'),
      [{ label: 'NDCG@5', data: pq.map(q => q['ndcg@5']), backgroundColor: '#58a6ffcc', borderColor: '#58a6ff', borderWidth: 2, borderRadius: 5 }],
      { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 1 } } });
  }

  const t = $('retrieval-table');
  t.innerHTML = `<div class="table-title">🔎 Résultats de retrieval</div>
  <table><thead><tr><th>Modèle</th><th>NDCG@5</th><th>NDCG@3</th><th>Recall@5</th><th>MRR</th><th>Requêtes</th></tr></thead>
  <tbody>${good.map(m => {
    const r = state.results[m].retrieval;
    const cls = v => v > 0.8 ? 'good' : v > 0.5 ? 'ok' : 'bad';
    return `<tr><td class="mono">${m}</td><td class="num ${cls(r.ndcg_at_5)}">${r.ndcg_at_5.toFixed(4)}</td><td class="num">${r.ndcg_at_3.toFixed(4)}</td><td class="num ${cls(r.recall_at_5)}">${r.recall_at_5.toFixed(4)}</td><td class="num ${cls(r.mrr)}">${r.mrr.toFixed(4)}</td><td class="num">${r.num_queries}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Classification tab ──────────────────────────────────────────── */
function renderClassificationTab(models) {
  const c = $('cls-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.classification?.nearest_centroid_accuracy != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données.</p>'; return; }

  const canvas = makeChartCard('cls-charts', '📦 Métriques de classification', 'Précision centroïde & score Silhouette');
  const ch = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: good.map(s),
      datasets: [
        { label: 'Précision (%)', data: good.map(m => +(state.results[m].classification.nearest_centroid_accuracy * 100).toFixed(1)), backgroundColor: '#58a6ff88', borderColor: '#58a6ff', borderWidth: 2, borderRadius: 5 },
        { label: 'Silhouette ×100', data: good.map(m => +(state.results[m].classification.silhouette_score * 100).toFixed(1)), backgroundColor: '#3fb95088', borderColor: '#3fb950', borderWidth: 2, borderRadius: 5 },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } },
  });
  state.charts.push(ch);

  $('cls-table').innerHTML = `<div class="table-title">📦 Résultats de classification</div>
  <table><thead><tr><th>Modèle</th><th>Précision centroïde</th><th>Silhouette</th><th>Textes</th><th>Classes</th></tr></thead>
  <tbody>${good.map(m => {
    const cl = state.results[m].classification;
    const cls = cl.nearest_centroid_accuracy > 0.8 ? 'good' : cl.nearest_centroid_accuracy > 0.5 ? 'ok' : 'bad';
    return `<tr><td class="mono">${m}</td><td class="num ${cls}">${(cl.nearest_centroid_accuracy*100).toFixed(1)}%</td><td class="num">${cl.silhouette_score.toFixed(4)}</td><td class="num">${cl.num_texts}</td><td>${cl.classes.join(', ')}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Robustness tab ──────────────────────────────────────────────── */
function renderRobustnessTab(models) {
  const c = $('rob-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.robustness?.discrimination_ratio != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données.</p>'; return; }

  const canvas = makeChartCard('rob-charts', '🛡 Similarité intra vs inter-groupe', 'Paraphrases (intra) doivent être plus similaires que les phrases différentes (inter)');
  const ch = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: good.map(s),
      datasets: [
        { label: 'Intra-groupe', data: good.map(m => state.results[m].robustness.avg_intra_similarity), backgroundColor: '#3fb95088', borderColor: '#3fb950', borderWidth: 2, borderRadius: 5 },
        { label: 'Inter-groupe', data: good.map(m => state.results[m].robustness.avg_inter_similarity), backgroundColor: '#f8514988', borderColor: '#f85149', borderWidth: 2, borderRadius: 5 },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 1 } } },
  });
  state.charts.push(ch);

  bar(makeChartCard('rob-charts', '📐 Ratio de discrimination', 'intra_sim / inter_sim — plus = mieux (>1.5 bon, >2.0 excellent)'), good,
    [{ label: 'Ratio', data: good.map(m => state.results[m].robustness.discrimination_ratio), backgroundColor: good.map((_, i) => color(i)), borderColor: good.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 5 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } });

  $('rob-table').innerHTML = `<div class="table-title">🛡 Résultats de robustesse</div>
  <table><thead><tr><th>Modèle</th><th>Sim. intra</th><th>Sim. inter</th><th>Ratio discrimination</th><th>Groupes</th></tr></thead>
  <tbody>${good.map(m => {
    const r = state.results[m].robustness;
    const cls = r.discrimination_ratio > 2 ? 'good' : r.discrimination_ratio > 1.5 ? 'ok' : 'bad';
    return `<tr><td class="mono">${m}</td><td class="num">${r.avg_intra_similarity.toFixed(4)}</td><td class="num">${r.avg_inter_similarity.toFixed(4)}</td><td class="num ${cls}">${r.discrimination_ratio.toFixed(4)}</td><td class="num">${r.num_groups}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Multilingual tab ────────────────────────────────────────────── */
function renderMultilingualTab(models) {
  const c = $('multi-charts');
  c.innerHTML = '';
  const good = models.filter(m => state.results[m]?.multilingual?.alignment_score != null);
  if (!good.length) { c.innerHTML = '<p class="muted-sm">Pas de données.</p>'; return; }

  const canvas = makeChartCard('multi-charts', '🌍 Similarité des paires de traduction', 'Similarité cosinus entre phrases de même sens dans des langues différentes');
  const ch = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: good.map(s),
      datasets: [
        { label: 'Sim. traduction', data: good.map(m => state.results[m].multilingual.avg_similarity), backgroundColor: '#58a6ff88', borderColor: '#58a6ff', borderWidth: 2, borderRadius: 5 },
        { label: 'Sim. non-traduction', data: good.map(m => state.results[m].multilingual.avg_non_translation_sim), backgroundColor: '#f8514988', borderColor: '#f85149', borderWidth: 2, borderRadius: 5 },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 1 } } },
  });
  state.charts.push(ch);

  bar(makeChartCard('multi-charts', '📐 Score d\'alignement', 'sim_traduction − sim_non_traduction · >0.3 = bon modèle multilingue'), good,
    [{ label: 'Alignement', data: good.map(m => state.results[m].multilingual.alignment_score), backgroundColor: good.map((_, i) => color(i)), borderColor: good.map((_, i) => PALETTE[i % PALETTE.length]), borderWidth: 2, borderRadius: 5 }],
    { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y } } });

  // Per-pair details for first model
  const fm = good[0];
  const details = state.results[fm].multilingual.details;
  if (details?.length) {
    const pairs = details.slice(0, 15);
    const pairC = makeChartCard('multi-charts', `🔠 Similarité par paire — ${s(fm)}`, 'Chaque paire de traduction (EN → autre langue)');
    pairC.style.gridColumn = '1 / -1';
    bar(pairC, pairs.map(p => p.lang),
      [{ label: 'Similarité', data: pairs.map(p => p.similarity), backgroundColor: pairs.map(p => p.similarity > 0.7 ? '#3fb95088' : p.similarity > 0.5 ? '#d2992288' : '#f8514988'), borderColor: pairs.map(p => p.similarity > 0.7 ? '#3fb950' : p.similarity > 0.5 ? '#d29922' : '#f85149'), borderWidth: 2, borderRadius: 5 }],
      { plugins: { ...chartDefaults().plugins, legend: { display: false } }, scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true, max: 1 } } });
  }

  $('multi-table').innerHTML = `<div class="table-title">🌍 Résultats multilingues</div>
  <table><thead><tr><th>Modèle</th><th>Sim. traduction</th><th>Sim. non-traduction</th><th>Score alignement</th><th>Paires</th></tr></thead>
  <tbody>${good.map(m => {
    const r = state.results[m].multilingual;
    const cls = r.alignment_score > 0.3 ? 'good' : r.alignment_score > 0.1 ? 'ok' : 'bad';
    return `<tr><td class="mono">${m}</td><td class="num">${r.avg_similarity.toFixed(4)}</td><td class="num">${r.avg_non_translation_sim.toFixed(4)}</td><td class="num ${cls}">${r.alignment_score.toFixed(4)}</td><td class="num">${r.num_pairs}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ══════════════════════════════════════════════════════════════════
   PCA Visualization
══════════════════════════════════════════════════════════════════ */
function initViz() {
  $('viz-run-btn').addEventListener('click', runVisualization);
}

function initVizForResults(models) {
  const sel = $('viz-model-select');
  sel.innerHTML = '';
  models.forEach(m => sel.insertAdjacentHTML('beforeend', `<option value="${m}">${m}</option>`));
}

async function runVisualization() {
  const model = $('viz-model-select').value;
  if (!model) return alert('Sélectionnez un modèle.');
  const btn = $('viz-run-btn');
  btn.textContent = '⏳ Calcul…';
  btn.disabled = true;
  try {
    const r = await fetch('/api/visualize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const d = await r.json();
    renderPCA(d, model);
  } catch (e) {
    alert('Erreur : ' + e.message);
  } finally {
    btn.textContent = '↻ Recalculer';
    btn.disabled = false;
  }
}

function renderPCA(data, model) {
  if (state.pcaChart) { state.pcaChart.destroy(); state.pcaChart = null; }
  const { points, labels, texts } = data;
  const uniqueLabels = [...new Set(labels)];
  const datasets = uniqueLabels.map((lbl, i) => {
    const idxs = labels.map((l, j) => l === lbl ? j : -1).filter(j => j >= 0);
    return {
      label: lbl,
      data: idxs.map(j => ({ x: points[j][0], y: points[j][1], text: texts[j] })),
      backgroundColor: PALETTE[i % PALETTE.length] + 'bb',
      pointRadius: 7,
    };
  });
  const canvas = $('pca-chart');
  state.pcaChart = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#1c2128', titleColor: '#e6edf3', bodyColor: '#8b949e',
          callbacks: { label: ctx => ctx.raw.text?.slice(0, 60) || '' },
        },
        title: { display: true, text: `PCA 2D — ${model}`, color: '#8b949e', font: { size: 12 } },
      },
      scales: {
        x: { ticks: { display: false }, grid: { color: '#21262d' } },
        y: { ticks: { display: false }, grid: { color: '#21262d' } },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════════════
   Similarity Explorer
══════════════════════════════════════════════════════════════════ */
function initExplorer() {
  const modelSel = $('exp-model');
  const runBtn   = $('exp-run');
  modelSel.addEventListener('change', () => { runBtn.disabled = !modelSel.value; });
  runBtn.addEventListener('click', runExplorer);
}

async function runExplorer() {
  const model = $('exp-model').value;
  const ta    = $('exp-text-a').value.trim();
  const tb    = $('exp-text-b').value.trim();
  if (!model || !ta || !tb) return alert('Remplissez tous les champs.');
  const btn = $('exp-run');
  btn.textContent = '⏳ Calcul…'; btn.disabled = true;
  try {
    const r = await fetch('/api/explore/similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, text_a: ta, text_b: tb }),
    });
    const d = await r.json();
    renderGauge(d);
  } catch (e) {
    alert('Erreur : ' + e.message);
  } finally {
    btn.textContent = 'Calculer la similarité'; btn.disabled = !model;
  }
}

function renderGauge(data) {
  const { similarity, magnitude_a, magnitude_b, dim } = data;
  const sim = similarity;
  $('exp-result').classList.remove('hidden');
  $('sim-score-txt').textContent = sim.toFixed(3);
  const col = sim > 0.7 ? '#3fb950' : sim > 0.4 ? '#d29922' : '#f85149';

  if (state.gaugeChart) state.gaugeChart.destroy();
  state.gaugeChart = new Chart($('sim-gauge'), {
    type: 'doughnut',
    data: {
      datasets: [{ data: [Math.max(0, sim), Math.max(0, 1 - sim)], backgroundColor: [col, '#21262d'], borderWidth: 0, circumference: 270, rotation: 225 }],
    },
    options: { responsive: false, cutout: '78%', plugins: { legend: { display: false }, tooltip: { enabled: false } } },
  });

  const interp = sim > 0.85 ? 'Quasi-identique' : sim > 0.7 ? 'Très similaire' : sim > 0.5 ? 'Modérément similaire' : sim > 0.3 ? 'Peu similaire' : 'Non similaire';
  $('sim-meta').innerHTML = `
    <div>Interprétation : <span>${interp}</span></div>
    <div>Dimension : <span>${dim}</span></div>
    <div>‖A‖ : <span>${magnitude_a}</span> · ‖B‖ : <span>${magnitude_b}</span></div>`;
}

/* ══════════════════════════════════════════════════════════════════
   History (localStorage)
══════════════════════════════════════════════════════════════════ */
function initHistory() {
  renderHistoryList();
  $('clear-history').addEventListener('click', () => {
    if (!confirm('Effacer tout l\'historique ?')) return;
    localStorage.removeItem('eb_history');
    renderHistoryList();
  });
  $('save-history').addEventListener('click', saveToHistory);
}

function saveToHistory() {
  if (!Object.keys(state.results).length) return alert('Aucun résultat à sauvegarder.');
  const history = JSON.parse(localStorage.getItem('eb_history') || '[]');
  history.unshift({ date: new Date().toISOString(), results: state.results });
  if (history.length > 20) history.length = 20;
  localStorage.setItem('eb_history', JSON.stringify(history));
  renderHistoryList();
  alert('Run sauvegardé dans l\'historique.');
}

function renderHistoryList() {
  const history = JSON.parse(localStorage.getItem('eb_history') || '[]');
  const el = $('history-list');
  if (!history.length) { el.innerHTML = '<p class="muted-sm">Aucun run enregistré.</p>'; return; }
  el.innerHTML = '';
  history.forEach((run, i) => {
    const models = Object.keys(run.results);
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
      <div class="history-item-date">${new Date(run.date).toLocaleString('fr-FR')}</div>
      <div class="history-item-models">${models.join(', ')}</div>`;
    div.addEventListener('click', () => {
      state.results = run.results;
      state.charts.forEach(c => c.destroy());
      state.charts = [];
      hero.classList.add('hidden');
      progressView.classList.add('hidden');
      resultsView.classList.remove('hidden');
      renderResults();
      document.querySelector('.nav-btn[data-view="benchmark"]').click();
    });
    el.appendChild(div);
  });
}

/* ══════════════════════════════════════════════════════════════════
   Export
══════════════════════════════════════════════════════════════════ */
$('export-json').addEventListener('click', () => {
  if (!Object.keys(state.results).length) return;
  dl(new Blob([JSON.stringify(state.results, null, 2)], { type: 'application/json' }), 'benchmark.json');
});

$('export-csv').addEventListener('click', () => {
  if (!Object.keys(state.results).length) return;
  const rows = [['model','overall_score','latency_ms','throughput','dim','pearson_r','ndcg5','recall5','mrr','cls_accuracy','silhouette','discrim_ratio','multilingual_alignment']];
  Object.entries(state.results).forEach(([m, r]) => rows.push([
    m, r.overall_score ?? '',
    r.speed?.latency_mean_ms ?? '', r.speed?.throughput_per_sec ?? '', r.speed?.embedding_dim ?? '',
    r.sts?.pearson_r ?? '',
    r.retrieval?.ndcg_at_5 ?? '', r.retrieval?.recall_at_5 ?? '', r.retrieval?.mrr ?? '',
    r.classification?.nearest_centroid_accuracy ?? '', r.classification?.silhouette_score ?? '',
    r.robustness?.discrimination_ratio ?? '',
    r.multilingual?.alignment_score ?? '',
  ]));
  dl(new Blob([rows.map(r => r.join(',')).join('\n')], { type: 'text/csv' }), 'benchmark.csv');
});

function dl(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}
