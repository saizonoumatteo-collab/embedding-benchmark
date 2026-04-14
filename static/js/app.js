/* ── State ─────────────────────────────────────────────────────────────────── */
const state = {
  models: [],
  selected: new Set(),
  results: null,
  charts: [],
};

const PALETTE = ['#6c63ff','#00d4aa','#ff6b6b','#ffd93d','#ff9f43','#48dbfb'];

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const ollamaStatus  = $('ollama-status');
const modelList     = $('model-list');
const refreshBtn    = $('refresh-models');
const runBtn        = $('run-btn');
const hero          = $('hero');
const progressView  = $('progress-view');
const progressLabel = $('progress-label');
const resultsView   = $('results-view');
const summaryCards  = $('summary-cards');
const chartsGrid    = $('charts-grid');
const detailTables  = $('detail-tables');
const exportJSON    = $('export-json');
const exportCSV     = $('export-csv');

/* ── Init ──────────────────────────────────────────────────────────────────── */
(async () => {
  await checkHealth();
  await loadModels();
})();

/* ── Health check ──────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();
    if (d.ollama === 'ok') {
      ollamaStatus.className = 'status-badge status-ok';
      ollamaStatus.innerHTML = '<span class="status-dot"></span> Ollama connecté';
    } else {
      throw new Error(d.error || 'unreachable');
    }
  } catch (e) {
    ollamaStatus.className = 'status-badge status-error';
    ollamaStatus.innerHTML = '<span class="status-dot"></span> Ollama hors ligne';
  }
}

/* ── Load models ───────────────────────────────────────────────────────────── */
async function loadModels() {
  modelList.innerHTML = '<p class="muted">Chargement…</p>';
  try {
    const r = await fetch('/api/models');
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();
    state.models = d.models;
    state.embeddingTagged = new Set(d.embedding_tagged || []);
    renderModelList();
  } catch (e) {
    modelList.innerHTML = `<p class="error-text">Erreur : ${e.message}</p>`;
  }
}

function renderModelList() {
  if (!state.models.length) {
    modelList.innerHTML = '<p class="muted">Aucun modèle trouvé dans Ollama.</p>';
    return;
  }
  modelList.innerHTML = '';
  state.models.forEach(name => {
    const item = document.createElement('label');
    item.className = 'model-item' + (state.selected.has(name) ? ' selected' : '');
    item.innerHTML = `
      <input type="checkbox" ${state.selected.has(name) ? 'checked' : ''} />
      <span class="model-check">${state.selected.has(name) ? '✓' : ''}</span>
      <span class="model-name">${name}</span>
      ${state.embeddingTagged.has(name) ? '<span class="model-tag">embed</span>' : ''}
    `;
    item.addEventListener('change', () => toggleModel(name, item));
    modelList.appendChild(item);
  });
}

function toggleModel(name, item) {
  if (state.selected.has(name)) {
    state.selected.delete(name);
    item.classList.remove('selected');
    item.querySelector('.model-check').textContent = '';
  } else {
    state.selected.add(name);
    item.classList.add('selected');
    item.querySelector('.model-check').textContent = '✓';
  }
  runBtn.disabled = state.selected.size === 0;
}

refreshBtn.addEventListener('click', loadModels);

/* ── Run benchmark ─────────────────────────────────────────────────────────── */
runBtn.addEventListener('click', runBenchmark);

async function runBenchmark() {
  const models = [...state.selected];
  const runSpeed = $('chk-speed').checked;
  const runSTS   = $('chk-sts').checked;
  const runCls   = $('chk-cls').checked;

  if (!runSpeed && !runSTS && !runCls) {
    alert('Sélectionnez au moins un test.');
    return;
  }

  const rawCustom = $('custom-texts').value.trim();
  const customTexts = rawCustom
    ? rawCustom.split('\n').map(l => l.trim()).filter(Boolean)
    : null;

  // Show progress
  hero.classList.add('hidden');
  resultsView.classList.add('hidden');
  progressView.classList.remove('hidden');
  runBtn.disabled = true;

  const tests = [runSpeed && 'vitesse', runSTS && 'STS', runCls && 'classification']
    .filter(Boolean).join(', ');
  progressLabel.textContent = `Benchmark en cours sur ${models.length} modèle(s) : ${tests}…`;

  try {
    const r = await fetch('/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        models, run_speed: runSpeed, run_sts: runSTS,
        run_classification: runCls,
        custom_texts: customTexts,
      }),
    });
    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || r.statusText);
    }
    const d = await r.json();
    state.results = d.results;
    renderResults(d.results);
  } catch (e) {
    progressLabel.textContent = '❌ Erreur : ' + e.message;
  } finally {
    runBtn.disabled = false;
  }
}

/* ── Render results ────────────────────────────────────────────────────────── */
function renderResults(results) {
  progressView.classList.add('hidden');
  resultsView.classList.remove('hidden');

  // Destroy previous charts
  state.charts.forEach(c => c.destroy());
  state.charts = [];

  summaryCards.innerHTML = '';
  chartsGrid.innerHTML = '';
  detailTables.innerHTML = '';

  const models = Object.keys(results);

  renderSummaryCards(results, models);
  renderCharts(results, models);
  renderDetailTables(results, models);
}

/* ── Summary cards ─────────────────────────────────────────────────────────── */
function renderSummaryCards(results, models) {
  models.forEach((model, idx) => {
    const r = results[model];
    const card = document.createElement('div');
    card.className = 'model-summary-card';
    card.style.borderTopColor = PALETTE[idx % PALETTE.length];
    card.style.borderTopWidth = '3px';

    let html = `<div class="card-model-name">⚡ ${model}</div>`;

    if (r.speed && !r.speed.error) {
      const latency = r.speed.latency_mean_ms;
      const quality = latency < 100 ? 'good' : latency < 500 ? '' : 'bad';
      html += `
        <div class="card-metric">
          <span class="card-metric-label">Latence moy.</span>
          <span class="card-metric-value ${quality}">${latency} ms</span>
        </div>
        <div class="card-metric">
          <span class="card-metric-label">Throughput</span>
          <span class="card-metric-value">${r.speed.throughput_texts_per_sec} t/s</span>
        </div>`;
      if (r.speed.embedding_dim) {
        html += `<div class="card-dim">${r.speed.embedding_dim}d</div>`;
      }
    }

    if (r.sts && !r.sts.error) {
      const pr = r.sts.pearson_r;
      const cls = pr > 0.7 ? 'good' : pr > 0.4 ? '' : 'bad';
      html += `
        <div class="card-metric">
          <span class="card-metric-label">Pearson r (STS)</span>
          <span class="card-metric-value ${cls}">${pr.toFixed(3)}</span>
        </div>`;
    }

    if (r.classification && !r.classification.error) {
      const acc = r.classification.nearest_centroid_accuracy;
      const sil = r.classification.silhouette_score;
      const cls = acc > 0.8 ? 'good' : acc > 0.5 ? '' : 'bad';
      html += `
        <div class="card-metric">
          <span class="card-metric-label">Précision centroïde</span>
          <span class="card-metric-value ${cls}">${(acc * 100).toFixed(1)}%</span>
        </div>
        <div class="card-metric">
          <span class="card-metric-label">Score Silhouette</span>
          <span class="card-metric-value">${sil.toFixed(3)}</span>
        </div>`;
    }

    if (r.error) {
      html += `<p class="error-text">Erreur : ${r.error}</p>`;
    }

    card.innerHTML = html;
    summaryCards.appendChild(card);
  });
}

/* ── Charts ────────────────────────────────────────────────────────────────── */
function makeChartCard(title, subtitle) {
  const card = document.createElement('div');
  card.className = 'chart-card';
  const wrapper = document.createElement('div');
  wrapper.className = 'chart-wrapper';
  const canvas = document.createElement('canvas');
  wrapper.appendChild(canvas);
  card.innerHTML = `<div class="chart-title">${title}</div><div class="chart-subtitle">${subtitle}</div>`;
  card.appendChild(wrapper);
  chartsGrid.appendChild(card);
  return canvas;
}

function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#9ca3c0', font: { size: 11 } } },
      tooltip: { backgroundColor: '#1e2333', titleColor: '#e8eaf0', bodyColor: '#9ca3c0' },
    },
    scales: {
      x: { ticks: { color: '#6b7394', font: { size: 11 } }, grid: { color: '#2a3047' } },
      y: { ticks: { color: '#6b7394', font: { size: 11 } }, grid: { color: '#2a3047' } },
    },
  };
}

function renderCharts(results, models) {
  const hasSpeed = models.some(m => results[m].speed && !results[m].speed.error);
  const hasSTS   = models.some(m => results[m].sts   && !results[m].sts.error);
  const hasCls   = models.some(m => results[m].classification && !results[m].classification.error);

  // 1. Latency bar chart
  if (hasSpeed) {
    const canvas = makeChartCard('🚀 Latence moyenne', 'Temps moyen pour générer un embedding (ms, moins = mieux)');
    const chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: models.map(shortName),
        datasets: [{
          label: 'Latence (ms)',
          data: models.map(m => results[m].speed?.latency_mean_ms ?? null),
          backgroundColor: models.map((_, i) => PALETTE[i % PALETTE.length] + 'cc'),
          borderColor: models.map((_, i) => PALETTE[i % PALETTE.length]),
          borderWidth: 2,
          borderRadius: 6,
        }],
      },
      options: {
        ...chartDefaults(),
        plugins: { ...chartDefaults().plugins, legend: { display: false } },
        scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } },
      },
    });
    state.charts.push(chart);
  }

  // 2. Latency range (min/p95/max)
  if (hasSpeed) {
    const canvas = makeChartCard('📊 Distribution de latence', 'Min / P95 / Max par modèle');
    const chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: models.filter(m => results[m].speed && !results[m].speed.error).map(shortName),
        datasets: [
          {
            label: 'Min',
            data: models.filter(m => results[m].speed && !results[m].speed.error).map(m => results[m].speed.latency_min_ms),
            backgroundColor: '#00d4aa88',
            borderColor: '#00d4aa',
            borderWidth: 2, borderRadius: 6,
          },
          {
            label: 'P95',
            data: models.filter(m => results[m].speed && !results[m].speed.error).map(m => results[m].speed.latency_p95_ms),
            backgroundColor: '#6c63ff88',
            borderColor: '#6c63ff',
            borderWidth: 2, borderRadius: 6,
          },
          {
            label: 'Max',
            data: models.filter(m => results[m].speed && !results[m].speed.error).map(m => results[m].speed.latency_max_ms),
            backgroundColor: '#ff6b6b88',
            borderColor: '#ff6b6b',
            borderWidth: 2, borderRadius: 6,
          },
        ],
      },
      options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } },
    });
    state.charts.push(chart);
  }

  // 3. STS Pearson r
  if (hasSTS) {
    const canvas = makeChartCard('🎯 Corrélation STS (Pearson r)', 'Corrélation avec les scores de similarité humains (plus élevé = mieux)');
    const chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: models.map(shortName),
        datasets: [{
          label: 'Pearson r',
          data: models.map(m => results[m].sts?.pearson_r ?? null),
          backgroundColor: models.map((_, i) => PALETTE[i % PALETTE.length] + 'cc'),
          borderColor: models.map((_, i) => PALETTE[i % PALETTE.length]),
          borderWidth: 2, borderRadius: 6,
        }],
      },
      options: {
        ...chartDefaults(),
        plugins: { ...chartDefaults().plugins, legend: { display: false } },
        scales: {
          ...chartDefaults().scales,
          y: { ...chartDefaults().scales.y, min: -1, max: 1 },
        },
      },
    });
    state.charts.push(chart);
  }

  // 4. STS scatter: predicted vs ground truth (first model only)
  if (hasSTS) {
    const firstModel = models.find(m => results[m].sts && !results[m].sts.error);
    if (firstModel) {
      const sts = results[firstModel].sts;
      const canvas = makeChartCard(
        `🔬 Prédictions STS — ${shortName(firstModel)}`,
        'Similarité prédite vs score humain (idéalement proche de y=x)'
      );
      const chart = new Chart(canvas, {
        type: 'scatter',
        data: {
          datasets: [{
            label: firstModel,
            data: sts.ground_truth_similarities.map((gt, i) => ({ x: gt, y: sts.predicted_similarities[i] })),
            backgroundColor: '#6c63ffcc',
            pointRadius: 5,
          }],
        },
        options: {
          ...chartDefaults(),
          scales: {
            x: { ...chartDefaults().scales.x, title: { display: true, text: 'Score humain', color: '#6b7394' }, min: 0, max: 1 },
            y: { ...chartDefaults().scales.y, title: { display: true, text: 'Cosine sim.', color: '#6b7394' }, min: 0, max: 1 },
          },
        },
      });
      state.charts.push(chart);
    }
  }

  // 5. Classification metrics
  if (hasCls) {
    const canvas = makeChartCard('📦 Classification', 'Précision centroïde & score Silhouette');
    const clsModels = models.filter(m => results[m].classification && !results[m].classification.error);
    const chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: clsModels.map(shortName),
        datasets: [
          {
            label: 'Précision (%)',
            data: clsModels.map(m => +(results[m].classification.nearest_centroid_accuracy * 100).toFixed(1)),
            backgroundColor: '#6c63ffcc', borderColor: '#6c63ff', borderWidth: 2, borderRadius: 6,
          },
          {
            label: 'Silhouette (×100)',
            data: clsModels.map(m => +(results[m].classification.silhouette_score * 100).toFixed(1)),
            backgroundColor: '#00d4aa88', borderColor: '#00d4aa', borderWidth: 2, borderRadius: 6,
          },
        ],
      },
      options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, beginAtZero: true } } },
    });
    state.charts.push(chart);
  }
}

/* ── Detail tables ─────────────────────────────────────────────────────────── */
function renderDetailTables(results, models) {
  const hasSpeed = models.some(m => results[m].speed && !results[m].speed.error);
  const hasSTS   = models.some(m => results[m].sts   && !results[m].sts.error);
  const hasCls   = models.some(m => results[m].classification && !results[m].classification.error);

  if (hasSpeed) {
    const card = document.createElement('div');
    card.className = 'table-card';
    card.innerHTML = `
      <div class="table-title">🚀 Résultats de vitesse</div>
      <table>
        <thead><tr>
          <th>Modèle</th><th>Latence moy.</th><th>Latence min</th>
          <th>P95</th><th>Max</th><th>Throughput</th><th>Dimensions</th>
        </tr></thead>
        <tbody id="speed-tbody"></tbody>
      </table>`;
    detailTables.appendChild(card);

    const tbody = card.querySelector('#speed-tbody');
    const speedModels = models.filter(m => results[m].speed && !results[m].speed.error);
    const bestThroughput = Math.max(...speedModels.map(m => results[m].speed.throughput_texts_per_sec));

    speedModels.forEach(model => {
      const s = results[model].speed;
      const isBest = s.throughput_texts_per_sec === bestThroughput;
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${model}</td>
        <td class="td-num">${s.latency_mean_ms} ms</td>
        <td class="td-num">${s.latency_min_ms} ms</td>
        <td class="td-num">${s.latency_p95_ms} ms</td>
        <td class="td-num">${s.latency_max_ms} ms</td>
        <td class="td-num ${isBest ? 'td-good' : ''}">${s.throughput_texts_per_sec} t/s</td>
        <td class="td-num">${s.embedding_dim ?? '—'}</td>`;
      tbody.appendChild(row);
    });

    // Show errors
    models.filter(m => results[m].speed?.error).forEach(m => {
      const row = document.createElement('tr');
      row.innerHTML = `<td>${m}</td><td colspan="6" class="error-text">${results[m].speed.error}</td>`;
      tbody.appendChild(row);
    });
  }

  if (hasSTS) {
    const card = document.createElement('div');
    card.className = 'table-card';
    card.innerHTML = `
      <div class="table-title">🎯 Résultats STS</div>
      <table>
        <thead><tr>
          <th>Modèle</th><th>Pearson r</th><th>Paires testées</th><th>Interprétation</th>
        </tr></thead>
        <tbody id="sts-tbody"></tbody>
      </table>`;
    detailTables.appendChild(card);

    const tbody = card.querySelector('#sts-tbody');
    models.filter(m => results[m].sts && !results[m].sts.error).forEach(model => {
      const s = results[model].sts;
      const pr = s.pearson_r;
      const interpretation = pr > 0.7 ? 'Excellente' : pr > 0.5 ? 'Bonne' : pr > 0.3 ? 'Modérée' : 'Faible';
      const cls = pr > 0.7 ? 'td-good' : pr < 0.3 ? 'td-bad' : '';
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${model}</td>
        <td class="td-num ${cls}">${pr.toFixed(4)}</td>
        <td class="td-num">${s.num_pairs}</td>
        <td class="${cls}">${interpretation}</td>`;
      tbody.appendChild(row);
    });
  }

  if (hasCls) {
    const card = document.createElement('div');
    card.className = 'table-card';
    card.innerHTML = `
      <div class="table-title">📦 Résultats de classification</div>
      <table>
        <thead><tr>
          <th>Modèle</th><th>Précision centroïde</th><th>Score Silhouette</th>
          <th>Textes</th><th>Classes</th>
        </tr></thead>
        <tbody id="cls-tbody"></tbody>
      </table>`;
    detailTables.appendChild(card);

    const tbody = card.querySelector('#cls-tbody');
    models.filter(m => results[m].classification && !results[m].classification.error).forEach(model => {
      const c = results[model].classification;
      const acc = c.nearest_centroid_accuracy;
      const cls = acc > 0.8 ? 'td-good' : acc < 0.4 ? 'td-bad' : '';
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${model}</td>
        <td class="td-num ${cls}">${(acc * 100).toFixed(1)}%</td>
        <td class="td-num">${c.silhouette_score.toFixed(4)}</td>
        <td class="td-num">${c.num_texts}</td>
        <td>${c.classes.join(', ')}</td>`;
      tbody.appendChild(row);
    });
  }
}

/* ── Export ────────────────────────────────────────────────────────────────── */
exportJSON.addEventListener('click', () => {
  if (!state.results) return;
  const blob = new Blob([JSON.stringify(state.results, null, 2)], { type: 'application/json' });
  downloadBlob(blob, 'benchmark_results.json');
});

exportCSV.addEventListener('click', () => {
  if (!state.results) return;
  const rows = [['model','latency_mean_ms','throughput_t_s','embedding_dim','pearson_r','centroid_accuracy','silhouette']];
  Object.entries(state.results).forEach(([model, r]) => {
    rows.push([
      model,
      r.speed?.latency_mean_ms ?? '',
      r.speed?.throughput_texts_per_sec ?? '',
      r.speed?.embedding_dim ?? '',
      r.sts?.pearson_r ?? '',
      r.classification?.nearest_centroid_accuracy ?? '',
      r.classification?.silhouette_score ?? '',
    ]);
  });
  const csv = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  downloadBlob(blob, 'benchmark_results.csv');
});

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */
function shortName(name) {
  return name.length > 22 ? name.slice(0, 20) + '…' : name;
}
