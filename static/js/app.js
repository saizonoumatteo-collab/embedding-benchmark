/* ══════════════════════════════════════════════════════════
   State
══════════════════════════════════════════════════════════ */
const state = {
  models: [], embTagged: new Set(), selected: new Set(),
  results: {}, charts: [], pcaChart: null, gaugeChart: null,
  vizMethod: 'pca', activeTab: 'overview',
};

const PALETTE = ['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#39d353','#f78166','#79c0ff'];
const ALL_TESTS = [
  { key:'speed',          label:'Vitesse',        icon:'🚀', chk:'chk-speed'     },
  { key:'sts',            label:'STS',             icon:'🎯', chk:'chk-sts'       },
  { key:'retrieval',      label:'Retrieval',       icon:'🔎', chk:'chk-retrieval' },
  { key:'classification', label:'Classification',  icon:'📦', chk:'chk-cls'       },
  { key:'robustness',     label:'Robustesse',      icon:'🛡',  chk:'chk-rob'       },
  { key:'multilingual',   label:'Multilingue',     icon:'🌍', chk:'chk-multi'     },
  { key:'negation',       label:'Négation',        icon:'🔄', chk:'chk-negation'  },
  { key:'topic_drift',    label:'Topic Drift',     icon:'📉', chk:'chk-drift'     },
];

const PRESETS = {
  quick:    { speed:1, sts:1, retrieval:0, cls:0, rob:0, multi:0, negation:0, drift:0 },
  standard: { speed:1, sts:1, retrieval:1, cls:1, rob:0, multi:0, negation:0, drift:0 },
  full:     { speed:1, sts:1, retrieval:1, cls:1, rob:1, multi:1, negation:1, drift:1 },
  quality:  { speed:0, sts:1, retrieval:1, cls:1, rob:1, multi:1, negation:1, drift:1 },
};

/* ══════════════════════════════════════════════════════════
   Init
══════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);

(async () => {
  await checkHealth();
  await loadModels();
  initNav();
  initTabs();
  initTooltips();
  initExplorer();
  initHeatmapPanel();
  initHistory();
  initViz();
  initH2H();
  initPresets();
})();

/* ══════════════════════════════════════════════════════════
   Health
══════════════════════════════════════════════════════════ */
async function checkHealth() {
  const el = $('ollama-status');
  try {
    const d = await (await fetch('/api/health')).json();
    if (d.ollama === 'ok') {
      el.className = 'badge badge-ok';
      el.innerHTML = '<span class="dot"></span><span class="badge-label">Ollama connecté</span>';
    } else throw new Error();
  } catch {
    el.className = 'badge badge-error';
    el.innerHTML = '<span class="dot"></span><span class="badge-label">Ollama hors ligne</span>';
  }
}

/* ══════════════════════════════════════════════════════════
   Models
══════════════════════════════════════════════════════════ */
async function loadModels() {
  $('model-list').innerHTML = '<p class="muted-sm">Chargement…</p>';
  try {
    const d = await (await fetch('/api/models')).json();
    state.models = d.models; state.embTagged = new Set(d.embedding_tagged);
    renderModels(); populateAllSelects();
  } catch (e) {
    $('model-list').innerHTML = `<p class="muted-sm" style="color:var(--red)">Erreur : ${e.message}</p>`;
  }
}

function renderModels() {
  const ml = $('model-list'); ml.innerHTML = '';
  if (!state.models.length) { ml.innerHTML = '<p class="muted-sm">Aucun modèle trouvé.</p>'; return; }
  state.models.forEach(name => {
    const lbl = document.createElement('label');
    lbl.className = 'model-item' + (state.selected.has(name) ? ' selected' : '');
    lbl.innerHTML = `<input type="checkbox" ${state.selected.has(name)?'checked':''}/><span class="chk">${state.selected.has(name)?'✓':''}</span><span class="model-name">${name}</span>${state.embTagged.has(name)?'<span class="embed-tag">embed</span>':''}`;
    lbl.addEventListener('change', () => {
      const on = !state.selected.has(name);
      on ? state.selected.add(name) : state.selected.delete(name);
      lbl.classList.toggle('selected', on);
      lbl.querySelector('.chk').textContent = on ? '✓' : '';
      $('run-btn').disabled = state.selected.size === 0;
    });
    ml.appendChild(lbl);
  });
}

function populateAllSelects() {
  ['exp-model','viz-model-select','hm-model','h2h-a','h2h-b'].forEach(id => {
    const el = $(id); if (!el) return;
    const prev = el.value;
    el.innerHTML = `<option value="">— sélectionner —</option>`;
    state.models.forEach(m => el.insertAdjacentHTML('beforeend', `<option ${m===prev?'selected':''} value="${m}">${m}</option>`));
  });
  $('exp-run') && ($('exp-run').disabled = !$('exp-model')?.value);
  $('hm-run')  && ($('hm-run').disabled  = !$('hm-model')?.value);
}

$('refresh-models').addEventListener('click', loadModels);

/* ══════════════════════════════════════════════════════════
   Presets
══════════════════════════════════════════════════════════ */
function initPresets() {
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = PRESETS[btn.dataset.preset];
      $('chk-speed').checked     = !!p.speed;
      $('chk-sts').checked       = !!p.sts;
      $('chk-retrieval').checked = !!p.retrieval;
      $('chk-cls').checked       = !!p.cls;
      $('chk-rob').checked       = !!p.rob;
      $('chk-multi').checked     = !!p.multi;
      $('chk-negation').checked  = !!p.negation;
      $('chk-drift').checked     = !!p.drift;
    });
  });
}

/* ══════════════════════════════════════════════════════════
   Nav
══════════════════════════════════════════════════════════ */
function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const v = btn.dataset.view;
      ['panel-benchmark','panel-explorer','panel-heatmap','panel-history'].forEach(p => $(p).classList.add('hidden'));
      $(`panel-${v}`).classList.remove('hidden');
      if (v === 'heatmap') showHeatmapView();
      else if (state.results && Object.keys(state.results).length) showResultsView();
    });
  });
}

function showResultsView() {
  $('hero').classList.add('hidden');
  $('heatmap-view').classList.add('hidden');
  if (Object.keys(state.results).length) {
    $('results-view').classList.remove('hidden');
  } else {
    $('hero').classList.remove('hidden');
  }
}
function showHeatmapView() {
  $('hero').classList.add('hidden');
  $('results-view').classList.add('hidden');
  $('heatmap-view').classList.remove('hidden');
}

/* ══════════════════════════════════════════════════════════
   Tabs
══════════════════════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab===name));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
  $(`tab-${name}`)?.classList.remove('hidden');
  state.activeTab = name;
}

/* ══════════════════════════════════════════════════════════
   Tooltips
══════════════════════════════════════════════════════════ */
function initTooltips() {
  const tip = $('tooltip');
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-tip]');
    if (!el) { tip.classList.add('hidden'); return; }
    tip.textContent = el.dataset.tip; tip.classList.remove('hidden');
  });
  document.addEventListener('mousemove', e => { tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px'; });
  document.addEventListener('mouseout',  e => { if (!e.target.closest('[data-tip]')) tip.classList.add('hidden'); });
}

/* ══════════════════════════════════════════════════════════
   Run benchmark (SSE)
══════════════════════════════════════════════════════════ */
$('run-btn').addEventListener('click', runBenchmark);

async function runBenchmark() {
  const models = [...state.selected];
  if (!models.length) return;

  // Custom STS pairs
  const rawSTS = $('custom-sts').value.trim();
  const customSTS = rawSTS ? rawSTS.split('\n').map(l => {
    const parts = l.split('|').map(p => p.trim());
    return parts.length === 3 ? [parts[0], parts[1], parseFloat(parts[2])] : null;
  }).filter(Boolean) : null;

  const cfg = {
    models,
    run_speed:          $('chk-speed').checked,
    run_sts:            $('chk-sts').checked,
    run_retrieval:      $('chk-retrieval').checked,
    run_classification: $('chk-cls').checked,
    run_robustness:     $('chk-rob').checked,
    run_multilingual:   $('chk-multi').checked,
    run_negation:       $('chk-negation').checked,
    run_topic_drift:    $('chk-drift').checked,
    custom_texts:  (() => { const v=$('custom-texts').value.trim(); return v?v.split('\n').map(l=>l.trim()).filter(Boolean):null; })(),
    custom_sts_pairs: customSTS,
  };
  if (!Object.values(cfg).slice(1,9).some(Boolean)) return alert('Sélectionnez au moins un test.');

  // Reset & show progress
  state.results = {}; state.charts.forEach(c=>c.destroy()); state.charts=[];
  $('hero').classList.add('hidden'); $('results-view').classList.add('hidden'); $('heatmap-view').classList.add('hidden');
  $('progress-view').classList.remove('hidden');
  $('run-btn').disabled = true;

  const steps = {}; $('progress-steps').innerHTML = '';
  models.forEach(m => {
    steps[m] = {};
    const mDiv = document.createElement('div');
    mDiv.className='progress-step';
    mDiv.innerHTML=`<span style="font-size:.85rem">📦</span><span style="flex:1;font-weight:700;font-size:.8rem">${m}</span>`;
    $('progress-steps').appendChild(mDiv);
    ALL_TESTS.forEach(t => {
      if (!cfg[`run_${t.key}`]) return;
      const div=document.createElement('div'); div.className='progress-step'; div.id=`step-${m}-${t.key}`;
      div.innerHTML=`<span style="margin-left:16px">${t.icon}</span><span class="step-label step-pending" style="flex:1">${t.label}</span><span class="step-status"></span>`;
      $('progress-steps').appendChild(div);
    });
  });

  function setStep(model,test,status,icon='') {
    const el=$(`step-${model}-${test}`); if(!el) return;
    el.querySelector('.step-label').className=`step-label step-${status}`;
    el.querySelector('.step-status').textContent=icon;
  }

  try {
    const resp = await fetch('/api/benchmark/stream', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
    const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(true) {
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n'); buf=lines.pop();
      for(const line of lines) {
        if(!line.startsWith('data: ')) continue;
        const ev=JSON.parse(line.slice(6));
        if(ev.type==='model_start') { $('progress-model').textContent=ev.model; $('progress-test').textContent='Démarrage…'; }
        else if(ev.type==='test_start') { $('progress-test').textContent=ALL_TESTS.find(t=>t.key===ev.test)?.label+'…'; setStep(ev.model,ev.test,'running','⏳'); }
        else if(ev.type==='test_done')  { setStep(ev.model,ev.test,'done','✓'); if(!state.results[ev.model]) state.results[ev.model]={}; state.results[ev.model][ev.test]=ev.data; }
        else if(ev.type==='test_error') { setStep(ev.model,ev.test,'error','✗'); if(!state.results[ev.model]) state.results[ev.model]={}; state.results[ev.model][ev.test]={error:ev.error}; }
        else if(ev.type==='model_done') { state.results[ev.model]=ev.result; }
        else if(ev.type==='done') { $('progress-view').classList.add('hidden'); $('results-view').classList.remove('hidden'); renderResults(); }
      }
    }
  } catch(e) { $('progress-test').textContent='❌ '+e.message; }
  finally { $('run-btn').disabled=false; }
}

/* ══════════════════════════════════════════════════════════
   Render results
══════════════════════════════════════════════════════════ */
function renderResults() {
  const models = Object.keys(state.results);
  renderOverview(models);
  renderSpeedTab(models);
  renderSTSTab(models);
  renderRetrievalTab(models);
  renderClassificationTab(models);
  renderRobustnessTab(models);
  renderMultilingualTab(models);
  renderNegationTab(models);
  renderDriftTab(models);
  initVizForResults(models);
  populateH2H(models);
  switchTab('overview');
}

/* ── Chart helpers ───────────────────────────────────────── */
const CD = () => ({
  responsive:true, maintainAspectRatio:false,
  plugins:{
    legend:{labels:{color:'#8b949e',font:{size:11}}},
    tooltip:{backgroundColor:'#1c2128',titleColor:'#e6edf3',bodyColor:'#8b949e',borderColor:'#30363d',borderWidth:1},
  },
  scales:{
    x:{ticks:{color:'#6e7681',font:{size:11}},grid:{color:'#21262d'}},
    y:{ticks:{color:'#6e7681',font:{size:11}},grid:{color:'#21262d'}},
  },
});
const s = str => str.length>16 ? str.slice(0,14)+'…' : str;
const cc = (i,a='cc') => PALETTE[i%PALETTE.length]+(a==='full'?'':a);

function makeCard(parentId, title, sub, height='210px') {
  const card=document.createElement('div'); card.className='chart-card';
  const wrap=document.createElement('div'); wrap.className='chart-wrap'; wrap.style.height=height;
  const canvas=document.createElement('canvas'); wrap.appendChild(canvas);
  card.innerHTML=`<div class="chart-title">${title}</div><div class="chart-sub">${sub}</div>`;
  card.appendChild(wrap); $(parentId).appendChild(card); return canvas;
}

function barChart(canvas, labels, datasets, extraOpts={}) {
  const c=new Chart(canvas,{type:'bar',data:{labels:labels.map(s),datasets},options:{...CD(),...extraOpts}});
  state.charts.push(c); return c;
}

/* ── Overview ────────────────────────────────────────────── */
function renderOverview(models) {
  // Leaderboard
  const lb = $('leaderboard'); lb.innerHTML='';
  const sorted = [...models].sort((a,b)=>(state.results[b]?.overall_score??0)-(state.results[a]?.overall_score??0));
  const maxSc = state.results[sorted[0]]?.overall_score??100;
  sorted.forEach((m,i) => {
    const r=state.results[m]; const sc=r?.overall_score??null;
    const rank=['🥇','🥈','🥉'][i]||(i+1+'.');
    const pills=ALL_TESTS.filter(t=>r?.[t.key]&&!r[t.key].error).map(t=>`<span class="lb-pill">${t.icon}</span>`).join('');
    const row=document.createElement('div'); row.className='lb-row';
    row.innerHTML=`<div class="lb-rank lb-rank-${i+1}">${rank}</div><div class="lb-name">${m}</div><div class="lb-bar-wrap"><div class="lb-bar" style="width:${sc?sc/maxSc*100:0}%;background:${PALETTE[i%PALETTE.length]}"></div></div><div class="lb-pills">${pills}</div><div class="lb-score">${sc??'—'}<small style="font-size:.58rem;color:var(--muted)">/100</small></div>`;
    lb.appendChild(row);
  });

  // Radar
  const rc=$('radar-chart'); if(rc._chart) rc._chart.destroy();
  const radarLabels=['STS','Retrieval','Classif.','Robustesse','Multilingue','Vitesse','Négation','Drift'];
  const rDat=models.map((m,i)=>{
    const r=state.results[m]||{};
    return {label:m,data:[
      r.sts?.pearson_r!=null?Math.max(0,r.sts.pearson_r):null,
      r.retrieval?.ndcg_at_5??null,
      r.classification?.nearest_centroid_accuracy??null,
      r.robustness?.discrimination_ratio!=null?Math.min(1,Math.max(0,r.robustness.discrimination_ratio-1)):null,
      r.multilingual?.alignment_score!=null?Math.min(1,Math.max(0,r.multilingual.alignment_score+0.1)):null,
      r.speed?.latency_mean_ms!=null?Math.min(1,1/(1+r.speed.latency_mean_ms/80)):null,
      r.negation?.negation_awareness??null,
      r.topic_drift?.monotonicity_score??null,
    ],borderColor:PALETTE[i%PALETTE.length],backgroundColor:PALETTE[i%PALETTE.length]+'22',pointBackgroundColor:PALETTE[i%PALETTE.length],borderWidth:2};
  });
  const radarC=new Chart(rc,{type:'radar',data:{labels:radarLabels,datasets:rDat},options:{responsive:true,maintainAspectRatio:false,scales:{r:{min:0,max:1,ticks:{display:false},grid:{color:'#30363d'},pointLabels:{color:'#8b949e',font:{size:10}}}},plugins:{legend:{labels:{color:'#8b949e',font:{size:10}}},tooltip:{backgroundColor:'#1c2128',titleColor:'#e6edf3',bodyColor:'#8b949e'}}}});
  rc._chart=radarC; state.charts.push(radarC);

  // Overall bar
  const oc=$('overall-chart'); if(oc._chart) oc._chart.destroy();
  const ovC=barChart(oc,sorted,
    [{label:'Score /100',data:sorted.map(m=>state.results[m]?.overall_score??0),backgroundColor:sorted.map((_,i)=>cc(i)),borderColor:sorted.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:6}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:100}}});
  oc._chart=ovC;
}

/* ── Speed ───────────────────────────────────────────────── */
function renderSpeedTab(models) {
  const c=$('speed-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.speed?.latency_mean_ms!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  barChart(makeCard('speed-charts','⏱ Latence moyenne','ms — moins = mieux'), good,
    [{label:'Latence moy.',data:good.map(m=>state.results[m].speed.latency_mean_ms),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:5}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true}}});

  // Grouped latency (min/P50/P95/max)
  const dc=makeCard('speed-charts','📊 Distribution latence','Min / P50 / P95 / Max');
  const distC=new Chart(dc,{type:'bar',data:{labels:good.map(s),datasets:[
    {label:'Min', data:good.map(m=>state.results[m].speed.latency_min_ms), backgroundColor:'#3fb95055',borderColor:'#3fb950',borderWidth:2,borderRadius:4},
    {label:'P50', data:good.map(m=>state.results[m].speed.latency_p50_ms), backgroundColor:'#58a6ff55',borderColor:'#58a6ff',borderWidth:2,borderRadius:4},
    {label:'P95', data:good.map(m=>state.results[m].speed.latency_p95_ms), backgroundColor:'#d2992255',borderColor:'#d29922',borderWidth:2,borderRadius:4},
    {label:'Max', data:good.map(m=>state.results[m].speed.latency_max_ms), backgroundColor:'#f8514955',borderColor:'#f85149',borderWidth:2,borderRadius:4},
  ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true}}}});
  state.charts.push(distC);

  barChart(makeCard('speed-charts','⚡ Throughput','Textes/sec — plus = mieux'), good,
    [{label:'Throughput',data:good.map(m=>state.results[m].speed.throughput_per_sec),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:5}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true}}});

  // Latency scatter (all individual measurements)
  if(good[0] && state.results[good[0]].speed.all_latencies) {
    const sc=makeCard('speed-charts','🔵 Toutes les latences — '+s(good[0]),'Chaque point = une requête');
    const scC=new Chart(sc,{type:'scatter',data:{datasets:good.map((m,i)=>({label:m,data:state.results[m].speed.all_latencies?.map((v,j)=>({x:j+1,y:v}))||[],backgroundColor:cc(i),pointRadius:4}))},options:{...CD(),scales:{x:{...CD().scales.x,title:{display:true,text:'#',color:'#6e7681'}},y:{...CD().scales.y,title:{display:true,text:'ms',color:'#6e7681'},beginAtZero:true}}}});
    state.charts.push(scC);
  }

  const best=Math.max(...good.map(m=>state.results[m].speed.throughput_per_sec));
  $('speed-table').innerHTML=`<div class="table-title">🚀 Résultats de vitesse</div>
  <table><thead><tr><th>Modèle</th><th>Moy.</th><th>P50</th><th>P95</th><th>Max</th><th>Throughput</th><th>Dim.</th></tr></thead><tbody>${good.map(m=>{
    const sp=state.results[m].speed; const b=sp.throughput_per_sec===best;
    return `<tr><td class="mono">${m}</td><td class="num">${sp.latency_mean_ms}ms</td><td class="num">${sp.latency_p50_ms}ms</td><td class="num">${sp.latency_p95_ms}ms</td><td class="num">${sp.latency_max_ms}ms</td><td class="num ${b?'good':''}">${sp.throughput_per_sec}t/s</td><td class="num">${sp.embedding_dim??'—'}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── STS ─────────────────────────────────────────────────── */
function renderSTSTab(models) {
  const c=$('sts-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.sts?.pearson_r!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  barChart(makeCard('sts-charts','📈 Corrélation de Pearson','Vs annotations humaines · plus proche de 1 = mieux'), good,
    [{label:'Pearson r',data:good.map(m=>state.results[m].sts.pearson_r),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:5}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,min:-1,max:1}}});

  // Scatter predicted vs ground truth
  const fm=good[0]; const sts=state.results[fm].sts;
  const sc=makeCard('sts-charts',`🔬 Prédictions vs réalité — ${s(fm)}`,'Cosine sim. prédite vs score annoté');
  const scC=new Chart(sc,{type:'scatter',data:{datasets:[{label:fm,data:sts.ground_truth.map((gt,i)=>({x:gt,y:sts.predicted[i]})),backgroundColor:'#58a6ffbb',pointRadius:5}]},options:{...CD(),scales:{x:{...CD().scales.x,title:{display:true,text:'Score humain',color:'#6e7681'},min:0,max:1},y:{...CD().scales.y,title:{display:true,text:'Cosine sim.',color:'#6e7681'},min:0,max:1}}}});
  state.charts.push(scC);

  // Distribution of similarity scores histogram-like
  if(good.length) {
    const bins=10; const binEdges=Array.from({length:bins+1},(_,i)=>i/bins);
    const hc=makeCard('sts-charts','📊 Distribution des similarités prédites','Histogramme des scores cosinus');
    const hDat=good.map((m,i)=>{
      const sims=state.results[m].sts.predicted;
      const counts=new Array(bins).fill(0);
      sims.forEach(v=>{const b=Math.min(Math.floor(v*bins),bins-1);counts[b]++;});
      return {label:m,data:counts,backgroundColor:cc(i),borderColor:PALETTE[i%PALETTE.length],borderWidth:2,borderRadius:3};
    });
    const hC=new Chart(hc,{type:'bar',data:{labels:binEdges.slice(0,-1).map((v,i)=>`${v.toFixed(1)}–${binEdges[i+1].toFixed(1)}`),datasets:hDat},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,title:{display:true,text:'Fréquence',color:'#6e7681'}}}}});
    state.charts.push(hC);
  }

  $('sts-table').innerHTML=`<div class="table-title">🎯 Résultats STS</div>
  <table><thead><tr><th>Modèle</th><th>Pearson r</th><th>Paires</th><th>Interprétation</th></tr></thead><tbody>${good.map(m=>{
    const pr=state.results[m].sts.pearson_r; const cls=pr>0.7?'good':pr>0.5?'ok':'bad';
    const lbl=pr>0.7?'Excellent':pr>0.5?'Bon':pr>0.3?'Modéré':'Faible';
    return `<tr><td class="mono">${m}</td><td class="num ${cls}">${pr.toFixed(4)}</td><td class="num">${state.results[m].sts.num_pairs}</td><td class="${cls}">${lbl}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Retrieval ───────────────────────────────────────────── */
function renderRetrievalTab(models) {
  const c=$('retrieval-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.retrieval?.ndcg_at_5!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  const rc=makeCard('retrieval-charts','🔎 Métriques de retrieval','NDCG@5 · Recall@5 · MRR — plus = mieux');
  const rC=new Chart(rc,{type:'bar',data:{labels:good.map(s),datasets:[
    {label:'NDCG@5',  data:good.map(m=>state.results[m].retrieval.ndcg_at_5),  backgroundColor:'#58a6ff88',borderColor:'#58a6ff',borderWidth:2,borderRadius:4},
    {label:'Recall@5',data:good.map(m=>state.results[m].retrieval.recall_at_5),backgroundColor:'#3fb95088',borderColor:'#3fb950',borderWidth:2,borderRadius:4},
    {label:'MRR',     data:good.map(m=>state.results[m].retrieval.mrr),         backgroundColor:'#d2992288',borderColor:'#d29922',borderWidth:2,borderRadius:4},
  ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}}});
  state.charts.push(rC);

  // Per-query heatmap-bar for first model
  const fm=good[0]; const pq=state.results[fm].retrieval.per_query;
  if(pq?.length) {
    const qc=makeCard('retrieval-charts',`📋 NDCG@5 par requête — ${s(fm)}`,'Performance individuelle sur chaque requête');
    barChart(qc, pq.map(q=>q.query.slice(0,30)),
      [{label:'NDCG@5',data:pq.map(q=>q['ndcg@5']),backgroundColor:pq.map(q=>q['ndcg@5']>0.8?'#3fb95088':q['ndcg@5']>0.5?'#d2992288':'#f8514988'),borderColor:pq.map(q=>q['ndcg@5']>0.8?'#3fb950':q['ndcg@5']>0.5?'#d29922':'#f85149'),borderWidth:2,borderRadius:4}],
      {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}});
  }

  $('retrieval-table').innerHTML=`<div class="table-title">🔎 Résultats de retrieval</div>
  <table><thead><tr><th>Modèle</th><th>NDCG@5</th><th>NDCG@3</th><th>Recall@5</th><th>MRR</th></tr></thead><tbody>${good.map(m=>{
    const r=state.results[m].retrieval; const v=x=>x>0.8?'good':x>0.5?'ok':'bad';
    return `<tr><td class="mono">${m}</td><td class="num ${v(r.ndcg_at_5)}">${r.ndcg_at_5.toFixed(4)}</td><td class="num">${r.ndcg_at_3.toFixed(4)}</td><td class="num ${v(r.recall_at_5)}">${r.recall_at_5.toFixed(4)}</td><td class="num ${v(r.mrr)}">${r.mrr.toFixed(4)}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Classification ──────────────────────────────────────── */
function renderClassificationTab(models) {
  const c=$('cls-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.classification?.nearest_centroid_accuracy!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  const mc=makeCard('cls-charts','📦 Métriques de classification','Précision centroïde & Silhouette');
  const mC=new Chart(mc,{type:'bar',data:{labels:good.map(s),datasets:[
    {label:'Précision (%)',   data:good.map(m=>+(state.results[m].classification.nearest_centroid_accuracy*100).toFixed(1)),backgroundColor:'#58a6ff88',borderColor:'#58a6ff',borderWidth:2,borderRadius:4},
    {label:'Silhouette ×100',data:good.map(m=>+(state.results[m].classification.silhouette_score*100).toFixed(1)),backgroundColor:'#3fb95088',borderColor:'#3fb950',borderWidth:2,borderRadius:4},
  ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true}}}});
  state.charts.push(mC);

  // Confusion matrix for first model
  const fm=good[0]; const clsData=state.results[fm].classification;
  if(clsData.confusion_matrix) {
    const cmCard=document.createElement('div'); cmCard.className='chart-card';
    cmCard.innerHTML=`<div class="chart-title">🗃 Matrice de confusion — ${s(fm)}</div><div class="chart-sub">Lignes = vrai label · Colonnes = prédit</div>`;
    const canvas=document.createElement('canvas'); canvas.style.maxWidth='100%';
    cmCard.appendChild(canvas); c.appendChild(cmCard);
    drawConfusionMatrix(canvas, clsData.confusion_matrix, clsData.classes);
  }

  $('cls-table').innerHTML=`<div class="table-title">📦 Résultats de classification</div>
  <table><thead><tr><th>Modèle</th><th>Précision</th><th>Silhouette</th><th>Textes</th><th>Classes</th></tr></thead><tbody>${good.map(m=>{
    const cl=state.results[m].classification; const cls=cl.nearest_centroid_accuracy>0.8?'good':cl.nearest_centroid_accuracy>0.5?'ok':'bad';
    return `<tr><td class="mono">${m}</td><td class="num ${cls}">${(cl.nearest_centroid_accuracy*100).toFixed(1)}%</td><td class="num">${cl.silhouette_score.toFixed(4)}</td><td class="num">${cl.num_texts}</td><td>${cl.classes.join(', ')}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Robustness ──────────────────────────────────────────── */
function renderRobustnessTab(models) {
  const c=$('rob-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.robustness?.discrimination_ratio!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  const rc=makeCard('rob-charts','🛡 Intra vs inter-groupe','Paraphrases (intra) doivent être plus proches');
  const rC=new Chart(rc,{type:'bar',data:{labels:good.map(s),datasets:[
    {label:'Intra-groupe',data:good.map(m=>state.results[m].robustness.avg_intra_similarity),backgroundColor:'#3fb95088',borderColor:'#3fb950',borderWidth:2,borderRadius:4},
    {label:'Inter-groupe',data:good.map(m=>state.results[m].robustness.avg_inter_similarity),backgroundColor:'#f8514988',borderColor:'#f85149',borderWidth:2,borderRadius:4},
  ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}}});
  state.charts.push(rC);

  barChart(makeCard('rob-charts','📐 Ratio de discrimination','intra/inter · >1.5 bon · >2 excellent'), good,
    [{label:'Ratio',data:good.map(m=>state.results[m].robustness.discrimination_ratio),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:4}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true}}});

  $('rob-table').innerHTML=`<div class="table-title">🛡 Robustesse</div>
  <table><thead><tr><th>Modèle</th><th>Sim. intra</th><th>Sim. inter</th><th>Ratio</th></tr></thead><tbody>${good.map(m=>{
    const r=state.results[m].robustness; const cls=r.discrimination_ratio>2?'good':r.discrimination_ratio>1.5?'ok':'bad';
    return `<tr><td class="mono">${m}</td><td class="num">${r.avg_intra_similarity.toFixed(4)}</td><td class="num">${r.avg_inter_similarity.toFixed(4)}</td><td class="num ${cls}">${r.discrimination_ratio.toFixed(4)}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Multilingual ────────────────────────────────────────── */
function renderMultilingualTab(models) {
  const c=$('multi-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.multilingual?.alignment_score!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  const mc=makeCard('multi-charts','🌍 Sim. traduction vs non-traduction','Bleu = paires traduites · Rouge = paires aléatoires');
  const mC=new Chart(mc,{type:'bar',data:{labels:good.map(s),datasets:[
    {label:'Traduction',     data:good.map(m=>state.results[m].multilingual.avg_similarity),         backgroundColor:'#58a6ff88',borderColor:'#58a6ff',borderWidth:2,borderRadius:4},
    {label:'Non-traduction', data:good.map(m=>state.results[m].multilingual.avg_non_translation_sim),backgroundColor:'#f8514988',borderColor:'#f85149',borderWidth:2,borderRadius:4},
  ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}}});
  state.charts.push(mC);

  barChart(makeCard('multi-charts','📐 Score d\'alignement','sim_traduction − sim_non-traduction · >0.3 = bon modèle multilingue'), good,
    [{label:'Alignement',data:good.map(m=>state.results[m].multilingual.alignment_score),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:4}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y}}});

  $('multi-table').innerHTML=`<div class="table-title">🌍 Multilingue</div>
  <table><thead><tr><th>Modèle</th><th>Sim. trad.</th><th>Sim. non-trad.</th><th>Alignement</th></tr></thead><tbody>${good.map(m=>{
    const r=state.results[m].multilingual; const cls=r.alignment_score>0.3?'good':r.alignment_score>0.1?'ok':'bad';
    return `<tr><td class="mono">${m}</td><td class="num">${r.avg_similarity.toFixed(4)}</td><td class="num">${r.avg_non_translation_sim.toFixed(4)}</td><td class="num ${cls}">${r.alignment_score.toFixed(4)}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Negation ────────────────────────────────────────────── */
function renderNegationTab(models) {
  const c=$('neg-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.negation?.avg_negation_similarity!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  barChart(makeCard('neg-charts','🔄 Similarité cosinus des paires opposées','Bas = le modèle distingue bien les opposés (moins = mieux ici)'), good,
    [{label:'Sim. moy. négation',data:good.map(m=>state.results[m].negation.avg_negation_similarity),backgroundColor:good.map(m=>state.results[m].negation.avg_negation_similarity>0.7?'#f8514988':'#3fb95088'),borderColor:good.map(m=>state.results[m].negation.avg_negation_similarity>0.7?'#f85149':'#3fb950'),borderWidth:2,borderRadius:4}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}});

  barChart(makeCard('neg-charts','🎯 Score de conscience des négations','1 − sim_moy · plus élevé = mieux'), good,
    [{label:'Négation awareness',data:good.map(m=>state.results[m].negation.negation_awareness),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:4}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}});

  // By category for first model
  const fm=good[0]; const byCat=state.results[fm].negation.by_category;
  if(byCat) {
    const cats=Object.keys(byCat);
    barChart(makeCard('neg-charts',`📊 Par catégorie — ${s(fm)}`,'Sentiment · Factuel · Logique · Moral'), cats,
      [{label:'Sim. moy.',data:cats.map(c=>byCat[c]),backgroundColor:cats.map((_,i)=>cc(i)),borderColor:cats.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:4}],
      {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}});
  }

  $('neg-table').innerHTML=`<div class="table-title">🔄 Négation</div>
  <table><thead><tr><th>Modèle</th><th>Sim. moy. opposés</th><th>Score conscience</th><th>Paires</th><th>Interprétation</th></tr></thead><tbody>${good.map(m=>{
    const n=state.results[m].negation; const cls=n.negation_awareness>0.6?'good':n.negation_awareness>0.4?'ok':'bad';
    const lbl=n.negation_awareness>0.6?'Excellente':n.negation_awareness>0.4?'Bonne':'Faible';
    return `<tr><td class="mono">${m}</td><td class="num ${n.avg_negation_similarity>0.7?'bad':''}">${n.avg_negation_similarity.toFixed(4)}</td><td class="num ${cls}">${n.negation_awareness.toFixed(4)}</td><td class="num">${n.num_pairs}</td><td class="${cls}">${lbl}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ── Topic Drift ─────────────────────────────────────────── */
function renderDriftTab(models) {
  const c=$('drift-charts'); c.innerHTML='';
  const good=models.filter(m=>state.results[m]?.topic_drift?.monotonicity_score!=null);
  if(!good.length){c.innerHTML='<p class="muted-sm">Pas de données.</p>';return;}

  // Line charts for each drift set (first model)
  const fm=good[0]; const sets=state.results[fm].topic_drift.per_set;
  if(sets?.length) {
    sets.forEach((ds,si) => {
      const cv=makeCard('drift-charts',`📉 Drift ${si+1} — ${ds.anchor.slice(0,45)}…`,'Similarité décroissante avec la distance sémantique','200px');
      const labels=['Très proche','Proche','Moyen','Lointain','Très lointain'];
      const lC=new Chart(cv,{type:'line',data:{labels,datasets:[
        {label:'Observé',     data:ds.level_sims, borderColor:'#58a6ff',backgroundColor:'#58a6ff22',tension:.3,fill:true,pointRadius:5},
        {label:'Attendu',     data:ds.expected,   borderColor:'#3fb95066',borderDash:[5,5],tension:.3,fill:false,pointRadius:3},
      ]},options:{...CD(),scales:{...CD().scales,y:{...CD().scales.y,min:0,max:1}}}});
      state.charts.push(lC);
    });
  }

  barChart(makeCard('drift-charts','📊 Score de monotonicité','Fraction de niveaux consécutifs où la sim. décroît · 1.0 = parfait'), good,
    [{label:'Monotonicité',data:good.map(m=>state.results[m].topic_drift.monotonicity_score),backgroundColor:good.map((_,i)=>cc(i)),borderColor:good.map((_,i)=>PALETTE[i%PALETTE.length]),borderWidth:2,borderRadius:4}],
    {plugins:{...CD().plugins,legend:{display:false}},scales:{...CD().scales,y:{...CD().scales.y,beginAtZero:true,max:1}}});

  $('drift-table').innerHTML=`<div class="table-title">📉 Topic Drift</div>
  <table><thead><tr><th>Modèle</th><th>Monotonicité</th><th>Corrélation vs attendu</th><th>Sets</th></tr></thead><tbody>${good.map(m=>{
    const d=state.results[m].topic_drift; const cls=d.monotonicity_score>0.8?'good':d.monotonicity_score>0.5?'ok':'bad';
    return `<tr><td class="mono">${m}</td><td class="num ${cls}">${d.monotonicity_score.toFixed(4)}</td><td class="num">${d.correlation_with_expected.toFixed(4)}</td><td class="num">${d.num_drift_sets}</td></tr>`;
  }).join('')}</tbody></table>`;
}

/* ══════════════════════════════════════════════════════════
   Confusion Matrix (canvas)
══════════════════════════════════════════════════════════ */
function drawConfusionMatrix(canvas, matrix, labels) {
  const n = labels.length;
  const margin = 70; const cell = 48;
  canvas.width  = margin + n * cell;
  canvas.height = margin + n * cell;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1c2128';
  ctx.fillRect(0,0,canvas.width,canvas.height);

  const maxVal = Math.max(...matrix.flat());

  for(let i=0;i<n;i++) {
    for(let j=0;j<n;j++) {
      const v=matrix[i][j]; const t=v/maxVal;
      ctx.fillStyle = i===j ? `rgba(63,185,80,${0.15+t*0.7})` : `rgba(248,81,73,${t*0.7})`;
      ctx.fillRect(margin+j*cell, margin+i*cell, cell-1, cell-1);
      ctx.fillStyle = t>0.5?'#0d1117':'#e6edf3';
      ctx.font='bold 13px Segoe UI'; ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(v, margin+j*cell+cell/2, margin+i*cell+cell/2);
    }
  }
  ctx.fillStyle='#8b949e'; ctx.font='11px Segoe UI';
  labels.forEach((lbl,i)=>{
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(lbl.slice(0,8), margin-4, margin+i*cell+cell/2);
    ctx.save(); ctx.translate(margin+i*cell+cell/2, margin-4);
    ctx.rotate(-Math.PI/3); ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(lbl.slice(0,8),0,0); ctx.restore();
  });
  ctx.fillStyle='#6e7681'; ctx.font='10px Segoe UI';
  ctx.textAlign='center'; ctx.fillText('Prédit →',canvas.width/2+margin/2, 10);
  ctx.save(); ctx.translate(10,canvas.height/2+margin/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Réel →',0,0); ctx.restore();
}

/* ══════════════════════════════════════════════════════════
   Head-to-head
══════════════════════════════════════════════════════════ */
function initH2H() { $('h2h-run').addEventListener('click', renderH2H); }
function populateH2H(models) {
  ['h2h-a','h2h-b'].forEach(id => {
    $(id).innerHTML='<option value="">— Modèle —</option>';
    models.forEach((m,i)=>$(id).insertAdjacentHTML('beforeend',`<option ${i===(id==='h2h-a'?0:1)?'selected':''} value="${m}">${m}</option>`));
  });
}
function renderH2H() {
  const a=$('h2h-a').value; const b=$('h2h-b').value;
  if(!a||!b||a===b) return alert('Sélectionnez deux modèles différents.');
  const ra=state.results[a]; const rb=state.results[b];
  const METRICS=[
    {label:'Score global',  fn:r=>r.overall_score??0, fmt:v=>v.toFixed(1), max:100},
    {label:'STS Pearson r', fn:r=>r.sts?.pearson_r??0, fmt:v=>v.toFixed(3), max:1},
    {label:'NDCG@5',        fn:r=>r.retrieval?.ndcg_at_5??0, fmt:v=>v.toFixed(3), max:1},
    {label:'MRR',           fn:r=>r.retrieval?.mrr??0, fmt:v=>v.toFixed(3), max:1},
    {label:'Précision cls', fn:r=>(r.classification?.nearest_centroid_accuracy??0)*100, fmt:v=>v.toFixed(1)+'%', max:100},
    {label:'Robustesse',    fn:r=>r.robustness?.discrimination_ratio??0, fmt:v=>v.toFixed(3), max:3},
    {label:'Alignement ML', fn:r=>r.multilingual?.alignment_score??0, fmt:v=>v.toFixed(3), max:1},
    {label:'Négation',      fn:r=>r.negation?.negation_awareness??0, fmt:v=>v.toFixed(3), max:1},
    {label:'Drift mono.',   fn:r=>r.topic_drift?.monotonicity_score??0, fmt:v=>v.toFixed(3), max:1},
    {label:'Latence (ms)',  fn:r=>r.speed?.latency_mean_ms??0, fmt:v=>v.toFixed(1)+'ms', max:null, invert:true},
  ];
  let aWins=0, bWins=0;
  const rows=METRICS.map(m=>{
    const va=m.fn(ra); const vb=m.fn(rb);
    const aW = m.invert ? va<vb : va>vb;
    const bW = m.invert ? vb<va : vb>va;
    if(aW) aWins++; else if(bW) bWins++;
    const maxV = m.max ?? Math.max(va,vb)*1.2 || 1;
    const pctA=Math.min(100,(va/maxV)*100); const pctB=Math.min(100,(vb/maxV)*100);
    return `<div class="h2h-metric-row">
      <div class="h2h-bar-left ${aW?'h2h-winner':'h2h-loser'}" style="display:flex;align-items:center;justify-content:flex-end;padding-right:5px;background:${aW?'#3fb95033':'var(--bg-h)'};height:26px;border-radius:4px">
        <span style="font-size:.72rem;font-weight:700;color:${aW?'#3fb950':'#8b949e'}">${m.fmt(va)}</span>
        <div style="width:${pctA*0.6}%;height:6px;background:${aW?'#3fb950':'#6e7681'};border-radius:2px;margin-left:5px"></div>
      </div>
      <div style="text-align:center;font-size:.68rem;color:var(--muted);white-space:nowrap;padding:0 6px">${m.label}</div>
      <div class="${bW?'h2h-winner':'h2h-loser'}" style="display:flex;align-items:center;padding-left:5px;background:${bW?'#3fb95033':'var(--bg-h)'};height:26px;border-radius:4px">
        <div style="width:${pctB*0.6}%;height:6px;background:${bW?'#3fb950':'#6e7681'};border-radius:2px;margin-right:5px"></div>
        <span style="font-size:.72rem;font-weight:700;color:${bW?'#3fb950':'#8b949e'}">${m.fmt(vb)}</span>
      </div>
    </div>`;
  }).join('');
  $('h2h-result').innerHTML=`
    <div style="display:flex;justify-content:space-between;margin-bottom:10px;font-size:.82rem;font-weight:700">
      <span style="color:var(--accent)">${a} <small style="color:var(--muted)">${aWins} victoires</small></span>
      <span style="color:var(--muted)">vs</span>
      <span style="color:var(--purple)">${b} <small style="color:var(--muted)">${bWins} victoires</small></span>
    </div>
    <div class="h2h-result-grid">${rows}</div>`;
  $('h2h-result').classList.remove('hidden');
}

/* ══════════════════════════════════════════════════════════
   Visualization (PCA / t-SNE)
══════════════════════════════════════════════════════════ */
function initViz() {
  document.querySelectorAll('.method-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.method-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active'); state.vizMethod=btn.dataset.method;
    });
  });
  $('viz-run-btn').addEventListener('click', runVisualization);
}
function initVizForResults(models) {
  const sel=$('viz-model-select'); sel.innerHTML='';
  models.forEach(m=>sel.insertAdjacentHTML('beforeend',`<option value="${m}">${m}</option>`));
  // Auto-render PCA from already computed classification data
  const fm=models[0]; const cls=state.results[fm]?.classification;
  if(cls?.pca_points) renderPCADirect(cls.pca_points, cls.pca_labels, `PCA — ${fm}`);
}
async function runVisualization() {
  const model=$('viz-model-select').value; if(!model) return alert('Sélectionnez un modèle.');
  const method=state.vizMethod;
  const btn=$('viz-run-btn'); btn.textContent='⏳…'; btn.disabled=true;
  $('viz-status').textContent=method==='tsne'?'Calcul t-SNE en cours (peut prendre 10-20 sec)…':'Calcul PCA…';
  try {
    const r=await fetch('/api/visualize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,mode:method})});
    const d=await r.json();
    const pts=method==='tsne'?(d.tsne_points||d.pca_points):d.pca_points;
    renderPCADirect(pts, d.labels, `${method.toUpperCase()} — ${model}`, d.texts);
    $('viz-status').textContent='';
  } catch(e){alert('Erreur: '+e.message); $('viz-status').textContent='';}
  finally{btn.textContent='↻ Calculer'; btn.disabled=false;}
}
function renderPCADirect(points, labels, title, texts=[]) {
  if(state.pcaChart){state.pcaChart.destroy();state.pcaChart=null;}
  const unique=[...new Set(labels)];
  const datasets=unique.map((lbl,i)=>{
    const idxs=labels.map((l,j)=>l===lbl?j:-1).filter(j=>j>=0);
    return {label:lbl,data:idxs.map(j=>({x:points[j][0],y:points[j][1],text:texts[j]||''})),backgroundColor:PALETTE[i%PALETTE.length]+'bb',pointRadius:7};
  });
  state.pcaChart=new Chart($('pca-chart'),{type:'scatter',data:{datasets},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}},tooltip:{backgroundColor:'#1c2128',callbacks:{label:ctx=>ctx.raw.text?.slice(0,60)||''}},title:{display:true,text:title,color:'#8b949e',font:{size:12}}},scales:{x:{ticks:{display:false},grid:{color:'#21262d'}},y:{ticks:{display:false},grid:{color:'#21262d'}}}}});
}

/* ══════════════════════════════════════════════════════════
   Similarity Explorer
══════════════════════════════════════════════════════════ */
function initExplorer() {
  $('exp-model').addEventListener('change', ()=>{ $('exp-run').disabled=!$('exp-model').value; });
  $('exp-run').addEventListener('click', async()=>{
    const model=$('exp-model').value; const ta=$('exp-text-a').value.trim(); const tb=$('exp-text-b').value.trim();
    if(!model||!ta||!tb) return alert('Remplissez tous les champs.');
    const btn=$('exp-run'); btn.textContent='⏳…'; btn.disabled=true;
    try {
      const d=await (await fetch('/api/explore/similarity',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,text_a:ta,text_b:tb})})).json();
      renderGauge(d);
    } catch(e){alert('Erreur: '+e.message);}
    finally{btn.textContent='Calculer la similarité'; btn.disabled=!model;}
  });
}
function renderGauge(data) {
  const {similarity:sim, magnitude_a, magnitude_b, dim}=data;
  $('exp-result').classList.remove('hidden');
  $('sim-score-txt').textContent=sim.toFixed(3);
  const col=sim>0.7?'#3fb950':sim>0.4?'#d29922':'#f85149';
  if(state.gaugeChart) state.gaugeChart.destroy();
  state.gaugeChart=new Chart($('sim-gauge'),{type:'doughnut',data:{datasets:[{data:[Math.max(0,sim),Math.max(0,1-sim)],backgroundColor:[col,'#21262d'],borderWidth:0,circumference:270,rotation:225}]},options:{responsive:false,cutout:'78%',plugins:{legend:{display:false},tooltip:{enabled:false}}}});
  $('sim-bar-fill').style.width=(Math.max(0,sim)*100)+'%'; $('sim-bar-fill').style.background=col;
  const interp=sim>0.9?'Quasi-identique':sim>0.75?'Très similaire':sim>0.5?'Similaire':sim>0.3?'Peu similaire':'Non similaire';
  $('sim-interp').textContent=interp;
  $('sim-meta').innerHTML=`<div>Interprétation : <span>${interp}</span></div><div>Dimension : <span>${dim}</span></div><div>‖A‖ : <span>${magnitude_a}</span> · ‖B‖ : <span>${magnitude_b}</span></div>`;
}

/* ══════════════════════════════════════════════════════════
   Heatmap panel
══════════════════════════════════════════════════════════ */
function initHeatmapPanel() {
  $('hm-model').addEventListener('change', ()=>{ $('hm-run').disabled=!$('hm-model').value; });
  $('hm-source').addEventListener('change', ()=>{
    $('hm-custom-area').classList.toggle('hidden', $('hm-source').value!=='custom');
  });
  $('hm-run').addEventListener('click', generateHeatmap);
}
async function generateHeatmap() {
  const model=$('hm-model').value; if(!model) return;
  const source=$('hm-source').value;
  let texts=[], labels=[];
  if(source==='custom') {
    texts=$('hm-custom-texts').value.split('\n').map(l=>l.trim()).filter(Boolean).slice(0,40);
    labels=$('hm-custom-labels').value.split('\n').map(l=>l.trim()).filter(Boolean);
    if(texts.length<2) return alert('Entrez au moins 2 textes.');
  } else {
    texts=source==='classification'?null:null;  // fetched server-side
    labels=[];
  }
  const btn=$('hm-run'); btn.textContent='⏳…'; btn.disabled=true;
  $('heatmap-loading').classList.remove('hidden');
  try {
    const body=source==='custom'?{model,texts,labels:labels.length?labels:null}:{model,texts:null,source};
    // For built-in sources, pass source hint or let server pick defaults
    const actualTexts = source==='custom' ? texts : await getBuiltinTexts(source);
    const actualLabels = source==='custom' ? (labels.length?labels:null) : await getBuiltinLabels(source);
    const r=await fetch('/api/heatmap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,texts:actualTexts,labels:actualLabels})});
    const d=await r.json();
    $('heatmap-info').textContent=`${d.texts.length} textes · Modèle : ${model}`;
    drawHeatmap($('heatmap-canvas'), d.matrix, d.labels||d.texts);
  } catch(e){alert('Erreur: '+e.message);}
  finally{btn.textContent='Générer la heatmap'; btn.disabled=!model; $('heatmap-loading').classList.add('hidden');}
}
async function getBuiltinTexts(source) {
  if(source==='classification') {
    // Use first 25 texts (5 per class)
    const cls=[["The quarterback threw a game-winning touchdown.","sports"],["The tennis player won the Grand Slam.","sports"],["She completed the marathon in under three hours.","sports"],["The basketball team scored in overtime.","sports"],["The cyclist dominated the mountain stage.","sports"],["The smartphone features a 200MP camera.","technology"],["Researchers built a faster AI architecture.","technology"],["The software update fixes vulnerabilities.","technology"],["Quantum computing reached a milestone.","technology"],["The EV can travel 800km on one charge.","technology"],["Simmer the sauce over low heat for 20 min.","cooking"],["Fold egg whites gently for the soufflé.","cooking"],["Marinate chicken in lemon juice and herbs.","cooking"],["Blend until the batter is smooth.","cooking"],["Toast spices in a dry pan before grinding.","cooking"],["The senator introduced a healthcare bill.","politics"],["The election results were contested.","politics"],["Parliament approved the budget.","politics"],["The PM announced climate commitments.","politics"],["Diplomatic talks resumed after a decade.","politics"],["Scientists discovered a deep-sea species.","science"],["The telescope captured an exoplanet image.","science"],["The vaccine reduced hospitalisations by 94%.","science"],["Sleep deprivation links to cognitive decline.","science"],["CRISPR corrected a hereditary mutation.","science"]];
    return cls.map(([t])=>t);
  }
  if(source==='sts') {
    const pairs=[["A man is playing guitar.","A person is playing an instrument."],["The stock market crashed.","Financial markets declined sharply."],["Water boils at 100°C.","H2O reaches boiling point at 373K."],["The book was interesting.","I found the novel captivating."],["A cat sits on a mat.","A dog runs in a park."],["The baby is sleeping.","The infant is awake and crying."],["He failed the exam.","She passed with distinction."],["Children play in the park.","Kids have fun outdoors."],["The car engine stopped.","The vehicle broke down."],["The meeting was postponed.","The conference was rescheduled."]];
    return pairs.flat();
  }
  return [];
}
async function getBuiltinLabels(source) {
  if(source==='classification') return Array.from({length:25},(_,i)=>['sports','technology','cooking','politics','science'][Math.floor(i/5)]);
  if(source==='sts') return Array.from({length:20},(_,i)=>`Paire ${Math.floor(i/2)+1} ${i%2===0?'A':'B'}`);
  return null;
}

/* ─── Heatmap canvas rendering ───────────────────────────── */
function drawHeatmap(canvas, matrix, labels) {
  const n=matrix.length; const margin=90; const cell=Math.max(18,Math.min(36,Math.floor((600-margin)/n)));
  canvas.width=margin+n*cell; canvas.height=margin+n*cell;
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#1c2128'; ctx.fillRect(0,0,canvas.width,canvas.height);
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) {
    const v=Math.max(0,Math.min(1,matrix[i][j]));
    ctx.fillStyle=simColor(v); ctx.fillRect(margin+j*cell,margin+i*cell,cell-1,cell-1);
    if(cell>=22){
      ctx.fillStyle=v>0.5?'#0d1117':'#e6edf3'; ctx.font=`bold ${Math.min(10,cell*0.35)}px Consolas`;
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(v.toFixed(2),margin+j*cell+cell/2,margin+i*cell+cell/2);
    }
  }
  ctx.fillStyle='#8b949e'; ctx.font=`${Math.min(10,cell*0.55)}px Segoe UI`;
  labels.forEach((lbl,i)=>{
    const short=typeof lbl==='string'?lbl.slice(0,12):String(lbl);
    ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(short,margin-4,margin+i*cell+cell/2);
    ctx.save(); ctx.translate(margin+i*cell+cell/2,margin-4);
    ctx.rotate(-Math.PI/2.5); ctx.textAlign='left'; ctx.textBaseline='middle';
    ctx.fillText(short,0,0); ctx.restore();
  });
}
function simColor(t) {
  // Cold (0) → dark blue → purple → warm red (1)
  if(t<0.5) { const u=t*2; return `rgb(${Math.round(26+u*(108-26))},${Math.round(42+u*(99-42))},${Math.round(74+u*(255-74))})` ; }
  const u=(t-0.5)*2; return `rgb(${Math.round(108+u*(248-108))},${Math.round(99+u*(81-99))},${Math.round(255+u*(73-255))})`;
}

/* ══════════════════════════════════════════════════════════
   History
══════════════════════════════════════════════════════════ */
function initHistory() {
  renderHistoryList();
  $('clear-history').addEventListener('click',()=>{if(!confirm('Effacer tout ?')) return; localStorage.removeItem('eb_history'); renderHistoryList();});
  $('save-history').addEventListener('click',()=>{
    if(!Object.keys(state.results).length) return alert('Aucun résultat.');
    const h=JSON.parse(localStorage.getItem('eb_history')||'[]');
    h.unshift({date:new Date().toISOString(),results:state.results});
    if(h.length>20) h.length=20;
    localStorage.setItem('eb_history',JSON.stringify(h));
    renderHistoryList(); alert('Sauvegardé.');
  });
}
function renderHistoryList() {
  const h=JSON.parse(localStorage.getItem('eb_history')||'[]'); const el=$('history-list');
  if(!h.length){el.innerHTML='<p class="muted-sm">Aucun run enregistré.</p>';return;}
  el.innerHTML='';
  h.forEach(run=>{
    const div=document.createElement('div'); div.className='history-item';
    div.innerHTML=`<div class="history-item-date">${new Date(run.date).toLocaleString('fr-FR')}</div><div class="history-item-models">${Object.keys(run.results).join(', ')}</div>`;
    div.addEventListener('click',()=>{
      state.results=run.results; state.charts.forEach(c=>c.destroy()); state.charts=[];
      $('hero').classList.add('hidden'); $('progress-view').classList.add('hidden'); $('results-view').classList.remove('hidden');
      renderResults(); document.querySelector('.nav-btn[data-view="benchmark"]').click();
    });
    el.appendChild(div);
  });
}

/* ══════════════════════════════════════════════════════════
   Export
══════════════════════════════════════════════════════════ */
$('export-json').addEventListener('click',()=>{ if(!Object.keys(state.results).length) return; dl(new Blob([JSON.stringify(state.results,null,2)],{type:'application/json'}),'benchmark.json'); });
$('export-csv').addEventListener('click',()=>{
  if(!Object.keys(state.results).length) return;
  const rows=[['model','overall','latency_ms','throughput','dim','pearson_r','ndcg5','recall5','mrr','cls_acc','silhouette','discrimination','multilingual','negation_awareness','monotonicity']];
  Object.entries(state.results).forEach(([m,r])=>rows.push([m,r.overall_score??'',r.speed?.latency_mean_ms??'',r.speed?.throughput_per_sec??'',r.speed?.embedding_dim??'',r.sts?.pearson_r??'',r.retrieval?.ndcg_at_5??'',r.retrieval?.recall_at_5??'',r.retrieval?.mrr??'',r.classification?.nearest_centroid_accuracy??'',r.classification?.silhouette_score??'',r.robustness?.discrimination_ratio??'',r.multilingual?.alignment_score??'',r.negation?.negation_awareness??'',r.topic_drift?.monotonicity_score??'']));
  dl(new Blob([rows.map(r=>r.join(',')).join('\n')],{type:'text/csv'}),'benchmark.csv');
});
function dl(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href);}
