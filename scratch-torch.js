(function (Scratch) {
  'use strict';

  // ============================================================
  //  Scratch Torch — смесь pandas × torch × sklearn в виде кубиков
  //  Грузится в TurboWarp:  ?extension=URL_ЭТОГО_ФАЙЛА
  // ============================================================

  const vm = Scratch.vm;

  // Аргумент, через который можно передавать объекты/JSON от вложенных
  // репортёров. В TurboWarp есть ANY, в стоковом Scratch его нет.
  const ANY = Scratch.ArgumentType.ANY || Scratch.ArgumentType.STRING;

  // ---------------- состояние расширения ----------------
  const store = {
    tables: {},   // имя -> { names: [], columns: { name: [] }, rows: N }
    models: {},   // имя -> { spec, tfModel, preShape, features, history, mapping, trained, loss }
  };

  // ---------------- загрузка tensorflow.js ----------------
  const TF_CDN = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  let tfPromise = null;

  function loadTF() {
    if (window.tf) return Promise.resolve();
    if (tfPromise) return tfPromise;
    tfPromise = new Promise(function (resolve, reject) {
      const s = document.createElement('script');
      s.src = TF_CDN;
      s.onload = resolve;
      s.onerror = function () {
        tfPromise = null;
        reject(new Error('не смог загрузить tensorflow.js'));
      };
      document.head.appendChild(s);
    });
    return tfPromise;
  }

  // ---------------- мелкие хелперы ----------------
  function isNum(v) {
    if (typeof v === 'number') return true;
    if (typeof v === 'string') {
      const t = v.trim();
      if (t === '') return false;
      return !isNaN(Number(t));
    }
    return false;
  }
  function toNum(v) { return Number(String(v).trim()); }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  // Распарсить значение аргумента в «слой». Вложенные репортёры отдают
  // JSON-строку, поэтому объект приходит строкой.
  function parseLayerArg(v) {
    if (v === null || v === undefined || v === '') return null;
    if (typeof v === 'object') return v;
    try { return JSON.parse(String(v)); } catch (e) { return null; }
  }

  // Развернуть цепочку слоёв (связный список через .next) в плоский массив.
  function flattenChain(head) {
    const out = [];
    let cur = head;
    let guard = 0;
    while (cur && typeof cur === 'object' && cur.type && guard < 100) {
      const next = cur.next;
      const c = Object.assign({}, cur);
      delete c.next;
      out.push(c);
      cur = next;
      guard++;
    }
    return out;
  }

  function warn(msg) {
    setHUD('⚠ ' + msg);
    try { console.warn('[ScratchTorch] ' + msg); } catch (e) {}
  }

  // ---------------- таблицы (DataFrame-стиль) ----------------
  function makeTable(rows, names) {
    const columns = {};
    names.forEach(function (n) { columns[n] = []; });
    rows.forEach(function (r) {
      names.forEach(function (n, i) { columns[n].push(r[i]); });
    });
    return { names: names, columns: columns, rows: rows.length };
  }

  function parseCSV(text, sep) {
    if (sep === 'запятая ,') sep = ',';
    else if (sep === 'табуляция') sep = '\t';
    else if (sep === 'точка с запятой') sep = ';';
    else if (sep === 'пробел') sep = /\s+/;
    const lines = String(text).split(/\r?\n/).filter(function (l) { return l.trim() !== ''; });
    if (lines.length === 0) return { names: [], rows: [] };
    const splitLine = function (l) { return String(l).split(sep).map(function (x) { return x.trim(); }); };
    const names = splitLine(lines[0]);
    const rows = lines.slice(1).map(splitLine);
    // выравниваем по ширине заголовка
    for (let i = 0; i < rows.length; i++) {
      while (rows[i].length < names.length) rows[i].push('');
    }
    return { names: names, rows: rows };
  }

  function cloneTable(t) {
    return makeTable(rowsOf(t), t.names);
  }

  function rowsOf(t) {
    const rows = [];
    for (let r = 0; r < t.rows; r++) {
      rows.push(t.names.map(function (n) { return t.columns[n][r]; }));
    }
    return rows;
  }

  function normalizeTable(t, method) {
    const out = makeTable(rowsOf(t), t.names);
    t.names.forEach(function (n) {
      const col = out.columns[n];
      const numeric = col.filter(isNum);
      if (numeric.length === 0) return;
      if (method === 'z-score') {
        const mean = numeric.reduce(function (a, b) { return a + toNum(b); }, 0) / numeric.length;
        const sd = Math.sqrt(numeric.reduce(function (a, b) { return a + (toNum(b) - mean) * (toNum(b) - mean); }, 0) / Math.max(1, numeric.length));
        for (let i = 0; i < col.length; i++) {
          if (isNum(col[i])) col[i] = sd > 0 ? (toNum(col[i]) - mean) / sd : 0;
        }
      } else {
        let mn = Infinity, mx = -Infinity;
        numeric.forEach(function (x) { mn = Math.min(mn, toNum(x)); mx = Math.max(mx, toNum(x)); });
        for (let i = 0; i < col.length; i++) {
          if (isNum(col[i])) col[i] = mx > mn ? (toNum(col[i]) - mn) / (mx - mn) : 0;
        }
      }
    });
    return out;
  }

  function splitTable(t, frac) {
    const idx = [];
    for (let i = 0; i < t.rows; i++) idx.push(i);
    // перемешаем (фиксированный «фан» — с перемешиванием по перемешиванию)
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    const k = Math.max(1, Math.round(idx.length * (1 - frac)));
    const pick = function (set) {
      const rows = set.map(function (i) { return t.names.map(function (n) { return t.columns[n][i]; }); });
      return makeTable(rows, t.names);
    };
    return { train: pick(idx.slice(0, k)), test: pick(idx.slice(k)) };
  }

  function featuresOf(table, featuresArg, label) {
    let f = String(featuresArg).split(/[,\s]+/).filter(function (s) { return s !== ''; });
    if (f.length === 1 && (f[0] === '*' || f[0] === 'все')) {
      f = table.names.filter(function (n) { return n !== label; });
    }
    f = f.filter(function (n) { return table.names.indexOf(n) !== -1; });
    return f;
  }

  // ---------------- сборка tf.js-модели из спецификации ----------------
  function dimsOf(spec) {
    if (!spec) return null;
    switch (spec.type) {
      case 'dense': return [spec.units];
      case 'dropout': return null; // не меняет размерность
      case 'flatten': return [spec.inDim];
      case 'lstm': return [spec.units];
      case 'reshape': return spec.shape;
      case 'conv2d': return [spec.H, spec.W, spec.C];
      case 'maxpool2d': return [spec.H, spec.W, spec.C];
    }
    return null;
  }

  function buildTfModel(chain, features) {
    const tf = window.tf;
    let preShape = null;           // во что решэпать вход [batch, features]
    const layersSpec = [];
    let curDims = [features];      // текущая «виртуальная» размерность

    // проход 1: вычислить размерности, придумать preShape для 2D/3D слоёв
    for (let i = 0; i < chain.length; i++) {
      const s = chain[i];
      const lastDim = curDims[curDims.length - 1];
      if (s.type === 'dense') {
        layersSpec.push({ type: 'dense', units: s.units, act: s.act });
        curDims = [s.units];
      } else if (s.type === 'dropout') {
        layersSpec.push({ type: 'dropout', rate: s.rate });
      } else if (s.type === 'lstm') {
        layersSpec.push({ type: 'lstm', units: s.units, act: s.act });
        curDims = [s.units];
      } else if (s.type === 'flatten') {
        layersSpec.push({ type: 'flatten' });
        curDims = [curDims.reduce(function (a, b) { return a * b; }, 1)];
      } else if (s.type === 'reshape') {
        const dims = s.shape;
        layersSpec.push({ type: 'reshape', target: dims });
        curDims = dims;
      } else if (s.type === 'conv2d') {
        if (curDims.length < 2) curDims = [curDims[0], 1, 1];
        const C = curDims.length === 3 ? curDims[2] : 1;
        const dims = curDims.length === 3 ? curDims : [curDims[0], curDims[1], C];
        layersSpec.push({ type: 'reshape', target: dims });
        layersSpec.push({ type: 'conv2d', filters: s.filters, kernel: s.kernel, act: s.act });
        curDims = [dims[0], dims[1], s.filters];
      } else if (s.type === 'maxpool2d') {
        if (curDims.length < 2) curDims = [curDims[0], 1, 1];
        const dims = curDims.length === 3 ? curDims : [curDims[0], curDims[1], 1];
        layersSpec.push({ type: 'reshape', target: dims });
        layersSpec.push({ type: 'maxpool2d', size: s.size });
        curDims = [Math.ceil(dims[0] / s.size), Math.ceil(dims[1] / s.size), dims[2]];
      }
    }

    // если первый слой требует не-плоский вход, а мы не решэпили — засунем
    // решэп самым первым
    const firstReal = chain[0];
    if (firstReal && (firstReal.type === 'conv2d' || firstReal.type === 'maxpool2d')) {
      const dims = [features, 1, 1];
      if (preShape === null) preShape = dims;
    }

    // если где-то внутри есть conv — вход должен быть 2D
    for (let i = 0; i < chain.length; i++) {
      if ((chain[i].type === 'conv2d' || chain[i].type === 'maxpool2d') && preShape === null) {
        preShape = [features, 1, 1];
        break;
      }
    }

    // смотрим, начинается ли цепочка с reshape: тогда вход надо решэпить
    if (chain[0] && chain[0].type === 'reshape') {
      preShape = chain[0].shape;
    }

    // LSTM первым слоем требует [batch, время, признаки] — добавим измерение времени
    if (chain[0] && chain[0].type === 'lstm') {
      preShape = [1, features];
    }

    const inputShape = preShape ? preShape : [features];
    const layers = [];
    for (let i = 0; i < layersSpec.length; i++) {
      const s = layersSpec[i];
      let cfg = {};
      if (s.type === 'dense') {
        cfg = { units: s.units, activation: s.act };
      } else if (s.type === 'dropout') {
        cfg = { rate: clamp(s.rate, 0, 0.99) };
      } else if (s.type === 'lstm') {
        cfg = { units: s.units, activation: s.act, returnSequences: false };
      } else if (s.type === 'flatten') {
        cfg = {};
      } else if (s.type === 'reshape') {
        cfg = { targetShape: s.target };
      } else if (s.type === 'conv2d') {
        cfg = { filters: s.filters, kernelSize: [s.kernel, s.kernel], activation: s.act, padding: 'same' };
      } else if (s.type === 'maxpool2d') {
        cfg = { poolSize: [s.size, s.size], strides: [s.size, s.size], padding: 'same' };
      }
      if (i === 0) cfg.inputShape = inputShape; // tf.js требует inputShape на первом слое
      let L;
      if (s.type === 'dense') L = tf.layers.dense(cfg);
      else if (s.type === 'dropout') L = tf.layers.dropout(cfg);
      else if (s.type === 'lstm') L = tf.layers.lstm(cfg);
      else if (s.type === 'flatten') L = tf.layers.flatten(cfg);
      else if (s.type === 'reshape') L = tf.layers.reshape(cfg);
      else if (s.type === 'conv2d') L = tf.layers.conv2d(cfg);
      else if (s.type === 'maxpool2d') L = tf.layers.maxPooling2d(cfg);
      if (L) layers.push(L);
    }

    const model = tf.sequential({
      layers: layers,
      inputShape: inputShape
    });
    return { tfModel: model, preShape: preShape, features: features };
  }

  // ---------------- скейлеры (авто-нормализация внутри fit) ----------------
  // модель хранит скейлеры признаков и метки, поэтому predict/score
  // принимают и возвращают значения в исходных единицах.
  function makeScaler(colVals) {
    const nums = colVals.filter(isNum).map(toNum);
    if (nums.length === 0) return { type: 'identity' };
    let mn = Infinity, mx = -Infinity;
    nums.forEach(function (v) { mn = Math.min(mn, v); mx = Math.max(mx, v); });
    if (mx > mn) return { type: 'minmax', min: mn, max: mx };
    const mean = nums.reduce(function (a, b) { return a + b; }, 0) / nums.length;
    const sd = Math.sqrt(nums.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / Math.max(1, nums.length)) || 1;
    return { type: 'std', mean: mean, sd: sd };
  }
  function scaleVal(v, s) {
    if (!s || s.type === 'identity') return v;
    if (s.type === 'minmax') return (v - s.min) / (s.max - s.min);
    return (v - s.mean) / s.sd;
  }
  function unscaleVal(v, s) {
    if (!s || s.type === 'identity') return v;
    if (s.type === 'minmax') return v * (s.max - s.min) + s.min;
    return v * s.sd + s.mean;
  }
  function computeScalers(table, feats, label) {
    const featScalers = feats.map(function (f) { return makeScaler(table.columns[f]); });
    const labelCol = table.columns[label];
    const yScaler = labelCol && labelCol.every(isNum) ? makeScaler(labelCol) : null;
    return { featScalers: featScalers, yScaler: yScaler };
  }

  // подготовка X, y для обучения (с авто-скейлингом)
  function prepareXY(table, feats, label, loss, scalers) {
    const tf = window.tf;
    const X = [], yRaw = [];
    for (let r = 0; r < table.rows; r++) {
      const row = feats.map(function (f, i) {
        const v = table.columns[f][r];
        return isNum(v) ? scaleVal(toNum(v), scalers.featScalers[i]) : 0;
      });
      X.push(row);
      yRaw.push(table.columns[label][r]);
    }
    // кодируем метки: строки -> индексы, числа -> скейлим
    let mapping = null;
    let y;
    if (yRaw.some(function (v) { return !isNum(v); })) {
      const uniq = [];
      yRaw.forEach(function (v) {
        if (uniq.indexOf(v) === -1) uniq.push(v);
      });
      mapping = uniq;
      y = yRaw.map(function (v) { return uniq.indexOf(v); });
    } else {
      y = yRaw.map(function (v) { return scaleVal(toNum(v), scalers.yScaler); });
    }
    let yTensor;
    if (loss === 'categoricalCrossentropy') {
      const classes = Math.max(2, Math.max.apply(null, y) + 1);
      yTensor = tf.oneHot(tf.tensor1d(y, 'int32'), classes);
    } else {
      yTensor = tf.tensor2d(y.map(function (v) { return [v]; }), [y.length, 1], 'float32');
    }
    return {
      xs: tf.tensor2d(X, [X.length, feats.length], 'float32'),
      ys: yTensor,
      mapping: mapping,
      classes: mapping ? mapping.length : null
    };
  }

  function reshapeInputTensor(tf, t, model) {
    if (model.preShape) {
      const sample = t.shape[0];
      return t.reshape([sample].concat(model.preShape));
    }
    return t;
  }

  // ---------------- оверлей-канвас (рисование схемы) ----------------
  const overlay = {
    canvas: null,
    ctx: null,
    hud: ''
  };

  function ensureCanvas() {
    try {
      const r = vm.runtime.renderer;
      if (!r || !r.canvas) return false;
      const stage = r.canvas;
      if (!overlay.canvas || !overlay.canvas.isConnected) {
        overlay.canvas = document.createElement('canvas');
        overlay.canvas.style.position = 'absolute';
        overlay.canvas.style.left = '0';
        overlay.canvas.style.top = '0';
        overlay.canvas.style.width = '100%';
        overlay.canvas.style.height = '100%';
        overlay.canvas.style.pointerEvents = 'none';
        overlay.canvas.style.zIndex = '9999';
        (stage.parentElement || document.body).appendChild(overlay.canvas);
      }
      overlay.canvas.width = stage.width;
      overlay.canvas.height = stage.height;
      overlay.ctx = overlay.canvas.getContext('2d');
      return true;
    } catch (e) {
      return false;
    }
  }

  function setHUD(msg) {
    overlay.hud = String(msg);
    redraw();
  }

  function layerUnits(layer) {
    switch (layer.type) {
      case 'dense': return layer.units;
      case 'lstm': return layer.units;
      case 'dropout': return null; // пассквозной
      case 'flatten': return null;
      case 'reshape': return null;
      case 'conv2d': return layer.filters;
      case 'maxpool2d': return null;
    }
    return null;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawModel(name) {
    const m = store.models[name];
    if (!m) return;
    const chain = flattenChain(m.spec);
    if (chain.length === 0) return;
    const ctx = overlay.ctx;
    const W = overlay.canvas.width, H = overlay.canvas.height;
    const topPad = 40, botPad = 70;
    const areaH = H - topPad - botPad;

    // узлы на слой (с капом, чтобы не месить кашу)
    const CAP = 36;
    const unitsPer = chain.map(function (l) {
      const u = layerUnits(l);
      if (u === null) return null;
      return { total: u, shown: Math.min(u, CAP) };
    });

    const xPad = 46;
    const xStep = (W - 2 * xPad) / Math.max(1, chain.length - 1);

    // позиции узлов
    const nodes = [];
    for (let i = 0; i < chain.length; i++) {
      const up = unitsPer[i];
      const x = chain.length === 1 ? W / 2 : xPad + i * xStep;
      if (!up) { nodes.push([]); continue; }
      const ys = [];
      for (let j = 0; j < up.shown; j++) {
        ys.push(topPad + (j + 0.5) * (areaH / up.shown));
      }
      nodes.push({ x: x, ys: ys, total: up.total });
    }

    // подписи слоёв
    ctx.font = Math.max(10, Math.min(13, H / 28)) + 'px monospace';
    for (let i = 0; i < chain.length; i++) {
      const l = chain[i];
      let label = '';
      if (l.type === 'dense') label = 'Dense ' + l.units;
      else if (l.type === 'dropout') label = 'Dropout';
      else if (l.type === 'flatten') label = 'Flatten';
      else if (l.type === 'lstm') label = 'LSTM ' + l.units;
      else if (l.type === 'reshape') label = 'Reshape ' + l.shape.join(',');
      else if (l.type === 'conv2d') label = 'Conv2d ' + l.filters;
      else if (l.type === 'maxpool2d') label = 'MaxPool';
      if (!nodes[i] || nodes[i].ys.length === 0) {
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
        ctx.textAlign = 'center';
        ctx.fillText(label, nodes[i] ? nodes[i].x : xPad + i * xStep, topPad + 12);
        continue;
      }
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.textAlign = 'center';
      const yTop = nodes[i].ys[0];
      const yBot = nodes[i].ys[nodes[i].ys.length - 1];
      ctx.fillText(label, nodes[i].x, yTop - 8);
      if (nodes[i].total > CAP) {
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.fillText('+ ещё ' + (nodes[i].total - CAP), nodes[i].x, yBot + 12);
      }
    }

    // рёбра: прозрачность ∝ |вес| после обучения
    // заранее вытащим весовые матрицы dense-слоёв (индекс слоя -> данные)
    const wmat = {};
    if (m.trained && m.tfModel) {
      try {
        const weights = m.tfModel.getWeights();
        const flat = weights.filter(function (w, z) { return z % 2 === 0; });
        let denseIdx = 0;
        for (let i = 0; i < chain.length; i++) {
          if (chain[i].type === 'dense') {
            const w = flat[denseIdx];
            denseIdx++;
            if (w) {
              wmat[i] = { arr: w.dataSync(), inU: w.shape[0], outU: w.shape[1] };
            }
          }
        }
      } catch (e) { /* без весов — рисуем равномерно */ }
    }
    const edgeAlpha = function (aLayer, aNode, bNode) {
      const wm = wmat[aLayer];
      if (!wm) return 0.10;
      const a = Math.min(aNode, wm.inU - 1), b = Math.min(bNode, wm.outU - 1);
      const val = Math.abs(wm.arr[a * wm.outU + b]);
      return clamp(0.04 + val * 0.35, 0.04, 0.85);
    };

    for (let i = 0; i < chain.length - 1; i++) {
      const A = nodes[i], B = nodes[i + 1];
      if (!A || !B) continue;
      const nA = A.ys.length, nB = B.ys.length;
      const product = nA * nB;
      ctx.lineWidth = 1;
      if (product <= 700) {
        for (let a = 0; a < nA; a++) {
          for (let b = 0; b < nB; b++) {
            const alpha = edgeAlpha(i, a, b);
            ctx.strokeStyle = 'rgba(238,76,44,' + alpha.toFixed(3) + ')';
            ctx.beginPath();
            ctx.moveTo(A.x + 7, A.ys[a]);
            ctx.quadraticCurveTo((A.x + B.x) / 2, (A.ys[a] + B.ys[b]) / 2, B.x - 7, B.ys[b]);
            ctx.stroke();
          }
        }
      } else {
        // «воронка» — слишком много рёбер
        ctx.fillStyle = 'rgba(238,76,44,0.08)';
        ctx.beginPath();
        ctx.moveTo(A.x, A.ys[0]);
        ctx.lineTo(B.x, B.ys[0]);
        ctx.lineTo(B.x, B.ys[nB - 1]);
        ctx.lineTo(A.x, A.ys[nA - 1]);
        ctx.closePath();
        ctx.fill();
      }
    }

    // узлы
    for (let i = 0; i < nodes.length; i++) {
      const N = nodes[i];
      if (!N) continue;
      const r = Math.max(2.5, Math.min(5, areaH / 60));
      for (let j = 0; j < N.ys.length; j++) {
        // подсветка входа по последнему предсказанию
        let fill = 'rgba(255,255,255,0.85)';
        if (i === 0 && m.lastInput && m.lastInput.length) {
          const v = m.lastInput[Math.min(j, m.lastInput.length - 1)];
          if (isNum(v)) {
            const a = clamp(0.15 + Math.abs(toNum(v)) * 0.7, 0.15, 0.95);
            fill = toNum(v) >= 0 ? 'rgba(76,175,80,' + a + ')' : 'rgba(244,67,54,' + a + ')';
          }
        }
        ctx.beginPath();
        ctx.arc(N.x, N.ys[j], r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.35)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // график loss (если обучали)
    if (m.history && m.history.length > 1) {
      const gw = Math.min(180, W * 0.35), gh = Math.min(46, H * 0.14);
      const gx = W - gw - 10, gy = H - gh - 8;
      roundRect(ctx, gx - 4, gy - 4, gw + 8, gh + 8, 6);
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fill();
      let mn = Infinity, mx = -Infinity;
      m.history.forEach(function (v) { mn = Math.min(mn, v); mx = Math.max(mx, v); });
      const span = (mx - mn) || 1;
      ctx.strokeStyle = '#7ee787';
      ctx.lineWidth = 2;
      ctx.beginPath();
      m.history.forEach(function (v, i) {
        const x = gx + (i / Math.max(1, m.history.length - 1)) * gw;
        const y = gy + gh - ((v - mn) / span) * gh;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      const lastLoss = m.history[m.history.length - 1];
      ctx.fillText('loss: ' + lastLoss.toFixed(4), gx, gy + gh + 12);
      ctx.fillText('epoch ' + m.epochsDone + '/' + m.epochsTotal, gx, gy + gh + 24);
    }

    // маппинг классов (для классификации)
    if (m.mapping) {
      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(m.mapping.map(function (c, i) { return i + ':' + c; }).join('  '), 10, H - 10);
    }
  }

  function redraw() {
    if (!ensureCanvas()) return;
    const ctx = overlay.ctx;
    ctx.clearRect(0, 0, overlay.canvas.width, overlay.canvas.height);
    Object.keys(store.models).forEach(function (name) {
      if (store.models[name].visible !== false) drawModel(name);
    });
    Object.keys(store.tables).forEach(function (name) {
      if (store.tables[name].show) drawTable(name);
    });
    // HUD
    if (overlay.hud) {
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      const w = ctx.measureText ? ctx.measureText(overlay.hud).width + 16 : 200;
      roundRect(ctx, 8, 8, w, 26, 6);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '12px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(overlay.hud, 16, 26);
    }
  }

  function drawTable(name) {
    const t = store.tables[name];
    if (!t) return;
    const ctx = overlay.ctx;
    const W = overlay.canvas.width;
    const rows = Math.min(t.rows, 12);
    const shownNames = t.names.slice(0, 6);
    const cellW = Math.max(50, W / (shownNames.length + 1));
    const cellH = 18;
    const x0 = 10, y0 = 40;
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    roundRect(ctx, x0 - 4, y0 - 4, cellW * shownNames.length + 8, cellH * (rows + 1) + 8, 6);
    ctx.fill();
    ctx.font = '11px monospace';
    shownNames.forEach(function (n, c) {
      ctx.fillStyle = '#82b1ff';
      ctx.fillText(n, x0 + c * cellW, y0 + 4);
    });
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    for (let r = 0; r < rows; r++) {
      shownNames.forEach(function (n, c) {
        const v = t.columns[n][r];
        ctx.fillText(String(v).slice(0, 14), x0 + c * cellW, y0 + cellH * (r + 1) + 4);
      });
    }
  }

  function drawAll() {
    redraw();
  }

  // автоперерисовка при ресайзе сцены
  setInterval(function () {
    if (overlay.canvas && overlay.canvas.isConnected) {
      try {
        const r = vm.runtime.renderer;
        if (r && r.canvas && (r.canvas.width !== overlay.canvas.width || r.canvas.height !== overlay.canvas.height)) {
          redraw();
        }
      } catch (e) {}
    }
  }, 500);

  // ---------------- блоки ----------------
  class ScratchTorch {
    getInfo() {
      const menus = {
        sep: {
          acceptReporters: true,
          items: ['запятая ,', 'табуляция', 'точка с запятой', 'пробел']
        },
        method: {
          acceptReporters: true,
          items: [
            { text: 'min-max (0…1)', value: 'min-max' },
            { text: 'z-score (стандартизация)', value: 'z-score' }
          ]
        },
        act: {
          acceptReporters: true,
          items: ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'gelu', 'elu']
        },
        loss: {
          acceptReporters: true,
          items: [
            { text: 'MSE (средний квадрат)', value: 'mse' },
            { text: 'MAE (средняя абсолютная)', value: 'mae' },
            { text: 'BinaryCrossentropy', value: 'binaryCrossentropy' },
            { text: 'CategoricalCrossentropy', value: 'categoricalCrossentropy' }
          ]
        },
        opt: {
          acceptReporters: true,
          items: ['adam', 'sgd', 'rmsprop', 'adamax', 'adagrad']
        }
      };

      const layerColor = { color1: '#EE4C2C', color2: '#C23B1F', color3: '#F2A292' };
      const modelColor = { color1: '#F59E0B', color2: '#B9770A', color3: '#FBCF86' };
      const dataColor = { color1: '#4F46E5', color2: '#3730A3', color3: '#A5B4FC' };
      const trainColor = { color1: '#16A34A', color2: '#15803D', color3: '#86EFAC' };
      const vizColor = { color1: '#9333EA', color2: '#6B21A8', color3: '#D8B4FE' };

      return {
        id: 'scratchtorch',
        name: 'Scratch Torch (pandas×torch×sklearn)',
        color1: '#EE4C2C',
        color2: '#C23B1F',
        color3: '#F2A292',
        blocks: [
          // ============ pandas: данные ============
          {
            opcode: 'dfCreate', blockType: Scratch.BlockType.COMMAND,
            text: 'создай DataFrame [name] из [csv] разделитель [sep]',
            arguments: {
              name: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' },
              csv: { type: Scratch.ArgumentType.STRING, defaultValue: 'sepal,petal,species\n5.1,1.4,setosa\n4.9,1.4,versicolor\n6.2,2.2,virginica' },
              sep: { type: Scratch.ArgumentType.STRING, menu: 'sep', defaultValue: 'запятая ,' }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },
          {
            opcode: 'dfColumn', blockType: Scratch.BlockType.REPORTER,
            text: 'колонка [col] из [table]',
            arguments: {
              col: { type: Scratch.ArgumentType.STRING, defaultValue: 'sepal' },
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },
          {
            opcode: 'dfRows', blockType: Scratch.BlockType.REPORTER,
            text: 'строк в [table]',
            arguments: {
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },
          {
            opcode: 'dfNormalize', blockType: Scratch.BlockType.COMMAND,
            text: 'нормализуй [table] способом [method] → сохрани как [newName]',
            arguments: {
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' },
              method: { type: Scratch.ArgumentType.STRING, menu: 'method', defaultValue: 'min-max (0…1)' },
              newName: { type: Scratch.ArgumentType.STRING, defaultValue: 'df2' }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },
          {
            opcode: 'dfSplit', blockType: Scratch.BlockType.COMMAND,
            text: 'раздели [table] на [trainName] и [testName] тестовая доля [frac]',
            arguments: {
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' },
              trainName: { type: Scratch.ArgumentType.STRING, defaultValue: 'train' },
              testName: { type: Scratch.ArgumentType.STRING, defaultValue: 'test' },
              frac: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.2 }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },
          {
            opcode: 'dfShow', blockType: Scratch.BlockType.COMMAND,
            text: 'покажи таблицу [table]',
            arguments: {
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df1' }
            },
            color1: dataColor.color1, color2: dataColor.color2, color3: dataColor.color3
          },

          '--- torch: слои (вкладываются друг в друга) ---',
          {
            opcode: 'nnDense', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.Dense [units] [act] → [next]',
            arguments: {
              units: { type: Scratch.ArgumentType.NUMBER, defaultValue: 16 },
              act: { type: Scratch.ArgumentType.STRING, menu: 'act', defaultValue: 'relu' },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnDropout', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.Dropout [rate] → [next]',
            arguments: {
              rate: { type: Scratch.ArgumentType.NUMBER, defaultValue: 0.2 },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnFlatten', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.Flatten → [next]',
            arguments: {
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnLSTM', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.LSTM [units] [act] → [next]',
            arguments: {
              units: { type: Scratch.ArgumentType.NUMBER, defaultValue: 8 },
              act: { type: Scratch.ArgumentType.STRING, menu: 'act', defaultValue: 'tanh' },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnConv2d', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.Conv2d [filters] [kernel] [act] → [next]',
            arguments: {
              filters: { type: Scratch.ArgumentType.NUMBER, defaultValue: 16 },
              kernel: { type: Scratch.ArgumentType.NUMBER, defaultValue: 3 },
              act: { type: Scratch.ArgumentType.STRING, menu: 'act', defaultValue: 'relu' },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnMaxPool', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.MaxPool2d [size] → [next]',
            arguments: {
              size: { type: Scratch.ArgumentType.NUMBER, defaultValue: 2 },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },
          {
            opcode: 'nnReshape', blockType: Scratch.BlockType.REPORTER,
            text: 'nn.Reshape [shape] → [next]',
            arguments: {
              shape: { type: Scratch.ArgumentType.STRING, defaultValue: '1,28,28' },
              next: { type: ANY, defaultValue: '' }
            },
            color1: layerColor.color1, color2: layerColor.color2, color3: layerColor.color3
          },

          {
            opcode: 'modelCreate', blockType: Scratch.BlockType.COMMAND,
            text: 'модель [name] = Sequential( [first] )',
            arguments: {
              name: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' },
              first: { type: ANY, defaultValue: '' }
            },
            color1: modelColor.color1, color2: modelColor.color2, color3: modelColor.color3
          },

          '--- sklearn: обучение и метрики ---',
          {
            opcode: 'fit', blockType: Scratch.BlockType.COMMAND,
            text: 'обучи [model] на [table] признаки [features] метка [label] эпох [epochs] батч [batch] loss [loss] оптимизатор [opt]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' },
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'df2' },
              features: { type: Scratch.ArgumentType.STRING, defaultValue: '*' },
              label: { type: Scratch.ArgumentType.STRING, defaultValue: 'species' },
              epochs: { type: Scratch.ArgumentType.NUMBER, defaultValue: 50 },
              batch: { type: Scratch.ArgumentType.NUMBER, defaultValue: 16 },
              loss: { type: Scratch.ArgumentType.STRING, menu: 'loss', defaultValue: 'MSE (средний квадрат)' },
              opt: { type: Scratch.ArgumentType.STRING, menu: 'opt', defaultValue: 'adam' }
            },
            color1: trainColor.color1, color2: trainColor.color2, color3: trainColor.color3
          },
          {
            opcode: 'predict', blockType: Scratch.BlockType.REPORTER,
            text: 'предскажи [model] для [input]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' },
              input: { type: Scratch.ArgumentType.STRING, defaultValue: '5.1,1.4' }
            },
            color1: trainColor.color1, color2: trainColor.color2, color3: trainColor.color3
          },
          {
            opcode: 'score', blockType: Scratch.BlockType.REPORTER,
            text: 'метрика [model] на [table] признаки [features] метка [label]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' },
              table: { type: Scratch.ArgumentType.STRING, defaultValue: 'test' },
              features: { type: Scratch.ArgumentType.STRING, defaultValue: '*' },
              label: { type: Scratch.ArgumentType.STRING, defaultValue: 'species' }
            },
            color1: trainColor.color1, color2: trainColor.color2, color3: trainColor.color3
          },
          {
            opcode: 'lastLoss', blockType: Scratch.BlockType.REPORTER,
            text: 'последний loss модели [model]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' }
            },
            color1: trainColor.color1, color2: trainColor.color2, color3: trainColor.color3
          },

          '--- визуализация ---',
          {
            opcode: 'drawNetwork', blockType: Scratch.BlockType.COMMAND,
            text: 'нарисуй схему [model]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' }
            },
            color1: vizColor.color1, color2: vizColor.color2, color3: vizColor.color3
          },
          {
            opcode: 'hideNetwork', blockType: Scratch.BlockType.COMMAND,
            text: 'спрячь схему [model]',
            arguments: {
              model: { type: Scratch.ArgumentType.STRING, defaultValue: 'm1' }
            },
            color1: vizColor.color1, color2: vizColor.color2, color3: vizColor.color3
          },
          {
            opcode: 'clearAll', blockType: Scratch.BlockType.COMMAND,
            text: 'сбрось всё (таблицы и модели)',
            color1: vizColor.color1, color2: vizColor.color2, color3: vizColor.color3
          }
        ]
      };
    }

    // ============ pandas: данные ============

    dfCreate(args) {
      const parsed = parseCSV(args.csv, args.sep);
      if (parsed.names.length === 0) {
        warn('пустой DataFrame');
        return;
      }
      const t = makeTable(parsed.rows, parsed.names);
      store.tables[args.name] = t;
      setHUD('DataFrame "' + args.name + '": ' + t.rows + ' строк × ' + t.names.length + ' колонок');
    }

    dfColumn(args) {
      const t = store.tables[args.table];
      if (!t) { warn('нет таблицы "' + args.table + '"'); return ''; }
      if (t.columns[args.col] === undefined) { warn('нет колонки "' + args.col + '"'); return ''; }
      return t.columns[args.col];
    }

    dfRows(args) {
      const t = store.tables[args.table];
      if (!t) return 0;
      return t.rows;
    }

    dfNormalize(args) {
      const t = store.tables[args.table];
      if (!t) { warn('нет таблицы "' + args.table + '"'); return; }
      const method = args.method.indexOf('z-score') === -1 ? 'min-max' : 'z-score';
      store.tables[args.newName] = normalizeTable(t, method);
      setHUD('нормализована → "' + args.newName + '" (' + method + ')');
    }

    dfSplit(args) {
      const t = store.tables[args.table];
      if (!t) { warn('нет таблицы "' + args.table + '"'); return; }
      const frac = clamp(toNum(args.frac) || 0.2, 0.05, 0.95);
      const r = splitTable(t, frac);
      store.tables[args.trainName] = r.train;
      store.tables[args.testName] = r.test;
      setHUD('train=' + r.train.rows + ' / test=' + r.test.rows);
    }

    dfShow(args) {
      const t = store.tables[args.table];
      if (!t) { warn('нет таблицы "' + args.table + '"'); return; }
      t.show = true;
      setHUD('показываю "' + args.table + '"');
      redraw();
    }

    // ============ torch: слои ============

    nnDense(args) {
      return JSON.stringify({
        type: 'dense',
        units: Math.max(1, Math.round(toNum(args.units) || 1)),
        act: args.act,
        next: parseLayerArg(args.next)
      });
    }

    nnDropout(args) {
      return JSON.stringify({
        type: 'dropout',
        rate: clamp(toNum(args.rate) || 0.2, 0, 0.99),
        next: parseLayerArg(args.next)
      });
    }

    nnFlatten(args) {
      return JSON.stringify({ type: 'flatten', next: parseLayerArg(args.next) });
    }

    nnLSTM(args) {
      return JSON.stringify({
        type: 'lstm',
        units: Math.max(1, Math.round(toNum(args.units) || 1)),
        act: args.act,
        next: parseLayerArg(args.next)
      });
    }

    nnConv2d(args) {
      return JSON.stringify({
        type: 'conv2d',
        filters: Math.max(1, Math.round(toNum(args.filters) || 1)),
        kernel: Math.max(1, Math.round(toNum(args.kernel) || 1)),
        act: args.act,
        next: parseLayerArg(args.next)
      });
    }

    nnMaxPool(args) {
      return JSON.stringify({
        type: 'maxpool2d',
        size: Math.max(1, Math.round(toNum(args.size) || 1)),
        next: parseLayerArg(args.next)
      });
    }

    nnReshape(args) {
      const dims = String(args.shape).split(/[x\s,]+/).filter(function (s) { return s !== ''; }).map(function (s) { return Math.max(1, Math.round(toNum(s) || 1)); });
      return JSON.stringify({ type: 'reshape', shape: dims.length ? dims : [1], next: parseLayerArg(args.next) });
    }

    modelCreate(args) {
      const head = parseLayerArg(args.first);
      const chain = flattenChain(head);
      if (chain.length === 0) {
        warn('Sequential пустой — добавь хотя бы один слой');
        return;
      }
      const prev = store.models[args.name];
      store.models[args.name] = {
        spec: head,
        chain: chain,
        tfModel: null,
        preShape: null,
        features: 0,
        history: [],
        mapping: null,
        trained: false,
        visible: true,
        lastInput: [],
        scalers: null,
        featureNames: [],
        epochsDone: 0,
        epochsTotal: 0
      };
      if (prev && prev.tfModel) { try { prev.tfModel.dispose(); } catch (e) {} }
      setHUD('модель "' + args.name + '": ' + chain.map(function (l) { return l.type; }).join(' → '));
      redraw();
    }

    // ============ sklearn: обучение ============

    async fit(args) {
      const model = store.models[args.model];
      if (!model) { warn('нет модели "' + args.model + '"'); return; }
      const table = store.tables[args.table];
      if (!table) { warn('нет таблицы "' + args.table + '"'); return; }

      try { await loadTF(); } catch (e) { warn(e.message); return; }
      const tf = window.tf;

      const feats = featuresOf(table, args.features, args.label);
      if (feats.length === 0) {
        warn('признаки не найдены в таблице');
        return;
      }
      if (table.columns[args.label] === undefined) {
        warn('нет колонки метки "' + args.label + '"');
        return;
      }

      const epochs = Math.max(1, Math.round(toNum(args.epochs) || 1));
      const batch = Math.max(1, Math.round(toNum(args.batch) || 1));
      const lossMap = {
        'MSE (средний квадрат)': 'meanSquaredError',
        'MAE (средняя абсолютная)': 'meanAbsoluteError',
        'BinaryCrossentropy': 'binaryCrossentropy',
        'CategoricalCrossentropy': 'categoricalCrossentropy'
      };
      const loss = lossMap[args.loss] || args.loss;

      // пересобираем модель, если архитектура/признаки поменялись
      const needRebuild = !model.tfModel || model.features !== feats.length;
      if (needRebuild) {
        const built = buildTfModel(model.chain, feats.length);
        if (model.tfModel) { try { model.tfModel.dispose(); } catch (e) {} }
        model.tfModel = built.tfModel;
        model.preShape = built.preShape;
        model.features = feats.length;
      }

      const data = prepareXY(table, feats, args.label, loss, computeScalers(table, feats, args.label));
      model.mapping = data.mapping;
      model.scalers = computeScalers(table, feats, args.label);
      model.featureNames = feats;
      model.history = [];
      model.epochsTotal = epochs;
      model.trained = true;
      model.lastInput = [];
      setHUD('обучаю "' + args.model + '" (features: ' + feats.join(',') + ')…');

      // compile обязателен в tf.js — без него веса не обучаются
      try {
        model.tfModel.compile({ optimizer: args.opt, loss: loss });
      } catch (e) {
        warn('compile упал: ' + e.message);
        data.xs.dispose(); data.ys.dispose();
        return;
      }
      const redrawEvery = Math.max(1, Math.floor(epochs / 20));
      let xsT = data.xs;
      try {
        // LSTM/conv требуют не-плоский вход — решэпим и на обучении
        if (model.preShape) xsT = reshapeInputTensor(tf, xsT, model);
        await model.tfModel.fit(xsT, data.ys, {
          epochs: epochs,
          batchSize: batch,
          shuffle: true,
          callbacks: {
            onEpochEnd: async function (epoch, logs) {
              model.history.push(logs.loss);
              model.epochsDone = epoch + 1;
              if (epoch % redrawEvery === 0 || epoch === epochs - 1) {
                redraw();
                await new Promise(function (r) { setTimeout(r, 0); });
              }
            }
          }
        });
      } catch (e) {
        warn('обучение упало: ' + e.message);
        data.xs.dispose(); data.ys.dispose();
        return;
      }
      const finalLoss = model.history[model.history.length - 1];
      setHUD('обучено "' + args.model + '": loss=' + (finalLoss !== undefined ? finalLoss.toFixed(4) : '?'));
      data.xs.dispose(); data.ys.dispose();
      if (xsT !== data.xs) { try { xsT.dispose(); } catch (e) {} }
      redraw();
    }

    predict(args) {
      const model = store.models[args.model];
      if (!model || !model.tfModel) { warn('сначала обучи "' + args.model + '"'); return ''; }
      const vals = String(args.input).split(/[,\s]+/).filter(function (s) { return s !== ''; }).map(function (s) { return toNum(s); });
      if (vals.length !== model.features) {
        warn('вход должен быть из ' + model.features + ' чисел');
        return '';
      }
      model.lastInput = vals;
      try {
        const tf = window.tf;
        // авто-скейлинг входа так же, как при обучении
        const scaled = vals.map(function (v, i) {
          return model.scalers && model.scalers.featScalers[i] ? scaleVal(v, model.scalers.featScalers[i]) : v;
        });
        let t = tf.tensor2d([scaled], [1, model.features]);
        t = reshapeInputTensor(tf, t, model);
        const out = model.tfModel.predict(t);
        const arr = Array.from(out.dataSync());
        t.dispose(); out.dispose();
        redraw();
        if (model.mapping && arr.length === model.mapping.length) {
          const cls = model.mapping[arr.indexOf(Math.max.apply(null, arr))];
          return cls !== undefined ? String(cls) : arr.join(',');
        }
        // возвращаем в исходных единицах
        if (model.scalers && model.scalers.yScaler) {
          return arr.map(function (x) { return Math.round(unscaleVal(x, model.scalers.yScaler) * 1000) / 1000; }).join(',');
        }
        return arr.map(function (x) { return Math.round(x * 1000) / 1000; }).join(',');
      } catch (e) {
        warn('предсказание упало: ' + e.message);
        return '';
      }
    }

    score(args) {
      const model = store.models[args.model];
      if (!model || !model.tfModel) { warn('сначала обучи "' + args.model + '"'); return 0; }
      const table = store.tables[args.table];
      if (!table) { warn('нет таблицы "' + args.table + '"'); return 0; }
      const feats = featuresOf(table, args.features, args.label);
      if (feats.length === 0) return 0;
      try {
        const tf = window.tf;
        const X = [];
        for (let r = 0; r < table.rows; r++) {
          X.push(feats.map(function (f, i) {
            const v = table.columns[f][r];
            if (!isNum(v)) return 0;
            // скейлим так же, как при обучении
            if (model.scalers && model.scalers.featScalers[i]) return scaleVal(toNum(v), model.scalers.featScalers[i]);
            return toNum(v);
          }));
        }
        let t = tf.tensor2d(X, [X.length, feats.length], 'float32');
        t = reshapeInputTensor(tf, t, model);
        const out = model.tfModel.predict(t);
        const arr = Array.from(out.dataSync());
        t.dispose(); out.dispose();

        const yRaw = [];
        for (let r = 0; r < table.rows; r++) yRaw.push(table.columns[args.label][r]);

        // классификация?
        const isCat = model.mapping || yRaw.some(function (v) { return !isNum(v); });
        if (isCat) {
          const uniq = model.mapping || [];
          if (uniq.length === 0) {
            yRaw.forEach(function (v) { if (uniq.indexOf(v) === -1) uniq.push(v); });
          }
          // сколько выходов у модели — берём из последнего dense-слоя
          let outDim = uniq.length;
          for (let i = model.chain.length - 1; i >= 0; i--) {
            if (model.chain[i].type === 'dense') { outDim = model.chain[i].units; break; }
          }
          if (outDim !== uniq.length) {
            // модель не соответствует числу классов — считаем по числу классов
            outDim = uniq.length;
          }
          let ok = 0;
          for (let r = 0; r < table.rows; r++) {
            const preds = arr.slice(r * outDim, (r + 1) * outDim);
            const predIdx = preds.indexOf(Math.max.apply(null, preds));
            const trueIdx = uniq.indexOf(yRaw[r]);
            if (predIdx === trueIdx) ok++;
          }
          return Math.round((ok / table.rows) * 1000) / 1000;
        }
        // регрессия: R² (сравниваем с предсказаниями в исходных единицах)
        const yN = yRaw.map(toNum);
        const predVals = arr.map(function (x) {
          return model.scalers && model.scalers.yScaler ? unscaleVal(x, model.scalers.yScaler) : x;
        });
        const meanY = yN.reduce(function (a, b) { return a + b; }, 0) / yN.length;
        let ssRes = 0, ssTot = 0;
        for (let r = 0; r < table.rows; r++) {
          const d = yN[r] - predVals[r];
          ssRes += d * d;
          const dd = yN[r] - meanY;
          ssTot += dd * dd;
        }
        const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 1;
        return Math.round(r2 * 1000) / 1000;
      } catch (e) {
        warn('метрика упала: ' + e.message);
        return 0;
      }
    }

    lastLoss(args) {
      const m = store.models[args.model];
      if (!m || m.history.length === 0) return 0;
      return Math.round(m.history[m.history.length - 1] * 100000) / 100000;
    }

    // ============ визуализация ============

    drawNetwork(args) {
      const m = store.models[args.model];
      if (!m) { warn('нет модели "' + args.model + '"'); return; }
      m.visible = true;
      redraw();
    }

    hideNetwork(args) {
      const m = store.models[args.model];
      if (!m) return;
      m.visible = false;
      redraw();
    }

    clearAll() {
      Object.keys(store.models).forEach(function (k) {
        const m = store.models[k];
        if (m.tfModel) { try { m.tfModel.dispose(); } catch (e) {} }
      });
      store.models = {};
      store.tables = {};
      overlay.hud = '';
      redraw();
    }
  }

  Scratch.extensions.register(new ScratchTorch());
})(Scratch);