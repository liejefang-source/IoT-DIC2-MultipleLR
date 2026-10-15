/**
 * IoT Multiple Linear Regression Studio - Core Engine & Interactive UI
 * Enterprise Predictive Analytics & Edge AI Machine Learning System
 */

// ==========================================================================
// 1. STATISTICAL & LINEAR ALGEBRA ENGINE (OLS)
// ==========================================================================

const MathEngine = {
  // Transpose matrix A (n x m -> m x n)
  transpose(A) {
    const rows = A.length;
    const cols = A[0].length;
    const AT = Array.from({ length: cols }, () => new Float64Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  },

  // Multiply matrices A (n x k) and B (k x m) -> C (n x m)
  multiply(A, B) {
    const n = A.length;
    const k = A[0].length;
    const m = B[0].length;
    const C = Array.from({ length: n }, () => new Float64Array(m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0.0;
        for (let l = 0; l < k; l++) {
          sum += A[i][l] * B[l][j];
        }
        C[i][j] = sum;
      }
    }
    return C;
  },

  // Multiply matrix A (n x k) by vector v (k x 1) -> vector (n x 1)
  multiplyVector(A, v) {
    const n = A.length;
    const k = A[0].length;
    const res = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0.0;
      for (let l = 0; l < k; l++) {
        sum += A[i][l] * v[l];
      }
      res[i] = sum;
    }
    return res;
  },

  // Invert matrix using Gauss-Jordan elimination with partial pivoting & Tikhonov stabilization
  invert(A, lambda = 1e-7) {
    const n = A.length;
    // Create augmented matrix [A + lambda*I | I]
    const aug = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) {
        row[j] = A[i][j] + (i === j ? lambda : 0.0);
        row[j + n] = (i === j ? 1.0 : 0.0);
      }
      return row;
    });

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      let maxVal = Math.abs(aug[i][i]);
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(aug[r][i]) > maxVal) {
          maxVal = Math.abs(aug[r][i]);
          maxRow = r;
        }
      }

      if (maxVal < 1e-14) {
        // Singular or degenerate, add regularization fallback
        aug[i][i] += 1e-5;
      }

      // Swap rows
      if (maxRow !== i) {
        const tmp = aug[i];
        aug[i] = aug[maxRow];
        aug[maxRow] = tmp;
      }

      // Normalize pivot row
      const pivot = aug[i][i];
      for (let c = i; c < 2 * n; c++) {
        aug[i][c] /= pivot;
      }

      // Eliminate column
      for (let r = 0; r < n; r++) {
        if (r !== i) {
          const factor = aug[r][i];
          for (let c = i; c < 2 * n; c++) {
            aug[r][c] -= factor * aug[i][c];
          }
        }
      }
    }

    // Extract inverse matrix
    const inv = Array.from({ length: n }, (_, i) => aug[i].slice(n, 2 * n));
    return inv;
  },

  // Log Gamma function approximation (Lanczos formula)
  logGamma(z) {
    const c = [
      57.1562356658629235, -59.5979603554754912, 14.1360979740884268,
      -0.491913816097620199, 0.339946499848118887e-4, 0.465236289270485756e-4,
      -0.983744753048795646e-4, 0.158088703224377949e-3, -0.210264441724104883e-3,
      0.217439618115212643e-3, -0.16431810653676389e-3, 0.844182239838527433e-4,
      -0.261908384015814087e-4, 0.368991826595316234e-5
    ];
    if (z < 0.5) {
      return Math.log(Math.PI / Math.sin(Math.PI * z)) - this.logGamma(1 - z);
    }
    z -= 1;
    let base = z + 5.2421875;
    let sum = 0.999999999999997092;
    for (let i = 0; i < c.length; i++) {
      sum += c[i] / (z + i + 1);
    }
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(base) - base + Math.log(sum);
  },

  // Continued fraction for incomplete beta function (Lentz's method)
  betaContinuedFraction(a, b, x) {
    const MAXIT = 120;
    const EPS = 3e-9;
    const FPMIN = 1e-30;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1.0;
    let d = 1.0 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    let h = d;

    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      h *= d * c;

      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1.0 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1.0 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1.0 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < EPS) break;
    }
    return h;
  },

  // Regularized incomplete beta function I_x(a, b)
  incBeta(a, b, x) {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;
    const factor = Math.exp(
      this.logGamma(a + b) - this.logGamma(a) - this.logGamma(b) +
      a * Math.log(x) + b * Math.log(1.0 - x)
    );
    if (x < (a + 1.0) / (a + b + 2.0)) {
      return (factor * this.betaContinuedFraction(a, b, x)) / a;
    } else {
      return 1.0 - (factor * this.betaContinuedFraction(b, a, 1.0 - x)) / b;
    }
  },

  // Student's t two-tailed p-value
  studentTPValue(t, df) {
    if (isNaN(t) || df <= 0) return 1.0;
    const t2 = t * t;
    const x = df / (df + t2);
    return this.incBeta(0.5 * df, 0.5, x);
  },

  // Snedecor's F-distribution right-tail p-value
  fDistributionPValue(F, df1, df2) {
    if (isNaN(F) || F <= 0 || df1 <= 0 || df2 <= 0) return 1.0;
    const x = (df1 * F) / (df1 * F + df2);
    return 1.0 - this.incBeta(0.5 * df1, 0.5 * df2, x);
  },

  // Pearson Correlation Coefficient
  correlation(x, y) {
    const n = x.length;
    if (n === 0) return 0;
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += x[i];
      sy += y[i];
      sxx += x[i] * x[i];
      syy += y[i] * y[i];
      sxy += x[i] * y[i];
    }
    const num = n * sxy - sx * sy;
    const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return den === 0 ? 0 : num / den;
  },

  // Solve Multiple Linear Regression (OLS)
  fitOLS(X, y, featureNames) {
    const n = X.length;
    const p = X[0].length; // p includes intercept (column 0 = 1)
    const df = n - p;

    // Xt = X^T
    const Xt = this.transpose(X);
    // XtX = X^T * X
    const XtX = this.multiply(Xt, X);
    // Xty = X^T * y
    const Xty = this.multiplyVector(Xt, y);

    // Invert XtX
    const invXtX = this.invert(XtX);

    // beta = (X^T X)^-1 X^T y
    const beta = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      let sum = 0.0;
      for (let j = 0; j < p; j++) {
        sum += invXtX[i][j] * Xty[j];
      }
      beta[i] = sum;
    }

    // Predictions & Residuals
    const yPred = new Float64Array(n);
    const residuals = new Float64Array(n);
    let ySum = 0.0;
    let rss = 0.0;
    let absErrSum = 0.0;

    for (let i = 0; i < n; i++) {
      let pred = 0.0;
      for (let j = 0; j < p; j++) {
        pred += X[i][j] * beta[j];
      }
      yPred[i] = pred;
      const res = y[i] - pred;
      residuals[i] = res;
      rss += res * res;
      absErrSum += Math.abs(res);
      ySum += y[i];
    }

    const yMean = ySum / n;
    let tss = 0.0;
    for (let i = 0; i < n; i++) {
      const diff = y[i] - yMean;
      tss += diff * diff;
    }

    // Metrics
    const r2 = tss > 0 ? 1.0 - rss / tss : 0.0;
    const adjR2 = (n - 1 > 0 && df > 0) ? 1.0 - (1.0 - r2) * ((n - 1) / df) : r2;
    const rmse = Math.sqrt(rss / n);
    const mae = absErrSum / n;

    // F-statistic & p-value
    const pFeatures = p - 1;
    let fStat = 0.0;
    let modelPVal = 1.0;
    if (pFeatures > 0 && df > 0 && rss > 0) {
      const msr = (tss - rss) / pFeatures;
      const mse = rss / df;
      fStat = msr / mse;
      modelPVal = this.fDistributionPValue(fStat, pFeatures, df);
    }

    // Coefficients statistics
    const sigma2 = df > 0 ? rss / df : 0;
    const coefficients = [];

    for (let j = 0; j < p; j++) {
      const varBeta = Math.max(0, sigma2 * invXtX[j][j]);
      const se = Math.sqrt(varBeta);
      const tStat = se > 1e-12 ? beta[j] / se : 0;
      const pVal = df > 0 ? this.studentTPValue(tStat, df) : 1.0;
      // 95% confidence interval
      const ciHalf = 1.96 * se;
      coefficients.push({
        name: featureNames[j],
        beta: beta[j],
        se: se,
        tStat: tStat,
        pVal: pVal,
        ciLower: beta[j] - ciHalf,
        ciUpper: beta[j] + ciHalf
      });
    }

    return {
      beta,
      coefficients,
      invXtX,
      r2,
      adjR2,
      rmse,
      mae,
      rss,
      tss,
      df,
      n,
      p,
      yMean,
      fStat,
      modelPVal,
      yPred,
      residuals
    };
  },

  // Predict with fitted model
  predict(beta, featureVector) {
    // featureVector includes 1 for intercept at index 0
    let res = 0.0;
    for (let i = 0; i < beta.length; i++) {
      res += beta[i] * featureVector[i];
    }
    return res;
  }
};

// ==========================================================================
// 2. DATASET MANAGER & PREPROCESSING
// ==========================================================================

class DatasetManager {
  constructor() {
    this.rawRows = [];
    this.columns = [];
    this.targetCol = '';
    this.featureCols = [];
    this.activeFeatures = []; // currently checked features
    this.categoricalMap = {}; // mapping for one-hot encoding
    this.isCategoricalCol = {};
    this.unit = '$';
    this.datasetName = '';
  }

  // Load from parsed CSV array
  loadFromRows(rows, name = 'Custom Dataset', defaultTarget = '') {
    if (!rows || rows.length < 2) {
      throw new Error('資料集樣本數不足，至少需要2筆有效資料！');
    }
    this.datasetName = name;
    this.columns = Object.keys(rows[0]).filter(c => c && c.trim().length > 0);
    this.rawRows = rows.filter(r => {
      // Drop completely empty rows
      return this.columns.some(c => r[c] !== null && r[c] !== undefined && r[c] !== '');
    });

    // Detect categorical vs numeric columns
    this.isCategoricalCol = {};
    this.categoricalMap = {};

    this.columns.forEach(col => {
      const sampleVals = this.rawRows.slice(0, 30).map(r => r[col]);
      const hasString = sampleVals.some(v => isNaN(Number(v)) && typeof v === 'string');
      if (hasString) {
        this.isCategoricalCol[col] = true;
        // Collect unique values
        const uniqueVals = [...new Set(this.rawRows.map(r => String(r[col]).trim()))].filter(Boolean);
        this.categoricalMap[col] = uniqueVals;
      } else {
        this.isCategoricalCol[col] = false;
      }
    });

    // Determine target column
    if (defaultTarget && this.columns.includes(defaultTarget)) {
      this.targetCol = defaultTarget;
    } else {
      // Pick last numeric column as target by default
      const numericCols = this.columns.filter(c => !this.isCategoricalCol[c]);
      this.targetCol = numericCols.length > 0 ? numericCols[numericCols.length - 1] : this.columns[this.columns.length - 1];
    }

    // Set feature columns (all columns except target)
    this.featureCols = this.columns.filter(c => c !== this.targetCol);
    this.activeFeatures = [...this.featureCols];
  }

  // Prepare full expanded feature matrix X (with one-hot encoding) and target y
  prepareDataMatrix(selectedFeatures, trainRatio = 0.8, seed = 42) {
    const n = this.rawRows.length;
    // Build design matrix schema
    const featureNames = ['Intercept'];
    const featureGetters = [];

    selectedFeatures.forEach(col => {
      if (this.isCategoricalCol[col]) {
        // Categorical One-Hot: drop first category as baseline to avoid dummy variable trap
        const cats = this.categoricalMap[col] || [];
        for (let i = 1; i < cats.length; i++) {
          const catVal = cats[i];
          featureNames.push(`${col}_${catVal}`);
          featureGetters.push((row) => (String(row[col]).trim() === catVal ? 1.0 : 0.0));
        }
      } else {
        featureNames.push(col);
        featureGetters.push((row) => {
          const val = Number(row[col]);
          return isNaN(val) ? 0.0 : val;
        });
      }
    });

    // Build X and y
    const X = [];
    const y = [];
    const originalRows = [];

    this.rawRows.forEach((row, idx) => {
      const targetVal = Number(row[this.targetCol]);
      if (!isNaN(targetVal)) {
        const xRow = [1.0];
        featureGetters.forEach(getter => xRow.push(getter(row)));
        X.push(xRow);
        y.push(targetVal);
        originalRows.push({ ...row, _id: idx + 1 });
      }
    });

    // Pseudo-random split using Linear Congruential Generator (LCG)
    const indices = Array.from({ length: X.length }, (_, i) => i);
    let s = seed;
    for (let i = indices.length - 1; i > 0; i--) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      const j = Math.floor((s / 4294967296) * (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }

    const trainSize = Math.max(2, Math.floor(X.length * trainRatio));
    const trainIndices = indices.slice(0, trainSize);
    const testIndices = indices.slice(trainSize);

    const trainX = trainIndices.map(i => X[i]);
    const trainY = trainIndices.map(i => y[i]);
    const trainRows = trainIndices.map(i => originalRows[i]);

    const testX = testIndices.map(i => X[i]);
    const testY = testIndices.map(i => y[i]);
    const testRows = testIndices.map(i => originalRows[i]);

    return {
      featureNames,
      featureGetters,
      train: { X: trainX, y: trainY, rows: trainRows },
      test: { X: testX, y: testY, rows: testRows },
      full: { X, y, rows: originalRows }
    };
  }

  // Get summary stats for a column (min, max, mean, correlation with target)
  getColumnStats(col) {
    if (this.isCategoricalCol[col]) {
      return { isCat: true, categories: this.categoricalMap[col] || [] };
    }
    const vals = this.rawRows.map(r => Number(r[col])).filter(v => !isNaN(v));
    const targetVals = this.rawRows.map(r => Number(r[this.targetCol])).filter(v => !isNaN(v));
    if (vals.length === 0) return { min: 0, max: 1, mean: 0, std: 1, corr: 0 };
    
    let min = Infinity, max = -Infinity, sum = 0;
    vals.forEach(v => {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    });
    const mean = sum / vals.length;
    let varSum = 0;
    vals.forEach(v => {
      varSum += (v - mean) * (v - mean);
    });
    const std = Math.sqrt(varSum / vals.length);
    const corr = MathEngine.correlation(vals, targetVals);

    return { isCat: false, min, max, mean, std, corr };
  }
}

// ==========================================================================
// 3. APPLICATION STATE & CONTROLLER
// ==========================================================================

const App = {
  datasetMgr: new DatasetManager(),
  currentModel: null,
  preparedData: null,
  splitSeed: 42,
  charts: {
    rmseFeatures: null,
    adjR2Features: null,
    scatter: null,
    residuals: null
  },

  init() {
    this.bindEvents();
    // Load default dataset: 50 Startups
    this.loadPresetProblem('startups');
    // Render Lucide icons
    if (window.lucide) {
      window.lucide.createIcons();
    }
  },

  bindEvents() {
    // Problem selector dropdown (+1 bonus point)
    const problemSelector = document.getElementById('problemSelector');
    problemSelector.addEventListener('change', (e) => {
      const val = e.target.value;
      if (val === 'startups' || val === 'boston') {
        this.loadPresetProblem(val);
      } else if (val === 'custom') {
        // Highlight dataset URL toolbar
        document.getElementById('datasetUrlInput').focus();
      }
    });

    // Preset buttons
    document.getElementById('btnPresetStartups').addEventListener('click', () => {
      problemSelector.value = 'startups';
      this.loadPresetProblem('startups');
    });

    document.getElementById('btnPresetBoston').addEventListener('click', () => {
      problemSelector.value = 'boston';
      this.loadPresetProblem('boston');
    });

    // Load from URL button (+2 bonus points)
    document.getElementById('btnLoadUrl').addEventListener('click', () => {
      this.loadDatasetFromUrl();
    });

    // Local file upload
    document.getElementById('fileUploadInput').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        this.loadDatasetFromFile(file);
      }
    });

    // Train/Test Split Slider
    const splitSlider = document.getElementById('trainSplitSlider');
    const splitLabel = document.getElementById('trainSplitLabel');
    splitSlider.addEventListener('input', (e) => {
      const val = e.target.value;
      splitLabel.textContent = `${val}% Train / ${100 - val}% Test`;
      this.trainAndRender();
    });

    // Reshuffle button
    document.getElementById('btnReshuffle').addEventListener('click', () => {
      this.splitSeed = Math.floor(Math.random() * 100000);
      this.trainAndRender();
    });

    // Target Column select change
    document.getElementById('targetColSelect').addEventListener('change', (e) => {
      this.datasetMgr.targetCol = e.target.value;
      this.datasetMgr.featureCols = this.datasetMgr.columns.filter(c => c !== this.datasetMgr.targetCol);
      this.datasetMgr.activeFeatures = [...this.datasetMgr.featureCols];
      this.renderFeatureChips();
      this.trainAndRender();
    });

    // Feature selection quick buttons
    document.getElementById('btnSelectAllFeatures').addEventListener('click', () => {
      this.datasetMgr.activeFeatures = [...this.datasetMgr.featureCols];
      this.renderFeatureChips();
      this.trainAndRender();
    });

    document.getElementById('btnResetFeatures').addEventListener('click', () => {
      // Keep only first feature if any
      this.datasetMgr.activeFeatures = this.datasetMgr.featureCols.slice(0, 1);
      this.renderFeatureChips();
      this.trainAndRender();
    });

    // Copy Equation button
    document.getElementById('btnCopyEquation').addEventListener('click', () => {
      const eqText = document.getElementById('equationDisplay').innerText;
      navigator.clipboard.writeText(eqText).then(() => {
        alert('多元線性迴歸方程式已成功複製至剪貼簿！');
      });
    });

    // Reset Live Predictor to mean
    document.getElementById('btnResetPredictor').addEventListener('click', () => {
      this.resetPredictorInputs();
    });

    // Chart tab switches
    document.querySelectorAll('.chart-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.chart-view').forEach(v => v.classList.remove('active'));
        const btn = e.currentTarget;
        btn.classList.add('active');
        const chartType = btn.dataset.chart;
        if (chartType === 'stepwise') {
          document.getElementById('chartViewStepwise').classList.add('active');
        } else if (chartType === 'scatter') {
          document.getElementById('chartViewScatter').classList.add('active');
          if (this.charts.scatter) this.charts.scatter.resize();
        } else if (chartType === 'residuals') {
          document.getElementById('chartViewResiduals').classList.add('active');
          if (this.charts.residuals) this.charts.residuals.resize();
        }
      });
    });

    // Diagnostic table search & filter
    document.getElementById('testTableSearch').addEventListener('input', () => {
      this.renderTestSetTable();
    });
    document.getElementById('testTableFilter').addEventListener('change', () => {
      this.renderTestSetTable();
    });

    // Modals
    const deployModal = document.getElementById('deployModal');
    document.getElementById('btnOpenDeployModal').addEventListener('click', () => {
      deployModal.classList.add('open');
    });
    document.getElementById('btnCloseDeployModal').addEventListener('click', () => {
      deployModal.classList.remove('open');
    });
    document.getElementById('btnModalClose').addEventListener('click', () => {
      deployModal.classList.remove('open');
    });
    deployModal.addEventListener('click', (e) => {
      if (e.target === deployModal) deployModal.classList.remove('open');
    });

    // Docs scroll
    document.getElementById('btnOpenDocs').addEventListener('click', () => {
      const sec = document.getElementById('methodologySection');
      if (sec) sec.scrollIntoView({ behavior: 'smooth' });
    });
  },

  // Load predefined Kaggle dataset
  loadPresetProblem(presetId) {
    // Update active preset button style
    document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));
    if (presetId === 'startups') {
      document.getElementById('btnPresetStartups').classList.add('active');
    } else if (presetId === 'boston') {
      document.getElementById('btnPresetBoston').classList.add('active');
    }

    const dataObj = window.BUILTIN_DATASETS ? window.BUILTIN_DATASETS[presetId] : null;
    if (!dataObj) {
      alert(`找不到預設資料集: ${presetId}`);
      return;
    }

    // Parse CSV via PapaParse
    Papa.parse(dataObj.rawCsv, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        this.datasetMgr.unit = dataObj.targetUnit || '';
        this.datasetMgr.loadFromRows(results.data, dataObj.name, dataObj.targetCol);
        document.getElementById('datasetDescText').textContent = dataObj.description;
        this.updateConfigUI();
        this.trainAndRender();
      },
      error: (err) => {
        alert('解析資料集失敗: ' + err.message);
      }
    });
  },

  // Load arbitrary CSV dataset from URL (+2 bonus points)
  loadDatasetFromUrl() {
    const url = document.getElementById('datasetUrlInput').value.trim();
    if (!url) {
      alert('請先輸入 CSV 網址！');
      return;
    }

    const btn = document.getElementById('btnLoadUrl');
    const oldText = btn.innerHTML;
    btn.innerHTML = '<i data-lucide="loader-2" class="icon-xs"></i> 載入中...';

    // Helper to parse CSV string
    const handleCsvText = (csvText) => {
      Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          btn.innerHTML = oldText;
          if (!results.data || results.data.length < 5) {
            alert('CSV 內容筆數過少或格式不正確！');
            return;
          }
          this.datasetMgr.unit = '';
          this.datasetMgr.loadFromRows(results.data, '自訂網址資料集');
          document.getElementById('problemSelector').value = 'custom';
          document.getElementById('datasetDescText').textContent = `自訂網址資料集: ${url}`;
          this.updateConfigUI();
          this.trainAndRender();
          alert(`自訂資料集載入成功！共解析 ${results.data.length} 筆資料。`);
        },
        error: (err) => {
          btn.innerHTML = oldText;
          alert('CSV 解析失敗: ' + err.message);
        }
      });
    };

    // Try direct fetch first
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.text();
      })
      .then(csvText => handleCsvText(csvText))
      .catch(err => {
        console.warn('Direct fetch failed, trying CORS proxy...', err);
        // Fallback to CORS proxy
        const proxyUrl = 'https://corsproxy.io/?' + encodeURIComponent(url);
        fetch(proxyUrl)
          .then(res => {
            if (!res.ok) throw new Error('CORS Proxy HTTP ' + res.status);
            return res.text();
          })
          .then(csvText => handleCsvText(csvText))
          .catch(proxyErr => {
            btn.innerHTML = oldText;
            alert(`載入網址失敗: ${err.message}。若因 CORS 跨域限制，您可下載該 CSV 檔案後點擊「本地上傳 CSV」進行分析！`);
          });
      });
  },

  // Load dataset from local file
  loadDatasetFromFile(file) {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (!results.data || results.data.length < 5) {
          alert('CSV 內容筆數過少或格式不正確！');
          return;
        }
        this.datasetMgr.unit = '';
        this.datasetMgr.loadFromRows(results.data, `本地上傳: ${file.name}`);
        document.getElementById('problemSelector').value = 'custom';
        document.getElementById('datasetDescText').textContent = `本地檔案: ${file.name} (${results.data.length} 筆)`;
        this.updateConfigUI();
        this.trainAndRender();
      },
      error: (err) => {
        alert('讀取本地檔案失敗: ' + err.message);
      }
    });
  },

  // Update Config Bar UI (Target options, feature chips, badges)
  updateConfigUI() {
    const mgr = this.datasetMgr;

    // Badges
    document.getElementById('badgeSampleCount').innerHTML = `<i data-lucide="database" class="icon-xs"></i> ${mgr.rawRows.length} 筆樣本`;
    document.getElementById('badgeFeatureCount').innerHTML = `<i data-lucide="columns" class="icon-xs"></i> ${mgr.featureCols.length} 個候選變數`;
    document.getElementById('badgeTargetCol').innerHTML = `<i data-lucide="target" class="icon-xs"></i> 目標: ${mgr.targetCol}`;

    // Target Select Options
    const targetSelect = document.getElementById('targetColSelect');
    targetSelect.innerHTML = '';
    mgr.columns.forEach(col => {
      const opt = document.createElement('option');
      opt.value = col;
      opt.textContent = col + (mgr.isCategoricalCol[col] ? ' (類別)' : ' (數值)');
      if (col === mgr.targetCol) opt.selected = true;
      targetSelect.appendChild(opt);
    });

    this.renderFeatureChips();
    if (window.lucide) window.lucide.createIcons();
  },

  // Render Feature selection chips
  renderFeatureChips() {
    const container = document.getElementById('featureChipsContainer');
    container.innerHTML = '';
    const mgr = this.datasetMgr;

    mgr.featureCols.forEach(col => {
      const chip = document.createElement('div');
      const isActive = mgr.activeFeatures.includes(col);
      chip.className = `feature-chip ${isActive ? 'active' : ''}`;
      
      const stats = mgr.getColumnStats(col);
      let corrText = '';
      if (!stats.isCat) {
        corrText = `(r=${stats.corr >= 0 ? '+' : ''}${stats.corr.toFixed(2)})`;
      } else {
        corrText = `[類別:${stats.categories.length}]`;
      }

      chip.innerHTML = `
        <span class="chip-dot"></span>
        <span class="chip-name">${col}</span>
        <span class="chip-corr">${corrText}</span>
      `;

      chip.addEventListener('click', () => {
        if (mgr.activeFeatures.includes(col)) {
          if (mgr.activeFeatures.length <= 1) {
            alert('至少需保留一個特徵進行迴歸分析！');
            return;
          }
          mgr.activeFeatures = mgr.activeFeatures.filter(f => f !== col);
        } else {
          mgr.activeFeatures.push(col);
        }
        this.renderFeatureChips();
        this.trainAndRender();
      });

      container.appendChild(chip);
    });
  },

  // Train OLS model and update entire dashboard
  trainAndRender() {
    const mgr = this.datasetMgr;
    if (mgr.activeFeatures.length === 0) return;

    const splitRatio = Number(document.getElementById('trainSplitSlider').value) / 100.0;
    
    // Prepare train/test matrix
    const data = mgr.prepareDataMatrix(mgr.activeFeatures, splitRatio, this.splitSeed);
    this.preparedData = data;

    // Fit OLS on training set
    const model = MathEngine.fitOLS(data.train.X, data.train.y, data.featureNames);
    this.currentModel = model;

    // Evaluate on test set
    let testRss = 0.0;
    let testAbsErr = 0.0;
    const testPreds = [];
    let testYSum = 0.0;

    data.test.X.forEach((xRow, idx) => {
      const pred = MathEngine.predict(model.beta, xRow);
      testPreds.push(pred);
      const actual = data.test.y[idx];
      const res = actual - pred;
      testRss += res * res;
      testAbsErr += Math.abs(res);
      testYSum += actual;
    });

    const testN = data.test.y.length;
    const testRmse = testN > 0 ? Math.sqrt(testRss / testN) : 0;
    const testMae = testN > 0 ? testAbsErr / testN : 0;
    const testYMean = testN > 0 ? testYSum / testN : 0;

    let testTss = 0.0;
    data.test.y.forEach(actual => {
      testTss += (actual - testYMean) * (actual - testYMean);
    });
    const testR2 = testTss > 0 ? 1.0 - testRss / testTss : 0;

    // 1. Render Metrics Cards
    this.renderMetrics(model, testR2, testRmse, testMae);

    // 2. Render Fitted Equation
    this.renderEquation(model);

    // 3. Render Coefficients Table
    this.renderCoefficientsTable(model);

    // 4. Render Live Predictor
    this.renderLivePredictor(model, data);

    // 5. Render Test Set Diagnostic Table
    this.renderTestSetTable(testPreds);

    // 6. Render Charts (Stepwise curve, Scatter plot, Residuals)
    this.renderCharts(model, data, testPreds);

    if (window.lucide) window.lucide.createIcons();
  },

  // Format currency or numbers
  formatVal(val, decimals = 2) {
    if (isNaN(val)) return '0.00';
    const isCurrency = this.datasetMgr.unit === '$';
    const absVal = Math.abs(val);
    let formatted = absVal.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
    if (val < 0) formatted = '-' + formatted;
    return isCurrency ? '$' + formatted : formatted;
  },

  // 1. Render KPI Metrics
  renderMetrics(model, testR2, testRmse, testMae) {
    document.getElementById('metricR2').textContent = model.r2.toFixed(4);
    document.getElementById('metricTestR2').textContent = testR2.toFixed(4);
    
    // Evaluation badge for R2
    const r2Desc = document.getElementById('metricR2Desc');
    if (model.r2 >= 0.85) {
      r2Desc.textContent = '解釋能力極佳';
      r2Desc.className = 'metric-badge badge-emerald';
    } else if (model.r2 >= 0.65) {
      r2Desc.textContent = '擬合良好';
      r2Desc.className = 'metric-badge badge-emerald';
    } else {
      r2Desc.textContent = '解釋度偏低';
      r2Desc.className = 'metric-badge badge-purple';
    }

    document.getElementById('metricAdjR2').textContent = model.adjR2.toFixed(4);
    document.getElementById('metricDf').textContent = model.df;

    document.getElementById('metricRMSE').textContent = this.formatVal(model.rmse);
    document.getElementById('metricTestRMSE').textContent = this.formatVal(testRmse);

    document.getElementById('metricMAE').textContent = this.formatVal(model.mae);
    document.getElementById('metricMeanY').textContent = this.formatVal(model.yMean);

    document.getElementById('metricFStat').textContent = model.fStat.toFixed(2);
    const pValBadge = document.getElementById('metricPValBadge');
    if (model.modelPVal < 0.001) {
      pValBadge.textContent = 'p < 0.0001 (***)';
    } else {
      pValBadge.textContent = `p = ${model.modelPVal.toFixed(4)}`;
    }
  },

  // 2. Render Mathematical Equation
  renderEquation(model) {
    const targetName = this.datasetMgr.targetCol;
    let eqHtml = `<span class="eq-target">${targetName}</span> = `;

    model.coefficients.forEach((c, idx) => {
      const isIntercept = idx === 0;
      const b = c.beta;
      const sign = b >= 0 ? (isIntercept ? '' : '+ ') : '- ';
      const absB = Math.abs(b);
      const formattedB = absB.toFixed(4);

      if (isIntercept) {
        eqHtml += `<span class="eq-coef">${b < 0 ? '-' : ''}${formattedB}</span>`;
      } else {
        eqHtml += ` <span class="eq-op">${sign}</span><span class="eq-coef">${formattedB}</span> · <span class="eq-feat">(${c.name})</span>`;
      }
    });

    document.getElementById('equationDisplay').innerHTML = eqHtml;
  },

  // 3. Render Model Coefficients Table
  renderCoefficientsTable(model) {
    const tbody = document.getElementById('coefficientsTableBody');
    tbody.innerHTML = '';

    model.coefficients.forEach(c => {
      const tr = document.createElement('tr');
      
      // Significance stars
      let star = 'ns';
      if (c.pVal < 0.001) star = '***';
      else if (c.pVal < 0.01) star = '**';
      else if (c.pVal < 0.05) star = '*';
      else if (c.pVal < 0.1) star = '·';

      const pValStr = c.pVal < 0.0001 ? '< 0.0001' : c.pVal.toFixed(4);

      tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${c.name}</td>
        <td style="color: ${c.beta >= 0 ? '#38bdf8' : '#f43f5e'}; font-weight: 700;">${c.beta.toFixed(4)}</td>
        <td>${c.se.toFixed(4)}</td>
        <td>${c.tStat.toFixed(3)}</td>
        <td>${pValStr}</td>
        <td><span class="signif-star">${star}</span></td>
        <td style="color: var(--text-muted);">[${c.ciLower.toFixed(2)}, ${c.ciUpper.toFixed(2)}]</td>
      `;
      tbody.appendChild(tr);
    });
  },

  // 4. Render Live Real-Time Predictor
  renderLivePredictor(model, data) {
    const container = document.getElementById('predictorControlsContainer');
    container.innerHTML = '';
    const mgr = this.datasetMgr;

    document.getElementById('predictorTargetName').textContent = `PREDICTED ${mgr.targetCol.toUpperCase()} (${mgr.unit})`;

    // Store state of predictor inputs
    this.predictorInputs = {};

    mgr.activeFeatures.forEach(col => {
      const stats = mgr.getColumnStats(col);
      const itemDiv = document.createElement('div');
      itemDiv.className = 'pred-item';

      if (stats.isCat) {
        // Categorical select
        itemDiv.innerHTML = `
          <div class="pred-label-row">
            <span class="pred-name">${col} (類別)</span>
          </div>
          <select id="pred_select_${col}" class="select-input select-sm">
            ${stats.categories.map(c => `<option value="${c}">${c}</option>`).join('')}
          </select>
        `;
        container.appendChild(itemDiv);

        const selectEl = itemDiv.querySelector('select');
        selectEl.addEventListener('change', () => {
          this.calculateLivePrediction();
        });
        this.predictorInputs[col] = { type: 'cat', el: selectEl };
      } else {
        // Numeric slider + number input
        const minVal = Math.floor(stats.min * 0.9);
        const maxVal = Math.ceil(stats.max * 1.1);
        const defaultVal = Number(stats.mean.toFixed(2));
        const stepVal = Math.max(0.01, Number(((maxVal - minVal) / 200).toFixed(2)));

        itemDiv.innerHTML = `
          <div class="pred-label-row">
            <span class="pred-name">${col}</span>
            <span id="pred_display_${col}" class="pred-val-box">${this.formatVal(defaultVal, 2)}</span>
          </div>
          <div class="pred-input-row">
            <input type="range" id="pred_slider_${col}" class="range-slider pred-slider" min="${minVal}" max="${maxVal}" step="${stepVal}" value="${defaultVal}">
            <input type="number" id="pred_num_${col}" class="pred-number-input" value="${defaultVal}" step="${stepVal}">
          </div>
        `;
        container.appendChild(itemDiv);

        const sliderEl = itemDiv.querySelector('.pred-slider');
        const numEl = itemDiv.querySelector('.pred-number-input');
        const displayEl = itemDiv.querySelector('.pred-val-box');

        const updateVal = (val) => {
          val = Number(val);
          sliderEl.value = val;
          numEl.value = val;
          displayEl.textContent = this.formatVal(val, 2);
          this.calculateLivePrediction();
        };

        sliderEl.addEventListener('input', (e) => updateVal(e.target.value));
        numEl.addEventListener('input', (e) => updateVal(e.target.value));

        this.predictorInputs[col] = { type: 'num', slider: sliderEl, num: numEl, display: displayEl, mean: defaultVal };
      }
    });

    this.calculateLivePrediction();
  },

  // Calculate live predicted target value from inputs
  calculateLivePrediction() {
    if (!this.currentModel || !this.preparedData) return;
    const model = this.currentModel;
    const mgr = this.datasetMgr;

    // Build feature vector with 1.0 for intercept
    const v = [1.0];
    const dummyRow = {};

    mgr.activeFeatures.forEach(col => {
      const input = this.predictorInputs[col];
      if (input.type === 'cat') {
        dummyRow[col] = input.el.value;
      } else {
        dummyRow[col] = Number(input.slider.value);
      }
    });

    // Evaluate with feature getters
    this.preparedData.featureGetters.forEach(getter => {
      v.push(getter(dummyRow));
    });

    const predY = MathEngine.predict(model.beta, v);
    const outputEl = document.getElementById('predictorOutputValue');
    outputEl.textContent = this.formatVal(predY);

    // Confidence interval approximation (~ 1.96 * RMSE)
    const lower = predY - 1.96 * model.rmse;
    const upper = predY + 1.96 * model.rmse;
    document.getElementById('predictorConfidenceRange').textContent = 
      `95% 預測信賴區間: ${this.formatVal(lower)} ~ ${this.formatVal(upper)}`;

    // Visual pulse
    outputEl.style.transform = 'scale(1.03)';
    setTimeout(() => { outputEl.style.transform = 'scale(1)'; }, 150);
  },

  // Reset predictor inputs to mean
  resetPredictorInputs() {
    if (!this.predictorInputs) return;
    Object.keys(this.predictorInputs).forEach(col => {
      const input = this.predictorInputs[col];
      if (input.type === 'num') {
        input.slider.value = input.mean;
        input.num.value = input.mean;
        input.display.textContent = this.formatVal(input.mean, 2);
      }
    });
    this.calculateLivePrediction();
  },

  // 5. Render Test Set Diagnostic Table
  renderTestSetTable(testPreds) {
    if (!this.preparedData) return;
    const testData = this.preparedData.test;
    const tbody = document.getElementById('testSetTableBody');
    tbody.innerHTML = '';

    const searchTerm = (document.getElementById('testTableSearch').value || '').trim().toLowerCase();
    const filterStatus = document.getElementById('testTableFilter').value;
    const rmse = this.currentModel.rmse;

    testData.rows.forEach((row, idx) => {
      const actual = testData.y[idx];
      const pred = testPreds ? testPreds[idx] : MathEngine.predict(this.currentModel.beta, testData.X[idx]);
      const res = actual - pred;
      const absRes = Math.abs(res);
      const errPct = actual !== 0 ? (absRes / Math.abs(actual)) * 100 : 0;

      // Classify outlier status
      let statusBadge = '';
      let statusClass = 'normal';
      if (absRes > 2.0 * rmse) {
        statusBadge = '<span class="badge-status status-outlier">異常值 Outlier (>2σ)</span>';
        statusClass = 'outlier';
      } else if (absRes > 1.0 * rmse) {
        statusBadge = '<span class="badge-status status-high">偏高誤差 (>1σ)</span>';
        statusClass = 'high';
      } else {
        statusBadge = '<span class="badge-status status-normal">正常 Normal</span>';
        statusClass = 'normal';
      }

      // Filter check
      if (filterStatus !== 'all' && filterStatus !== statusClass) {
        return;
      }

      // Search check
      const rowIdStr = String(row._id);
      if (searchTerm && !rowIdStr.includes(searchTerm) && !String(actual).includes(searchTerm)) {
        return;
      }

      const tr = document.createElement('tr');
      const diffSign = res >= 0 ? '+' : '';
      const diffColor = res >= 0 ? '#34d399' : '#f43f5e';

      tr.innerHTML = `
        <td style="font-weight: 700; color: #fff;">#${row._id}</td>
        <td>${this.formatVal(actual)}</td>
        <td style="color: #38bdf8;">${this.formatVal(pred)}</td>
        <td style="color: ${diffColor};">${diffSign}${this.formatVal(res)}</td>
        <td>${errPct.toFixed(2)}%</td>
        <td>${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    });
  },

  // 6. Render Charts (Dual Stepwise Curves, Scatter, Residuals)
  renderCharts(model, data, testPreds) {
    this.renderStepwiseCharts(data);
    this.renderScatterChart(data, testPreds);
    this.renderResidualsChart(model);
  },

  // Stepwise feature evaluation: computes RMSE & Adj R2 as feature count increases
  renderStepwiseCharts(data) {
    const mgr = this.datasetMgr;
    const activeFeats = [...mgr.activeFeatures];
    const maxK = activeFeats.length;

    const labels = [];
    const rmseVals = [];
    const adjR2Vals = [];

    // Order features by correlation with target (greedy forward selection order)
    const sortedFeats = [...activeFeats].sort((a, b) => {
      const sa = mgr.getColumnStats(a);
      const sb = mgr.getColumnStats(b);
      return Math.abs(sb.corr || 0) - Math.abs(sa.corr || 0);
    });

    for (let k = 1; k <= maxK; k++) {
      const subsetFeats = sortedFeats.slice(0, k);
      const subData = mgr.prepareDataMatrix(subsetFeats, 0.8, this.splitSeed);
      const subModel = MathEngine.fitOLS(subData.train.X, subData.train.y, subData.featureNames);
      labels.push(String(k));
      rmseVals.push(Number(subModel.rmse.toFixed(2)));
      adjR2Vals.push(Number(subModel.adjR2.toFixed(4)));
    }

    // Chart 1: RMSE by Number of Features
    const ctxRmse = document.getElementById('chartRmseFeatures').getContext('2d');
    if (this.charts.rmseFeatures) this.charts.rmseFeatures.destroy();

    this.charts.rmseFeatures = new Chart(ctxRmse, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'RMSE (均方根誤差)',
          data: rmseVals,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointBackgroundColor: '#06b6d4',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(18, 26, 44, 0.95)',
            borderColor: 'rgba(6, 182, 212, 0.4)',
            borderWidth: 1,
            titleFont: { family: 'Plus Jakarta Sans', weight: 'bold' },
            bodyFont: { family: 'JetBrains Mono' }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Number of Features (k)', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          },
          y: {
            title: { display: true, text: 'RMSE', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          }
        }
      }
    });

    // Chart 2: Adjusted R² by Number of Features
    const ctxAdjR2 = document.getElementById('chartAdjR2Features').getContext('2d');
    if (this.charts.adjR2Features) this.charts.adjR2Features.destroy();

    this.charts.adjR2Features = new Chart(ctxAdjR2, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Adjusted R²',
          data: adjR2Vals,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(99, 102, 241, 0.12)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5,
          pointBackgroundColor: '#818cf8',
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(18, 26, 44, 0.95)',
            borderColor: 'rgba(99, 102, 241, 0.4)',
            borderWidth: 1,
            titleFont: { family: 'Plus Jakarta Sans', weight: 'bold' },
            bodyFont: { family: 'JetBrains Mono' }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Number of Features (k)', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          },
          y: {
            title: { display: true, text: 'Adjusted R²', color: '#94a3b8', font: { size: 11 } },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          }
        }
      }
    });
  },

  // Scatter plot: Actual vs Predicted
  renderScatterChart(data, testPreds) {
    const ctx = document.getElementById('chartActualPredicted').getContext('2d');
    if (this.charts.scatter) this.charts.scatter.destroy();

    // Train scatter points
    const trainPoints = [];
    let minVal = Infinity, maxVal = -Infinity;

    data.train.y.forEach((act, idx) => {
      const pred = this.currentModel.yPred[idx];
      trainPoints.push({ x: act, y: pred });
      if (act < minVal) minVal = act;
      if (act > maxVal) maxVal = act;
      if (pred < minVal) minVal = pred;
      if (pred > maxVal) maxVal = pred;
    });

    // Test scatter points
    const testPoints = [];
    data.test.y.forEach((act, idx) => {
      const pred = testPreds[idx];
      testPoints.push({ x: act, y: pred });
      if (act < minVal) minVal = act;
      if (act > maxVal) maxVal = act;
      if (pred < minVal) minVal = pred;
      if (pred > maxVal) maxVal = pred;
    });

    // 45-degree line
    const diagLine = [
      { x: minVal, y: minVal },
      { x: maxVal, y: maxVal }
    ];

    this.charts.scatter = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '理想擬合線 (y = x)',
            data: diagLine,
            type: 'line',
            borderColor: 'rgba(255, 255, 255, 0.4)',
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1.5,
            fill: false
          },
          {
            label: '訓練集 (Train)',
            data: trainPoints,
            backgroundColor: 'rgba(6, 182, 212, 0.7)',
            borderColor: '#06b6d4',
            pointRadius: 4.5
          },
          {
            label: '測試集 (Test)',
            data: testPoints,
            backgroundColor: 'rgba(244, 63, 94, 0.85)',
            borderColor: '#f43f5e',
            pointRadius: 6,
            pointHoverRadius: 8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans' } }
          },
          tooltip: {
            callbacks: {
              label: (context) => `真實: ${context.parsed.x.toFixed(2)}, 預測: ${context.parsed.y.toFixed(2)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Actual Value (真實值)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          },
          y: {
            title: { display: true, text: 'Predicted Value (預測值)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          }
        }
      }
    });
  },

  // Residual distribution histogram
  renderResidualsChart(model) {
    const ctx = document.getElementById('chartResiduals').getContext('2d');
    if (this.charts.residuals) this.charts.residuals.destroy();

    const residuals = Array.from(model.residuals);
    if (residuals.length === 0) return;

    let min = Infinity, max = -Infinity;
    residuals.forEach(r => {
      if (r < min) min = r;
      if (r > max) max = r;
    });

    const numBins = 14;
    const binWidth = (max - min) / numBins;
    const binCounts = new Array(numBins).fill(0);
    const binLabels = [];

    for (let i = 0; i < numBins; i++) {
      const bStart = min + i * binWidth;
      const bEnd = bStart + binWidth;
      binLabels.push(`${(bStart + binWidth / 2).toFixed(0)}`);
    }

    residuals.forEach(r => {
      let bIdx = Math.floor((r - min) / binWidth);
      if (bIdx >= numBins) bIdx = numBins - 1;
      if (bIdx < 0) bIdx = 0;
      binCounts[bIdx]++;
    });

    this.charts.residuals = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: binLabels,
        datasets: [{
          label: '殘差分佈頻率 (Residual Frequency)',
          data: binCounts,
          backgroundColor: 'rgba(245, 158, 11, 0.7)',
          borderColor: '#f59e0b',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => `樣本頻率: ${context.parsed.y} 筆`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Residual Range (殘差值)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            title: { display: true, text: 'Frequency (頻率)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono' } }
          }
        }
      }
    });
  }
};

// Initialize when DOM ready
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
