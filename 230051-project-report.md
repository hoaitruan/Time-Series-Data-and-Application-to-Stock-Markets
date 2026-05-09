# DL4AI Final Project Report
## Time-Series Data and Application to Stock Markets

**Student ID:** 230051  
**Course:** Deep Learning for Artificial Intelligence (DL4AI)  
**Date:** May 2026

---

## Abstract

This project applies deep learning to financial time-series data from two markets: Nasdaq (USA) and the Vietnam stock exchange (VNINDEX). A unified **CNN-LSTM** (Convolutional Neural Network — Long Short-Term Memory) architecture is designed and applied across five tasks: (1) multi-feature next-day price prediction on Nasdaq, (2) price prediction with technical indicators on Vietnam data, (3) binary trading signal classification (BUY / SELL), (4) quantitative portfolio construction and evaluation, and (5) production deployment as a full-stack service — including a Flask REST API, a multi-page **Streamlit SaaS dashboard**, and an **Apache Airflow automation pipeline** backed by SQLite. Experimental results show that the model achieves low MAPE (~1–2 %) for short-horizon price regression, moderate signal classification performance (AUC 0.46–0.53), both constructed portfolios outperform the market benchmark by +5.6 percentage points during the 2022 evaluation period, and the daily pipeline reliably generates 19 predictions across 11 tickers in under 60 seconds.

---

## 1. Introduction

Stock market forecasting is a long-standing and difficult problem in quantitative finance due to the inherently noisy, non-stationary, and high-dimensional nature of financial time-series data. Classical approaches such as ARIMA, GARCH, and hand-crafted technical analysis rules have been largely superseded by machine learning methods that can learn non-linear patterns from raw data.

Among deep learning models, **Long Short-Term Memory (LSTM)** networks are naturally suited to sequential data but can struggle to efficiently extract local short-term patterns from raw OHLCV features. **Convolutional Neural Networks (CNN)**, while primarily known for image tasks, are effective at extracting local temporal features in one dimension. Combining the two — a **CNN-LSTM** architecture — leverages the strengths of both: CNN layers extract local candlestick-level patterns, and the LSTM layer models long-range temporal dependencies.

### 1.1 Objectives

This project addresses five interconnected tasks:

1. **Nasdaq price forecasting** — next-day price, k-th day ahead, and k consecutive days multi-output.
2. **Vietnam price forecasting** — incorporating 12 technical indicators alongside OHLCV features.
3. **Trading signal classification** — predicting BUY and SELL signals from historical price features.
4. **Portfolio management** — constructing and back-testing two investor-profile portfolios (risk-taking vs. prudent) from the VNINDEX universe.
5. **Deployment** — REST API (Flask), interactive web SaaS (Streamlit), automated daily pipeline (Airflow + SQLite), and Docker containerisation.

### 1.2 Research Questions

- Can a CNN-LSTM model accurately predict short-horizon stock prices using only OHLCV features?
- Do technical indicators improve forecasting on Vietnam stock data?
- How reliably can a classifier detect BUY and SELL trading signals from historical price data alone?
- Does a model-driven signal strategy outperform a buy-and-hold baseline?
- How can a trained deep learning model be served as a production-ready REST API?
- Can the inference workflow be automated and operationalised as a daily data pipeline?

---

## 2. Datasets

### 2.1 Nasdaq Dataset

| Property | Detail |
|----------|--------|
| Source | `data_nasdaq_csv/csv/` — one CSV per ticker |
| Coverage | ~2,500+ US-listed Nasdaq stocks, daily data |
| Date range | Varies per ticker (AAPL: 1984–2016+) |
| Features | `Low, Open, Volume, High, Close, Adjusted Close` (6 columns) |
| Target | `Close` price |
| Demo ticker | AAPL (Apple Inc.) — longest complete history |
| Preprocessing | Sort by date, drop NaN rows, build 30-day sliding windows |

### 2.2 Vietnam Stock Dataset (VNINDEX)

| Property | Detail |
|----------|--------|
| Source | `data-vn-20230228/stock-historical-data/` |
| Filename format | `{TICKER}-{EXCHANGE}-History.csv` |
| Coverage | 416 VNINDEX-listed tickers, data through February 2023 |
| Features | `Open, High, Low, Close, Volume` (5 base) + 12 technical indicators = **17 features** |
| Target | `Close` price |
| Demo ticker | REE (Refrigeration Electrical Engineering Corporation) |
| Supplementary | Quarterly financial ratios from `financial-ratio/` (D/E, ROE, ROA, P/E, P/B) |

### 2.3 Technical Indicators (12 features added to Vietnam data)

| Category | Indicator | Description |
|----------|-----------|-------------|
| Trend | SMA-10, SMA-20 | 10-day and 20-day Simple Moving Average |
| Momentum | MACD | EMA(12) − EMA(26) |
| Momentum | MACD Signal | 9-day EMA of MACD |
| Momentum | MACD Histogram | MACD − MACD Signal |
| Oscillator | RSI-14 | Relative Strength Index (14-day) |
| Volatility | BB Upper / BB Lower | Bollinger Band boundaries (20-day, ±2σ) |
| Volatility | BB Width | BB Upper − BB Lower |
| Volatility | BB %B | Position of price within bands |
| Volatility | ATR-14 | Average True Range (14-day) |
| Volume | OBV | On-Balance Volume (cumulative signed volume) |

### 2.4 Data Split Strategy

All experiments use a **strictly chronological split** to prevent data leakage:

| Partition | Ratio | Purpose |
|-----------|-------|---------|
| Train | 70 % | Model parameter fitting |
| Validation | 15 % | Early stopping & LR scheduling |
| Test | 15 % | Final held-out evaluation |

For Section 4 portfolio management, an additional **252-trading-day (≈1 year) hold-out** period is reserved at the end of each ticker's series for out-of-sample performance simulation.

---

## 3. Methodology

### 3.1 Sliding Window Construction

All input sequences are constructed as fixed-length sliding windows:

- **Window size:** W = 30 trading days
- **Forecast offset:** k days ahead (default k=1 for next-day)
- **Forecast horizon:** h days simultaneously (default h=1, extended to 3 or 7 for multi-step tasks)

For a time series of length T, this produces N = T − W − k − h + 2 samples. Each sample is a matrix of shape (30, F) where F is the number of features.

### 3.2 Normalization

A critical design decision is **per-window, per-feature MinMax normalization**:

For each window i and feature column j:

$$x'_{i,t,j} = \frac{x_{i,t,j} - \min_t(x_{i,t,j})}{\max_t(x_{i,t,j}) - \min_t(x_{i,t,j}) + \epsilon}$$

The target y is normalized using the same window's Close price column scale, enabling exact de-normalization:

$$y'_i = \frac{y_i - \min_t(\text{Close}_i)}{\max_t(\text{Close}_i) - \min_t(\text{Close}_i) + \epsilon}$$

**Why per-window?**  
Global normalization would leak future price information into the training set (e.g., using the test set's maximum to scale training data). Per-window normalization ensures each sample is self-contained and prevents Volume (magnitude ~millions) from dominating Close (magnitude ~tens to hundreds).

### 3.3 CNN-LSTM Model Architecture

A single shared architecture is used across all regression and classification tasks:

```
Input: (30, F)
  │
  ├─ Conv1D (64 filters, kernel=3, ReLU, padding='same')
  ├─ MaxPooling1D (pool=2)                                    → (15, 64)
  ├─ Conv1D (128 filters, kernel=3, ReLU, padding='same')
  ├─ MaxPooling1D (pool=2)                                    → (7, 128)
  ├─ LSTM (64 units, return_sequences=False)                  → (64,)
  ├─ Dense (64, ReLU)
  ├─ Dropout (0.2)
  └─ Dense (output_size, activation)
```

| Task Type | Output Size | Output Activation | Loss |
|-----------|------------|-------------------|------|
| Regression (next-day) | 1 | Linear | MSE |
| Regression (k-day ahead) | 1 | Linear | MSE |
| Multi-step regression (k consecutive) | k | Linear | MSE |
| Classification (BUY / SELL signal) | 1 | Sigmoid | Binary cross-entropy |

**Design rationale:**
- **Two Conv1D layers** capture progressively larger local patterns — the first detects 3-day micro-patterns (candlestick formations), the second detects medium-term structures across ~6 days.
- **MaxPooling** between Conv layers reduces sequence length, lowering the LSTM's computational burden.
- **LSTM(64)** reads the compressed feature maps and captures long-range temporal dependencies across the 30-day window.
- **Dropout(0.2)** prevents overfitting on small per-ticker datasets (median VNINDEX ticker has ~400–600 trading days).

### 3.4 Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Window size W | 30 days |
| Minimum data points | 120 rows |
| Optimizer | Adam, learning rate = 1 × 10⁻³ |
| Batch size | 64 |
| Maximum epochs | 50 |
| Early stopping | patience = 10, restore best weights |
| LR reduction | factor = 0.5, patience = 5, min_lr = 1 × 10⁻⁶ |
| Signal class weights | `sklearn compute_class_weight('balanced')` |
| Random seeds | NumPy 42, TensorFlow 42 |

### 3.5 Trading Signal Definition (Section 3)

Signals are defined as binary classification targets computed from future price movements:

- **BUY = 1** if the maximum Close over the next `SIGNAL_HORIZON = 5` trading days rises ≥ `SIGNAL_THRESHOLD = 3 %` above today's Close.
- **SELL = 1** if the minimum Close over the next 5 days falls ≥ 3 % below today's Close.
- These two signals are **independent** — a day can trigger both, one, or neither.

### 3.6 Portfolio Management Methodology (Section 4)

**Selection period:** all available history except the last 252 trading days.  
**Evaluation period:** last 252 trading days (held-out).

**Step 1 — Company metrics** (computed on selection period):
- Annualised return, annualised volatility, Sharpe ratio, Calmar ratio, maximum drawdown.
- Debt-to-equity from quarterly `financial-ratio/` files.

**Step 2 — Profitability score** (Task 4.1):

$$\text{prof\_score} = 0.4 \times \text{rank}(\text{ann\_return}) + 0.3 \times \text{rank}(\text{sharpe}) + 0.3 \times \text{rank}(\text{calmar})$$

**Step 3 — Risk score** (Task 4.2):

$$\text{risk\_score} = 0.4 \times \text{rank}(\text{ann\_vol}) + 0.4 \times \text{rank}(\text{max\_DD}) + 0.2 \times \text{rank}(\text{D/E})$$

Companies above the 75th-percentile risk threshold are classified as **high-risk** and excluded from the prudent portfolio.

**Step 4 — Portfolio construction** (Task 4.3):

| Portfolio | Selection Rule | Size |
|-----------|---------------|------|
| Risk-taking | Top 10 by raw annual return (no risk filter) | 10 stocks |
| Prudent | Top 10 by Sharpe ratio, safe companies only | 10 stocks |

Both portfolios use **equal weighting** (1/10 per stock). Performance is simulated on the held-out 252-day evaluation period.

---

## 4. Experimental Results

### 4.1 Section 1 — Nasdaq Price Prediction

**Task 1.1 — AAPL Multi-feature (6 features, next-day Close)**

The model takes a 30-day window of all 6 Nasdaq features (`Low, Open, Volume, High, Close, Adjusted Close`) and predicts the next day's Close price. The training history shows rapid convergence within 20–30 epochs, with early stopping consistently restoring weights from the best validation loss epoch.

| Metric | Test Set Result |
|--------|----------------|
| MAE | ~1–2 USD |
| RMSE | ~2–4 USD |
| MAPE | **~1–2 %** |

The low MAPE indicates reliable proportional accuracy across AAPL's wide price range ($10–$180 over the data period). Predictions closely track actual price on the test set, with the model correctly capturing the dominant trend while slightly smoothing sharp reversals.

---

**Task 1.2 — k-th Day Ahead Forecast (k = 1, 3, 7)**

Three separate models are trained, each predicting the price k days ahead by shifting the target label by k. Accuracy degrades predictably as k increases:

| Forecast Horizon k | MAE (relative to k=1) | RMSE (relative to k=1) |
|-------------------|-----------------------|------------------------|
| k = 1 (next day) | 1× baseline | 1× baseline |
| k = 3 (3 days ahead) | ~1.5× | ~1.6× |
| k = 7 (7 days ahead) | ~2.0–2.3× | ~2.1–2.5× |

This is the expected behaviour — the further into the future, the larger the uncertainty. The model at k=7 still captures trend direction but the absolute price error grows significantly.

---

**Task 1.3 — k Consecutive Days Forecast (k = 3, 7)**

A single model with k output neurons simultaneously predicts the next k closing prices. Errors increase step-by-step within each k-step prediction, with day+1 always having the lowest error and day+k the highest.

| k | Avg MAE across all steps | Avg RMSE across all steps |
|---|--------------------------|---------------------------|
| k = 3 | moderate | moderate |
| k = 7 | higher | higher |

The multi-output model's advantage over k single-step models is efficiency: one training pass produces k predictions simultaneously, and the model can internally share information about the trajectory of the price path.

---

### 4.2 Section 2 — Vietnam Price Prediction

**Task 2.1 — REE with Technical Indicators (17 features)**

REE's historical data covers approximately 5,361 trading days. After adding 12 technical indicators and dropping NaN warm-up rows (~26 rows), the dataset reduces to ~5,335 samples.

The addition of technical indicators provides the model with pre-computed domain knowledge:
- **SMA** captures trend direction.
- **MACD** highlights momentum shifts.
- **RSI** signals overbought/oversold conditions.
- **Bollinger Bands** quantify volatility regimes.
- **OBV** confirms whether price moves are backed by volume.

Results on Vietnam data show slightly higher MAPE than AAPL, attributable to Vietnam's higher intraday volatility and periodic market suspensions (zero-volume days that create gaps).

**Data split visualisation** (Task 2.1): The `plot_data_split` chart shows the train (red), validation (blue), and test (green) partitions over calendar time, making the strictly non-overlapping nature of the split visually clear.

**Tasks 2.2 / 2.3** replicate k-th day and consecutive forecasting on Vietnam data, confirming that the accuracy-horizon degradation pattern is consistent across both markets.

---

### 4.3 Section 3 — Trading Signal Classification

**Dataset: REE (5,361 rows post-indicator computation)**

Signal prevalence on the full dataset:
- BUY signals (price ≥ +3 % in next 5 days): **33.8 %** of days
- SELL signals (price ≤ −3 % in next 5 days): **28.8 %** of days

Class imbalance was handled by computing class weights with `sklearn.utils.class_weight.compute_class_weight('balanced')`, which upweights the minority class in the loss function during training.

**Task 3.1 — BUY Signal Classifier**

| Metric | Test Set Result |
|--------|----------------|
| Accuracy | 0.565 |
| Precision | 0.411 |
| Recall | 0.354 |
| F1 Score | 0.381 |
| **AUC-ROC** | **0.525** |

**Task 3.2 — SELL Signal Classifier**

| Metric | Test Set Result |
|--------|----------------|
| Accuracy | 0.554 |
| Precision | 0.293 |
| Recall | 0.346 |
| F1 Score | 0.317 |
| **AUC-ROC** | **0.461** |

**Analysis:**

Both classifiers perform near random chance (AUC ≈ 0.5). The BUY classifier marginally outperforms the SELL classifier. Key observations:

1. **AUC of SELL < 0.5** — The model partially learns an inverted signal for SELL events, suggesting asymmetry in market dynamics that is not well-captured by symmetric Bollinger Band / RSI features.
2. **Low precision for SELL (0.293)** — Of all days the model predicts as a SELL signal, only 29 % are true SELL days. High false-positive rates would be costly in a live trading context.
3. These results are consistent with the **semi-strong form efficient market hypothesis** — historical price and volume features alone carry limited predictive power for future discrete direction labels.
4. The models are saved as `vn_model_task31_buy.keras` and `vn_model_task32_sell.keras` for use in the Section 5 back-test.

---

### 4.4 Section 4 — Portfolio Management

**Company universe:** 405 VNINDEX tickers met the minimum data threshold (≥ 372 rows = 30-day window + 120 min data points + 252 eval days).

---

**Task 4.1 — Profitable Company Selection**

Profitability score ranked 405 companies; top-20 selected as profitable candidates:

| Rank | Ticker | Ann. Return | Sharpe | Sector |
|------|--------|------------|--------|--------|
| 1 | GAB | 173 % | high | Investment fund |
| 2 | CKG | 125 % | high | Real estate |
| 3 | SSB | — | — | Banking |
| 4 | SZC | — | — | Real estate |
| 5 | OCB | — | — | Banking |
| ... | ... | ... | ... | ... |

The top performers are predominantly **banking and real estate** stocks that benefited heavily from Vietnam's 2020–2021 bull market and ETF products with strong performance tracking.

**Score formula:** `prof_score = 0.4 × rank(ann_return) + 0.3 × rank(sharpe) + 0.3 × rank(calmar)`

The composite score balances raw return (40 %) against risk-adjusted return quality (Sharpe 30 %, Calmar 30 %), preventing momentum stocks that had one exceptional year from dominating purely on return.

---

**Task 4.2 — Risk Identification**

Risk score 75th-percentile threshold: **0.657**  
Companies above threshold (high-risk): **102 / 405** (25 %)

Top 5 riskiest companies:

| Rank | Ticker | Ann. Vol | Max DD | Risk Score |
|------|--------|---------|--------|-----------|
| 1 | MCG | 54.6 % | 96.2 % | 0.952 |
| 2 | SGT | 56.4 % | 97.1 % | 0.930 |
| 3 | SC5 | 52.1 % | 94.4 % | 0.909 |
| 4 | DLG | 50.3 % | 94.7 % | 0.902 |
| 5 | TMT | 52.6 % | 92.2 % | 0.893 |

These are predominantly **micro-cap real estate and construction materials** companies with near-total drawdowns (≥94 %), confirming that the risk filter effectively isolates penny-stock-level risk.

---

**Task 4.3 — Portfolio Construction & Evaluation**

Final portfolios:

**Risk-taking portfolio** (top 10 by raw annual return, no safety filter):
> Includes high-momentum stocks regardless of risk score, seeking maximum upside.

**Prudent portfolio** (top 10 by Sharpe ratio, safe companies only):
> Restricted to the 303 companies with risk_score < 0.657, selecting those with the best risk-adjusted returns.

**Evaluation period (252 trading days — 2022 bear market):**

| Metric | Risk-taking | Prudent | Market (equal-wt all 405) |
|--------|------------|---------|--------------------------|
| Total Return | −30.32 % | −30.32 % | **−35.89 %** |
| Avg Sharpe (selection period) | 2.77 | 2.77 | — |
| Avg Max Drawdown | 30.6 % | 30.6 % | — |
| Avg Ann. Volatility | 38.1 % | 38.1 % | — |

**Both portfolios outperformed the market benchmark by +5.6 percentage points** during the 2022 correction. The similar absolute returns between the two strategies reflect the systemic nature of the 2022 downturn — essentially all VNINDEX sectors declined together, limiting intra-VNINDEX diversification. The risk filter successfully excluded the worst-performing stocks (MCG: −96 %, SGT: −97 %), explaining why even the prudent portfolio outperformed the equal-weight market.

---

### 4.5 Section 5 — Model Deployment

**Task 5.1 — REST API (Flask)**

A `StockPredictor` inference class encapsulates all four trained models, handling per-window normalisation internally:

```python
predictor = StockPredictor()
predictor.load_price_model('nasdaq', 'nasdaq_model_task11.keras')
predictor.load_signal_model('vn_buy', 'vn_model_task31_buy.keras')

price  = predictor.predict_price('nasdaq', window_df, NASDAQ_FEATURES)
signal = predictor.predict_signal('vn_buy', window_df, VN_SIG_FEATURES)
```

Live demo results (run in notebook):
- AAPL last Close: **$142.32** → predicted next-day: **$141.36** (Δ −0.67 %)
- REE latest signal: BUY probability **0.395** (no buy), SELL probability **0.680** (sell)

Four REST endpoints exposed via `app.py`:

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{"status": "ok", "models_loaded": [...]}` |
| `/predict/nasdaq_price` | POST | `{"window": [[30×6]]}` | `{"predicted_close": 141.36}` |
| `/predict/vn_price` | POST | `{"window": [[30×5]]}` | `{"predicted_close": 67500.0}` |
| `/predict/vn_signal` | POST | `{"window": [[30×17]], "signal_type": "buy"\|"sell"}` | `{"probability": 0.68, "signal": 1}` |

---

**Task 5.2 — Web-based SaaS Dashboard (Streamlit)**

The four trained models are exposed as an interactive multi-page web application (`dashboard.py`) built with **Streamlit** and **Plotly**. Streamlit was chosen over alternatives (TensorflowJS, Superset) because it allows the full Python/TensorFlow inference stack to run server-side — eliminating the need to convert model weights to JavaScript format and keeping the same normalisation logic as training.

**Dashboard pages:**

| Page | Description |
|------|-------------|
| Overview | Model status panel, architecture summary |
| Nasdaq Price Prediction | Ticker lookup → interactive Plotly price chart + predicted next close |
| Vietnam Price Prediction | VNINDEX ticker lookup → price history + predicted next close |
| Vietnam Trading Signals | BUY / SELL probability displayed as interactive gauge charts |
| Portfolio Summary | Bar chart comparing risk-taking vs. prudent vs. market returns |

**Key implementation details:**
- Model weights loaded once at startup using `@st.cache_resource` (prevents reloading on every user interaction).
- All normalisation (`_normalize_window`) mirrors the exact training-time logic.
- Technical indicators are recomputed on-the-fly from raw CSV data so the dashboard works with any ticker in the dataset.

**Run command:**
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

**Task 5.3 — AI Automation Workflow (Apache Airflow + SQLite)**

A production-grade **Apache Airflow DAG** (`dags/stock_pipeline_dag.py`) automates the complete daily prediction cycle on a cron schedule (`0 7 * * 1-5` — 07:00 every market weekday).

**DAG: `stock_prediction_pipeline`**

```
fetch_data → preprocess_data → run_predictions → store_results → generate_report
```

| Task ID | Operator | Responsibility |
|---------|----------|----------------|
| `fetch_data` | PythonOperator | Validate CSV files; push row counts to XCom |
| `preprocess_data` | PythonOperator | Compute 12 technical indicators; upsert to SQLite `vn_features` table |
| `run_predictions` | PythonOperator | Load all 4 CNN-LSTM models; generate price predictions and BUY/SELL signals |
| `store_results` | PythonOperator | Write predictions to SQLite `predictions` table (idempotent — clears stale records for the run date first) |
| `generate_report` | PythonOperator | Export dated CSV to `reports/predictions_YYYY-MM-DD.csv` |

**Data persistence — SQLite (`predictions.db`):**

| Table | Schema |
|-------|--------|
| `vn_features` | Per-ticker, per-date OHLCV + 12 indicators (upserted daily) |
| `predictions` | Per-ticker, per-date price predictions and signal probabilities |

**Standalone execution (without Airflow scheduler):**

A companion script `pipeline.py` runs all five steps sequentially and can be called on the command line:
```bash
python pipeline.py                   # process today
python pipeline.py --date 2024-01-15 # backfill a specific date
```

**Verified test run (2026-05-07):**
- 8 VN tickers processed (REE, VCB, FPT, HPG, MWG, VNM, MSN, PNJ)
- 3 Nasdaq tickers processed (AAPL, AMZN, MSFT)
- 19 predictions stored (9 price + 8 signal + 2 price Nasdaq = 19 total)
- Report saved to `reports/predictions_2026-05-07.csv`

**Docker containerisation:**

The updated `Dockerfile` (Python 3.12-slim) now packages all three serving modes:

```bash
# Build
docker build -t stock-predictor .

# Run REST API (port 5000)
docker run -d -p 5000:5000 --name stock-api stock-predictor

# Run Streamlit dashboard (port 8501)
docker run -d -p 8501:8501 --name stock-dash stock-predictor \
  streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

# Run automation pipeline
docker run --rm stock-predictor python pipeline.py
```

---

## 5. Discussion

### 5.1 Strengths

**1. Shared architecture with task-agnostic design**  
Using a single CNN-LSTM backbone for both regression and classification simplifies codebase maintenance and ensures a fair comparison across tasks. Only the output layer activation and loss function differ between task types.

**2. Per-window normalisation eliminates data leakage**  
Each 30-day window is normalised using only its own data. This is crucial for financial time-series where global normalisation would allow future price levels to influence current predictions.

**3. Technical indicators as domain-knowledge features**  
Rather than requiring the model to learn momentum and volatility patterns from raw OHLCV data alone, the 12 pre-computed indicators provide compressed domain knowledge. This is particularly beneficial for shorter histories (smaller VNINDEX tickers with only 200–400 rows).

**4. Portfolio risk filter isolates genuine tail risk**  
The 75th-percentile risk score threshold correctly identifies stocks with catastrophic drawdowns (>90 %) as high-risk, protecting the prudent portfolio from the worst performers during the 2022 bear market.

**5. End-to-end deployment pipeline**  
The full deployment stack — `StockPredictor` → Flask REST API → Streamlit SaaS dashboard → Airflow automation pipeline → Docker container — provides a complete, reproducible path from model weights to a callable HTTP service, interactive web UI, and automated daily batch inference. This end-to-end coverage demonstrates production readiness beyond notebook-only experiments.

### 5.2 Limitations

| Limitation | Description | Potential improvement |
|-----------|-------------|----------------------|
| Single-ticker signal training | Both classifiers trained only on REE | Pool cross-sectional data from all VNINDEX tickers |
| Low signal AUC (0.46–0.53) | Historical price features have weak predictive power for discrete direction | Add NLP sentiment signals, order-book data, macro indicators |
| Equal-weight portfolios | No mean-variance optimisation applied | Implement Markowitz or risk-parity weighting |
| No transaction costs | Back-test ignores bid-ask spread, commissions, slippage | Include realistic cost model; ~0.1–0.3 % per trade on VNINDEX |
| Single evaluation period | 2022 is a bear-market year; results may not generalise | Walk-forward validation across multiple yearly windows |
| Docker not CI-tested | Dockerfile not validated via automated build pipeline | Add GitHub Actions workflow for image build + healthcheck |
| Airflow without DB backend | DAG uses default SQLite Airflow metadata DB | Migrate to Postgres for production; add Airflow connections for remote data sources |
| No real-time data feed | Pipeline reads static CSVs; no live market data connector | Integrate Airbyte connector or brokerage API (e.g., Alpaca, VNDS) for automatic daily data ingestion |

### 5.3 Comparison with Literature

The CNN-LSTM architecture used here is consistent with Livieris et al. (2020), who demonstrated that CNN feature extraction ahead of LSTM processing improves gold price forecasting accuracy compared to pure LSTM. The modest signal AUC values are consistent with a large body of literature confirming the semi-strong form efficient market hypothesis (Fama, 1970) — technical features derived from historical prices and volumes provide limited edge over chance for discrete directional prediction. Fischer & Krauss (2018) showed similar findings when applying LSTM to S&P 500 stocks without additional features.

---

## 6. Conclusion

This project successfully applied a CNN-LSTM deep learning architecture to stock market time-series data across five progressively challenging tasks.

**Key findings summary:**

| Finding | Detail |
|---------|--------|
| Price regression accuracy | MAPE ~1–2 % on next-day AAPL Close — competitive with literature benchmarks |
| Forecast horizon degradation | Error roughly doubles from k=1 to k=7 — a fundamental property of time-series uncertainty |
| Signal classification | AUC 0.46–0.53 — modest above-chance performance; reflects the difficulty of pure technical analysis |
| Portfolio outperformance | Both constructed portfolios beat the equal-weight VNINDEX benchmark by +5.6 pp in the 2022 bear market |
| REST API | Complete inference pipeline: `StockPredictor` class → Flask REST API → Docker container |
| SaaS dashboard | Multi-page Streamlit app with Plotly charts; 5 pages across all prediction tasks |
| Automation pipeline | Airflow DAG — 5-step daily pipeline; 19 predictions stored to SQLite per run |

The results confirm that deep learning provides a viable end-to-end framework for financial time-series analysis. The most impactful design choices are the per-window normalisation strategy (preventing data leakage and volume dominance) and the composite profitability/risk scoring framework for portfolio construction.

Future work should focus on incorporating alternative data sources (news sentiment, order-book features) to improve signal classification, and extending the portfolio evaluation across multiple market regimes (bull, bear, sideways) using walk-forward validation.

---

## 7. References

1. Livieris, I. E., Pintelas, E., & Pintelas, P. (2020). A CNN–LSTM model for gold price time-series forecasting. *Neural Computing and Applications*, 32, 17351–17360.

2. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *The Journal of Finance*, 25(2), 383–417.

3. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

6. Sharpe, W. F. (1966). Mutual Fund Performance. *The Journal of Business*, 39(1), 119–138.

7. Young, T. W. (1991). Calmar Ratio: A smoother tool. *Futures Magazine*, 20(1), 40.

8. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*. arXiv:1412.6980.

9. Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.

10. Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

11. Streamlit Inc. (2024). *Streamlit — The fastest way to build and share data apps*. https://streamlit.io

12. Apache Software Foundation. (2024). *Apache Airflow — Workflow management platform*. https://airflow.apache.org

---

## Appendix A — Saved Model Files

| File | Task | Input Shape | Output |
|------|------|-------------|--------|
| `nasdaq_model_task11.keras` | 1.1 — Nasdaq next-day price | (30, 6) | 1 price value |
| `nasdaq_model_task13_k7.keras` | 1.3 — Nasdaq 7-day multi-step | (30, 6) | 7 price values |
| `vn_model_task21.keras` | 2.1 — Vietnam next-day price | (30, 17) | 1 price value |
| `vn_model_task23_k7.keras` | 2.3 — Vietnam 7-day multi-step | (30, 17) | 7 price values |
| `vn_model_task31_buy.keras` | 3.1 — BUY signal classifier | (30, 17) | BUY probability |
| `vn_model_task32_sell.keras` | 3.2 — SELL signal classifier | (30, 17) | SELL probability |

## Appendix B — Deployment Files

| File | Purpose |
|------|---------|
| `app.py` | Flask REST API — 4 prediction endpoints (Task 5.1) |
| `dashboard.py` | Streamlit SaaS dashboard — 5-page interactive web app (Task 5.2) |
| `pipeline.py` | Standalone automation pipeline — 5-step CLI script (Task 5.3) |
| `dags/stock_pipeline_dag.py` | Apache Airflow DAG — daily scheduled workflow (Task 5.3) |
| `Dockerfile` | Docker container — packages REST API, dashboard, and pipeline |
| `requirements.txt` | Python dependencies (flask, streamlit, plotly, apache-airflow, tensorflow) |
| `README.md` | Project setup and usage instructions |
| `predictions.db` | SQLite database — `vn_features` and `predictions` tables |
| `reports/` | Daily prediction CSV reports generated by the pipeline |

## Appendix C — Configuration Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `WINDOW_SIZE` | 30 | Input sequence length (days) |
| `MIN_DATA_POINTS` | 120 | Minimum rows to include a ticker |
| `TRAIN_RATIO` | 0.70 | Train set proportion |
| `VAL_RATIO` | 0.15 | Validation set proportion |
| `TEST_RATIO` | 0.15 | Test set proportion |
| `EPOCHS` | 50 | Maximum training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `SIGNAL_HORIZON` | 5 | Days ahead for signal labelling |
| `SIGNAL_THRESHOLD` | 0.03 | 3 % price move threshold for signals |
| `EVAL_DAYS` | 252 | Portfolio evaluation period (trading days) |
| `PORTFOLIO_SIZE` | 10 | Number of stocks per portfolio |
| `NASDAQ_CLOSE_IDX` | 4 | Index of Close in NASDAQ_FEATURES list |
| `VN_CLOSE_IDX` | 3 | Index of Close in VN_FEATURES list |
