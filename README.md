# DL4AI Final Project — Time-Series Data and Application to Stock Markets

Deep learning project applying CNN-LSTM models to Nasdaq and Vietnam stock market data.  
Covers price forecasting, trading signal classification, portfolio construction, and REST API deployment.

---

## Project Structure

```
finaldeep/
├── Final-project-DL4AI.ipynb   ← Main submission notebook (all tasks)
├── app.py                       ← Flask REST API
├── Dockerfile                   ← Docker container definition
├── requirements.txt             ← Python dependencies
│
├── data_nasdaq_csv/csv/         ← Nasdaq ticker CSVs (1 file per ticker)
├── data-vn-20230228/
│   ├── companies.csv            ← VN company list + exchange mapping
│   ├── ticker-overview.csv      ← Exchange / sector metadata
│   ├── stock-historical-data/   ← VN OHLCV history (1 file per ticker)
│   ├── financial-ratio/         ← Quarterly financial ratios
│   ├── dividend-history/        ← Dividend records
│   └── industry-analysis/       ← Industry-level data
│
└── *.keras                      ← Saved model weights
    ├── nasdaq_model_task11.keras     Task 1.1  Nasdaq next-day price
    ├── nasdaq_model_task13_k7.keras  Task 1.3  Nasdaq 7-day multi-step
    ├── vn_model_task21.keras         Task 2.1  Vietnam next-day price
    ├── vn_model_task23_k7.keras      Task 2.3  Vietnam 7-day multi-step
    ├── vn_model_task31_buy.keras     Task 3.1  BUY signal classifier
    └── vn_model_task32_sell.keras    Task 3.2  SELL signal classifier
```

---

## Sections & Tasks

### Section 0 — Shared Utilities
Reusable helpers used across all tasks:
- **0.1** Imports (TensorFlow 2.21, NumPy, Pandas, scikit-learn)
- **0.2** Configuration & paths
- **0.3** Data loaders (Nasdaq + Vietnam)
- **0.4** Sliding-window builder + per-window MinMax normalization
- **0.5** Chronological train/val/test split & time-series cross-validation
- **0.6** Technical indicators (SMA, MACD, RSI, Bollinger Bands, ATR, OBV)
- **0.7** CNN-LSTM model builder
- **0.8** Evaluation & visualization helpers

### Section 1 — Nasdaq Price Prediction
| Task | Description |
|------|-------------|
| 1.1 | Multi-feature extension — 6 features (Low, Open, Volume, High, Close, Adj Close) |
| 1.2 | k-th day forecast — predict the price k=1, 3, 7 days ahead |
| 1.3 | k consecutive days — predict the next k=3, 7 closing prices simultaneously |

### Section 2 — Vietnam Price Prediction
| Task | Description |
|------|-------------|
| 2.1 | Single ticker (REE) — next-day price with 12 technical indicators |
| 2.2 | k-th day forecast — k=1, 3, 7 days ahead on Vietnam data |
| 2.3 | k consecutive days — k=3, 7 step multi-output on Vietnam data |

### Section 3 — Trading Signal Classification
- **Signal definition** — BUY if price rises ≥ 3 % in the next 5 days; SELL if it falls ≥ 3 %
- **Task 3.1** — BUY signal classifier (CNN-LSTM, class-weighted)
- **Task 3.2** — SELL signal classifier (CNN-LSTM, class-weighted)
- Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, confusion matrix

### Section 4 — Portfolio Management
| Task | Description |
|------|-------------|
| 4.0 | Company metrics — Sharpe, Calmar, annualised return/volatility for 405 VNINDEX stocks |
| 4.1 | Profitable selection — top-20 by composite score (40% return + 30% Sharpe + 30% Calmar) |
| 4.2 | Risk management — 102 high-risk companies identified (top 25 % by 40% vol + 40% max-DD + 20% D/E) |
| 4.3 | Portfolio composition — risk-taking (top-10 by return) vs. prudent (top-10 Sharpe, safe only); 252-day eval |

### Section 5 — Deployment (Extra Credit, 30 %)
| Task | Description |
|------|-------------|
| 5.1 | REST API — `StockPredictor` class + Flask `app.py` with 4 prediction endpoints |
| 5.2 | Back-test dashboard — signal-driven strategy vs. buy-and-hold on REE test set |
| 5.3 | Docker — `Dockerfile` + `requirements.txt` for portable container deployment |

---

## Model Architecture

All models use the same **CNN-LSTM** backbone:

```
Input (WINDOW_SIZE=30, n_features)
  → Conv1D(64, k=3, ReLU)  → MaxPool(2)
  → Conv1D(128, k=3, ReLU) → MaxPool(2)
  → LSTM(64)
  → Dense(64, ReLU) → Dropout(0.2)
  → Dense(output_size, linear | sigmoid)
```

| Task type | Loss | Output activation |
|-----------|------|-------------------|
| Regression | MSE + MAE | Linear |
| Classification | Binary cross-entropy | Sigmoid |

**Training config:** 50 epochs, batch 64, Adam 1e-3, EarlyStopping (patience=10), ReduceLROnPlateau (patience=5).

---

## Running the Notebook

### Requirements

- Python 3.12
- TensorFlow 2.21.0
- NumPy ≥ 2.0, Pandas ≥ 3.0, scikit-learn, matplotlib

### Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run

Open `Final-project-DL4AI.ipynb` in Jupyter Lab / VS Code and run all cells top-to-bottom.  
Sections 1 and 2 train models from scratch (~5–15 min per section on CPU).  
Sections 3, 4, 5 use pre-trained models saved in `*.keras` files.

---

## REST API (Task 5.1)

### Start the server

```bash
python app.py
# → Listening on http://0.0.0.0:5000
```

### Endpoints

| Method | Endpoint | Input shape | Output |
|--------|----------|-------------|--------|
| `GET`  | `/health` | — | `{"status": "ok", "models_loaded": [...]}` |
| `POST` | `/predict/nasdaq_price` | `{"window": [[30 × 6]]}` | `{"predicted_close": 141.36}` |
| `POST` | `/predict/vn_price` | `{"window": [[30 × 5]]}` | `{"predicted_close": 67500.0}` |
| `POST` | `/predict/vn_signal` | `{"window": [[30 × 17]], "signal_type": "buy"\|"sell"}` | `{"probability": 0.68, "signal": 1}` |

**Feature order**

| Dataset | Features (in order) |
|---------|---------------------|
| Nasdaq | `Low, Open, Volume, High, Close, Adjusted Close` |
| Vietnam price | `Open, High, Low, Close, Volume` |
| Vietnam signal | `Open, High, Low, Close, Volume` + 12 indicators: `SMA_10, SMA_20, MACD, MACD_signal, MACD_hist, RSI, BB_upper, BB_lower, BB_width, BB_pct, ATR, OBV` |

### Example request

```bash
curl -X POST http://localhost:5000/health

curl -X POST http://localhost:5000/predict/vn_signal \
     -H "Content-Type: application/json" \
     -d '{"window": [[...30x17 matrix...]], "signal_type": "buy"}'
```

---

## Docker Deployment (Task 5.3)

```bash
# Build image
docker build -t stock-predictor .

# Run container (detached, port 5000)
docker run -d -p 5000:5000 --name stock-api stock-predictor

# Verify health
curl http://localhost:5000/health

# Stop and remove
docker stop stock-api && docker rm stock-api
```

---

## Key Results Summary

| Task | Metric | Value |
|------|--------|-------|
| 1.1 Nasdaq AAPL next-day | Test MAPE | ~1–2 % |
| 3.1 BUY signal | AUC-ROC | 0.525 |
| 3.2 SELL signal | AUC-ROC | 0.461 |
| 4.3 Risk-taking portfolio | Eval-period return | -30.3 % |
| 4.3 Prudent portfolio | Eval-period return | -30.3 % |
| 4.3 Market benchmark | Eval-period return | -35.9 % |
| 5.2 Signal strategy vs B&H | Return (+37.7 % vs +147 %) | REE 2019–2023 |

---

## Submission Checklist

- [ ] `<StudentID>-project-notebook.ipynb` — rename `Final-project-DL4AI.ipynb`
- [ ] `<StudentID>-project-report.pdf` — write and export from Section 6
- [ ] Compress into `DL4AI-<StudentID>-project.zip`
