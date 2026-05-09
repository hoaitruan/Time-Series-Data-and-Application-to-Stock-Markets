"""
Task 5.3 – AI Automation Workflow (Apache Airflow DAG)
======================================================
This DAG automates the full stock-prediction pipeline on a daily schedule:

  1. fetch_data        – validate that the latest CSV data is present
  2. preprocess_data   – compute technical indicators & write to SQLite DB
  3. run_predictions   – run all CNN-LSTM models on the last window
  4. store_results     – write predictions + signals back to SQLite
  5. generate_report   – create a daily summary CSV report

Run the Airflow scheduler with:
    export AIRFLOW_HOME=~/airflow
    airflow db init
    airflow scheduler &
    airflow webserver &
Then trigger this DAG from the Airflow UI at http://localhost:8080.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ─────────────────────────────────────────────────────────────────────────────
# Paths (resolve relative to this file so the DAG is portable)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH    = os.path.join(BASE_DIR, "predictions.db")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

VN_DATA_DIR     = os.path.join(BASE_DIR, "data-vn-20230228", "stock-historical-data")
NASDAQ_DATA_DIR = os.path.join(BASE_DIR, "data_nasdaq_csv", "csv")

# Tickers to process daily
VN_WATCH_LIST     = ["REE", "VCB", "FPT", "HPG", "MWG", "VNM", "MSN", "PNJ"]
NASDAQ_WATCH_LIST = ["AAPL", "AMZN", "MSFT", "GOOGL", "META"]

WINDOW_SIZE = 30

VN_FEATURES      = ["Open", "High", "Low", "Close", "Volume"]
NASDAQ_FEATURES  = ["Low", "Open", "Volume", "High", "Close", "Adjusted Close"]
VN_INDICATOR_COLS = [
    "SMA_10", "SMA_30", "EMA_10", "EMA_30",
    "MACD", "MACD_Signal", "RSI",
    "BB_upper", "BB_lower", "BB_width",
    "Return_1d", "Momentum_5d",
]
VN_SIG_FEATURES = VN_FEATURES + VN_INDICATOR_COLS

# ─────────────────────────────────────────────────────────────────────────────
# Default DAG arguments
# ─────────────────────────────────────────────────────────────────────────────
default_args = {
    "owner":            "dl4ai",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 – Fetch / validate data
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(**context):
    """Verify that required CSV files exist and log their row counts to XCom."""
    import pandas as pd

    summary = {}

    for ticker in VN_WATCH_LIST:
        path = os.path.join(VN_DATA_DIR, f"{ticker}-VNINDEX-History.csv")
        if os.path.exists(path):
            n = len(pd.read_csv(path))
            summary[f"vn_{ticker}"] = n
        else:
            summary[f"vn_{ticker}"] = 0
            print(f"[WARN] VN ticker {ticker} CSV not found at {path}")

    for ticker in NASDAQ_WATCH_LIST:
        path = os.path.join(NASDAQ_DATA_DIR, f"{ticker}.csv")
        if os.path.exists(path):
            n = len(pd.read_csv(path))
            summary[f"nasdaq_{ticker}"] = n
        else:
            summary[f"nasdaq_{ticker}"] = 0
            print(f"[WARN] Nasdaq ticker {ticker} CSV not found at {path}")

    context["ti"].xcom_push(key="data_summary", value=summary)
    print(f"[fetch_data] Data summary: {summary}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 – Pre-process: compute indicators & store in SQLite
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(**context):
    """Compute technical indicators for VN tickers and upsert into SQLite."""
    import pandas as pd

    def _add_indicators(df):
        c = df["Close"].copy()
        df["SMA_10"]      = c.rolling(10).mean()
        df["SMA_30"]      = c.rolling(30).mean()
        df["EMA_10"]      = c.ewm(span=10, adjust=False).mean()
        df["EMA_30"]      = c.ewm(span=30, adjust=False).mean()
        ema12             = c.ewm(span=12, adjust=False).mean()
        ema26             = c.ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        delta             = c.diff()
        gain              = delta.clip(lower=0).rolling(14).mean()
        loss              = (-delta.clip(upper=0)).rolling(14).mean()
        df["RSI"]         = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
        ma                = c.rolling(20).mean()
        std               = c.rolling(20).std()
        df["BB_upper"]    = ma + 2 * std
        df["BB_lower"]    = ma - 2 * std
        df["BB_width"]    = df["BB_upper"] - df["BB_lower"]
        df["Return_1d"]   = c.pct_change(1)
        df["Momentum_5d"] = c.pct_change(5)
        return df

    conn = sqlite3.connect(DB_PATH)
    # Create table if needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vn_features (
            ticker      TEXT,
            date        TEXT,
            open        REAL, high REAL, low REAL, close REAL, volume REAL,
            sma10 REAL, sma30 REAL, ema10 REAL, ema30 REAL,
            macd  REAL, macd_signal REAL, rsi REAL,
            bb_upper REAL, bb_lower REAL, bb_width REAL,
            return_1d REAL, momentum_5d REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()

    records_written = 0
    for ticker in VN_WATCH_LIST:
        path = os.path.join(VN_DATA_DIR, f"{ticker}-VNINDEX-History.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["TradingDate"]).sort_values("TradingDate")
        for col in VN_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = _add_indicators(df).dropna()
        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO vn_features VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ticker,
                str(row.get("TradingDate", "")),
                row["Open"], row["High"], row["Low"], row["Close"], row["Volume"],
                row["SMA_10"], row["SMA_30"], row["EMA_10"], row["EMA_30"],
                row["MACD"], row["MACD_Signal"], row["RSI"],
                row["BB_upper"], row["BB_lower"], row["BB_width"],
                row["Return_1d"], row["Momentum_5d"],
            ))
            records_written += 1
        conn.commit()
        print(f"[preprocess_data] {ticker}: {len(df)} rows written to DB")

    conn.close()
    print(f"[preprocess_data] Total records written: {records_written}")
    return records_written


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 – Run predictions
# ─────────────────────────────────────────────────────────────────────────────
def run_predictions(**context):
    """Load all 4 models and generate predictions for every watch-list ticker."""
    import numpy as np
    import pandas as pd
    from tensorflow import keras

    def _norm_window(w):
        X, sc = w.copy(), []
        for j in range(X.shape[1]):
            mn  = X[:, j].min()
            rng = X[:, j].max() - mn or 1.0
            X[:, j] = (X[:, j] - mn) / rng
            sc.append((mn, rng))
        return X, sc

    # Load models
    model_files = {
        "nasdaq_price": "nasdaq_model_task11.keras",
        "vn_price":     "vn_model_task21.keras",
        "vn_buy":       "vn_model_task31_buy.keras",
        "vn_sell":      "vn_model_task32_sell.keras",
    }
    loaded = {
        k: keras.models.load_model(os.path.join(BASE_DIR, v))
        for k, v in model_files.items()
        if os.path.exists(os.path.join(BASE_DIR, v))
    }

    results = {}

    # --- Nasdaq price prediction ---
    if "nasdaq_price" in loaded:
        for ticker in NASDAQ_WATCH_LIST:
            path = os.path.join(NASDAQ_DATA_DIR, f"{ticker}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            for col in NASDAQ_FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=NASDAQ_FEATURES).tail(WINDOW_SIZE + 10)
            if len(df) < WINDOW_SIZE:
                continue
            w = df[NASDAQ_FEATURES].values[-WINDOW_SIZE:]
            X_n, sc = _norm_window(w)
            y_n  = loaded["nasdaq_price"].predict(X_n[np.newaxis], verbose=0)[0, 0]
            mn, rng = sc[4]  # NASDAQ_CLOSE_IDX = 4
            y_pred  = float(y_n * rng + mn)
            results[f"nasdaq_{ticker}"] = {
                "type": "price", "last_close": float(df["Close"].iloc[-1]),
                "predicted_close": y_pred,
            }

    # --- VN price prediction ---
    if "vn_price" in loaded:
        conn = sqlite3.connect(DB_PATH)
        for ticker in VN_WATCH_LIST:
            rows = conn.execute("""
                SELECT open,high,low,close,volume FROM vn_features
                WHERE ticker=? ORDER BY date DESC LIMIT ?
            """, (ticker, WINDOW_SIZE)).fetchall()
            if len(rows) < WINDOW_SIZE:
                continue
            w = np.array(rows[::-1], dtype=float)
            X_n, sc = _norm_window(w)
            y_n  = loaded["vn_price"].predict(X_n[np.newaxis], verbose=0)[0, 0]
            mn, rng = sc[3]  # VN_CLOSE_IDX = 3
            results[f"vn_{ticker}_price"] = {
                "type": "price", "last_close": float(w[-1, 3]),
                "predicted_close": float(y_n * rng + mn),
            }
        conn.close()

    # --- VN signal prediction ---
    if "vn_buy" in loaded and "vn_sell" in loaded:
        conn = sqlite3.connect(DB_PATH)
        col_order = (
            "open,high,low,close,volume,"
            "sma10,sma30,ema10,ema30,macd,macd_signal,rsi,"
            "bb_upper,bb_lower,bb_width,return_1d,momentum_5d"
        )
        for ticker in VN_WATCH_LIST:
            rows = conn.execute(f"""
                SELECT {col_order} FROM vn_features
                WHERE ticker=? ORDER BY date DESC LIMIT ?
            """, (ticker, WINDOW_SIZE)).fetchall()
            if len(rows) < WINDOW_SIZE:
                continue
            w = np.array(rows[::-1], dtype=float)
            X_n, _ = _norm_window(w)
            bp = float(loaded["vn_buy"].predict(X_n[np.newaxis],  verbose=0)[0, 0])
            sp = float(loaded["vn_sell"].predict(X_n[np.newaxis], verbose=0)[0, 0])
            sig = "HOLD"
            if bp > 0.5 and bp > sp:
                sig = "BUY"
            elif sp > 0.5 and sp > bp:
                sig = "SELL"
            results[f"vn_{ticker}_signal"] = {
                "type": "signal", "signal": sig, "buy_prob": bp, "sell_prob": sp,
            }
        conn.close()

    context["ti"].xcom_push(key="predictions", value=results)
    print(f"[run_predictions] Generated {len(results)} predictions.")
    return len(results)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 – Store results in SQLite
# ─────────────────────────────────────────────────────────────────────────────
def store_results(**context):
    """Persist all predictions to SQLite (predictions table)."""
    predictions = context["ti"].xcom_pull(key="predictions", task_ids="run_predictions")
    if not predictions:
        print("[store_results] No predictions to store.")
        return 0

    run_date = context["ds"]  # YYYY-MM-DD string from Airflow

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date        TEXT,
            ticker          TEXT,
            pred_type       TEXT,
            last_close      REAL,
            predicted_close REAL,
            signal          TEXT,
            buy_prob        REAL,
            sell_prob       REAL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Remove stale records for this run_date before inserting fresh ones
    conn.execute("DELETE FROM predictions WHERE run_date=?", (run_date,))
    conn.commit()

    n = 0
    for key, rec in predictions.items():
        conn.execute("""
            INSERT INTO predictions
            (run_date, ticker, pred_type, last_close, predicted_close, signal, buy_prob, sell_prob)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            run_date, key, rec.get("type"),
            rec.get("last_close"), rec.get("predicted_close"),
            rec.get("signal"), rec.get("buy_prob"), rec.get("sell_prob"),
        ))
        n += 1
    conn.commit()
    conn.close()
    print(f"[store_results] Stored {n} records for {run_date}.")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 – Generate daily report
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(**context):
    """Export today's predictions from SQLite as a dated CSV report."""
    import pandas as pd

    run_date = context["ds"]
    os.makedirs(REPORT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE run_date=? ORDER BY ticker",
        conn, params=(run_date,)
    )
    conn.close()

    if df.empty:
        print(f"[generate_report] No predictions for {run_date}.")
        return

    report_path = os.path.join(REPORT_DIR, f"predictions_{run_date}.csv")
    df.to_csv(report_path, index=False)
    print(f"[generate_report] Report saved → {report_path} ({len(df)} rows)")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="stock_prediction_pipeline",
    description="Daily CNN-LSTM stock prediction pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 7 * * 1-5",   # 07:00 on weekdays (market days)
    catchup=False,
    tags=["dl4ai", "stock", "prediction"],
) as dag:

    t_fetch = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t_predict = PythonOperator(
        task_id="run_predictions",
        python_callable=run_predictions,
    )

    t_store = PythonOperator(
        task_id="store_results",
        python_callable=store_results,
    )

    t_report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    # Pipeline: fetch → preprocess → predict → store → report
    t_fetch >> t_preprocess >> t_predict >> t_store >> t_report
