"""
Task 5.3 – Standalone AI Automation Pipeline
============================================
Runs the full data → preprocess → predict → store → report pipeline
without requiring Airflow to be running. Useful for local testing.

Usage:
    python pipeline.py                   # process today
    python pipeline.py --date 2024-01-15 # backfill a specific date
"""

import argparse
import os
import sqlite3
import sys
from datetime import date

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "predictions.db")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

VN_DATA_DIR     = os.path.join(BASE_DIR, "data-vn-20230228", "stock-historical-data")
NASDAQ_DATA_DIR = os.path.join(BASE_DIR, "data_nasdaq_csv", "csv")

VN_WATCH_LIST     = ["REE", "VCB", "FPT", "HPG", "MWG", "VNM", "MSN", "PNJ"]
NASDAQ_WATCH_LIST = ["AAPL", "AMZN", "MSFT", "GOOGL", "META"]
WINDOW_SIZE       = 30

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
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def _norm_window(w: np.ndarray) -> tuple:
    X, sc = w.copy(), []
    for j in range(X.shape[1]):
        mn  = X[:, j].min()
        rng = X[:, j].max() - mn or 1.0
        X[:, j] = (X[:, j] - mn) / rng
        sc.append((mn, rng))
    return X, sc


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vn_features (
            ticker TEXT, date TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            sma10 REAL, sma30 REAL, ema10 REAL, ema30 REAL,
            macd REAL, macd_signal REAL, rsi REAL,
            bb_upper REAL, bb_lower REAL, bb_width REAL,
            return_1d REAL, momentum_5d REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT, ticker TEXT, pred_type TEXT,
            last_close REAL, predicted_close REAL,
            signal TEXT, buy_prob REAL, sell_prob REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Fetch / validate data
# ─────────────────────────────────────────────────────────────────────────────
def step_fetch_data() -> dict:
    print("\n[Step 1] Validating data files …")
    summary = {}
    for ticker in VN_WATCH_LIST:
        path = os.path.join(VN_DATA_DIR, f"{ticker}-VNINDEX-History.csv")
        n = len(pd.read_csv(path)) if os.path.exists(path) else 0
        summary[f"vn_{ticker}"] = n
        status = f"{n} rows" if n else "MISSING"
        print(f"  VN  {ticker:<8} {status}")
    for ticker in NASDAQ_WATCH_LIST:
        path = os.path.join(NASDAQ_DATA_DIR, f"{ticker}.csv")
        n = len(pd.read_csv(path)) if os.path.exists(path) else 0
        summary[f"nasdaq_{ticker}"] = n
        status = f"{n} rows" if n else "MISSING"
        print(f"  NDQ {ticker:<8} {status}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Pre-process & store features in SQLite
# ─────────────────────────────────────────────────────────────────────────────
def step_preprocess(conn: sqlite3.Connection) -> int:
    print("\n[Step 2] Computing indicators and storing in SQLite …")
    total = 0
    for ticker in VN_WATCH_LIST:
        path = os.path.join(VN_DATA_DIR, f"{ticker}-VNINDEX-History.csv")
        if not os.path.exists(path):
            print(f"  Skipping {ticker} (file not found)")
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
                ticker, str(row.get("TradingDate", "")),
                row["Open"], row["High"], row["Low"], row["Close"], row["Volume"],
                row["SMA_10"], row["SMA_30"], row["EMA_10"], row["EMA_30"],
                row["MACD"], row["MACD_Signal"], row["RSI"],
                row["BB_upper"], row["BB_lower"], row["BB_width"],
                row["Return_1d"], row["Momentum_5d"],
            ))
            total += 1
        conn.commit()
        print(f"  {ticker}: {len(df)} rows stored")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Run CNN-LSTM model inference
# ─────────────────────────────────────────────────────────────────────────────
def step_run_predictions(conn: sqlite3.Connection) -> dict:
    print("\n[Step 3] Running CNN-LSTM predictions …")
    from tensorflow import keras  # lazy import

    model_files = {
        "nasdaq_price": "nasdaq_model_task11.keras",
        "vn_price":     "vn_model_task21.keras",
        "vn_buy":       "vn_model_task31_buy.keras",
        "vn_sell":      "vn_model_task32_sell.keras",
    }
    models = {}
    for k, fname in model_files.items():
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            models[k] = keras.models.load_model(path)
            print(f"  Loaded {fname}")
        else:
            print(f"  [WARN] {fname} not found – skipping")

    results = {}

    # Nasdaq price
    if "nasdaq_price" in models:
        for ticker in NASDAQ_WATCH_LIST:
            path = os.path.join(NASDAQ_DATA_DIR, f"{ticker}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            for col in NASDAQ_FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=NASDAQ_FEATURES)
            if len(df) < WINDOW_SIZE:
                continue
            w = df[NASDAQ_FEATURES].values[-WINDOW_SIZE:]
            X_n, sc = _norm_window(w)
            y_n = models["nasdaq_price"].predict(X_n[np.newaxis], verbose=0)[0, 0]
            mn, rng = sc[4]
            y_pred = float(y_n * rng + mn)
            last   = float(df["Close"].iloc[-1])
            chg    = (y_pred - last) / last * 100
            results[f"nasdaq_{ticker}"] = {
                "type": "price", "last_close": last,
                "predicted_close": y_pred, "change_pct": chg,
            }
            print(f"  NASDAQ {ticker}: last={last:.2f}  pred={y_pred:.2f}  ({chg:+.2f}%)")

    # VN price
    if "vn_price" in models:
        for ticker in VN_WATCH_LIST:
            rows = conn.execute("""
                SELECT open,high,low,close,volume FROM vn_features
                WHERE ticker=? ORDER BY date DESC LIMIT ?
            """, (ticker, WINDOW_SIZE)).fetchall()
            if len(rows) < WINDOW_SIZE:
                continue
            w = np.array(rows[::-1], dtype=float)
            X_n, sc = _norm_window(w)
            y_n = models["vn_price"].predict(X_n[np.newaxis], verbose=0)[0, 0]
            mn, rng = sc[3]
            y_pred = float(y_n * rng + mn)
            last   = float(w[-1, 3])
            chg    = (y_pred - last) / last * 100
            results[f"vn_{ticker}_price"] = {
                "type": "price", "last_close": last,
                "predicted_close": y_pred, "change_pct": chg,
            }
            print(f"  VN    {ticker}: last={last:,.0f}  pred={y_pred:,.0f}  ({chg:+.2f}%)")

    # VN signal
    if "vn_buy" in models and "vn_sell" in models:
        col_q = ("open,high,low,close,volume,"
                 "sma10,sma30,ema10,ema30,macd,macd_signal,rsi,"
                 "bb_upper,bb_lower,bb_width,return_1d,momentum_5d")
        for ticker in VN_WATCH_LIST:
            rows = conn.execute(f"""
                SELECT {col_q} FROM vn_features
                WHERE ticker=? ORDER BY date DESC LIMIT ?
            """, (ticker, WINDOW_SIZE)).fetchall()
            if len(rows) < WINDOW_SIZE:
                continue
            w = np.array(rows[::-1], dtype=float)
            X_n, _ = _norm_window(w)
            bp = float(models["vn_buy"].predict(X_n[np.newaxis],  verbose=0)[0, 0])
            sp = float(models["vn_sell"].predict(X_n[np.newaxis], verbose=0)[0, 0])
            sig = "HOLD"
            if bp > 0.5 and bp > sp:
                sig = "BUY"
            elif sp > 0.5 and sp > bp:
                sig = "SELL"
            results[f"vn_{ticker}_signal"] = {
                "type": "signal", "signal": sig, "buy_prob": bp, "sell_prob": sp,
            }
            print(f"  VN    {ticker} signal: {sig:4s}  buy={bp:.3f}  sell={sp:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Store predictions in SQLite
# ─────────────────────────────────────────────────────────────────────────────
def step_store_results(conn: sqlite3.Connection, results: dict, run_date: str) -> int:
    print(f"\n[Step 4] Storing {len(results)} predictions for {run_date} …")
    # Remove any stale records for this run_date before inserting fresh ones
    conn.execute("DELETE FROM predictions WHERE run_date=?", (run_date,))
    conn.commit()
    n = 0
    for key, rec in results.items():
        conn.execute("""
            INSERT INTO predictions
            (run_date, ticker, pred_type, last_close, predicted_close,
             signal, buy_prob, sell_prob)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            run_date, key, rec.get("type"),
            rec.get("last_close"), rec.get("predicted_close"),
            rec.get("signal"), rec.get("buy_prob"), rec.get("sell_prob"),
        ))
        n += 1
    conn.commit()
    print(f"  {n} records written.")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Generate daily CSV report
# ─────────────────────────────────────────────────────────────────────────────
def step_generate_report(conn: sqlite3.Connection, run_date: str) -> str:
    print(f"\n[Step 5] Generating report for {run_date} …")
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE run_date=? ORDER BY ticker",
        conn, params=(run_date,)
    )
    if df.empty:
        print("  No data found.")
        return ""
    report_path = os.path.join(REPORT_DIR, f"predictions_{run_date}.csv")
    df.to_csv(report_path, index=False)
    print(f"  Report saved → {report_path}")
    print(df[["ticker", "pred_type", "predicted_close", "signal",
              "buy_prob", "sell_prob"]].to_string(index=False))
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline")
    parser.add_argument(
        "--date", default=str(date.today()),
        help="Run date in YYYY-MM-DD format (default: today)"
    )
    args = parser.parse_args()
    run_date = args.date

    print(f"=== Stock Prediction Pipeline | run_date={run_date} ===")

    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)

    try:
        step_fetch_data()
        step_preprocess(conn)
        results = step_run_predictions(conn)
        step_store_results(conn, results, run_date)
        step_generate_report(conn, run_date)
    finally:
        conn.close()

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
