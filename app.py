from flask import Flask, render_template, jsonify, request
import json, os
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# Paths & discovery
# ─────────────────────────────────────────────────────────
JSON_DIR = "forecast/json"
CSV_DIR  = "forecast/datasets"

LOG_DIR  = "logs"
LOG_ALL_FILE = os.path.join(LOG_DIR, "trades_all.csv")
def LOG_MODEL_FILE(model_key: str):
    return os.path.join(LOG_DIR, f"trades_{model_key}.csv")

DEFAULT_EDGE_PCT = 5.0
DEFAULT_START_CAPITAL = 10_000.0   # ทุนอ้างอิงต่อโมเดล
DEFAULT_NOTIONAL = 1_000.0         # ลงทุนต่อออเดอร์
DEFAULT_FEE_BPS = 10.0
DEFAULT_INTEGER_SHARES = False     # ✅ ใช้เศษหุ้นเป็นค่าเริ่มต้น

def _list_json_tickers():
    if not os.path.isdir(JSON_DIR): return set()
    return {os.path.splitext(fn)[0] for fn in os.listdir(JSON_DIR) if fn.lower().endswith(".json")}

def _list_csv_tickers():
    if not os.path.isdir(CSV_DIR): return set()
    out = set()
    for fn in os.listdir(CSV_DIR):
        if not fn.lower().endswith(".csv"): continue
        out.add(os.path.splitext(fn)[0].replace("_dataset", ""))
    return out

_available_json = _list_json_tickers()
_available_csv  = _list_csv_tickers()
TICKERS = sorted(_available_json & _available_csv) or sorted(_available_json) or ['AMZN','META','NVDA']

# ─────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────
def _read_prediction_df(ticker: str) -> pd.DataFrame:
    p = os.path.join(JSON_DIR, f"{ticker}.json")
    if not os.path.exists(p): raise FileNotFoundError(f"Prediction file not found for {ticker}")
    with open(p, "r", encoding="utf-8") as f: data = json.load(f)
    df = pd.DataFrame(data)
    if "Date" not in df: raise ValueError("JSON missing 'Date'")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["LSTM_Predicted_Price","Transformer_Predicted_Price","TCN_GRU_Predicted_Price"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) > 30: df = df.iloc[:30].copy()  # บังคับ 30 จุดถ้ามีเกิน
    for c in ["LSTM_Predicted_Price","Transformer_Predicted_Price","TCN_GRU_Predicted_Price"]:
        if c in df: df[c] = df[c].ffill().bfill()
    return df

def _read_actual_df(ticker: str) -> pd.DataFrame:
    p = os.path.join(CSV_DIR, f"{ticker}_dataset.csv")
    if not os.path.exists(p): raise FileNotFoundError(f"Dataset CSV not found for {ticker}")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if "Close" not in df.columns: raise ValueError("CSV missing 'Close'")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").ffill().bfill()
    return df
# ─────────────────────────────────────────────────────────
# Logs (all orders)
# ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join("forecast", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_ALL_FILE = os.path.join(LOG_DIR, "orders_all.csv")   # รวมทุกโมเดลทุกหุ้น
def _calc_realtime_pnl(row: dict) -> dict:
    # ✅ ถ้าเป็น CLOSED → ข้าม ไม่คำนวณซ้ำ
    if str(row.get("status", "")).upper() == "CLOSED":
        return row

    ticker = str(row.get("ticker"))
    side = (row.get("side") or "LONG").upper()
    shares = float(row.get("shares", 0.0))
    entry_price = float(row.get("entry_price", 0.0))
    invested_usd = entry_price * shares if shares and entry_price else 0.0

    # ใช้ราคาจริงล่าสุดเสมอ
    try:
        current_price, current_date = _load_current_price(ticker)
    except Exception:
        current_price, current_date = entry_price, row.get("entry_date")

    gross = (current_price - entry_price) * shares
    net = gross  # ไม่มี fees แล้ว
    return_pct = (net / invested_usd * 100.0) if invested_usd else 0.0

    row.update({
        "px_ref": clean_num(current_price, 6),
        "px_ref_date": current_date,
        "gross_pnl_usd": clean_num(gross, 6),
        "net_pnl_usd": clean_num(net, 6),
        "return_pct": clean_num(return_pct, 4),
        "expected_price": clean_num(row.get("planned_exit_price", 0.0), 6),
        "expected_date": row.get("planned_exit_date", "")
    })

    return row



def _load_current_price(ticker: str):
    """
    ดึงราคาปิดล่าสุดจาก CSV dataset ของ ticker นั้น
    - ไม่บังคับว่าต้องเป็นวันปัจจุบัน
    - ใช้แถวสุดท้ายของ DataFrame เสมอ (last available close)
    """
    df = _read_actual_df(ticker)
    last_close = float(df["Close"].iloc[-1])         # ราคาปิดล่าสุด
    last_date  = df.index[-1].strftime("%Y-%m-%d")   # วันที่ล่าสุดใน dataset
    return last_close, last_date


def clean_num(x, ndigits=8, decimal_cut=2):
    """
    - ndigits: จำนวนทศนิยมที่จะปัด
    - decimal_cut: ถ้าทศนิยม N หลักแรกเป็นศูนย์ → บังคับเป็น 0
    """
    try:
        v = float(x)
        if v == 0.0:
            return 0.0
        # เช็คว่าเล็กเกิน 1/(10^decimal_cut)
        if abs(v) < 10**(-decimal_cut):
            return 0.0
        return round(v, ndigits)
    except (TypeError, ValueError):
        return None


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2: return 0.0
    roll_max = np.maximum.accumulate(equity)
    return float((equity/roll_max - 1.0).min() * 100.0)

def _sharpe(daily_rets: np.ndarray) -> float:
    if daily_rets.size == 0: return 0.0
    mu, sd = float(np.mean(daily_rets)), float(np.std(daily_rets, ddof=1))
    if sd == 0: return 0.0
    return (mu/sd) * np.sqrt(252.0)


EQUITY_CURVE_FILE = os.path.join(LOG_DIR, "equity_curve.csv")
@app.route("/equity_curve")
def equity_curve():
    model = request.args.get("model", "all").lower()
    if not os.path.exists(EQUITY_CURVE_FILE):
        return jsonify({"dates": [], "equity": []})

    df = pd.read_csv(EQUITY_CURVE_FILE, parse_dates=["date"])
    if df.empty:
        return jsonify({"dates": [], "equity": []})

    # เอา record ล่าสุดต่อวัน+โมเดล
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date","model"], keep="last")

    start_date = pd.to_datetime("2025-08-01")
    end_date = df["date"].max()
    all_days = pd.date_range(start_date, end_date, freq="D")

    if model != "all":
        series = (
            df[df["model"] == model]
            .set_index("date")["equity"]
            .reindex(all_days)
            .ffill()
            .fillna(10000)   # baseline ถ้าไม่มีค่า → 10000
        )
    else:
        # 🔹 กำหนดโมเดลที่ต้องมีครบ
        models = ["lstm","trans","tcn"]
        pivot = df.pivot(index="date", columns="model", values="equity")
        pivot = pivot.reindex(all_days).ffill()

        # เติม baseline = 10000 สำหรับโมเดลที่ไม่เจอ
        for m in models:
            if m not in pivot.columns:
                pivot[m] = 10000

        pivot = pivot.fillna(10000)
        series = pivot[models].sum(axis=1)   # ✅ รวมทุกโมเดล

    dates = [d.strftime("%Y-%m-%d") for d in series.index]
    equity = [round(v,2) for v in series.values]

    return jsonify({"dates": dates, "equity": equity})


def _update_equity_curve():
    """Rebuild equity_curve.csv จาก orders_all.csv (เฉพาะ CLOSED orders)"""
    if not os.path.exists(LOG_ALL_FILE):
        print(f"No {LOG_ALL_FILE} found")
        return

    try:
        df = pd.read_csv(LOG_ALL_FILE, parse_dates=["exit_date"], low_memory=False)
    except Exception as e:
        print(f"Error reading {LOG_ALL_FILE}: {e}")
        return

    # เอาเฉพาะ CLOSED
    df = df[df["status"].str.upper() == "CLOSED"].copy()
    if df.empty:
        print("No closed trades yet")
        return

    # แปลง numeric
    df["net_pnl_usd"] = pd.to_numeric(df["net_pnl_usd"], errors="coerce").fillna(0.0)
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")

    out_rows = []
    for model, g in df.groupby("model"):
        g = g.sort_values("exit_date")
        equity = DEFAULT_START_CAPITAL
        for _, row in g.iterrows():
            equity += row["net_pnl_usd"]
            out_rows.append({
                "date": row["exit_date"].strftime("%Y-%m-%d"),
                "model": str(model).lower(),
                "equity": round(equity, 2)
            })

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values(["model", "date"]).reset_index(drop=True)
    out_df.to_csv(EQUITY_CURVE_FILE, index=False)
    print(f"✅ Equity curve saved to {EQUITY_CURVE_FILE}")


# ─────────────────────────────────────────────────────────
# Performance API (รวมทุกโมเดลหรือตามโมเดล)
# ─────────────────────────────────────────────────────────
@app.route("/performance")
def get_performance():
    model = request.args.get("model", "all").lower()
    path = LOG_ALL_FILE if model == "all" else LOG_MODEL_FILE(model)

    # 🔹 base capital แก้ตรงนี้
    base_capital = 30000 if model == "all" else DEFAULT_START_CAPITAL

    if not os.path.exists(path):
        return jsonify({
            "model": model,
            "final_equity": base_capital,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": 0.0
        })

    df = pd.read_csv(path)
    if df.empty:
        return jsonify({
            "model": model,
            "final_equity": base_capital,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "num_trades": 0,
            "win_rate_pct": 0.0
        })

    # ✅ refresh ทุก order ที่ยัง OPEN
    for i, row in df.iterrows():
        if str(row.get("status", "")).upper() == "OPEN":
            row = _calc_realtime_pnl(row.to_dict())
            df.loc[i, "net_pnl_usd"] = row["net_pnl_usd"]
            df.loc[i, "gross_pnl_usd"] = row["gross_pnl_usd"]
            df.loc[i, "return_pct"] = row["return_pct"]
            df.loc[i, "current_price"] = row["px_ref"]
            df.loc[i, "current_date"] = row["px_ref_date"]

    # 🔢 คำนวณค่าต่าง ๆ
    realized = df.loc[df["status"] == "CLOSED", "net_pnl_usd"].sum()

    final_equity = base_capital + realized
    total_return_pct = (final_equity / base_capital - 1.0) * 100.0

    # trades & winrate
    closed_trades = df[df["status"] == "CLOSED"]
    num_trades = len(closed_trades)
    win_trades = (closed_trades["net_pnl_usd"] > 0).sum()
    win_rate_pct = (win_trades / num_trades * 100.0) if num_trades > 0 else 0.0

    # max drawdown (ใช้ equity แบบสะสม)
    equity_curve = base_capital + df["net_pnl_usd"].cumsum()
    max_dd = _max_drawdown(equity_curve.values) if len(equity_curve) > 1 else 0.0

    return jsonify({
        "model": model,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "num_trades": int(num_trades),
        "win_rate_pct": round(win_rate_pct, 2)
    })


# ─────────────────────────────────────────────────────────
# UI routes
# ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", tickers=TICKERS)

@app.route("/trade")
def trade():
    selected_ticker = request.args.get("ticker") or (TICKERS[0] if TICKERS else None)
    model = request.args.get("model", "lstm").lower()
    # หน้า UI นี้ใช้พรีวิวอย่างเดียวแล้ว
    current_price = current_date = error = None
    if selected_ticker:
        try:
            current_price, current_date = _load_current_price(selected_ticker)
        except Exception as e:
            error = str(e)
    return render_template(
        "trade.html",
        tickers=TICKERS,
        selected_ticker=selected_ticker,
        params={"model": model},
        current_price=current_price,
        current_date=current_date,
        error=error,
    )

# ─────────────────────────────────────────────────────────
# Forecast data API (graph page uses)
# ─────────────────────────────────────────────────────────
@app.route("/data/<ticker>")
def get_prediction_data(ticker):
    try:
        df_pred = _read_prediction_df(ticker)
        df_act  = _read_actual_df(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    future_dates = df_pred["Date"].dt.strftime("%Y-%m-%d").tolist()
    lstm  = df_pred["LSTM_Predicted_Price"].tolist() if "LSTM_Predicted_Price" in df_pred else []
    trans = df_pred["Transformer_Predicted_Price"].tolist() if "Transformer_Predicted_Price" in df_pred else []
    tcn   = df_pred["TCN_GRU_Predicted_Price"].tolist() if "TCN_GRU_Predicted_Price" in df_pred else []

    actual_days = 504
    actual_dates  = df_act.index[-actual_days:].strftime("%Y-%m-%d").tolist()
    actual_prices = df_act["Close"].iloc[-actual_days:].tolist()
    latest_date   = df_act.index[-1].strftime("%Y-%m-%d")

    def ret(seq):
        if not seq or not seq[0]: return None
        return (float(seq[-1])/float(seq[0]) - 1.0)*100.0

    return jsonify({
        "ticker": ticker,
        "actual_dates": actual_dates, "actual_prices": actual_prices,
        "future_dates": future_dates,
        "lstm_prices": lstm, "trans_prices": trans, "tcn_prices": tcn,
        "lstm_return_pct": ret(lstm), "trans_return_pct": ret(trans), "tcn_return_pct": ret(tcn),
        "latest_date": latest_date,
    })
# เป้ากำไรอิง "ระยะทางจากราคาเข้า → ราคาทำนายสูงสุด"
# 100.0 = รอจนแตะค่าสูงสุดที่ทำนาย, 80.0 = ถึง 80% ของทางก็ขาย
DEFAULT_PROFIT_TO_PEAK_PCT = 100.0
# ─────────────────────────────────────────────────────────
# Simulator (buy notional at t=0 if passes edge; block SELL if future)
# ─────────────────────────────────────────────────────────
def _load_prediction_series(ticker: str, model_key: str):
    df = _read_prediction_df(ticker)
    col = {"lstm":"LSTM_Predicted_Price","trans":"Transformer_Predicted_Price","tcn":"TCN_GRU_Predicted_Price"}.get(model_key)
    if not col or col not in df: raise ValueError(f"Missing column for model {model_key}")
    dates  = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    prices = pd.to_numeric(df[col], errors="coerce").ffill().bfill().tolist()
    return dates, prices

def simulate_buy_hold_with_today(
    dates, prices, current_date_str,
    start_capital=DEFAULT_START_CAPITAL,
    notional=DEFAULT_NOTIONAL,
    fee_bps=DEFAULT_FEE_BPS,
    integer_shares=DEFAULT_INTEGER_SHARES,
    target_idx=None,
    profit_to_peak_pct=DEFAULT_PROFIT_TO_PEAK_PCT
):
    """
    กลยุทธ์: ซื้อที่ t=0 แล้วถือจนถึง 'วันที่ของค่าสูงสุด' (target_idx)
    หรือถ้าแตะกำไรที่เป็นสัดส่วนของทางไปถึงจุดสูงสุด (profit_to_peak_pct%) ก่อน ก็ออกวันนั้น
    - ปิดจริงได้ก็ต่อเมื่อ planned_exit_date <= current_date
    - Equity Curve = start_capital + P&L สะสม
    """
    n = len(prices)
    if n == 0:
        return {
            "future_dates": [], "prices": [], "equity_curve": [],
            "trades": [], "metrics": {"final_equity": start_capital, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "num_trades": 0, "win_rate_pct": 0.0, "sharpe": 0.0},
            "position": {"open": False}
        }

    fee = fee_bps/10000.0
    dates_dt = pd.to_datetime(dates)
    today_dt = pd.to_datetime(current_date_str)
    today_idx = int(max([i for i, d in enumerate(dates_dt) if d <= today_dt], default=0))

    # ซื้อที่ t=0
    entry_price = float(prices[0])
    size = notional / entry_price
    shares = float(np.floor(size)) if integer_shares else float(size)

    cash = float(start_capital)
    buy_notional = shares * entry_price
    buy_fee = abs(buy_notional) * fee
    cash -= buy_notional
    cash -= buy_fee

    trades = [{
        "date": dates[0], "action": "BUY", "price": round(entry_price,6), "shares": float(shares),
        "cash": round(cash,6), "equity": round(cash + shares*entry_price,6), "pnl": None,
        "reason": f"Enter long (${notional:.2f})"
    }]

    # ===== กำหนดเป้าออก =====
    if target_idx is None:
        target_idx = int(np.argmax(prices))
    target_idx = int(max(0, min(target_idx, n-1)))

    pred_max = float(prices[target_idx])

    profit_px = entry_price + (pred_max - entry_price) * (profit_to_peak_pct / 100.0)
    if pred_max < entry_price:
        profit_px = entry_price

    early_idx = next((i for i, p in enumerate(prices) if float(p) >= profit_px), None)
    planned_exit_idx = early_idx if (early_idx is not None and early_idx < target_idx) else target_idx

    equity_curve = []
    sell_executed, realized_pnl = False, None
    sell_idx = None

    for i in range(n):
        p = float(prices[i])

        # ปิดได้จริงเมื่อถึงวันตามแผน และวันนั้นไม่อยู่ในอนาคต
        if (i >= planned_exit_idx) and (dates_dt[planned_exit_idx] <= today_dt) and not sell_executed:
            if dates_dt[planned_exit_idx] < today_dt:
                sell_idx = today_idx
            else:
                sell_idx = planned_exit_idx

            sell_price = float(prices[sell_idx])
            sell_notional = shares * sell_price
            sell_fee = abs(sell_notional) * fee
            cash += sell_notional
            cash -= sell_fee
            realized_pnl = (sell_price - entry_price) * shares
            trades.append({
                "date": dates[sell_idx], "action": "SELL", "price": round(sell_price,6),
                "shares": 0.0, "cash": round(cash,6), "equity": round(cash,6),
                "pnl": round(realized_pnl,6),
                "reason": "Hit profit (to-peak %)" if (early_idx is not None and sell_idx==early_idx)
                          else "Exit at predicted peak"
            })
            sell_executed = True

        # ===== Equity Curve: start_capital + P&L =====
        if sell_executed and sell_idx is not None and i >= sell_idx:
            eq = cash  # หลังขาย → เงินสดคงที่
        else:
            pnl = (p - entry_price) * shares
            eq = start_capital + pnl
        equity_curve.append(eq)

    equity_arr = np.array(equity_curve, dtype=float)
    cut = today_idx if not sell_executed else n-1
    daily_rets = np.diff(equity_arr[:cut+1]) / equity_arr[:cut] if cut >= 1 else np.array([])

    metrics = {
        "final_equity": float(equity_arr[cut]),
        "total_return_pct": float((equity_arr[cut]/equity_arr[0]-1.0)*100.0) if equity_arr[0] != 0 else 0.0,
        "max_drawdown_pct": float(_max_drawdown(equity_arr[:cut+1] if cut+1>1 else equity_arr[:1])),
        "num_trades": int(len(trades)),
        "win_rate_pct": float(100.0 if (realized_pnl is not None and realized_pnl > 0) else 0.0) if sell_executed else 0.0,
        "sharpe": float(_sharpe(daily_rets)),
    }

    position = {
        "open": not sell_executed,
        "planned_exit_date": dates[planned_exit_idx],
        "planned_exit_price": float(prices[planned_exit_idx]),
        "peak_idx": int(target_idx),
        "profit_to_peak_pct": float(profit_to_peak_pct),
        "profit_target_price": float(profit_px),
        "today": str(today_dt.date()),
        "price_today": float(prices[today_idx]),
        "unrealized_pnl": float((float(prices[today_idx]) - entry_price) * shares) if not sell_executed else 0.0,
        "buy_fee": float(buy_fee),
        "notional": float(notional)
    }

    return {
        "future_dates": dates, "prices": prices,
        "equity_curve": [float(x) for x in equity_arr],
        "trades": trades, "metrics": metrics, "position": position
    }



# ─────────────────────────────────────────────────────────
# Logging (OPEN/CLOSED) to CSV
# ─────────────────────────────────────────────────────────
LOG_HEADER = [
    "trade_id","logged_at","status",
    "ticker","model",
    "entry_date","entry_price","shares","invested_usd",
    "exit_date","exit_price",
    "gross_pnl_usd","net_pnl_usd","return_pct",
    "expected_price","expected_date",
    "current_price","current_date"
]


def _ensure_dir(): os.makedirs(LOG_DIR, exist_ok=True)

def _upsert_row(csv_path: str, row: dict):
    _ensure_dir()
    if os.path.exists(csv_path):
        try: df = pd.read_csv(csv_path)
        except Exception: df = pd.DataFrame(columns=LOG_HEADER)
    else:
        df = pd.DataFrame(columns=LOG_HEADER)
    if "trade_id" in df.columns and not df.empty:
        df = df[df["trade_id"].astype(str) != str(row["trade_id"])]
    df = pd.concat([df, pd.DataFrame([row], columns=LOG_HEADER)], ignore_index=True)
    df.to_csv(csv_path, index=False)

def _upsert_trade_logs(row: dict, model_key: str):
    _upsert_row(LOG_ALL_FILE, row)
    _upsert_row(LOG_MODEL_FILE(model_key), row)
    _update_equity_curve()

def _has_open_position(ticker: str, model: str) -> bool:
    """เช็คว่ามีออเดอร์ OPEN ของ ticker+model อยู่หรือไม่ (กันเปิดซ้ำ)"""
    path = LOG_ALL_FILE
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return False
    if df.empty: return False
    df["status"] = df.get("status","OPEN")
    df["ticker"] = df.get("ticker","")
    df["model"]  = df.get("model","")
    mask = (df["status"] == "OPEN") & (df["ticker"] == ticker) & (df["model"] == model)
    return bool(mask.any())

def _log_open(sim: dict, ticker: str, model: str, start_capital: float, fee_bps: float, edge_pct: float):
    pos = sim.get("position", {})
    buy = next((t for t in sim.get("trades", []) if t.get("action")=="BUY"), None)
    if not buy: return
    shares = float(buy["shares"])
    entry_price = float(buy["price"])
    notional = float(pos.get("notional", shares*entry_price))
    price_today = float(pos.get("price_today", entry_price))
    final_equity_today = float(sim["metrics"]["final_equity"])

    net_pnl = final_equity_today - float(start_capital)
    gross_pnl = net_pnl
    ret_pct = (price_today/entry_price - 1.0)*100.0 if entry_price else 0.0
    trade_id = f"{ticker}-{model}-{buy['date']}-{int(notional)}"
    current_price, current_date = _load_current_price(ticker)

    row = {
        "trade_id": trade_id, "logged_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "status": "OPEN", "ticker": ticker, "model": model,
        "entry_date": buy["date"], "entry_price": clean_num(entry_price, 6),
        "shares": clean_num(shares, 6),
        "invested_usd": clean_num(notional, 6),
        "exit_date": "", "exit_price": "",
        "gross_pnl_usd": clean_num(gross_pnl, 6),
        "net_pnl_usd": clean_num(net_pnl, 6),
        "return_pct": clean_num(ret_pct, 6),
        "expected_price": clean_num(pos.get("planned_exit_price", 0.0), 6),
        "expected_date": pos.get("planned_exit_date", ""),
        "current_price": round(current_price, 6),
        "current_date": current_date
    }
    _upsert_trade_logs(row, model)



def _log_closed(sim: dict, ticker: str, model: str, start_capital: float, fee_bps: float, edge_pct: float):
    trades = sim.get("trades", [])
    pos = sim.get("position", {})
    buy  = next((t for t in trades if t.get("action")=="BUY"), None)
    sell = next((t for t in trades if t.get("action")=="SELL"), None)
    if not (buy and sell): return

    shares = float(buy["shares"])
    entry_price = float(buy["price"])
    exit_price  = float(sell["price"])
    notional = float(pos.get("notional", shares*entry_price))

    final_equity = float(sim["metrics"]["final_equity"])
    net_pnl = final_equity - float(start_capital)
    gross_pnl = (exit_price-entry_price)*shares
    ret_pct = (exit_price/entry_price - 1.0)*100.0 if entry_price else 0.0
    trade_id = f"{ticker}-{model}-{buy['date']}-{int(notional)}"
    current_price, current_date = _load_current_price(ticker)

    row = {
        "trade_id": trade_id, "logged_at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "status": "CLOSED", "ticker": ticker, "model": model,
        "entry_date": buy["date"], "entry_price": clean_num(entry_price, 6),
        "shares": clean_num(shares, 6),
        "invested_usd": clean_num(notional, 6),
        "exit_date": sell["date"], "exit_price": round(exit_price,6),
        "gross_pnl_usd": clean_num(gross_pnl, 6),
        "net_pnl_usd": clean_num(net_pnl, 6),
        "return_pct": clean_num(ret_pct, 6),
        "expected_price": clean_num(pos.get("planned_exit_price", 0.0), 6),
        "expected_date": pos.get("planned_exit_date", ""),
        "current_price": round(current_price, 6),
        "current_date": current_date
    }
    _upsert_trade_logs(row, model)


# ─────────────────────────────────────────────────────────
# Public simulate endpoint (preview หรือ log จริงถ้าไม่ preview)
# ─────────────────────────────────────────────────────────
@app.route("/simulate/<ticker>")
def simulate_endpoint(ticker):
    model = request.args.get("model","lstm").lower()
    preview = request.args.get("preview", "1").lower() in ("1","true","yes","on")

    start_capital = float(request.args.get("start_capital", DEFAULT_START_CAPITAL))
    notional      = float(request.args.get("notional", DEFAULT_NOTIONAL))
    fee_bps       = float(request.args.get("fee_bps", DEFAULT_FEE_BPS))
    edge_pct      = float(request.args.get("edge_pct", DEFAULT_EDGE_PCT))
    profit_to_peak_pct = float(request.args.get("profit_to_peak_pct", DEFAULT_PROFIT_TO_PEAK_PCT))

    try:
        dates, prices = _load_prediction_series(ticker, model)
        current_price, current_date = _load_current_price(ticker)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not prices:
        return jsonify({"error": "No predicted prices"}), 400

    # ใช้ค่าสูงสุดของ 30 จุดเป็นเป้าหมายหลัก
    idx_max = int(np.argmax(prices))
    pred_max = float(prices[idx_max])
    pred_min = float(prices[int(np.argmin(prices))])
    up_edge = (pred_max/current_price - 1.0) * 100.0

    if up_edge < edge_pct:
        n = len(prices)
        return jsonify({
            "ticker": ticker, "model": model,
            "future_dates": dates, "prices": prices,
            "equity_curve": [start_capital]*n,
            "trades": [], "metrics": {
                "final_equity": float(start_capital),
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "num_trades": 0, "win_rate_pct": 0.0, "sharpe": 0.0
            },
            "gate": {
                "engaged": False, "rule": "max-above-current",
                "edge_pct": edge_pct, "up_edge_pct": round(up_edge, 4),
                "current_price": round(current_price,6), "current_date": current_date,
                "pred_max_price": round(pred_max,6), "pred_max_date": dates[idx_max],
                "pred_min_price": round(pred_min,6)
            },
            "position": {"open": False}
        })

    # run sim
    sim = simulate_buy_hold_with_today(
        dates, prices, current_date,
        start_capital=start_capital, notional=notional, fee_bps=fee_bps,
        integer_shares=DEFAULT_INTEGER_SHARES,
        target_idx=idx_max,
        profit_to_peak_pct=profit_to_peak_pct
    )
    sim.update({
        "ticker": ticker, "model": model,
        "gate": {
            "engaged": True, "rule": "max-above-current",
            "edge_pct": edge_pct, "up_edge_pct": round(up_edge, 4),
            "current_price": round(current_price,6), "current_date": current_date,
            "pred_max_price": round(pred_max,6), "pred_max_date": dates[idx_max],
            "pred_min_price": round(pred_min,6),
            "profit_to_peak_pct": float(profit_to_peak_pct)
        }
    })

    if not preview:
        if sim.get("position",{}).get("open", True):
            _log_open(sim, ticker, model, start_capital, fee_bps, edge_pct)
        else:
            _log_closed(sim, ticker, model, start_capital, fee_bps, edge_pct)

    return jsonify(sim)


# ใช้วันล่าสุดจาก dataset CSV ของ ticker ที่มีอยู่
def _get_latest_dataset_date() -> str:
    dates = []
    for t in TICKERS:
        try:
            df = _read_actual_df(t)
            dates.append(df.index[-1])
        except Exception:
            continue
    return max(dates).strftime("%Y-%m-%d")

def _check_force_close():
    """ตรวจสอบออเดอร์ OPEN → force close
       และ forward-fill equity curve ทุกโมเดล"""
    if not os.path.exists(LOG_ALL_FILE):
        return

    df = pd.read_csv(LOG_ALL_FILE)
    if df.empty:
        return

    # 1) ปิด order ตาม expected px/date
    open_rows = df[df["status"] == "OPEN"].to_dict(orient="records")

    for row in open_rows:
        ticker = row["ticker"]
        model = row["model"]

        try:
            current_price, current_date = _load_current_price(ticker)
        except Exception:
            continue

        expected_px = float(row.get("expected_price") or 0.0)
        expected_date = row.get("expected_date") or ""

        force_close = False
        reason = ""

        if expected_px and current_price >= expected_px:
            force_close = True
            reason = "Current Px >= Expected Px"

        elif expected_date:
            try:
                if pd.to_datetime(current_date) > pd.to_datetime(expected_date):
                    force_close = True
                    reason = "Current Date > Expected Date"
            except Exception:
                pass

        if force_close:
            shares = float(row.get("shares", 0.0))
            entry_price = float(row.get("entry_price", 0.0))
            exit_price = float(current_price)

            gross_pnl = (exit_price - entry_price) * shares
            net_pnl = gross_pnl
            ret_pct = (exit_price / entry_price - 1.0) * 100.0 if entry_price else 0.0

            closed_row = row.copy()
            closed_row.update({
                "status": "CLOSED",
                "exit_date": current_date,
                "exit_price": round(exit_price, 6),
                "gross_pnl_usd": clean_num(gross_pnl, 6),
                "net_pnl_usd": clean_num(net_pnl, 6),
                "return_pct": clean_num(ret_pct, 6),
                "current_price": round(current_price, 6),
                "current_date": current_date
            })

            _upsert_trade_logs(closed_row, model)

@app.route("/markers/<ticker>")
def get_markers(ticker):
    if not os.path.exists(LOG_ALL_FILE):
        return jsonify({"lstm":{"buy":[],"exit":[]},
                        "trans":{"buy":[],"exit":[]},
                        "tcn":{"buy":[],"exit":[]}})
    df = pd.read_csv(LOG_ALL_FILE)
    if df.empty:
        return jsonify({"lstm":{"buy":[],"exit":[]},
                        "trans":{"buy":[],"exit":[]},
                        "tcn":{"buy":[],"exit":[]}})

    # กรองเฉพาะ ticker + ที่ OPEN
    df = df[(df["ticker"] == ticker) & (df["status"] == "OPEN")]

    # เตรียมโครงสร้างสำหรับแต่ละโมเดล
    markers = {
        "lstm": {"buy": [], "exit": []},
        "trans": {"buy": [], "exit": []},
        "tcn": {"buy": [], "exit": []}
    }

    for _, row in df.iterrows():
        model = str(row.get("model","")).lower()
        if model not in markers:
            continue
        if row.get("entry_date") and row.get("entry_price"):
            markers[model]["buy"].append({
                "date": row["entry_date"],
                "price": float(row["entry_price"])
            })
        if row.get("expected_date") and row.get("expected_price"):
            markers[model]["exit"].append({
                "date": row["expected_date"],
                "price": float(row["expected_price"])
            })

    return jsonify(markers)



# ─────────────────────────────────────────────────────────
# Logs API
# ─────────────────────────────────────────────────────────
@app.route("/logs")
def get_logs():
    _check_force_close()
    model = request.args.get("model", "all").lower()
    status = request.args.get("status", "all").lower()
    ticker = request.args.get("ticker", "all").upper()

    path = LOG_ALL_FILE if model == "all" else LOG_MODEL_FILE(model)
    if not os.path.exists(path):
        return jsonify({
            "model": model,
            "items": [],
            "totals": {"overall_pnl": 0.0, "realized_pnl": 0.0, "open_pnl": 0.0, "invested_usd": 0.0},
            "totals_by_model": {"lstm": 0.0, "trans": 0.0, "tcn": 0.0} if model == "all" else None
        })

    df = pd.read_csv(path)
    if df.empty:
        return jsonify({
            "model": model,
            "items": [],
            "totals": {"overall_pnl": 0.0, "realized_pnl": 0.0, "open_pnl": 0.0, "invested_usd": 0.0},
            "totals_by_model": {"lstm": 0.0, "trans": 0.0, "tcn": 0.0} if model == "all" else None
        })

    # convert numeric
    for col in ["net_pnl_usd", "gross_pnl_usd", "invested_usd"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["status"] = df["status"].fillna("OPEN").astype(str).str.strip().str.upper()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
    df["model"] = df["model"].fillna("").astype(str).str.strip().str.lower()

    # 🔍 apply filters
    if status and status.lower() != "all":
        df = df.loc[df["status"] == status.upper()]

    if ticker and ticker.upper() != "ALL":
        df = df.loc[df["ticker"] == ticker.upper()]

    if model and model.lower() != "all":
        df = df.loc[df["model"] == model.lower()]

    # 🔄 refresh PnL + ราคาล่าสุด
    for i, row in df.iterrows():
        if str(row.get("status", "")).upper() == "OPEN":
            row = _calc_realtime_pnl(row.to_dict())
            df.loc[i, "gross_pnl_usd"] = row["gross_pnl_usd"]
            df.loc[i, "net_pnl_usd"] = row["net_pnl_usd"]
            df.loc[i, "return_pct"] = row["return_pct"]
            df.loc[i, "current_price"] = row["px_ref"]
            df.loc[i, "current_date"] = row["px_ref_date"]

    # ✅ คำนวณ totals หลัง refresh
    overall = clean_num(df["net_pnl_usd"].sum())
    realized = clean_num(df.loc[df["status"] == "CLOSED", "net_pnl_usd"].sum())
    openp = clean_num(df.loc[df["status"] == "OPEN", "net_pnl_usd"].sum())
    invested = clean_num(df["invested_usd"].sum())

    totals_by_model = None
    if model == "all":
        totals_by_model = {k: float(df.loc[df["model"] == k, "net_pnl_usd"].sum()) for k in ["lstm", "trans", "tcn"]}

    # ✅ สร้าง items หลังอัปเดต df
    items = df.fillna("").sort_values("logged_at", ascending=False).to_dict(orient="records")

    return jsonify({
        "model": model,
        "status": status,
        "ticker": ticker,
        "items": items,
        "totals": {
            "overall_pnl": overall,
            "realized_pnl": realized,
            "open_pnl": openp,
            "invested_usd": invested
        },
        "totals_by_model": totals_by_model
    })
# ─────────────────────────────────────────────────────────
# Auto-trade on startup (enter for all models×tickers, skip duplicates)
# ─────────────────────────────────────────────────────────
def bootstrap_autotrade():
    models = ["lstm","trans","tcn"]
    for m in models:
        for t in TICKERS:
            try:
                if _has_open_position(t, m):
                    continue

                dates, prices = _load_prediction_series(t, m)
                current_price, current_date = _load_current_price(t)
                if not prices:
                    continue

                idx_max = int(np.argmax(prices))
                pred_max = float(prices[idx_max])
                up_edge = (pred_max/current_price - 1.0) * 100.0
                if up_edge < DEFAULT_EDGE_PCT:
                    continue

                sim = simulate_buy_hold_with_today(
                    dates, prices, current_date,
                    start_capital=DEFAULT_START_CAPITAL,
                    notional=DEFAULT_NOTIONAL,
                    fee_bps=DEFAULT_FEE_BPS,
                    integer_shares=DEFAULT_INTEGER_SHARES,
                    target_idx=idx_max,
                    profit_to_peak_pct=DEFAULT_PROFIT_TO_PEAK_PCT
                )

                if sim.get("position",{}).get("open", True):
                    _log_open(sim, t, m, DEFAULT_START_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_EDGE_PCT)
                else:
                    _log_closed(sim, t, m, DEFAULT_START_CAPITAL, DEFAULT_FEE_BPS, DEFAULT_EDGE_PCT)

            except Exception as e:
                print(f"[autotrade] {t}/{m} -> {e}")



