import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import holidays
from io import StringIO

# ---------------------------------------------------------------------
# 🌤️ Session + Header ปลอมเป็น Browser (กันโดน Yahoo บล็อก)
# ---------------------------------------------------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9"
})


# ---------------------------------------------------------------------
# 📅 เช็กตลาดเปิด
# ---------------------------------------------------------------------
def is_market_open():
    today = pd.Timestamp.today().normalize()
    return today.weekday() < 5 and today not in holidays.US()


# ---------------------------------------------------------------------
# 📈 ดาวน์โหลดข้อมูลหุ้น + VIX (มี retry + fallback)
# ---------------------------------------------------------------------
def safe_download(ticker, start, end, retries=3):
    for i in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                session=session
            )
            if not df.empty:
                print(f"✅ {ticker}: {len(df)} rows (try {i+1})")
                return df
        except Exception as e:
            print(f"⚠️ Retry {i+1}/{retries} for {ticker}: {e}")
        time.sleep(2)

    # 🔁 fallback: ดึงผ่าน Yahoo CSV URL
    try:
        print(f"🌐 Fallback mode for {ticker}")
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1=0&period2=9999999999&interval=1d&events=history"
        )
        r = session.get(url, timeout=10)
        if r.status_code == 200 and len(r.text) > 100:
            df = pd.read_csv(StringIO(r.text))
            df = df.set_index("Date")
            print(f"✅ Fallback OK for {ticker} ({len(df)} rows)")
            return df
        else:
            print(f"❌ Fallback failed ({r.status_code}): {r.text[:100]}")
    except Exception as e:
        print(f"❌ Fallback error for {ticker}: {e}")

    print(f"⛔ Failed to get data for {ticker} after {retries} retries.")
    return pd.DataFrame()


# ---------------------------------------------------------------------
# 🧠 ฟังก์ชันหลัก: สร้าง Dataset
# ---------------------------------------------------------------------
def generate_dataset_for_ticker(ticker, save_dir="forecast/datasets"):
    start_date = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # --- โหลดข้อมูลหุ้นและ VIX ---
    stock = safe_download(ticker, start_date, end_date)
    vix = safe_download("^VIX", start_date, end_date)[["Close"]].rename(columns={"Close": "VIX_Close"})

    if stock.empty or vix.empty:
        print(f"⚠️ Skipping {ticker} because data is empty.")
        return

    # --- เติม missing + join ---
    stock = stock.ffill().bfill()
    vix = vix.ffill().bfill()
    data = stock.join(vix, how="left").ffill().bfill()

    # --- ฟีเจอร์เทคนิค ---
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    for w in [10, 20, 30, 45]:
        data[f"MA_{w}"] = data["Close"].rolling(window=w).mean()
    sd_window = 20
    data[f"SD_{sd_window}"] = data["Close"].rolling(window=sd_window).std()
    data[f"Upper_{sd_window}"] = data[f"MA_{sd_window}"] + data[f"SD_{sd_window}"]
    data[f"Lower_{sd_window}"] = data[f"MA_{sd_window}"] - data[f"SD_{sd_window}"]

    # --- ทำความสะอาด + reset index ---
    data = data.dropna().reset_index()

    # --- เลือกฟีเจอร์ที่ใช้จริง ---
    features = [
        "Date", "Close", "High", "Low", "Open", "Volume", "VIX_Close",
        "EMA_12", "EMA_26", "RSI", "MACD",
        "MA_10", "MA_20", "MA_30", "MA_45",
        "SD_20", "Upper_20", "Lower_20"
    ]
    dataset = data[features]

    # --- บันทึกไฟล์ ---
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{ticker}_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"✅ บันทึกแล้ว: {output_path}")


# ---------------------------------------------------------------------
# 🚀 MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":

    tickers = ["AMZN", "TSLA", "GOOGL", "META", "AAPL", "NVDA", "MSFT"]
    for ticker in tickers:
        generate_dataset_for_ticker(ticker)
        time.sleep(3)  # ⏱ เว้นช่วงกัน rate-limit
