import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import holidays


# 🌍 สร้าง session ที่ปลอม header เป็น browser
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
})


def is_market_open():
    today = pd.Timestamp.today().normalize()
    return today.weekday() < 5 and today not in holidays.US()


# 📥 ฟังก์ชันโหลดข้อมูลแบบปลอดภัย (กันพังบน GitHub Actions)
def safe_download(ticker, start, end, retries=3):
    for i in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                session=session,
                timeout=15
            )
            if not df.empty:
                print(f"✅ {ticker}: {len(df)} rows (try {i+1})")
                return df
        except Exception as e:
            print(f"⚠️ Retry {i+1}/{retries} for {ticker}: {e}")
        time.sleep(2)

    # fallback → ใช้ .history() (ดึง crumb/cookie อัตโนมัติ)
    try:
        print(f"🌐 Fallback mode for {ticker} using .history() ...")
        ticker_obj = yf.Ticker(ticker, session=session)
        df = ticker_obj.history(period="5y", auto_adjust=False)
        if not df.empty:
            print(f"✅ {ticker}: {len(df)} rows via .history() fallback")
            return df
    except Exception as e:
        print(f"❌ Fallback failed for {ticker}: {e}")

    print(f"⛔ Failed to get data for {ticker} after all retries.")
    return pd.DataFrame()


def generate_dataset_for_ticker(ticker, save_dir="forecast/datasets"):
    start_date = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # โหลดหุ้นหลัก
    stock = safe_download(ticker, start_date, end_date)

    # โหลด VIX
    vix = safe_download("^VIX", start_date, end_date)
    if vix.empty or "Close" not in vix.columns:
        print("⚠️ No VIX data available — filling with NaN column instead.")
        stock["VIX_Close"] = np.nan
    else:
        vix = vix[["Close"]].rename(columns={"Close": "VIX_Close"})
        stock = stock.join(vix, how="left").ffill().bfill()

    # ถ้าไม่มีข้อมูลหุ้น ให้ข้าม
    if stock.empty:
        print(f"⚠️ Skipping {ticker} (no data downloaded).")
        return

    # 🧮 คำนวณฟีเจอร์ตามเดิม (ไม่เปลี่ยน)
    data = stock.ffill().bfill()
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

    data = data.dropna().reset_index()

    features = [
        "Date", "Close", "High", "Low", "Open", "Volume", "VIX_Close",
        "EMA_12", "EMA_26", "RSI", "MACD",
        "MA_10", "MA_20", "MA_30", "MA_45",
        "SD_20", "Upper_20", "Lower_20"
    ]
    dataset = data[features]

    # 💾 บันทึกข้อมูลตาม path เดิม
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{ticker}_dataset.csv")
    dataset.to_csv(output_path, index=False)
    print(f"✅ บันทึกแล้ว: {output_path}")


# ─────────────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tickers = ["AMZN", "TSLA", "GOOGL", "META", "AAPL", "NVDA", "MSFT"]

    for ticker in tickers:
        generate_dataset_for_ticker(ticker)
        time.sleep(3)  # ป้องกัน rate-limit
