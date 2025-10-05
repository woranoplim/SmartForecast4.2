import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import holidays

def is_market_open():
    today = pd.Timestamp.today().normalize()
    return today.weekday() < 5 and today not in holidays.US()

def generate_dataset_for_ticker(ticker, save_dir=f"forecast/datasets"):

    start_date = (datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    # end_date = "2025-10-04"
    # --- โหลดข้อมูลหุ้นและ VIX ---
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)[["Close"]].rename(columns={"Close": "VIX_Close"})

    # --- เติม missing + join ---
    stock = stock.ffill().bfill()
    vix = vix.ffill().bfill()
    data = stock.join(vix, how="left").ffill().bfill()

    # --- สร้างฟีเจอร์ทางเทคนิค ---
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

# ---------- MAIN ----------
if __name__ == "__main__":
    # if not is_market_open():
    #     print("⛔ ตลาดปิดวันนี้ ไม่โหลดข้อมูลใหม่")
    #     exit()

    tickers = ["AMZN", "TSLA", "GOOGL","META","AAPL","NVDA","MSFT"]
    for ticker in tickers:
        generate_dataset_for_ticker(ticker)



