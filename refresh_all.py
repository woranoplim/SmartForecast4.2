# refresh_all.py
"""
Run only the forecasting scripts located inside a folder (default: ./forecast).
Does NOT touch datasets.

Usage:
  python refresh_all.py
  python refresh_all.py --tickers AAPL,MSFT
  python refresh_all.py --forecast-dir forecast
  python refresh_all.py --stop-on-fail
  python refresh_all.py --dry-run
"""

import sys
import os
import glob
import argparse
import subprocess
from time import perf_counter
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TICKERS = ["AAPL","AMZN","GOOGL","META","MSFT","NVDA","TSLA"]

def parse_args():
    p = argparse.ArgumentParser(description="Run forecasting scripts only (update_forecast*.py) inside a folder.")
    p.add_argument("--tickers", "-t",
                   help="Comma-separated tickers to run (e.g., AAPL,MSFT). If omitted, run all update_forecast*.py found.")
    p.add_argument("--forecast-dir", "-f",
                   default="forecast",
                   help="Directory that contains update_forecast*.py (default: forecast)")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Stop immediately when any script fails (default: continue).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only show what would be executed.")
    return p.parse_args()

def ensure_abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)

def discover_scripts_from_dir(dir_path):
    """Find all update_forecast*.py under given dir."""
    pat = os.path.join(dir_path, "update_forecast*.py")
    paths = sorted(glob.glob(pat))
    if not paths:
        return []
    # sort by our preferred ticker order first
    order_index = {t: i for i, t in enumerate(DEFAULT_TICKERS)}
    def sort_key(p):
        name = os.path.basename(p)  # update_forecastAAPL.py
        ticker = name.replace("update_forecast", "").replace(".py", "").replace("_", "").upper()
        return (order_index.get(ticker, 999), name.lower())
    return sorted(paths, key=sort_key)

def scripts_for_tickers(tickers, dir_path):
    """Return script paths for selected tickers in the dir."""
    out = []
    for t in tickers:
        # support two common patterns: update_forecastAAPL.py or update_forecast_AAPL.py
        candidates = [
            os.path.join(dir_path, f"update_forecast{t}.py"),
            os.path.join(dir_path, f"update_forecast_{t}.py"),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path:
            out.append(path)
        else:
            print(f"⚠️  Missing script for {t} in {dir_path}: {os.path.basename(candidates[0])}")
    return out

def run(cmd, cwd):
    """Run command and stream output; return (ok, duration_seconds)."""
    print(f"\n▶ Running (cwd={cwd}): {' '.join(cmd)}")
    t0 = perf_counter()
    proc = subprocess.run(cmd, cwd=cwd)
    dt = perf_counter() - t0
    ok = (proc.returncode == 0)
    print(f"◀ Done ({'OK' if ok else 'FAIL'} in {dt:.2f}s)")
    return ok, dt

def main():
    args = parse_args()
    py = sys.executable

    forecast_dir = ensure_abs(args.forecast_dir)  # e.g., <project>/forecast
    if not os.path.isdir(forecast_dir):
        print(f"⛔ forecast-dir not found: {forecast_dir}")
        return

    print("="*70)
    print(f"SmartForecast — forecast refresh only @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Forecast dir: {forecast_dir}")
    print("="*70)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        scripts = scripts_for_tickers(tickers, forecast_dir)
    else:
        scripts = discover_scripts_from_dir(forecast_dir)

    if not scripts:
        print(f"⚠️  No update_forecast*.py scripts found under {forecast_dir}.")
        return

    print("Will run:")
    for p in scripts:
        print(f"  - {os.path.basename(p)}")
    if args.dry_run:
        print("\n(dry-run) Nothing executed.")
        return

    results = []
    # IMPORTANT: set cwd to forecast_dir so relative paths inside scripts work,
    # e.g. writing to ./json/, ./datasets/ from within forecast/
    for script in scripts:
        ok, dt = run([py, os.path.basename(script)], cwd=forecast_dir)
        results.append((os.path.basename(script), ok, dt))
        if not ok and args.stop_on_fail:
            print("\n⛔ Stopping due to failure (--stop-on-fail).")
            break

    print("\n" + "-"*70)
    print("Summary:")
    ok_total = sum(1 for _, ok, _ in results if ok)
    for name, ok, dt in results:
        print(f"  - {name:<28} {'OK' if ok else 'FAIL':<5} ({dt:.2f}s)")
    print("-"*70)
    print(f"Completed: {ok_total}/{len(results)} scripts OK")
    print("="*70)

if __name__ == "__main__":
    main()
