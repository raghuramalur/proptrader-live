"""
proptrader_signal.py — PropTrader Paper Trading Engine
=======================================================
Runs every 4H on Mon-Fri via GitHub Actions.

POSITION SAFETY RULES (learned from RAAM):
  1. Reads paper_positions.json BEFORE generating any signal
  2. Will NOT enter a new trade if position already open for that asset
  3. Tracks every open position as state — no re-entry bug possible
  4. Closes positions when stop/target/time-stop hit, THEN re-enables signals

DATA:
  Gold (XAUUSD): TwelveData  XAU/USD  4H
  Euro (EURUSD): TwelveData  EUR/USD  4H

SIGNAL (Donchian Volatility Breakout, both directions):
  LONG:  close > N-bar high AND ATR > rolling median ATR
  SHORT: close < N-bar low  AND ATR > rolling median ATR
  Stop:  K_stop × ATR from entry
  Target: K_target × ATR from entry
  Time stop: 90 bars (~15 days)
"""

import os, json, csv, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY        = os.environ["TWELVE_DATA_API_KEY"]
POSITION_FILE  = "paper_positions.json"
TRADE_LOG_FILE = "paper_trade_log.csv"
DASHBOARD_FILE = "proptrader_dashboard.json"
SIZING_FILE    = "data/stage_3/position_sizing.json"

HIST_BARS      = 200   # enough for channel_n=50 + ATR_n=50 + buffer
MAX_HOLD_BARS  = 90    # time stop: ~15 trading days

# ── LOAD POSITION SIZING FROM STAGE 4 ────────────────────────────────────────
with open(SIZING_FILE) as f:
    sizing = json.load(f)

# Asset config — params come from Stage 2, lots from Stage 4
# Load from positions_by_asset (keyed as "Gold"/"Euro", not "Gold_donchian_vbo")
# Takes the primary (first) strategy per asset — highest OOS Sortino from Stage 2
ASSETS = {}
for asset_name, strategy_list in sizing.get("positions_by_asset", {}).items():
    if not strategy_list:
        continue
    primary = strategy_list[0]   # primary = index 0 = best OOS Sortino
    ASSETS[asset_name] = {
        "symbol":    primary["symbol"],
        "lot_size":  primary["lot_size"],
        "k_stop":    primary["params"]["k_stop"],
        "k_target":  primary["params"]["k_target"],
        "channel_n": primary["params"]["channel_n"],
        "atr_n":     primary["params"]["atr_n"],
        "strategy":  primary["strategy"],
    }

# TwelveData symbol mapping
TD_SYMBOLS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "BTCUSD": "BTC/USD",
    "NVDA":   "NVDA",
    "NAS100": "QQQ",
}

ACCOUNT_SIZE   = sizing["account_size"]
OPTIMAL_RISK   = sizing["optimal_risk"]
DAILY_DD_LIMIT = ACCOUNT_SIZE * 0.05   # $250
MAX_DD_LIMIT   = ACCOUNT_SIZE * 0.10   # $500

print(f"{'='*65}")
print(f"PROPTRADER SIGNAL ENGINE | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Assets: {list(ASSETS.keys())} | Risk: {OPTIMAL_RISK*100:.1f}%")
print(f"{'='*65}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def strip_tz(ts):
    """Ensure UTC-naive Timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def fetch_bars(symbol, bars=HIST_BARS):
    """Fetch 4H OHLCV from TwelveData. Returns clean DataFrame."""
    td_sym = TD_SYMBOLS.get(symbol, symbol)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     td_sym,
        "interval":   "4h",
        "outputsize": bars,
        "apikey":     API_KEY,
        "format":     "JSON",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            data = r.json()
            if data.get("status") == "error":
                print(f"  [WARN] TwelveData {symbol}: {data.get('message')}")
                if "Rate limit" in data.get("message", ""):
                    time.sleep(70)
                    continue
                return None
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = df["datetime"].apply(
                    lambda x: strip_tz(x)  # UTC naive
                )
                df = df.set_index("datetime").sort_index()
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col].astype(float)
                if "volume" in df.columns:
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
                return df
        except Exception as e:
            print(f"  [WARN] {symbol} attempt {attempt+1}: {e}")
            time.sleep(5)
    return None


def compute_signals(df, channel_n, atr_n, k_stop, k_target):
    """
    Donchian Volatility Breakout signal on a bar DataFrame.
    Returns last-bar signal dict: side (+1/-1/0), entry, stop, target
    """
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_n).mean()

    ch_high = d["high"].rolling(channel_n).max().shift(1)
    ch_low  = d["low"].rolling(channel_n).min().shift(1)

    atr_med   = atr.rolling(50).median()
    vol_active = atr > atr_med

    long_entry  = (d["close"] > ch_high) & vol_active
    short_entry = (d["close"] < ch_low)  & vol_active

    # Debounce: don't re-signal on same condition consecutive bars
    long_entry  = long_entry  & ~long_entry.shift(1).fillna(False)
    short_entry = short_entry & ~short_entry.shift(1).fillna(False)

    last = d.iloc[-1]
    last_atr = atr.iloc[-1]
    close    = float(last["close"])

    if pd.isna(last_atr) or last_atr <= 0:
        return {"side": 0}

    if long_entry.iloc[-1]:
        return {
            "side":         +1,
            "direction":    "LONG",
            "entry_price":  close,
            "stop_price":   close - k_stop  * last_atr,
            "target_price": close + k_target * last_atr,
            "atr":          float(last_atr),
        }
    if short_entry.iloc[-1]:
        return {
            "side":         -1,
            "direction":    "SHORT",
            "entry_price":  close,
            "stop_price":   close + k_stop  * last_atr,
            "target_price": close - k_target * last_atr,
            "atr":          float(last_atr),
        }
    return {"side": 0}


# ── LOAD STATE ────────────────────────────────────────────────────────────────
# paper_positions.json: {asset_name: {side, entry_price, stop, target, lot,
#                                      entry_time, bars_held}}
if os.path.exists(POSITION_FILE):
    with open(POSITION_FILE) as f:
        open_positions = json.load(f)
else:
    open_positions = {}

print(f"\nOpen positions loaded: {list(open_positions.keys()) or 'NONE'}")
for asset, pos in open_positions.items():
    print(f"  {asset}: {pos['direction']} {pos['lot_size']}L @ "
          f"{pos['entry_price']:.4f} | "
          f"stop={pos['stop_price']:.4f} target={pos['target_price']:.4f} "
          f"| bars_held={pos.get('bars_held',0)}")

# ── INIT TRADE LOG ────────────────────────────────────────────────────────────
log_headers = ["date","asset","direction","entry_time","exit_time",
               "entry_price","exit_price","lot_size","pnl_pct",
               "pnl_dollar","exit_reason","bars_held"]

if not os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(log_headers)

completed_trades = []

# ── MAIN LOOP: check each asset ───────────────────────────────────────────────
price_cache = {}    # for dashboard P&L

for asset_name, cfg in ASSETS.items():
    symbol    = cfg["symbol"]
    lot_size  = cfg["lot_size"]
    channel_n = cfg["channel_n"]
    atr_n     = cfg["atr_n"]
    k_stop    = cfg["k_stop"]
    k_target  = cfg["k_target"]

    print(f"\n[{asset_name} / {symbol}]")

    df = fetch_bars(symbol)
    time.sleep(8)   # TwelveData free: 8 req/min

    if df is None or len(df) < max(channel_n, atr_n) + 10:
        print(f"  ✗ Insufficient data ({len(df) if df is not None else 0} bars)")
        continue

    current_price = float(df["close"].iloc[-1])
    current_high  = float(df["high"].iloc[-1])
    current_low   = float(df["low"].iloc[-1])
    price_cache[asset_name] = current_price
    print(f"  Current price: {current_price:.4f}")

    # ── CHECK EXISTING POSITION FIRST ─────────────────────────────────────────
    if asset_name in open_positions:
        pos       = open_positions[asset_name]
        direction = pos["direction"]
        bars_held = pos.get("bars_held", 0) + 1
        open_positions[asset_name]["bars_held"] = bars_held

        exit_price  = None
        exit_reason = None

        if direction == "LONG":
            if current_low <= pos["stop_price"]:
                exit_price, exit_reason = pos["stop_price"], "STOP"
            elif current_high >= pos["target_price"]:
                exit_price, exit_reason = pos["target_price"], "TARGET"
            elif bars_held >= MAX_HOLD_BARS:
                exit_price, exit_reason = current_price, "TIME"
        else:  # SHORT
            if current_high >= pos["stop_price"]:
                exit_price, exit_reason = pos["stop_price"], "STOP"
            elif current_low <= pos["target_price"]:
                exit_price, exit_reason = pos["target_price"], "TARGET"
            elif bars_held >= MAX_HOLD_BARS:
                exit_price, exit_reason = current_price, "TIME"

        if exit_price is not None:
            entry_price = pos["entry_price"]
            side_mult   = +1 if direction == "LONG" else -1
            pnl_pct     = (exit_price - entry_price) / entry_price * 100 * side_mult

            # Dollar P&L depends on asset pip value
            if symbol == "XAUUSD":
                pnl_dollar = (exit_price - entry_price) * side_mult * lot_size * 100
            elif symbol == "EURUSD":
                pnl_dollar = (exit_price - entry_price) * side_mult * lot_size * 100000
            else:
                pnl_dollar = pnl_pct / 100 * ACCOUNT_SIZE * OPTIMAL_RISK

            completed_trades.append({
                "date":         date.today().isoformat(),
                "asset":        asset_name,
                "direction":    direction,
                "entry_time":   pos["entry_time"],
                "exit_time":    datetime.utcnow().isoformat(),
                "entry_price":  entry_price,
                "exit_price":   exit_price,
                "lot_size":     lot_size,
                "pnl_pct":      round(pnl_pct, 4),
                "pnl_dollar":   round(pnl_dollar, 2),
                "exit_reason":  exit_reason,
                "bars_held":    bars_held,
            })

            print(f"  → CLOSED {direction} @ {exit_price:.4f} | "
                  f"P&L: ${pnl_dollar:+.2f} ({pnl_pct:+.2f}%) | {exit_reason}")
            del open_positions[asset_name]   # position closed — slot is now free
        else:
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
            pnl_pct *= (+1 if direction == "LONG" else -1)
            print(f"  → HOLDING {direction} @ {pos['entry_price']:.4f} | "
                  f"Unrealised: {pnl_pct:+.2f}% | bars={bars_held}")
        continue   # don't check for new signal while position is open

    # ── NO OPEN POSITION — CHECK FOR ENTRY ────────────────────────────────────
    signal = compute_signals(df, channel_n, atr_n, k_stop, k_target)

    if signal["side"] == 0:
        print(f"  → No signal this bar")
    else:
        direction   = signal["direction"]
        entry_price = signal["entry_price"]
        stop_price  = signal["stop_price"]
        target_price = signal["target_price"]

        # Dynamic lot: use LIVE ATR so dollar risk stays at target regardless of ATR drift
        live_atr = signal["atr"]
        if symbol == "XAUUSD":
            dynamic_lot = (ACCOUNT_SIZE * OPTIMAL_RISK) / (live_atr * k_stop * 100)
        elif symbol == "EURUSD":
            dynamic_lot = (ACCOUNT_SIZE * OPTIMAL_RISK) / ((live_atr * k_stop / 0.0001) * 10)
        else:
            dynamic_lot = lot_size
        dynamic_lot = max(round(round(dynamic_lot / 0.01) * 0.01, 2), 0.01)

        print(f"  → *** SIGNAL: {direction} ***")
        print(f"     Entry:  {entry_price:.4f}")
        print(f"     Stop:   {stop_price:.4f}  (dist={abs(entry_price-stop_price):.4f})")
        print(f"     Target: {target_price:.4f}  (dist={abs(entry_price-target_price):.4f})")
        print(f"     Lot:    {dynamic_lot} (live ATR={live_atr:.4f}) | Risk: ~${ACCOUNT_SIZE*OPTIMAL_RISK:.2f}")

        open_positions[asset_name] = {
            "symbol":       symbol,
            "direction":    direction,
            "side":         signal["side"],
            "entry_price":  entry_price,
            "stop_price":   stop_price,
            "target_price": target_price,
            "lot_size":     dynamic_lot,
            "entry_time":   datetime.utcnow().isoformat(),
            "bars_held":    0,
            "atr_at_entry": live_atr,
        }

# ── WRITE COMPLETED TRADES TO LOG ─────────────────────────────────────────────
if completed_trades:
    with open(TRADE_LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_headers)
        for trade in completed_trades:
            w.writerow(trade)
    print(f"\n✓ {len(completed_trades)} trade(s) logged → {TRADE_LOG_FILE}")

# ── SAVE POSITION STATE ────────────────────────────────────────────────────────
with open(POSITION_FILE, "w") as f:
    json.dump(open_positions, f, indent=2)
print(f"✓ Position state saved → {POSITION_FILE}")

# ── COMPUTE PAPER P&L FROM TRADE LOG ──────────────────────────────────────────
total_pnl = 0.0
total_trades = wins = losses = 0

if os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if row.get("pnl_dollar"):
            pnl = float(row["pnl_dollar"])
            total_pnl += pnl
            total_trades += 1
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1

win_rate = wins / total_trades if total_trades > 0 else 0

# Unrealised P&L from open positions
unrealised = 0.0
for asset_name, pos in open_positions.items():
    price = price_cache.get(asset_name, pos["entry_price"])
    entry = pos["entry_price"]
    side  = pos["side"]
    sym   = pos["symbol"]
    lot   = pos["lot_size"]
    if sym == "XAUUSD":
        unrealised += (price - entry) * side * lot * 100
    elif sym == "EURUSD":
        unrealised += (price - entry) * side * lot * 100000

# ── BUILD DASHBOARD JSON ──────────────────────────────────────────────────────
# Load recent trades for dashboard display
recent_trades = []
if os.path.exists(TRADE_LOG_FILE):
    with open(TRADE_LOG_FILE) as f:
        recent_trades = list(csv.DictReader(f))[-20:]
dashboard = {
    "generated_at":     datetime.utcnow().isoformat() + "Z",
    "recent_trades":    recent_trades,
    "account_size":     ACCOUNT_SIZE,
    "optimal_risk_pct": OPTIMAL_RISK * 100,
    "realised_pnl":     round(total_pnl, 2),
    "unrealised_pnl":   round(unrealised, 2),
    "total_pnl":        round(total_pnl + unrealised, 2),
    "total_trades":     total_trades,
    "wins":             wins,
    "losses":           losses,
    "win_rate":         round(win_rate, 4),
    "open_positions":   open_positions,
    "last_check":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    "assets":           list(ASSETS.keys()),
}

with open(DASHBOARD_FILE, "w") as f:
    json.dump(dashboard, f, indent=2)

print(f"\n{'='*65}")
print(f"PAPER TRADING SUMMARY")
print(f"  Realised P&L:   ${total_pnl:+.2f}")
print(f"  Unrealised P&L: ${unrealised:+.2f}")
print(f"  Total P&L:      ${total_pnl+unrealised:+.2f}")
print(f"  Trades:         {total_trades} | W={wins} L={losses} WR={win_rate:.1%}")
print(f"  Open positions: {list(open_positions.keys()) or 'NONE'}")
print(f"{'='*65}")
