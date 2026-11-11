"""Moving average crossover bot for an Alpaca paper trading account.

For each stock in config.json:
  1. Fetch the last N days of daily prices from yfinance.
  2. Compute the short and long simple moving averages.
  3. Compare the last two days. If the short MA moved from <= long MA to > long MA,
     that's a bullish crossover -> BUY. If it moved from >= to <, bearish -> SELL.
  4. Place a market order on Alpaca paper trading (sells first, then buys).

Usage:
  python bot.py                       # check signals and place orders
  python bot.py --report              # same, then print a summary of what it did
  python bot.py --preview --report    # print what the next run will do, no orders
  python bot.py --dry-run             # signals only, no Alpaca connection, no orders
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import yfinance as yf
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),
              # ~1,000 lines per run; keep the log at 5 MB plus three old copies.
              RotatingFileHandler(BASE_DIR / "bot.log", maxBytes=5_000_000, backupCount=3)],
)
log = logging.getLogger("bot")

LOG_DIR = BASE_DIR / "logs"
RUN_LOG_DAYS = 90


def start_run_log(mode):
    """Give each live/preview run its own complete log file in logs/, alongside
    the rolling bot.log. Files older than RUN_LOG_DAYS are deleted."""
    LOG_DIR.mkdir(exist_ok=True)
    cutoff = time.time() - RUN_LOG_DAYS * 86400
    for old in LOG_DIR.glob("*.log"):
        if old.stat().st_mtime < cutoff:
            old.unlink()
    path = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{mode}.log"
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    # A crash's traceback normally only goes to the terminal; log it as well.
    sys.excepthook = lambda *exc: log.critical("Run crashed", exc_info=exc)
    log.info("Run log: %s", path)
    return path


FINAL_ORDER_STATES = {"filled", "canceled", "expired", "rejected", "done_for_day"}


def load_config():
    with open(BASE_DIR / "config.json") as f:
        return json.load(f)


def rate_limit_wait(headers, default=3.0, cap=60.0):
    """Seconds a 429 response asks us to wait: Retry-After (seconds or an HTTP
    date), else X-RateLimit-Reset (Unix time the window resets), else default."""
    value = headers.get("Retry-After")
    if value:
        try:
            return min(cap, max(0.0, float(value)))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                return min(cap, max(0.0, (when - datetime.now(timezone.utc)).total_seconds()))
            except (TypeError, ValueError):
                pass
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        try:
            return min(cap, max(0.0, float(reset) - time.time()))
        except ValueError:
            pass
    return default


def _respect_rate_limit(response, *args, **kwargs):
    # requests response hook: runs before alpaca-py's own retry on a 429, so we
    # wait as long as the server asks instead of alpaca-py's fixed few seconds.
    if response.status_code == 429:
        wait = rate_limit_wait(response.headers)
        log.warning("Alpaca rate limit hit; waiting %.1fs as its headers ask, then retrying", wait)
        time.sleep(wait)
    return response


def get_trading_client():
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret or "your_paper" in key or "your_paper" in secret:
        sys.exit("Missing Alpaca credentials. Fill in ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    # paper=True is hard-coded so this bot can never touch a live account.
    # alpaca-py retries 429s itself; the hook sets the wait from the response headers.
    # TradingClient doesn't take retry settings, so set alpaca-py's (version-pinned)
    # attributes directly: 5 retries, 1s extra wait on top of the header's wait.
    client = TradingClient(key, secret, paper=True)
    client._retry, client._retry_wait = 5, 1
    client._session.hooks["response"].append(_respect_rate_limit)
    return client


class NotEnoughHistory(ValueError):
    """Recently listed stock without enough days for the long moving average yet."""


def _yf_symbol(symbol):
    # Config uses Alpaca's share-class format (BRK.B); yfinance wants BRK-B.
    return symbol.replace(".", "-")


def _clean_closes(symbol, close):
    if isinstance(close, pd.DataFrame):  # yfinance may return a one-column frame
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        raise ValueError(f"No price data returned for {symbol}")

    # Before the 4 pm ET close, today's bar is still moving. Drop it so signals
    # only use completed days (e.g. a run at the open acts on yesterday's close).
    now_et = pd.Timestamp.now(tz="America/New_York")
    if close.index[-1].date() == now_et.date() and now_et.hour < 16:
        close = close.iloc[:-1]
    return close


def fetch_closes(symbol, lookback_days):
    df = yf.download(_yf_symbol(symbol), period=f"{lookback_days}d", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {symbol}")
    return _clean_closes(symbol, df["Close"])


def fetch_all_closes(symbols, lookback_days, batch_size=200, retry_waits=(15, 45)):
    """Download closing prices in batches; symbols that still fail are left out
    and fetched one at a time later.

    yfinance doesn't expose Yahoo's response headers, so there's no Retry-After to
    honor. Instead, when a large share of a batch comes back empty (Yahoo's usual
    sign of throttling), wait and retry just the missing symbols."""
    closes = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        for attempt, wait in enumerate((0, *retry_waits)):
            todo = [s for s in batch if s not in closes]
            # A few gaps are normal (no data for that symbol); many mean throttling.
            if not todo or (attempt and len(todo) <= len(batch) * 0.1):
                break
            if wait:
                log.warning("No prices for %d of %d stocks in this batch (likely rate limited); "
                            "waiting %ss and retrying them", len(todo), len(batch), wait)
                time.sleep(wait)
            try:
                df = yf.download([_yf_symbol(s) for s in todo], period=f"{lookback_days}d",
                                 interval="1d", auto_adjust=True, progress=False,
                                 threads=True, group_by="ticker")
            except Exception as e:
                log.warning("Batch download failed (%s)", e)
                continue
            for s in todo:
                try:
                    closes[s] = _clean_closes(s, df[_yf_symbol(s)]["Close"])
                except Exception:
                    pass
    return closes


def detect_crossover(closes, short_window, long_window):
    """Return ('buy' | 'sell' | None, details) based on the last two days."""
    if len(closes) < long_window + 1:
        raise NotEnoughHistory(f"only {len(closes)} days of prices, need {long_window + 1}")

    short_ma = closes.rolling(short_window).mean()
    long_ma = closes.rolling(long_window).mean()

    prev_short, prev_long = short_ma.iloc[-2], long_ma.iloc[-2]
    curr_short, curr_long = short_ma.iloc[-1], long_ma.iloc[-1]

    details = {
        "date": closes.index[-1].date(),
        "close": closes.iloc[-1],
        "prev_short": prev_short, "prev_long": prev_long,
        "curr_short": curr_short, "curr_long": curr_long,
    }

    if prev_short <= prev_long and curr_short > curr_long:
        return "buy", details
    if prev_short >= prev_long and curr_short < curr_long:
        return "sell", details
    return None, details


def evaluate(stock, all_closes, lookback_days):
    """Compute one stock's signal. Never raises; problems are recorded on the result."""
    symbol = stock["symbol"]
    short_w, long_w = stock["short_window"], stock["long_window"]
    r = {"symbol": symbol, "short_w": short_w, "long_w": long_w, "size": stock["trade_size"],
         "signal": None, "action": None, "qty": 0}
    try:
        closes = all_closes.get(symbol)
        if closes is None:
            closes = fetch_closes(symbol, lookback_days)
        signal, d = detect_crossover(closes, short_w, long_w)
        r.update(signal=signal, d=d)
        log.info("%s %s close=%.2f | MA%d %.2f -> %.2f | MA%d %.2f -> %.2f | signal=%s",
                 symbol, d["date"], d["close"],
                 short_w, d["prev_short"], d["curr_short"],
                 long_w, d["prev_long"], d["curr_long"],
                 signal or "none")
    except NotEnoughHistory as e:
        r.update(action="new_listing", error=str(e))
        log.info("%s: skipped, %s", symbol, e)
    except Exception as e:
        r.update(action="error", error=str(e))
        log.error("%s: %s", symbol, e)
    return r


def flag_stale(results):
    """Don't trade on a signal whose prices stop before the latest trading day
    seen across all stocks (e.g. Yahoo hasn't updated that symbol yet)."""
    dates = [r["d"]["date"] for r in results if r.get("d")]
    if not dates:
        return
    latest = max(dates)
    for r in results:
        if r["signal"] and r["action"] is None and r["d"]["date"] < latest:
            r.update(action="stale", error=f"latest price is from {r['d']['date']}, not {latest}")
            log.warning("%s: %s signal ignored, %s", r["symbol"], r["signal"], r["error"])


def place_order(client, symbol, side, qty, signal_date):
    # One order per symbol/side/day: Alpaca rejects a repeated client_order_id,
    # so running the bot twice on the same day can't double-trade.
    client_order_id = f"mac-{symbol}-{side.value}-{signal_date}"
    order = client.submit_order(MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    ))
    log.info("%s: %s order for %s shares submitted (id=%s, status=%s)",
             symbol, side.value.upper(), qty, order.id, order.status)
    return order


def trade(client, config, cash, holdings, results, live):
    """Turn signals into orders. Sells go first so their estimated proceeds count
    toward the buys. In preview (live=False) nothing is submitted."""
    max_price = config.get("max_share_price")
    max_margin = config.get("max_margin")
    # Running cash estimate: negative cash is money borrowed on margin. Orders are
    # priced at the last close, since the actual fill price isn't known yet.
    signals = [r for r in results if r["signal"] and r["action"] is None]
    for r in sorted(signals, key=lambda r: r["signal"] != "sell"):  # stable: config order kept
        symbol, d, side = r["symbol"], r["d"], r["signal"]
        if side == "sell":
            # Only sell shares we actually hold; never open a short position.
            qty = min(r["size"], holdings.get(symbol, 0.0))
            if qty <= 0:
                r["action"] = "skip"
                log.info("%s: sell signal but no shares held, skipping", symbol)
                continue
        else:
            qty = r["size"]
            if max_price and d["close"] > max_price:
                r["action"] = "skip_price"
                log.info("%s: buy signal but last close $%.2f is over the $%s limit, skipping",
                         symbol, d["close"], max_price)
                continue
            if max_margin is not None and cash - d["close"] * qty < -max_margin:
                r["action"] = "skip_margin"
                log.info("%s: buy signal but ~$%.2f would take margin past the $%s limit "
                         "(cash left $%.2f), skipping", symbol, d["close"] * qty, max_margin, cash)
                continue

        if live:
            try:
                order = place_order(client, symbol, OrderSide(side), qty, d["date"])
            except APIError as e:
                r.update(action="error", error=f"Alpaca rejected the order: {e}")
                log.error("%s: Alpaca rejected the order: %s", symbol, e)
                continue  # nothing was spent, so cash is unchanged
            r.update(order_id=str(order.id), status=order.status.value)
        r.update(action=side, qty=qty)
        cash += d["close"] * qty if side == "sell" else -d["close"] * qty


def wait_for_fills(client, results, timeout=90):
    """Poll submitted orders until they reach a final state or the timeout passes.
    One request per poll for all orders keeps us well under Alpaca's rate limit."""
    pending = {r["order_id"]: r for r in results if r.get("order_id")}
    if not pending:
        return
    if not client.get_clock().is_open:
        log.info("Market is closed; %d orders will fill at the next open", len(pending))
        return
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        time.sleep(5)
        try:
            orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
        except Exception as e:
            log.warning("Couldn't check order status (%s); retrying", e)
            continue
        for order in orders:
            r = pending.get(str(order.id))
            if r is None:
                continue
            r["status"] = order.status.value
            if order.filled_avg_price:
                r["fill_price"] = float(order.filled_avg_price)
            if r["status"] in FINAL_ORDER_STATES:
                del pending[str(order.id)]
    for r in pending.values():
        log.info("%s: order still %s after %ss", r["symbol"], r["status"], timeout)


def run(mode="live", report=False):
    """mode: 'live' places orders, 'preview' reads the account but places none,
    'dry' skips Alpaca entirely."""
    config = load_config()
    client = None if mode == "dry" else get_trading_client()

    acct, holdings = None, {}
    if client:
        acct = client.get_account()
        holdings = {p.symbol: float(p.qty) for p in client.get_all_positions()}
        log.info("Connected to Alpaca PAPER account %s | cash $%s | buying power $%s | %d positions",
                 acct.account_number, acct.cash, acct.buying_power, len(holdings))

    stocks = config["stocks"]
    all_closes = fetch_all_closes([s["symbol"] for s in stocks], config["lookback_days"])
    log.info("Downloaded prices for %d of %d stocks in batches", len(all_closes), len(stocks))

    results = [evaluate(stock, all_closes, config["lookback_days"]) for stock in stocks]
    flag_stale(results)

    if client:
        trade(client, config, float(acct.cash), holdings, results, live=(mode == "live"))
    if mode == "live":
        wait_for_fills(client, results)

    if report:
        acct = client.get_account()  # refresh cash after fills
        positions = client.get_all_positions()
        if mode == "live":
            title, body = run_report(acct, results, positions)
        else:
            title, body = preview_report(acct, results, positions, client.get_clock())
        return f"{title}\n\n{body}"


# ---------- reports ----------

def _gap_text(r):
    d = r["d"]
    gap = d["curr_short"] - d["curr_long"]
    where = "above" if gap > 0 else "below"
    return (f"{r['short_w']}-day {d['curr_short']:.2f} is {abs(gap):.2f} {where} "
            f"{r['long_w']}-day {d['curr_long']:.2f}")


def _signal_text(r):
    d = r["d"]
    direction = "above" if r["signal"] == "buy" else "below"
    return (f"close {d['close']:.2f}; {r['short_w']}-day crossed {direction} {r['long_w']}-day "
            f"({d['prev_short']:.2f} -> {d['curr_short']:.2f} vs "
            f"{d['prev_long']:.2f} -> {d['curr_long']:.2f})")


def _near_crossovers(results, limit=5):
    """Stocks without a signal whose averages are converging, closest first."""
    near = []
    for r in results:
        if r.get("d") is None or r["signal"]:
            continue
        d = r["d"]
        prev_gap = d["prev_short"] - d["prev_long"]
        gap = d["curr_short"] - d["curr_long"]
        if abs(gap) < abs(prev_gap):
            near.append((abs(gap) / d["close"], r))
    return [r for _, r in sorted(near, key=lambda x: x[0])[:limit]]


def _common_sections(results, positions):
    by_symbol = {r["symbol"]: r for r in results}
    lines = []

    def section(action, heading, fmt=lambda r: f"  {r['symbol']}"):
        rows = [r for r in results if r["action"] == action]
        if rows:
            lines.append(f"{heading} ({len(rows)}):")
            lines.extend(fmt(r) for r in rows)
            lines.append("")

    price = lambda r: f"  {r['symbol']} (last close ${r['d']['close']:,.2f})"
    section("skip", "Sell signals skipped, no shares held")
    section("skip_price", "Buy signals skipped, share price over the max_share_price limit", price)
    section("skip_margin", "Buy signals skipped, would go past the max_margin limit", price)
    section("stale", "Signals ignored, price data out of date", lambda r: f"  {r['symbol']}: {r['error']}")
    section("new_listing", "Skipped, too new for the long moving average")

    errors = [r for r in results if r["action"] == "error"]
    lines.append(f"Errors ({len(errors)}):" if errors else "Errors: none")
    lines += [f"  {r['symbol']}: {r['error']}" for r in errors]
    lines.append("")

    lines.append(f"Holdings ({len(positions)}):" if positions else "Holdings: none")
    for p in positions:
        r = by_symbol.get(p.symbol)
        trend = _gap_text(r) if r and r.get("d") else "no MA data" if r else "not in config"
        lines.append(f"  {p.symbol:<6} {p.qty} @ ${float(p.avg_entry_price):.2f} | now "
                     f"${float(p.current_price):.2f} | P/L ${float(p.unrealized_pl):+.2f} | {trend}")
    lines.append("")

    near = _near_crossovers(results)
    if near:
        lines.append("Closest to a crossover (averages converging):")
        lines += [f"  {r['symbol']:<6} {_gap_text(r)}" for r in near]
        lines.append("")
    return lines


def _header(acct, results):
    dates = sorted({str(r["d"]["date"]) for r in results if r.get("d")})
    cash = float(acct.cash)
    margin = f" | margin in use ${-cash:,.2f}" if cash < 0 else ""
    return [f"Paper account {acct.account_number} | cash ${cash:,.2f}{margin} | "
            f"buying power ${float(acct.buying_power):,.2f}",
            f"Checked {len(results)} stocks using closing prices from {', '.join(dates) or 'n/a'}.",
            ""]


def _order_summary(orders):
    if len(orders) > 5:
        buys = sum(r["action"] == "buy" for r in orders)
        return f"{buys} buys, {len(orders) - buys} sells"
    return ", ".join(f"{r['action']} {r['symbol']}" for r in orders) or "no trades"


def run_report(acct, results, positions):
    orders = [r for r in results if r["action"] in ("buy", "sell")]
    lines = _header(acct, results)
    lines.append(f"Orders placed ({len(orders)}):" if orders else "Orders placed: none (no crossovers)")
    for r in orders:
        status = r.get("status", "unknown")
        fill = f" @ ${r['fill_price']:.2f}" if r.get("fill_price") else ""
        lines.append(f"  {r['action'].upper():<4} {r['qty']:g} {r['symbol']:<6} {status}{fill}")
        lines.append(f"       {_signal_text(r)}")
    lines.append("")
    lines += _common_sections(results, positions)

    title = f"Trading bot: {_order_summary(orders)} ({pd.Timestamp.now(tz='America/New_York'):%a %b %d})"
    return title, "\n".join(lines)


def preview_report(acct, results, positions, clock):
    next_open = pd.Timestamp(clock.next_open).tz_convert("America/New_York")
    planned = [r for r in results if r["action"] in ("buy", "sell")]
    lines = [f"Plan for the next run at the {next_open:%a %b %d} open "
             f"(orders go in around 7:40 am MT).", ""]
    lines += _header(acct, results)
    lines.append(f"Planned orders ({len(planned)}):" if planned else "Planned orders: none (no crossovers)")
    for r in planned:
        lines.append(f"  {r['action'].upper():<4} {r['qty']:g} {r['symbol']:<6} {_signal_text(r)}")
    lines.append("")
    lines += _common_sections(results, positions)
    lines.append("Signals near a tie can flip if the price data is revised before the morning run.")

    title = f"Trading bot plan for {next_open:%a %b %d}: {_order_summary(planned)}"
    return title, "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MA crossover bot (Alpaca paper trading)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="show signals without connecting to Alpaca")
    group.add_argument("--preview", action="store_true", help="work out the next run's orders without placing them")
    parser.add_argument("--report", action="store_true", help="print a summary report when done")
    args = parser.parse_args()
    if args.report and args.dry_run:
        parser.error("--report needs the Alpaca connection; use --preview --report instead of --dry-run")
    mode = "dry" if args.dry_run else "preview" if args.preview else "live"
    run_log = start_run_log(mode) if mode != "dry" else None
    text = run(mode=mode, report=args.report)
    if text:
        print(text)
        with open(run_log, "a") as f:
            f.write(f"\n{text}\n")
    if run_log:
        log.info("Run finished; full log saved to %s", run_log)
