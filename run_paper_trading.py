#!/usr/bin/env python3
"""
PAPER TRADING - Live Strategy Execution
====================================
Run the optimized strategy on T-Invest sandbox in real-time
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

import os
import urllib3
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from prometheus_client import start_http_server, Gauge, Counter, Histogram

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

PAPER_TRADE_SIGNALS = Counter("paper_trade_signals_total", "Total signals by outcome", ["ticker", "action", "reason"])
PAPER_TRADE_ORDERS = Counter("paper_trade_orders_total", "Total orders by status", ["ticker", "direction", "status"])
PAPER_TRADE_PNL = Gauge("paper_trade_pnl_rub", "Current paper trading PnL in RUB")
PAPER_TRADE_POSITIONS = Gauge("paper_trade_positions_count", "Number of open positions")
PAPER_TRADE_SIGNAL_LATENCY = Histogram("paper_trade_signal_latency_seconds", "Time to generate signal")
PAPER_TRADE_ACCOUNT_BALANCE = Gauge("paper_trade_account_balance_rub", "Current account balance in RUB")

PAPER_TRADE_DEALS_BUY = Counter("paper_trade_deals_buy_total", "Total BUY deals")
PAPER_TRADE_DEALS_SELL = Counter("paper_trade_deals_sell_total", "Total SELL deals")
PAPER_TRADE_DEALS_SL = Counter("paper_trade_deals_sl_total", "Deals closed by Stop Loss")
PAPER_TRADE_DEALS_TP = Counter("paper_trade_deals_tp_total", "Deals closed by Take Profit")
PAPER_TRADE_COMMISSION = Counter("paper_trade_commission_rub", "Cumulative commission in RUB")
PAPER_TRADE_SLIPPAGE = Counter("paper_trade_slippage_rub", "Cumulative slippage in RUB")
PAPER_TRADE_WINRATE = Gauge("paper_trade_winrate", "Winrate percentage")
PAPER_TRADE_CUMULATIVE_PNL = Gauge("paper_trade_cumulative_pnl_rub", "Cumulative PnL in RUB")
PAPER_TRADE_TOTAL_TRADES = Counter("paper_trade_total_trades", "Total closed trades")
PAPER_TRADE_WON_TRADES = Counter("paper_trade_won_trades", "Won trades")
PAPER_TRADE_LOST_TRADES = Counter("paper_trade_lost_trades", "Lost trades")

SESSION_STATS_FILE = Path("/tmp/paper_trading_session.json")


def load_session_stats():
    if SESSION_STATS_FILE.exists():
        return json.loads(SESSION_STATS_FILE.read_text())
    return {
        "total_trades": 0,
        "won_trades": 0,
        "lost_trades": 0,
        "commission": 0.0,
        "slippage": 0.0,
        "cumulative_pnl": 0.0,
        "buy_deals": 0,
        "sell_deals": 0,
        "sl_closed": 0,
        "tp_closed": 0,
    }


def save_session_stats(stats):
    SESSION_STATS_FILE.write_text(json.dumps(stats, indent=2))


def update_trade_stats(pnl, reason, commission=0.0, slippage=0.0):
    stats = load_session_stats()
    stats["cumulative_pnl"] += pnl
    stats["commission"] += commission
    stats["slippage"] += slippage
    stats["total_trades"] += 1
    
    if reason in ["SL", "TP_PARTIAL"]:
        stats["sl_closed"] += 1
    
    if pnl > 0:
        stats["won_trades"] += 1
    else:
        stats["lost_trades"] += 1
    
    winrate = (stats["won_trades"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0
    
    PAPER_TRADE_CUMULATIVE_PNL.set(stats["cumulative_pnl"])
    PAPER_TRADE_COMMISSION.set(stats["commission"])
    PAPER_TRADE_SLIPPAGE.set(stats["slippage"])
    PAPER_TRADE_WINRATE.set(winrate)
    PAPER_TRADE_TOTAL_TRADES.inc()
    if pnl > 0:
        PAPER_TRADE_WON_TRADES.inc()
    else:
        PAPER_TRADE_LOST_TRADES.inc()
    
    save_session_stats(stats)
    return stats


def get_positions_detail(client, account_id, positions):
    positions_detail = []
    ticker_figi = {}
    
    for pos in positions:
        ticker = pos.get('ticker')
        if not ticker or ticker == 'RUB000UTSTOM':
            continue
        
        quantity = pos.get('quantity', {})
        balance = int(quantity.get('units', '0')) if str(quantity.get('units', '0')).isdigit() else 0
        
        if balance > 0:
            avg_price = pos.get('averagePositionPrice', {})
            avg_price_val = float(avg_price.get('units', 0)) + float(avg_price.get('nano', 0)) / 1e9 if avg_price else 0
            figi = pos.get('figi')
            ticker_figi[ticker] = figi
            
            current_price = 0
            if figi:
                try:
                    last_price_data = client.get_last_price(figi)
                    if last_price_data:
                        current_price = last_price_data
                except:
                    pass
            
            pnl = 0
            if avg_price_val > 0 and current_price > 0:
                pnl = (current_price - avg_price_val) * balance
            
            positions_detail.append({
                "ticker": ticker,
                "quantity": balance,
                "avg_price": avg_price_val,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": ((current_price - avg_price_val) / avg_price_val * 100) if avg_price_val > 0 else 0
            })
    
    return positions_detail


def get_session_summary(client, account_id, positions):
    stats = load_session_stats()
    positions_detail = get_positions_detail(client, account_id, positions)
    
    portfolio = client.get_portfolio(account_id)
    balance = 0
    if portfolio:
        currencies = portfolio.get('totalAmountCurrencies', {})
        balance = float(currencies.get('units', 0)) + float(currencies.get('nano', 0)) / 1e9
    
    unrealized_pnl = sum(p["pnl"] for p in positions_detail)
    total_pnl = stats["cumulative_pnl"] + unrealized_pnl
    winrate = (stats["won_trades"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0
    
    return {
        "balance": balance,
        "positions": positions_detail,
        "unrealized_pnl": unrealized_pnl,
        "cumulative_pnl": stats["cumulative_pnl"],
        "total_pnl": total_pnl,
        "winrate": winrate,
        "total_trades": stats["total_trades"],
        "won_trades": stats["won_trades"],
        "lost_trades": stats["lost_trades"],
        "buy_deals": stats["buy_deals"],
        "sell_deals": stats["sell_deals"],
        "sl_closed": stats["sl_closed"],
        "tp_closed": stats["tp_closed"],
        "commission": stats["commission"],
        "slippage": stats["slippage"],
    }


def setup_prometheus_metrics():
    try:
        start_http_server(PROMETHEUS_PORT)
        print(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    except Exception as e:
        print(f"Failed to start Prometheus server: {e}")

print("=" * 70)
print("PAPER TRADING - LIVE STRATEGY")
print("=" * 70)

TOKEN = 't.9kPmBnJM7bBln56Nhtj_a3iu-aajTyoArtYKam7J_bmob_7jQXbQVzo4N2X9hyhwN-HyMyyzjQhS2YPPoZ4Owg'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = 'https://sandbox-invest-public-api.tbank.ru/rest'
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'Content-Type': 'application/json'
}

CONFIG = {
    'rsi_oversold': 30,
    'zscore_oversold': -1.8,
    'volume_ratio_min': 1.1,
    
    'allowed_tickers': [
        'TATN', 'MGNT', 'MAGN', 'SELG', 'APTK',
        'KMAZ', 'MRKZ', 'MRKS', 'KROT',
        'MSTT', 'OGKB', 'PRFN', 'TATNP', 'TGKA', 'TGKB', 'TGKN', 'POSI'
    ],
    
    'position_size_pct': 0.35,
    'stop_loss_pct': 0.008,
    'take_profit_pct': 0.025,
    'trailing_stop_pct': 0.008,
    
    'use_partial_tp': True,
    'partial_tp_level': 0.012,
    'partial_tp_pct': 0.50,
    
    'safe_hours': (10, 17),
    'market_momentum_threshold': -0.02,
}


class TInvestClient:
    def __init__(self, token: str):
        self.token = token
        self.api_base = 'https://sandbox-invest-public-api.tbank.ru/rest'
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.verify = False
        urllib3.disable_warnings()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _call(self, method: str, endpoint: str, data: dict = None, retries: int = None):
        if retries is None:
            retries = self.max_retries
        
        url = f"{self.api_base}/tinkoff.public.invest.api.contract.v1.{endpoint}"
        
        for attempt in range(retries):
            try:
                resp = self.session.post(url, headers=self.headers, json=data or {}, timeout=30)
                
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = (attempt + 1) * self.retry_delay
                    print(f"  Rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  API Error: {resp.status_code} - {resp.text[:200]}")
                    if attempt < retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * self.retry_delay
                    print(f"  Connection error: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(f"  Connection failed after {retries} attempts")
                return None
            except Exception as e:
                print(f"  Unexpected error: {e}")
                return None
        
        return None
    
    def get_accounts(self):
        result = self._call('POST', 'SandboxService/GetSandboxAccounts', {})
        if result and 'accounts' in result:
            return result['accounts']
        return []
    
    def get_portfolio(self, account_id: str):
        return self._call('POST', 'SandboxService/GetSandboxPortfolio', {'accountId': account_id})
    
    def get_positions(self, account_id: str):
        return self._call('POST', 'SandboxService/GetSandboxPositions', {'accountId': account_id})
    
    def find_instrument(self, ticker: str):
        result = self._call('POST', 'InstrumentsService/FindInstrument', {'query': ticker})
        if result and 'instruments' in result and result['instruments']:
            inst = result['instruments'][0]
            return {
                'figi': inst.get('figi'),
                'ticker': inst.get('ticker'),
                'lot': inst.get('lot', 1),
                'name': inst.get('name'),
            }
        return None
    
    def get_last_price(self, figi: str):
        result = self._call('POST', 'MarketDataService/GetLastPrices', {'figi': [figi]})
        if result and 'lastPrices' in result:
            for lp in result['lastPrices']:
                if lp.get('figi') == figi:
                    price_units = lp.get('price', {}).get('units', '0')
                    price_nano = lp.get('price', {}).get('nano', 0)
                    return float(f"{price_units}.{price_nano:09d}") if price_nano else float(price_units)
        return None
    
    def get_candles(self, figi: str, from_time: datetime, to_time: datetime, interval: str = 'CANDLE_INTERVAL_5_MIN'):
        result = self._call('POST', 'MarketDataService/GetCandles', {
            'figi': figi,
            'from': from_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'to': to_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'interval': interval
        })
        if result and 'candles' in result:
            return result['candles']
        return []
    
    def post_order(self, account_id: str, figi: str, direction: str, quantity: int, order_type: str = 'ORDER_TYPE_MARKET'):
        return self._call('POST', 'SandboxService/PostSandboxOrder', {
            'accountId': account_id,
            'figi': figi,
            'direction': direction,
            'quantity': str(quantity),
            'orderType': order_type,
        })
    
    def get_orders(self, account_id: str):
        return self._call('POST', 'SandboxService/GetSandboxOrders', {'accountId': account_id})
    
    def cancel_order(self, account_id: str, order_id: str):
        return self._call('POST', 'SandboxService/CancelSandboxOrder', {
            'accountId': account_id,
            'orderId': order_id
        })
    
    def get_operations(self, account_id: str, from_time: datetime, to_time: datetime):
        return self._call('POST', 'SandboxService/GetSandboxOperations', {
            'accountId': account_id,
            'from': from_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
            'to': to_time.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        })


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        return df
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    df['zscore_20'] = (df['close'] - df['ma20']) / df['std20']
    
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio_20'] = df['volume'] / df['vol_ma20']
    
    return df


def fund_account_if_needed(client, account_id, min_balance=500000):
    """Пополняет счёт если баланс меньше минимального"""
    portfolio = client.get_portfolio(account_id)
    if not portfolio:
        return False
    
    currencies = portfolio.get('totalAmountCurrencies', {})
    balance = float(currencies.get('units', 0)) + float(currencies.get('nano', 0)) / 1e9
    
    if balance < min_balance:
        print(f"  Баланс {balance:.0f} < {min_balance}, пополняем...")
        result = client._call('SandboxService/SandboxPayIn', {
            'accountId': account_id,
            'amount': {
                'currency': 'RUB',
                'units': str(min_balance),
                'nano': 0
            }
        })
        if result:
            print(f"  ✅ Счёт пополнен на {min_balance} RUB")
            return True
    return False


def run_paper_trading():
    setup_prometheus_metrics()
    
    print("\n[1/5] Connecting to T-Invest Sandbox...")
    
    client = TInvestClient(TOKEN)
    accounts = client.get_accounts()
    
    if not accounts:
        print("  No accounts found!")
        return
    
    account_id = accounts[0]['id']
    print(f"  Account: {account_id}")
    
    fund_account_if_needed(client, account_id)
    
    portfolio = client.get_portfolio(account_id)
    if portfolio:
        currencies = portfolio.get('totalAmountCurrencies', {})
        balance = float(currencies.get('units', 0)) + float(currencies.get('nano', 0)) / 1e9
        PAPER_TRADE_ACCOUNT_BALANCE.set(balance)
    
    print("\n[2/5] Fetching portfolio...")
    positions = portfolio.get('positions', []) if portfolio else []
    print(f"  Open positions: {len(positions)}")
    PAPER_TRADE_POSITIONS.set(len([p for p in positions if p.get('ticker') and p.get('ticker') != 'RUB000UTSTOM']))
    
    current_positions = {}
    current_positions_info = {}
    
    for pos in positions:
        ticker = pos.get('ticker')
        if not ticker or ticker == 'RUB000UTSTOM':
            continue
        
        quantity = pos.get('quantity', {})
        balance = quantity.get('units', '0') if quantity else '0'
        current_positions[ticker] = int(balance) if str(balance).isdigit() else 0
        
        avg_price = pos.get('averagePositionPrice', {})
        avg_price_val = float(avg_price.get('units', 0)) + float(avg_price.get('nano', 0)) / 1e9 if avg_price else 0
        
        current_positions_info[ticker] = {
            'balance': current_positions[ticker],
            'avg_price': avg_price_val,
            'figi': pos.get('figi')
        }
        print(f"    - {ticker}: {balance} shares @ {avg_price_val:.2f}")
    
    print("\n[3/5] Fetching market data...")
    
    ticker_figi = {}
    ticker_info = {}
    
    for ticker in CONFIG['allowed_tickers']:
        info = client.find_instrument(ticker)
        if info:
            ticker_figi[ticker] = info['figi']
            ticker_info[ticker] = info
            print(f"  {ticker}: FIGI={info['figi']}, Lot={info['lot']}")
    
    signals = []
    positions_to_close = []
    
    for ticker in list(current_positions.keys()):
        if ticker not in ticker_figi:
            continue
            
        pos_info = current_positions_info.get(ticker, {})
        balance = pos_info.get('balance', 0)
        avg_price = pos_info.get('avg_price', 0)
        figi = ticker_figi.get(ticker)
        
        if balance > 0 and avg_price > 0:
            current_price = client.get_last_price(figi)
            if current_price:
                pnl_pct = (current_price - avg_price) / avg_price
                
                sl_triggered = pnl_pct < -CONFIG['stop_loss_pct']
                tp_triggered = pnl_pct > CONFIG['take_profit_pct']
                partial_tp_triggered = CONFIG['use_partial_tp'] and pnl_pct > CONFIG['partial_tp_level']
                
                print(f"\n  {ticker}:")
                print(f"    Entry: {avg_price:.2f}, Current: {current_price:.2f}")
                print(f"    PnL: {pnl_pct*100:.2f}%")
                
                if sl_triggered:
                    print(f"    🛑 STOP LOSS triggered!")
                    positions_to_close.append({
                        'ticker': ticker, 'figi': figi, 
                        'quantity': balance, 'reason': 'SL',
                        'entry_price': avg_price
                    })
                elif tp_triggered:
                    print(f"    🎯 TAKE PROFIT triggered!")
                    positions_to_close.append({
                        'ticker': ticker, 'figi': figi, 
                        'quantity': balance, 'reason': 'TP',
                        'entry_price': avg_price
                    })
                elif partial_tp_triggered:
                    partial_qty = int(balance * CONFIG['partial_tp_pct'])
                    print(f"    🎯 Partial TP: selling {partial_qty} shares")
                    positions_to_close.append({
                        'ticker': ticker, 'figi': figi, 
                        'quantity': partial_qty, 'reason': 'PARTIAL_TP',
                        'entry_price': avg_price
                    })
    
    print("\n[3/5] Calculating new signals...")
    
    for ticker, figi in ticker_figi.items():
        if ticker in current_positions and current_positions[ticker] > 0:
            print(f"  {ticker}: Already have position, skipping signal check")
            continue
        
        last_price = client.get_last_price(figi)
        if not last_price:
            print(f"  {ticker}: No price data")
            continue
        
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(hours=6)
        
        candles = client.get_candles(figi, from_time, now, 'CANDLE_INTERVAL_5_MIN')
        
        if not candles:
            print(f"  {ticker}: No candles (market may be closed)")
            continue
        
        df = pd.DataFrame([
            {
                'close': float(c.get('close', {}).get('units', 0)) + float(c.get('close', {}).get('nano', 0)) / 1e9,
                'volume': float(c.get('volume', 0)),
                'time': c.get('time', ''),
            }
            for c in candles
        ])
        
        if len(df) < 20:
            print(f"  {ticker}: Only {len(df)} candles, need 20+")
            continue
        
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        
        rsi = latest.get('rsi_14', 50)
        zscore = latest.get('zscore_20', 0)
        vol_ratio = latest.get('volume_ratio_20', 1)
        
        hour = datetime.fromisoformat(latest['time'].replace('Z', '+00:00')).hour if latest.get('time') else 12
        
        print(f"\n  {ticker}:")
        print(f"    Price: {latest['close']:.2f}")
        print(f"    RSI: {rsi:.1f}")
        print(f"    Z-score: {zscore:.2f}")
        print(f"    Vol ratio: {vol_ratio:.2f}")
        
        if (rsi < CONFIG['rsi_oversold'] and 
            zscore < CONFIG['zscore_oversold'] and 
            vol_ratio >= CONFIG['volume_ratio_min'] and
            10 <= hour <= 17):
            print(f"    🟢 SIGNAL: BUY {ticker}")
            PAPER_TRADE_SIGNALS.labels(ticker=ticker, action="buy", reason="signal").inc()
            signals.append({
                'ticker': ticker,
                'figi': figi,
                'price': latest['close'],
                'lot': ticker_info[ticker]['lot'],
            })
        else:
            print(f"    🔴 No signal")
            PAPER_TRADE_SIGNALS.labels(ticker=ticker, action="none", reason="no_signal").inc()
    
    print("\n[4/5] Executing orders...")
    
    portfolio_total = 1000000
    position_value = portfolio_total * CONFIG['position_size_pct']
    
    stats = load_session_stats()
    stats["sell_deals"] += len(positions_to_close)
    save_session_stats(stats)
    PAPER_TRADE_DEALS_SELL.inc(len(positions_to_close))
    
    for close_info in positions_to_close:
        ticker = close_info['ticker']
        figi = close_info['figi']
        qty = close_info['quantity']
        reason = close_info['reason']
        entry_price = close_info.get('entry_price', 0)
        
        print(f"  Selling {ticker}: {qty} shares ({reason})")
        
        current_price = 0
        try:
            last_price_data = client.get_last_price(figi)
            if last_price_data:
                current_price = last_price_data
        except:
            pass
        
        result = client.post_order(account_id, figi, 'ORDER_DIRECTION_SELL', qty)
        
        if result:
            order_id = result.get('orderId')
            print(f"    ✅ Sell order placed: {order_id}")
            PAPER_TRADE_ORDERS.labels(ticker=ticker, direction="sell", status="success").inc()
            
            pnl = 0
            if entry_price > 0 and current_price > 0:
                pnl = (current_price - entry_price) * qty
            
            commission = current_price * qty * BROKER_COMMISSION_RATE
            slippage = abs(current_price - entry_price) * qty * 0.001
            
            update_trade_stats(pnl, reason, commission, slippage)
            
            if reason == "SL":
                PAPER_TRADE_DEALS_SL.inc()
            elif reason in ["TP", "PARTIAL_TP"]:
                PAPER_TRADE_DEALS_TP.inc()
        else:
            print(f"    ❌ Sell order failed")
            PAPER_TRADE_ORDERS.labels(ticker=ticker, direction="sell", status="failed").inc()
    
    BROKER_COMMISSION_RATE = 0.0005
    
    stats = load_session_stats()
    stats["buy_deals"] += len(signals)
    save_session_stats(stats)
    PAPER_TRADE_DEALS_BUY.inc(len(signals))
    
    for signal in signals:
        ticker = signal['ticker']
        figi = signal['figi']
        price = signal['price']
        lot = signal['lot']
        
        shares_to_buy = int(position_value / price)
        shares_to_buy = (shares_to_buy // lot) * lot
        
        if shares_to_buy < lot:
            print(f"  {ticker}: Not enough funds for minimum lot")
            continue
        
        print(f"  Buying {ticker}: {shares_to_buy} shares @ ~{price}")
        
        result = client.post_order(account_id, figi, 'ORDER_DIRECTION_BUY', shares_to_buy)
        
        if result:
            order_id = result.get('orderId')
            print(f"    ✅ Order placed: {order_id}")
            PAPER_TRADE_ORDERS.labels(ticker=ticker, direction="buy", status="success").inc()
            commission = price * shares_to_buy * BROKER_COMMISSION_RATE
            stats = load_session_stats()
            stats["commission"] += commission
            save_session_stats(stats)
            PAPER_TRADE_COMMISSION.inc(commission)
        else:
            print(f"    ❌ Order failed")
            PAPER_TRADE_ORDERS.labels(ticker=ticker, direction="buy", status="failed").inc()
    
    print("\n[5/5] Portfolio summary...")
    
    print("\n" + "=" * 50)
    print("PAPER TRADING SUMMARY")
    print("=" * 50)
    print(f"  Account: {account_id}")
    print(f"  Open positions: {len(positions)}")
    print(f"  New signals: {len(signals)}")
    
    if signals:
        print(f"\n  🟢 ORDERS PLACED:")
        for s in signals:
            print(f"    - {s['ticker']}")
    else:
        print(f"\n  🔴 No signals at this time")
    
    final_portfolio = client.get_portfolio(account_id)
    if final_portfolio:
        currencies = final_portfolio.get('totalAmountCurrencies', {})
        final_balance = float(currencies.get('units', 0)) + float(currencies.get('nano', 0)) / 1e9
        PAPER_TRADE_ACCOUNT_BALANCE.set(final_balance)
        pnl = final_balance - balance
        PAPER_TRADE_PNL.set(pnl)
        print(f"\n  PnL: {pnl:.2f} RUB")
    
    print("\n✅ Paper trading cycle complete!")


def run_continuous_trading(interval_minutes: int = 15):
    import sys
    
    print(f"\n🔄 Starting continuous trading (check every {interval_minutes} min)")
    print("Press Ctrl+C to stop\n")
    sys.stdout.flush()
    
    cycle = 0
    try:
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"Trading Cycle #{cycle}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*60)
            sys.stdout.flush()
            
            run_paper_trading()
            
            print(f"\n⏳ Next check in {interval_minutes} minutes...")
            sys.stdout.flush()
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping continuous trading...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', '-c', action='store_true', help='Run continuously')
    parser.add_argument('--interval', '-i', type=int, default=15, help='Interval in minutes (default: 15)')
    args = parser.parse_args()
    
    if args.continuous:
        run_continuous_trading(args.interval)
    else:
        run_paper_trading()
