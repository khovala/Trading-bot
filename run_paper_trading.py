#!/usr/bin/env python3
"""
PAPER TRADING - Live Strategy Execution
====================================
Run the optimized strategy on T-Invest sandbox in real-time
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

import os
from pathlib import Path
from dotenv import load_dotenv

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')
load_dotenv(workspace / '.env')

import urllib3
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prometheus_client import start_http_server, Gauge, Counter, Histogram

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")

VK_TOKEN = os.getenv("VK_TOKEN", "")
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "")

SMSRU_API_KEY = os.getenv("SMSRU_API_KEY", "")
SMSRU_PHONE = os.getenv("SMSRU_PHONE", "")

SMSC_LOGIN = os.getenv("SMSC_LOGIN", "")
SMSC_PASSWORD = os.getenv("SMSC_PASSWORD", "")
SMSC_PHONE = os.getenv("SMSC_PHONE", "")

# Max Messenger
MAX_TOKEN = os.getenv("MAX_TOKEN", "")
MAX_CHAT_ID = os.getenv("MAX_CHAT_ID", "")

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
PAPER_TRADE_COMMISSION = Gauge("paper_trade_commission_rub", "Cumulative commission in RUB")
PAPER_TRADE_SLIPPAGE = Gauge("paper_trade_slippage_rub", "Cumulative slippage in RUB")
PAPER_TRADE_WINRATE = Gauge("paper_trade_winrate", "Winrate percentage")
PAPER_TRADE_CUMULATIVE_PNL = Gauge("paper_trade_cumulative_pnl_rub", "Cumulative PnL in RUB")
PAPER_TRADE_TOTAL_TRADES = Counter("paper_trade_total_trades", "Total closed trades")
PAPER_TRADE_WON_TRADES = Counter("paper_trade_won_trades", "Won trades")
PAPER_TRADE_LOST_TRADES = Counter("paper_trade_lost_trades", "Lost trades")

PAPER_TRADE_POSITIONS_VALUE = Gauge("paper_trade_positions_value_rub", "Total positions value in RUB")
PAPER_TRADE_TOTAL_ASSETS = Gauge("paper_trade_total_assets_rub", "Total assets (cash + positions) in RUB")
PAPER_TRADE_UNREALIZED_PNL = Gauge("paper_trade_unrealized_pnl_rub", "Unrealized PnL in RUB")
PAPER_TRADE_REALIZED_PNL = Gauge("paper_trade_realized_pnl_rub", "Realized PnL in RUB")

PAPER_TRADE_DRAWDOWN = Gauge("paper_trade_drawdown_rub", "Current drawdown in RUB")
PAPER_TRADE_DRAWDOWN_PCT = Gauge("paper_trade_drawdown_percent", "Current drawdown percentage")
PAPER_TRADE_MAX_DRAWDOWN = Gauge("paper_trade_max_drawdown_rub", "Maximum drawdown in RUB")
PAPER_TRADE_MAX_DRAWDOWN_PCT = Gauge("paper_trade_max_drawdown_percent", "Maximum drawdown percentage")

PAPER_TRADE_AVG_WIN = Gauge("paper_trade_avg_win_rub", "Average win in RUB")
PAPER_TRADE_AVG_LOSS = Gauge("paper_trade_avg_loss_rub", "Average loss in RUB")
PAPER_TRADE_PROFIT_FACTOR = Gauge("paper_trade_profit_factor", "Profit factor (avg win / avg loss)")
PAPER_TRADE_RETURN_PCT = Gauge("paper_trade_return_percent", "Total return percentage")

SESSION_STATS_FILE = Path("/tmp/paper_trading_session.json")
PEAK_ASSETS_FILE = Path("/tmp/paper_trading_peak.json")


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
        balance = float(currencies.get('units', 0) or 0) + float(currencies.get('nano', 0) or 0) / 1e9
    
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


def update_portfolio_metrics(client, account_id, stats):
    try:
        portfolio = client.get_portfolio(account_id)
        if not portfolio:
            return
        
        # Get positions for accurate cash balance
        positions_response = client.get_positions(account_id)
        
        cash = 0.0
        if positions_response:
            money_positions = positions_response.get('money', [])
            for m in money_positions:
                currency = m.get('currency', '')
                if currency == 'RUB':
                    units = m.get('units', '0')
                    nano = m.get('nano', 0)
                    cash = float(units) + float(nano) / 1e9
        
        # Fallback to portfolio total if no cash from positions
        if cash == 0:
            currencies = portfolio.get('totalAmountCurrencies', {})
            cash = float(currencies.get('units', 0) or 0) + float(currencies.get('nano', 0) or 0) / 1e9
        
        portfolio_positions = portfolio.get('positions', []) if portfolio else []
        
        positions_value = 0
        unrealized_pnl = 0
        
        for pos in portfolio_positions:
            ticker = pos.get('ticker')
            if not ticker or ticker == 'RUB000UTSTOM':
                continue
            
            quantity = pos.get('quantity', {})
            qty = int(str(quantity.get('units', '0')))
            
            if qty > 0:
                avg_price_data = pos.get('averagePositionPrice', {})
                avg_price_val = 0
                if avg_price_data:
                    if isinstance(avg_price_data, dict):
                        avg_price_val = float(avg_price_data.get('units', 0) or 0)
                    else:
                        avg_price_val = float(avg_price_data)
                
                current_price_data = pos.get('currentPrice', {})
                current_price = 0
                if current_price_data:
                    if isinstance(current_price_data, dict):
                        current_price = float(current_price_data.get('units', 0) or 0)
                    else:
                        current_price = float(current_price_data)
                
                if current_price == 0:
                    current_price = avg_price_val
                
                if current_price > 0:
                    positions_value += current_price * qty
                    unrealized_pnl += (current_price - avg_price_val) * qty
        
        total_assets = cash + positions_value
        realized_pnl = stats.get('cumulative_pnl', 0)
        
        PAPER_TRADE_POSITIONS_VALUE.set(positions_value)
        PAPER_TRADE_TOTAL_ASSETS.set(total_assets)
        PAPER_TRADE_UNREALIZED_PNL.set(unrealized_pnl)
        PAPER_TRADE_REALIZED_PNL.set(realized_pnl)
        
        peak_data = {"peak": 1000000.0}
        if PEAK_ASSETS_FILE.exists():
            try:
                peak_data = json.loads(PEAK_ASSETS_FILE.read_text())
            except:
                pass
        
        peak = peak_data.get("peak", 1000000.0)
        if total_assets > peak:
            peak = total_assets
            PEAK_ASSETS_FILE.write_text(json.dumps({"peak": peak}))
        
        drawdown = max(0, peak - total_assets)
        drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0
        max_drawdown = peak_data.get("max_drawdown", 0)
        max_drawdown_pct = peak_data.get("max_drawdown_pct", 0)
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown_pct
            peak_data["max_drawdown"] = max_drawdown
            peak_data["max_drawdown_pct"] = max_drawdown_pct
            PEAK_ASSETS_FILE.write_text(json.dumps(peak_data))
        
        PAPER_TRADE_DRAWDOWN.set(drawdown)
        PAPER_TRADE_DRAWDOWN_PCT.set(drawdown_pct)
        PAPER_TRADE_MAX_DRAWDOWN.set(max_drawdown)
        PAPER_TRADE_MAX_DRAWDOWN_PCT.set(max_drawdown_pct)
        
        initial = 1000000.0
        return_pct = ((total_assets - initial) / initial * 100) if initial > 0 else 0
        PAPER_TRADE_RETURN_PCT.set(return_pct)
        
        won = stats.get("won_trades", 0)
        lost = stats.get("lost_trades", 0)
        total = stats.get("total_trades", 0)
        
        if won > 0 and total > 0:
            avg_win = realized_pnl / won if won > 0 else 0
            PAPER_TRADE_AVG_WIN.set(avg_win)
        
        if lost > 0:
            avg_loss = abs(realized_pnl) / lost if realized_pnl < 0 else 0
            PAPER_TRADE_AVG_LOSS.set(avg_loss)
            if avg_loss > 0 and avg_win > 0:
                profit_factor = avg_win / avg_loss
                PAPER_TRADE_PROFIT_FACTOR.set(profit_factor)
        
    except Exception as e:
        print(f"Failed to update portfolio metrics: {e}")


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
    
    # Get portfolio with total amounts
    portfolio = client.get_portfolio(account_id)
    
    # Get positions directly for detailed info
    positions_response = client.get_positions(account_id)
    
    # Calculate cash balance from positions (RUB currency)
    cash_balance = 0.0
    if positions_response:
        money_positions = positions_response.get('money', [])
        for m in money_positions:
            currency = m.get('currency', '')
            if currency == 'RUB':
                units = m.get('units', '0')
                nano = m.get('nano', 0)
                cash_balance = float(units) + float(nano) / 1e9
                print(f"  Cash (RUB): {cash_balance:,.2f}")
    
    # Also check totalAmountCurrencies as fallback
    if portfolio:
        currencies = portfolio.get('totalAmountCurrencies', {})
        total_currencies = float(currencies.get('units', 0) or 0) + float(currencies.get('nano', 0) or 0) / 1e9
        if cash_balance == 0:
            cash_balance = total_currencies
        print(f"  Total currencies (portfolio): {total_currencies:,.2f}")
    
    # Use cash balance for account balance metric
    PAPER_TRADE_ACCOUNT_BALANCE.set(cash_balance)
    
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
                    partial_qty = max(1, int(balance * CONFIG['partial_tp_pct'] + 0.9999))
                    print(f"    🎯 Partial TP: selling {partial_qty} of {balance} shares")
                    if partial_qty > 0 and partial_qty <= balance:
                        positions_to_close.append({
                            'ticker': ticker, 'figi': figi, 
                            'quantity': partial_qty, 'reason': 'PARTIAL_TP',
                            'entry_price': avg_price
                        })
                    elif partial_qty >= balance:
                        positions_to_close.append({
                            'ticker': ticker, 'figi': figi, 
                            'quantity': balance, 'reason': 'TP',
                            'entry_price': avg_price
                        })
                    else:
                        print(f"    ⚠️  Skipping sell: qty={partial_qty}, balance={balance}")
    
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
    
    BROKER_COMMISSION_RATE = 0.0005
    
    stats = load_session_stats()
    
    for close_info in positions_to_close:
        ticker = close_info['ticker']
        figi = close_info['figi']
        qty = close_info['quantity']
        reason = close_info['reason']
        entry_price = close_info.get('entry_price', 0)
        
        if qty <= 0:
            print(f"  ⚠️  Skipping {ticker}: qty={qty} is invalid")
            continue
        
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
            
            stats["sell_deals"] += 1
            save_session_stats(stats)
            PAPER_TRADE_DEALS_SELL.inc()
            
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
            PAPER_TRADE_DEALS_BUY.inc()
            stats = load_session_stats()
            stats["buy_deals"] += 1
            stats["commission"] += price * shares_to_buy * BROKER_COMMISSION_RATE
            save_session_stats(stats)
            PAPER_TRADE_COMMISSION.set(stats["commission"])
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
        final_balance = float(currencies.get('units', 0) or 0) + float(currencies.get('nano', 0) or 0) / 1e9
        PAPER_TRADE_ACCOUNT_BALANCE.set(final_balance)
        if 'balance' in dir() and isinstance(balance, float):
            pnl = final_balance - balance
            PAPER_TRADE_PNL.set(pnl)
            print(f"\n  PnL: {pnl:.2f} RUB")
        
        stats = load_session_stats()
        update_portfolio_metrics(client, account_id, stats)
    
    print("\n✅ Paper trading cycle complete!")


def send_telegram_report(client, account_id, stats):
    """Send trading report to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        portfolio = client.get_portfolio(account_id)
        balance = 0
        if portfolio:
            currencies = portfolio.get('totalAmountCurrencies', {})
            balance = float(currencies.get('units', 0) or 0) + float(currencies.get('nano', 0) or 0) / 1e9
        
        positions_result = client.get_positions(account_id)
        positions = positions_result.get('positions', []) if positions_result else []
        
        positions_detail = []
        unrealized_pnl = 0
        for pos in positions:
            ticker = pos.get('ticker')
            if not ticker or ticker == 'RUB000UTSTOM':
                continue
            quantity = pos.get('quantity', {})
            balance_pos = int(quantity.get('units', '0')) if str(quantity.get('units', '0')).isdigit() else 0
            if balance_pos > 0:
                avg_price = pos.get('averagePositionPrice', {})
                avg_price_val = float(avg_price.get('units', 0)) + float(avg_price.get('nano', 0)) / 1e9 if avg_price else 0
                figi = pos.get('figi')
                current_price = 0
                if figi:
                    try:
                        lp = client.get_last_price(figi)
                        if lp:
                            current_price = lp
                    except:
                        pass
                pnl = 0
                if avg_price_val > 0 and current_price > 0:
                    pnl = (current_price - avg_price_val) * balance_pos
                    unrealized_pnl += pnl
                positions_detail.append({
                    "ticker": ticker,
                    "quantity": balance_pos,
                    "avg_price": avg_price_val,
                    "current_price": current_price,
                    "pnl": pnl,
                    "pnl_pct": ((current_price - avg_price_val) / avg_price_val * 100) if avg_price_val > 0 else 0
                })
        
        total_pnl = stats['cumulative_pnl'] + unrealized_pnl
        winrate = (stats['won_trades']/stats['total_trades']*100) if stats['total_trades'] > 0 else 0
        
        now = datetime.now().strftime("%H:%M:%S")
        emoji_pnl = "📈" if total_pnl >= 0 else "📉"
        
        message = f"📊 *MOEX Trading Session Report*\n"
        message += f"🕐 *Время:* {now}\n\n"
        message += f"💰 *Баланс счета:* `{balance:,.2f} RUB`\n\n"
        message += f"{emoji_pnl} *P&L:*\n"
        message += f"   • Реализованный: `{stats['cumulative_pnl']:,.2f} RUB`\n"
        message += f"   • Нереализованный: `{unrealized_pnl:,.2f} RUB`\n"
        message += f"   • Общий: `{total_pnl:,.2f} RUB`\n\n"
        message += f"🎯 *Winrate:* `{stats['won_trades']}/{stats['total_trades']} = {winrate:.1f}%`\n\n"
        message += f"🔄 *Сделки:*\n"
        message += f"   • BUY: `{stats['buy_deals']}`\n"
        message += f"   • SELL: `{stats['sell_deals']}`\n"
        message += f"   • Закрыто по SL: `{stats['sl_closed']}`\n"
        message += f"   • Закрыто по TP: `{stats['tp_closed']}`\n\n"
        message += f"💳 *Комиссия:* `{stats['commission']:,.2f} RUB`\n"
        message += f"〰️ *Проскальзывание:* `{stats['slippage']:,.2f} RUB`\n\n"
        
        if positions_detail:
            message += f"📊 *Открытые позиции:*\n"
            for pos in sorted(positions_detail, key=lambda x: x['pnl'], reverse=True):
                emoji = "🟢" if pos['pnl'] >= 0 else "🔴"
                message += f"   {emoji} {pos['ticker']}: {pos['quantity']} шт @ {pos['avg_price']:.2f} → {pos['current_price']:.2f} ({pos['pnl_pct']:+.2f}%)\n"
        
        # Send to Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code == 200:
                    print(f"📱 Telegram report sent successfully")
                else:
                    print(f"📱 Telegram error: {response.status_code}")
            except Exception as e:
                print(f"📱 Failed to send Telegram: {e}")
        
        # Send to VK Group
        if VK_TOKEN and VK_GROUP_ID:
            import random
            try:
                vk_url = "https://api.vk.com/method/messages.send"
                vk_payload = {
                    "peer_id": f"-{VK_GROUP_ID}",
                    "message": message.replace("*", "").replace("`", ""),
                    "access_token": VK_TOKEN,
                    "random_id": random.randint(0, 2**31),
                    "v": "5.131"
                }
                response = requests.post(vk_url, data=vk_payload, timeout=30)
                result = response.json()
                if response.status_code == 200 and not result.get('error'):
                    print(f"📣 VK report sent to group successfully")
                else:
                    print(f"📣 VK error: {result.get('error', {}).get('error_msg', result)}")
            except Exception as e:
                print(f"📣 Failed to send VK: {e}")
        
        # Send to Email
        if SMTP_HOST and SMTP_USER and EMAIL_TO:
            try:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = f"MOEX Trading Report - {now}"
                msg['From'] = EMAIL_FROM or SMTP_USER
                msg['To'] = EMAIL_TO
                plain_text = message.replace("*", "").replace("`", "")
                msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
                
                if SMTP_PORT == 465:
                    server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
                else:
                    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
                    server.starttls()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())
                server.quit()
                print(f"📧 Email report sent to {EMAIL_TO}")
            except Exception as e:
                print(f"📧 Failed to send email: {e}")
        
        # Send SMS via SMSC
        if SMSC_LOGIN and SMSC_PASSWORD and SMSC_PHONE:
            try:
                import urllib.parse
                sms_text = f"MOEX PnL:{total_pnl:,.0f} RUB Win:{winrate:.0f}% Trades:{stats['total_trades']} Pos:{len(positions_detail)}"
                smsc_url = "https://smsc.ru/sys/send.php"
                smsc_data = {
                    "login": SMSC_LOGIN,
                    "psw": SMSC_PASSWORD,
                    "phones": SMSC_PHONE,
                    "mes": sms_text,
                    "fmt": 3,
                    "json": 1
                }
                response = requests.post(smsc_url, data=sms_data, timeout=15)
                result = response.json()
                if result.get('ack'):
                    print(f"📱 SMS report sent to {SMSC_PHONE}")
                else:
                    print(f"📱 SMS error: {result}")
            except Exception as e:
                print(f"📱 Failed to send SMS: {e}")
        
        # Send SMS via SMS.ru (alternative)
        elif SMSRU_API_KEY and SMSRU_PHONE:
            try:
                import urllib.parse
                sms_text = f"MOEX PnL:{total_pnl:,.0f} RUB Win:{winrate:.0f}% Trades:{stats['total_trades']} Pos:{len(positions_detail)}"
                sms_url = "https://sms.ru/sms/send"
                sms_data = {
                    "api_id": SMSRU_API_KEY,
                    "to": SMSRU_PHONE,
                    "msg": sms_text,
                    "json": 1
                }
                response = requests.post(sms_url, data=sms_data, timeout=10)
                result = response.json()
                if result.get('code') == 100:
                    print(f"📱 SMS report sent to {SMSRU_PHONE}")
                else:
                    print(f"📱 SMS error: {result}")
            except Exception as e:
                print(f"📱 Failed to send SMS: {e}")
                
    except Exception as e:
        print(f"📱 Failed to send report: {e}")


def get_prometheus_metrics_report():
    try:
        response = requests.get(f'http://localhost:{PROMETHEUS_PORT}/metrics', timeout=10)
        metrics_text = response.text
        
        report = {}
        
        for line in metrics_text.split('\n'):
            if line.startswith('paper_trade_') and '{' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].replace('paper_trade_', '')
                    try:
                        value = float(parts[1])
                        report[name] = value
                    except:
                        pass
        
        return report
    except Exception as e:
        print(f"Failed to get prometheus metrics: {e}")
        return {}


def send_max_report():
    if not MAX_TOKEN:
        return
    
    try:
        user_id = os.getenv("MAX_USER_ID", "")
        chat_id = os.getenv("MAX_CHAT_ID", "")
        
        if not user_id and not chat_id:
            try:
                response = requests.get(
                    "https://platform-api.max.ru/updates",
                    headers={"Authorization": MAX_TOKEN},
                    params={"timeout": 1},
                    timeout=5
                )
                if response.status_code == 200:
                    updates = response.json()
                    for update in updates.get('updates', []):
                        if update.get('update_type') == 'bot_started':
                            user_id = update.get('user', {}).get('user_id')
                            print(f"💬 Detected user: {user_id}")
                            break
                        elif update.get('type') == 'chat_add':
                            chat_id = update.get('chat_id')
                            print(f"💬 Detected new chat: {chat_id}")
                            break
            except:
                pass
        
        recipient_id = user_id or chat_id
        if not recipient_id:
            print("💬 No MAX_USER_ID or MAX_CHAT_ID configured")
            return
        
        report = get_prometheus_metrics_report()
        
        if not report:
            print("💬 No prometheus metrics to send")
            return
        
        now = datetime.now().strftime("%H:%M:%S")
        
        total_assets = report.get('total_assets_rub', 0)
        positions_value = report.get('positions_value_rub', 0)
        unrealized = report.get('unrealized_pnl_rub', 0)
        realized = report.get('realized_pnl_rub', 0)
        cash = total_assets - positions_value if total_assets > 0 else 0
        
        winrate = report.get('winrate', 0)
        return_pct = report.get('return_percent', 0)
        drawdown = report.get('drawdown_rub', 0)
        drawdown_pct = report.get('drawdown_percent', 0)
        commission = report.get('commission_rub', 0)
        
        message = f"📊 *MOEX Trading Report*\n"
        message += f"🕐 *Время:* {now}\n\n"
        
        message += f"💰 *Баланс:*\n"
        message += f"   • Денежные средства: `{cash:,.0f} RUB`\n"
        message += f"   • Стоимость позиций: `{positions_value:,.0f} RUB`\n"
        message += f"   • Общие активы: `{total_assets:,.0f} RUB`\n\n"
        
        emoji_pnl = "📈" if unrealized >= 0 else "📉"
        message += f"{emoji_pnl} *P&L:*\n"
        message += f"   • Реализованный: `{realized:,.0f} RUB`\n"
        message += f"   • Нереализованный: `{unrealized:,.0f} RUB`\n\n"
        
        message += f"📈 *Доходность:* `{return_pct:+.2f}%`\n"
        message += f"🎯 *Winrate:* `{winrate:.1f}%`\n\n"
        
        message += f"📉 *Риск:*\n"
        message += f"   • Просадка: `{drawdown:,.0f} RUB` ({drawdown_pct:.3f}%)\n\n"
        
        message += f"💳 *Комиссия:* `{commission:,.0f} RUB`\n"
        
        if recipient_id:
            try:
                if user_id:
                    max_url = f"https://platform-api.max.ru/messages?user_id={user_id}"
                else:
                    max_url = f"https://platform-api.max.ru/messages?chat_id={chat_id}"
                headers = {
                    "Authorization": MAX_TOKEN,
                    "Content-Type": "application/json"
                }
                payload = {
                    "text": message.replace("*", "").replace("`", ""),
                    "format": "markdown"
                }
                response = requests.post(max_url, json=payload, headers=headers, timeout=30)
                if response.status_code in [200, 201]:
                    result = response.json()
                    if 'message' in result:
                        print(f"💬 Max report sent successfully")
                else:
                    print(f"💬 Max error: {response.status_code} - {response.text[:100]}")
            except Exception as e:
                print(f"💬 Failed to send Max: {e}")
        
    except Exception as e:
        print(f"💬 Failed to generate report: {e}")


def telegram_reporter_thread():
    """Background thread that sends Telegram reports every 5 minutes."""
    while True:
        time.sleep(300)
        client = TInvestClient(TOKEN)
        accounts = client.get_accounts()
        if not accounts:
            continue
        account_id = accounts[0]['id']
        stats = load_session_stats()
        send_telegram_report(client, account_id, stats)


def max_reporter_thread():
    """Background thread that sends Max reports every 5 minutes."""
    while True:
        time.sleep(300)
        if MAX_TOKEN:
            send_max_report()


def run_continuous_trading(interval_minutes: int = 5):
    import sys
    
    print(f"\n🔄 Starting continuous trading (check every {interval_minutes} min)")
    print("Press Ctrl+C to stop\n")
    sys.stdout.flush()
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(f"📱 Telegram notifications enabled")
    if VK_TOKEN and VK_GROUP_ID:
        print(f"📣 VK notifications enabled (group)")
    if SMSRU_API_KEY and SMSRU_PHONE:
        print(f"📱 SMS notifications enabled to {SMSRU_PHONE}")
    if SMSC_LOGIN and SMSC_PASSWORD and SMSC_PHONE:
        print(f"📱 SMSC notifications enabled to {SMSC_PHONE}")
    if SMTP_HOST and SMTP_USER and EMAIL_TO:
        print(f"📧 Email notifications enabled to {EMAIL_TO}")
    if MAX_TOKEN:
        if MAX_CHAT_ID:
            print(f"💬 Max notifications enabled")
        else:
            print(f"💬 Max bot ready - write /start to @id710513232468_bot")
    
    cycle = 0
    stats = load_session_stats()
    
    try:
        while True:
            cycle += 1
            print(f"\n{'='*60}")
            print(f"Trading Cycle #{cycle}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print('='*60)
            sys.stdout.flush()
            
            result = run_paper_trading()
            
            stats = load_session_stats()
            
            if cycle == 1 and (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
                reporter = threading.Thread(target=telegram_reporter_thread, daemon=True)
                reporter.start()
                client = TInvestClient(TOKEN)
                accounts = client.get_accounts()
                if accounts:
                    account_id = accounts[0]['id']
                    send_telegram_report(client, account_id, stats)
            
            if cycle == 1 and MAX_TOKEN:
                max_reporter = threading.Thread(target=max_reporter_thread, daemon=True)
                max_reporter.start()
                send_max_report()
            
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
    parser.add_argument('--interval', '-i', type=int, default=5, help='Interval in minutes (default: 5)')
    args = parser.parse_args()
    
    if args.continuous:
        run_continuous_trading(args.interval)
    else:
        run_paper_trading()
