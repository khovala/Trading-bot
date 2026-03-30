#!/usr/bin/env python3
"""
Trading Session Monitor - Local Version
Запускать на локальном компьютере: python run_monitor_local.py
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread

import requests
import urllib3
urllib3.disable_warnings()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN", "")

SESSION_STATS_FILE = Path("/tmp/paper_trading_session.json")
CHECK_INTERVAL = 300

API_BASE = 'https://sandbox-invest-public-api.tbank.ru/rest'


class TInvestClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update(self.headers)

    def _call(self, method: str, endpoint: str, data: dict = None):
        url = f"{API_BASE}/tinkoff.public.invest.api.contract.v1.{endpoint}"
        try:
            resp = self.session.post(url, json=data or {}, timeout=30)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"API error: {e}")
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

    def get_last_price(self, figi: str):
        result = self._call('POST', 'MarketDataService/GetLastPrices', {'figi': [figi]})
        if result and 'lastPrices' in result:
            for lp in result['lastPrices']:
                if lp.get('figi') == figi:
                    price_units = lp.get('price', {}).get('units', '0')
                    price_nano = lp.get('price', {}).get('nano', 0)
                    return float(f"{price_units}.{price_nano:09d}") if price_nano else float(price_units)
        return None


def load_session_stats():
    if SESSION_STATS_FILE.exists():
        try:
            return json.loads(SESSION_STATS_FILE.read_text())
        except:
            pass
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


def get_positions_detail(client, account_id):
    positions_result = client.get_positions(account_id)
    if not positions_result:
        return [], 0
    
    positions = positions_result.get('positions', [])
    positions_detail = []
    total_unrealized_pnl = 0
    
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
            
            current_price = 0
            if figi:
                try:
                    current_price = client.get_last_price(figi) or 0
                except:
                    pass
            
            pnl = 0
            if avg_price_val > 0 and current_price > 0:
                pnl = (current_price - avg_price_val) * balance
                total_unrealized_pnl += pnl
            
            positions_detail.append({
                "ticker": ticker,
                "quantity": balance,
                "avg_price": avg_price_val,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": ((current_price - avg_price_val) / avg_price_val * 100) if avg_price_val > 0 else 0
            })
    
    return positions_detail, total_unrealized_pnl


def get_account_balance(client, account_id):
    portfolio = client.get_portfolio(account_id)
    if portfolio:
        currencies = portfolio.get('totalAmountCurrencies', {})
        return float(currencies.get('units', 0)) + float(currencies.get('nano', 0)) / 1e9
    return 0


def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def format_telegram_message(stats, positions, balance, unrealized_pnl, total_pnl):
    now = datetime.now().strftime("%H:%M:%S")
    
    emoji_balance = "💰"
    emoji_pnl = "📈" if total_pnl >= 0 else "📉"
    emoji_winrate = "🎯"
    emoji_trades = "🔄"
    emoji_sl = "🛑"
    emoji_tp = "💎"
    emoji_commission = "💳"
    emoji_slippage = "〰️"
    emoji_positions = "📊"
    
    message = f"📊 *MOEX Trading Session Report*\n"
    message += f"🕐 *Время:* {now}\n\n"
    
    message += f"{emoji_balance} *Баланс счета:* `{balance:,.2f} RUB`\n\n"
    
    message += f"{emoji_pnl} *P&L:*\n"
    message += f"   • Реализованный: `{stats['cumulative_pnl']:,.2f} RUB`\n"
    message += f"   • Нереализованный: `{unrealized_pnl:,.2f} RUB`\n"
    message += f"   • Общий: `{total_pnl:,.2f} RUB`\n\n"
    
    winrate = (stats['won_trades']/stats['total_trades']*100) if stats['total_trades'] > 0 else 0
    message += f"{emoji_winrate} *Winrate:* `{stats['won_trades']}/{stats['total_trades']} = {winrate:.1f}%`\n\n"
    
    message += f"{emoji_trades} *Сделки:*\n"
    message += f"   • BUY: `{stats['buy_deals']}`\n"
    message += f"   • SELL: `{stats['sell_deals']}`\n"
    message += f"   • Закрыто по SL: `{stats['sl_closed']}`\n"
    message += f"   • Закрыто по TP: `{stats['tp_closed']}`\n\n"
    
    message += f"{emoji_commission} *Комиссия:* `{stats['commission']:,.2f} RUB`\n"
    message += f"{emoji_slippage} *Проскальзывание:* `{stats['slippage']:,.2f} RUB`\n\n"
    
    if positions:
        message += f"{emoji_positions} *Открытые позиции:*\n"
        for pos in sorted(positions, key=lambda x: x['pnl'], reverse=True):
            emoji = "🟢" if pos['pnl'] >= 0 else "🔴"
            message += f"   {emoji} {pos['ticker']}: {pos['quantity']} шт @ {pos['avg_price']:.2f} → {pos['current_price']:.2f} ({pos['pnl_pct']:+.2f}%)\n"
    
    return message


def main():
    logger.info("Starting Trading Session Monitor (Local Version)")
    
    if not TINKOFF_TOKEN:
        logger.error("TINKOFF_TOKEN not set. Set: export TINKOFF_TOKEN=your_token")
        return
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return
    
    client = TInvestClient(TINKOFF_TOKEN)
    accounts = client.get_accounts()
    
    if not accounts:
        logger.error("No accounts found. Run paper trading first!")
        return
    
    account_id = accounts[0]['id']
    logger.info(f"Monitoring account: {account_id}")
    
    while True:
        try:
            stats = load_session_stats()
            positions, unrealized_pnl = get_positions_detail(client, account_id)
            balance = get_account_balance(client, account_id)
            total_pnl = stats['cumulative_pnl'] + unrealized_pnl
            
            message = format_telegram_message(stats, positions, balance, unrealized_pnl, total_pnl)
            send_telegram_message(message)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
