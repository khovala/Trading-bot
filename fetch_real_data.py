#!/usr/bin/env python3
"""
Fetch real market data from Tinkoff Invest API v2 using REST.
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import requests

TOKEN = 't.cL4pEV3CLodP2X3nTnQ3xaCsqvpO-LMCx3zBNQ5xKsKvn8USdNsT5858Ws4gMHao94JZTubIWkXMIcL-qqBQUw'

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

print("=" * 70)
print("FETCH REAL DATA FROM TINKOFF INVEST API v2")
print("=" * 70)

# API endpoints
BASE_URL = "https://invest-public-api.tinkoff.ru/rest"
SANDBOX_URL = "https://sandbox-invest-public-api.tinkoff.ru/rest"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

def api_request(url, method="GET", params=None, json_data=None):
    """Make API request."""
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        else:
            response = requests.post(url, headers=headers, json=json_data, timeout=30)
        
        print(f"  Request: {method} {url}")
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  Error: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

def main():
    # Test connection - get sandbox accounts
    print("\n[1/4] Testing connection to Tinkoff Sandbox...")
    
    url = f"{SANDBOX_URL}/tinkoff.invest.api.contract.v1.SandboxService/GetAccounts"
    result = api_request(url, method="POST", json_data={})
    
    if result:
        accounts = result.get("accounts", [])
        print(f"  Found {len(accounts)} accounts")
        for acc in accounts:
            print(f"    - {acc}")
    
    # Known FIGIs for MOEX stocks
    instruments = {
        'SBER': 'BBG004730N88',
        'GAZP': 'BBG00475KKY8',
        'LKOH': 'BBG004731428',
        'YNDX': 'BBG00B5QFGK3',
        'POLY': 'BBG00Y72NMN1',
        'NVTK': 'BBG004732LP8',
        'MGNT': 'BBG00475BF52',
        'SNGS': 'BBG004722R36',
        'SNGSP': 'BBG00475BD72',
        'TATN': 'BBG0047315D4',
    }
    
    print(f"\n[2/4] Instruments: {list(instruments.keys())}")
    
    print("\n[3/4] Fetching candles (last 7 days)...")
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    all_candles = {}
    
    for ticker, figi in instruments.items():
        url = f"{SANDBOX_URL}/tinkoff.invest.api.contract.v1.MarketDataService/GetCandles"
        
        params = {
            "figi": figi,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "interval": "1min",  # 1min, 5min, 15min, 1hour, 1day
        }
        
        result = api_request(url, method="POST", json_data=params)
        
        if result and "candles" in result:
            candles_data = result["candles"]
            candles_list = []
            for c in candles_data:
                candles_list.append({
                    'timestamp': c.get('time', ''),
                    'open': float(c.get('open', {}).get('units', 0)) + float(c.get('open', {}).get('nano', 0)) / 1e9,
                    'high': float(c.get('high', {}).get('units', 0)) + float(c.get('high', {}).get('nano', 0)) / 1e9,
                    'low': float(c.get('low', {}).get('units', 0)) + float(c.get('low', {}).get('nano', 0)) / 1e9,
                    'close': float(c.get('close', {}).get('units', 0)) + float(c.get('close', {}).get('nano', 0)) / 1e9,
                    'volume': c.get('volume', 0),
                })
            
            all_candles[ticker] = candles_list
            print(f"    {ticker}: {len(candles_list)} candles")
        else:
            print(f"    {ticker}: No candles (may be market closed)")
    
    total_candles = sum(len(c) for c in all_candles.values())
    print(f"\n  Total candles: {total_candles}")
    
    print("\n[4/4] Saving data...")
    
    candles_dir = workspace / "data/raw/market/candles_sandbox"
    candles_dir.mkdir(parents=True, exist_ok=True)
    
    for ticker, candles in all_candles.items():
        output_file = candles_dir / f"{ticker}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'ticker': ticker,
                'figi': instruments[ticker],
                'fetched_at': datetime.now(timezone.utc).isoformat(),
                'period': f"{start_date.date()} to {end_date.date()}",
                'candles': candles
            }, f, indent=2, default=str)
        print(f"    Saved {ticker}.json ({len(candles)} candles)")
    
    metadata = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
        'instruments': instruments,
        'total_candles': total_candles,
    }
    
    with open(candles_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n  Metadata saved to {candles_dir}/metadata.json")
    
    print("\n" + "=" * 70)
    print("DATA FETCH COMPLETE")
    print("=" * 70)
    print(f"\nTotal instruments: {len(all_candles)}")
    print(f"Total candles: {total_candles}")
    print(f"Data saved to: {candles_dir}")

if __name__ == "__main__":
    main()
