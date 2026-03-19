#!/usr/bin/env python3
"""
Tinkoff Invest API v2 client for real market data.
NOTE: API endpoints may have changed. Update URLs as needed.
"""

import sys
sys.path.insert(0, '/Users/sergeyeliseev/moex-sandbox-platform')

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import grpc
from typing import List, Dict, Any

TOKEN = 't.cL4pEV3CLodP2X3nTnQ3xaCsqvpO-LMCx3zBNQ5xKsKvn8USdNsT5858Ws4gMHao94JZTubIWkXMIcL-qqBQUw'

workspace = Path('/Users/sergeyeliseev/moex-sandbox-platform')

# Tinkoff Invest API v2 endpoints
SANDBOX_API = "sandbox-invest-public-api.tinkoff.ru:443"
PRODUCTION_API = "invest-public-api.tinkoff.ru:443"

# Known MOEX instrument FIGIs
INSTRUMENTS = {
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

def cast_money(value: Any) -> float:
    """Convert Quotation to float."""
    if hasattr(value, 'units') and hasattr(value, 'nano'):
        return float(value.units) + float(value.nano) / 1e9
    elif isinstance(value, dict):
        return float(value.get('units', 0)) + float(value.get('nano', 0)) / 1e9
    return float(value)

def cast_time(time_str: str) -> datetime:
    """Parse timestamp string."""
    if time_str.endswith('Z'):
        time_str = time_str[:-1] + '+00:00'
    return datetime.fromisoformat(time_str)

async def fetch_candles(figi: str, days: int = 7, interval: str = "1min") -> List[Dict]:
    """Fetch candles from Tinkoff API."""
    import asyncio
    
    # Try to use official SDK if available
    try:
        from tinkoff.invest import AsyncClient, CandleInterval
        
        interval_map = {
            "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
            "1hour": CandleInterval.CANDLE_INTERVAL_HOUR,
            "1day": CandleInterval.CANDLE_INTERVAL_DAY,
        }
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        async with AsyncClient(TOKEN) as client:
            candles_data = []
            async for candle in client.get_all_candles(
                figi=figi,
                from_=start_date,
                to=end_date,
                interval=interval_map.get(interval, CandleInterval.CANDLE_INTERVAL_1_MIN),
            ):
                candles_data.append({
                    'timestamp': candle.time.isoformat(),
                    'open': cast_money(candle.open),
                    'high': cast_money(candle.high),
                    'low': cast_money(candle.low),
                    'close': cast_money(candle.close),
                    'volume': candle.volume,
                })
            return candles_data
    except ImportError:
        print("  Tinkoff SDK not installed. Use: pip install tinkoff-invest-python")
        return []

async def main():
    print("=" * 70)
    print("TINKOFF INVEST API v2 - REAL DATA FETCH")
    print("=" * 70)
    
    print("\n[1/3] Checking API connection...")
    
    try:
        from tinkoff.invest import AsyncClient
        
        async with AsyncClient(TOKEN) as client:
            accounts = await client.users.get_accounts()
            print(f"  Connected! Found {len(accounts.accounts)} accounts")
            for acc in accounts.accounts:
                print(f"    - {acc.id}: {acc.name}")
    except ImportError:
        print("  ERROR: tinkoff-invest-python not installed")
        print("  Run: pip install tinkoff-invest-python")
        print("\n  Alternative: use REST API directly")
        return
    except Exception as e:
        print(f"  ERROR: {e}")
        print("\n  Possible issues:")
        print("  1. Token expired or invalid")
        print("  2. API endpoint changed")
        print("  3. Network issues")
        return
    
    print("\n[2/3] Fetching candles (7 days, 1min interval)...")
    
    candles_dir = workspace / "data/raw/market/candles_sandbox"
    candles_dir.mkdir(parents=True, exist_ok=True)
    
    all_candles = {}
    
    for ticker, figi in INSTRUMENTS.items():
        try:
            candles = await fetch_candles(figi, days=7, interval="1min")
            if candles:
                all_candles[ticker] = candles
                print(f"    {ticker}: {len(candles)} candles")
                
                # Save individual ticker data
                with open(candles_dir / f"{ticker}.json", 'w') as f:
                    json.dump({
                        'ticker': ticker,
                        'figi': figi,
                        'fetched_at': datetime.now(timezone.utc).isoformat(),
                        'candles': candles
                    }, f, indent=2, default=str)
            else:
                print(f"    {ticker}: No candles (market closed?)")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")
    
    print("\n[3/3] Saving metadata...")
    
    metadata = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'instruments': INSTRUMENTS,
        'total_candles': sum(len(c) for c in all_candles.values()),
    }
    
    with open(candles_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total candles: {sum(len(c) for c in all_candles.values())}")
    print(f"Saved to: {candles_dir}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
