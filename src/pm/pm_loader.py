"""
Polymarket data: Gamma API for market discovery, CLOB for order book and prices.
Target: Solana up/down markets (15min, 1h) with good liquidity.
Gamma returns outcomePrices and clobTokenIds as JSON strings; we parse them.
"""
from __future__ import annotations

import json
import re
from typing import Optional, List, Dict, Any

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _get(url: str, params: Optional[dict] = None, timeout: int = 15) -> Any:
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def list_events(
    closed: bool = False,
    limit: int = 100,
    offset: int = 0,
    order: str = "id",
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch events from Gamma API."""
    url = f"{GAMMA_BASE}/events"
    params = {
        "closed": str(closed).lower(),
        "limit": limit,
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    return _get(url, params)


def list_markets(
    closed: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch markets from Gamma API."""
    url = f"{GAMMA_BASE}/markets"
    params = {"closed": str(closed).lower(), "limit": limit, "offset": offset}
    return _get(url, params)


def search_events_or_markets(
    query: str,
    search_events: bool = True,
    closed: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search events (or markets) by text. Uses Gamma search if available."""
    # Gamma has GET /events and GET /markets; no explicit search endpoint in docs.
    # So we fetch recent events/markets and filter by title/description.
    if search_events:
        items = list_events(closed=closed, limit=limit * 2)
    else:
        items = list_markets(closed=closed, limit=limit * 2)
    q = query.lower()
    out = []
    for e in items:
        title = (e.get("title") or "").lower()
        desc = (e.get("description") or "").lower()
        if q in title or q in desc:
            out.append(e)
            if len(out) >= limit:
                break
    return out


def list_solana_markets(
    keywords: Optional[List[str]] = None,
    preferred_timeframes: Optional[List[str]] = None,
    closed: bool = False,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """
    Find Solana-related binary markets (e.g. SOL up/down 15min, 1h).
    Returns list of market/event objects with token IDs for CLOB.
    """
    keywords = keywords or ["solana", "sol ", "sol up", "sol down"]
    preferred_timeframes = preferred_timeframes or ["15", "15min", "1h", "1 hour"]
    events = list_events(closed=closed, limit=200)
    out = []
    for ev in events:
        title = (ev.get("title") or "").lower()
        desc = (ev.get("description") or "").lower()
        if not any(kw in title or kw in desc for kw in keywords):
            continue
        has_timeframe = any(tf in title or tf in desc for tf in preferred_timeframes)
        markets = ev.get("markets") or []
        for m in markets:
            if m.get("closed") or not m.get("acceptingOrders", True):
                continue
            m["_event_title"] = ev.get("title")
            m["_event_slug"] = ev.get("slug")
            m["_has_preferred_timeframe"] = has_timeframe
            # Parse JSON strings from Gamma
            for key, parse_key in [("outcomePrices", "outcomePrices"), ("clobTokenIds", "clobTokenIds")]:
                val = m.get(key)
                if isinstance(val, str):
                    try:
                        m[parse_key] = json.loads(val)
                    except json.JSONDecodeError:
                        m[parse_key] = val
            out.append(m)
            if len(out) >= limit:
                return out
    return out


def get_market_book(token_id: str) -> Dict[str, Any]:
    """CLOB order book for a token (outcome)."""
    url = f"{CLOB_BASE}/book"
    params = {"token_id": token_id}
    return _get(url, params)


def get_market_prices(token_id: str) -> Dict[str, Any]:
    """CLOB price for a token."""
    url = f"{CLOB_BASE}/price"
    params = {"token_id": token_id}
    return _get(url, params)


def get_midpoint(token_id: str) -> Optional[float]:
    """Midpoint price for token."""
    url = f"{CLOB_BASE}/midpoint"
    params = {"token_id": token_id}
    try:
        data = _get(url, params)
        return float(data.get("mid") or data.get("price", 0))
    except Exception:
        return None


def get_timeseries(
    market: str,
    interval: str = "max",
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Historical timeseries for a market (CLOB)."""
    url = f"{CLOB_BASE}/timeseries"
    params = {"market": market, "interval": interval}
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    try:
        return _get(url, params)
    except Exception:
        return []
