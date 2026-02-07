"""
Polymarket strategy: map SOL reversal signal to PM side (buy the "losing" outcome at ~0.1).
Optional: use PM timeseries / option-like payoff to suggest take-profit level.
"""
from __future__ import annotations

import json
from typing import Optional, List, Dict, Any

# Reversal: 1 = price will go up, -1 = price will go down.
# PM "Solana up" market: Yes token = up, No token = down.
# We trade the tail: buy the outcome that is currently cheap (~0.1); when SOL reverses, that outcome may go 0.6–1.
# So: reversal_up (1) -> we expect SOL to go up -> buy "Yes" on "Solana UP" market (currently "No" might be 0.9 and Yes 0.1 if market thinks down; we disagree and buy Yes at 0.1).
# Actually: "losing" side = the side the market prices low. If market thinks SOL will go UP, Yes is expensive and No is cheap. We want to buy the cheap side when we predict REVERSAL. So:
# - Predict reversal UP (SOL will go up): we want to buy the outcome that pays if SOL goes up = "Yes" on "Solana UP". The "losing" side that's at 0.1 is the one market thinks won't happen. If market thinks SOL will go DOWN, then Yes (up) is ~0.1. So we buy Yes at 0.1.
# - Predict reversal DOWN: we want to buy the outcome that pays if SOL goes down = "No" on "Solana UP" (or Yes on "Solana DOWN"). So we buy the outcome that is currently ~0.1 and that pays on SOL down.
# So mapping: reversal 1 (up) -> buy the token that pays if SOL goes UP (often labeled "Yes" on "Solana up" market). reversal -1 (down) -> buy the token that pays if SOL goes DOWN ("No" on "Solana up" or Yes on "Solana down").
# We need from Gamma/CLOB: for each market, which token is "up" and which is "down", and their prices. Then we choose the token that matches our reversal direction and has lower price (the "tail").
def pm_reversal_side(
    reversal_signal: int,
    market: Dict[str, Any],
    yes_token_id: Optional[str] = None,
    no_token_id: Optional[str] = None,
    yes_price: float = 0.5,
    no_price: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Given reversal_signal (1=up, -1=down), return which outcome to buy and at what limit price.
    market: Gamma market dict; may have outcomePrices, tokens, etc.
    Returns e.g. {"token_id": "...", "side": "BUY", "limit_price": 0.10, "outcome": "yes"|"no"}.
    """
    # Market structure: binary market has two tokens. Typically outcomePrices = ["0.45", "0.55"] for Yes/No.
    # We want to buy the outcome that corresponds to our reversal direction and is currently cheap (tail).
    # If reversal_signal == 1: we want the "SOL up" outcome -> typically "Yes" on "Solana up" market.
    # If reversal_signal == -1: we want the "SOL down" outcome -> "No" on "Solana up" or "Yes" on "Solana down".
    # So we need to know: is this market "up" or "down"? From title: "Solana up" -> Yes=up, No=down. "Solana down" -> Yes=down, No=up.
    title = (market.get("_event_title") or market.get("question") or "").lower()
    is_up_market = "up" in title and "down" not in title
    is_down_market = "down" in title

    # Resolve token IDs (Gamma: clobTokenIds list; outcomePrices list)
    tokens = market.get("clobTokenIds") or market.get("tokens") or market.get("outcomeTokens") or []
    if tokens and not yes_token_id and not no_token_id:
        yes_token_id = tokens[0] if len(tokens) > 0 else None
        no_token_id = tokens[1] if len(tokens) > 1 else None

    prices = market.get("outcomePrices") or market.get("prices") or []
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            prices = []
    if len(prices) >= 2:
        yes_price = float(prices[0]) if isinstance(prices[0], (int, float, str)) else 0.5
        no_price = float(prices[1]) if isinstance(prices[1], (int, float, str)) else 0.5
    # Prefer passed-in prices
    if reversal_signal == 1:
        # We want SOL up outcome
        if is_up_market:
            token_id = yes_token_id
            price = yes_price
            outcome = "yes"
        else:
            token_id = no_token_id
            price = no_price
            outcome = "no"
    elif reversal_signal == -1:
        if is_up_market:
            token_id = no_token_id
            price = no_price
            outcome = "no"
        else:
            token_id = yes_token_id
            price = yes_price
            outcome = "yes"
    else:
        return None

    if not token_id:
        return None
    # Place limit at 0.10 to catch the "tail"; if already above 0.2 we might still bid 0.10
    limit_price = 0.10
    return {
        "token_id": token_id,
        "side": "BUY",
        "limit_price": limit_price,
        "outcome": outcome,
        "current_price": price,
    }


def suggest_take_profit(
    entry_price: float,
    timeseries: Optional[List[Dict[str, Any]]] = None,
    target_multiple: float = 3.0,
    max_price: float = 0.95,
) -> float:
    """
    Suggest take-profit price for PM outcome. Option-like: we want to sell before resolution.
    target_multiple: e.g. 3 = sell when price is 3x entry (e.g. 0.1 -> 0.3).
    max_price: cap (e.g. 0.95) to avoid holding to expiry only.
    """
    raw = min(entry_price * target_multiple, max_price)
    # Round to 0.05 for PM
    return round(raw * 20) / 20.0
