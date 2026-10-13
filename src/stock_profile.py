"""Stock name search and a separate, current analyst opinion snapshot.

The local name index is for navigation only. Analyst figures come from Yahoo
Finance at request time and are never used as historical model features.
"""

from datetime import datetime, timezone
from functools import lru_cache
import math
import re
import time

import pandas as pd
import yfinance as yf


# Display names and familiar aliases help people who do not know tickers.
# Remote search covers the much larger universe outside this short index.
_POPULAR = (
    ("NVDA", "NVIDIA Corporation", ("nvidia",)),
    ("AAPL", "Apple Inc.", ("apple",)),
    ("MSFT", "Microsoft Corporation", ("microsoft",)),
    ("AMZN", "Amazon.com, Inc.", ("amazon",)),
    ("GOOGL", "Alphabet Inc. (Google)", ("google", "alphabet")),
    ("META", "Meta Platforms, Inc.", ("facebook", "meta")),
    ("TSLA", "Tesla, Inc.", ("tesla",)),
    ("AMD", "Advanced Micro Devices, Inc.", ("amd",)),
    ("AVGO", "Broadcom Inc.", ("broadcom",)),
    ("NFLX", "Netflix, Inc.", ("netflix",)),
    ("PLTR", "Palantir Technologies Inc.", ("palantir",)),
    ("INTC", "Intel Corporation", ("intel",)),
    ("ORCL", "Oracle Corporation", ("oracle",)),
    ("ADBE", "Adobe Inc.", ("adobe",)),
    ("CRM", "Salesforce, Inc.", ("salesforce",)),
    ("UBER", "Uber Technologies, Inc.", ("uber",)),
    ("SHOP", "Shopify Inc.", ("shopify",)),
    ("COIN", "Coinbase Global, Inc.", ("coinbase",)),
    ("DIS", "The Walt Disney Company", ("disney",)),
    ("JPM", "JPMorgan Chase & Co.", ("jpmorgan", "chase")),
    ("BAC", "Bank of America Corporation", ("bank of america",)),
    ("V", "Visa Inc.", ("visa",)),
    ("MA", "Mastercard Incorporated", ("mastercard",)),
    ("WMT", "Walmart Inc.", ("walmart",)),
    ("COST", "Costco Wholesale Corporation", ("costco",)),
    ("KO", "The Coca-Cola Company", ("coca cola", "coca-cola")),
    ("PEP", "PepsiCo, Inc.", ("pepsi", "pepsico")),
    ("NKE", "NIKE, Inc.", ("nike",)),
    ("MCD", "McDonald's Corporation", ("mcdonalds", "mcdonald's")),
    ("XOM", "Exxon Mobil Corporation", ("exxon", "exxonmobil")),
    ("QCOM", "QUALCOMM Incorporated", ("qualcomm",)),
    ("ARM", "Arm Holdings plc", ("arm",)),
    ("SOFI", "SoFi Technologies, Inc.", ("sofi",)),
    ("BRK-B", "Berkshire Hathaway Inc. Class B", ("berkshire",)),
    ("SPY", "SPDR S&P 500 ETF Trust", ("s&p 500", "spy")),
    ("QQQ", "Invesco QQQ Trust", ("nasdaq 100", "qqq")),
)

_NAMES = {symbol: name for symbol, name, _ in _POPULAR}
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


def _clean_symbol(value):
    symbol = str(value or "").strip().upper()
    return symbol if _SYMBOL.fullmatch(symbol) else ""


def _clean_name(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("|", " ").split())[:100]


def decode_selection(value):
    """Return ``(symbol, company_name)`` from a dropdown value or symbol.

    Invalid or empty selections return two empty strings. A bare known symbol
    resolves to its display name; an unknown symbol has no invented name.
    """
    if not isinstance(value, str):
        return "", ""
    raw_symbol, separator, raw_name = value.partition("|")
    symbol = _clean_symbol(raw_symbol)
    if not symbol:
        return "", ""
    return symbol, _clean_name(raw_name) if separator and _clean_name(raw_name) else _NAMES.get(symbol, "")


def _option(symbol, name):
    return {
        "label": f"{name} ({symbol})" if name else f"{symbol} (ticker)",
        "value": f"{symbol}|{name}",
    }


def local_search_options(query, selected=None, limit=8):
    """Instant matches from the small name/alias index, without network use."""
    needle = " ".join(str(query or "").casefold().split())[:80]
    if not needle:
        symbol, name = decode_selection(selected)
        return [_option(symbol, name)] if symbol else []
    scored = []
    for symbol, name, aliases in _POPULAR:
        terms = (symbol.casefold(), name.casefold(), *aliases)
        if not any(term.startswith(needle) or (len(needle) >= 3 and needle in term)
                   for term in terms):
            continue
        score = 0 if symbol.casefold().startswith(needle) else 1 if any(
            term.startswith(needle) for term in terms[1:]
        ) else 2
        scored.append((score, symbol, name))
    scored.sort()
    return [_option(symbol, name) for _, symbol, name in scored[:max(0, int(limit))]]


@lru_cache(maxsize=256)
def _remote_search(query, ten_minute_bucket):
    """Yahoo search results cached for ten minutes to reduce request volume."""
    del ten_minute_bucket
    try:
        quotes = yf.Search(
            query, max_results=12, news_count=0, lists_count=0,
            include_cb=False, timeout=5,
        ).quotes
    except Exception:
        return ()
    results = []
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        kind = str(quote.get("quoteType") or "").upper()
        if kind and kind not in {"EQUITY", "ETF"}:
            continue
        symbol = _clean_symbol(quote.get("symbol"))
        name = _clean_name(quote.get("longname") or quote.get("shortname"))
        if symbol and name:
            results.append((symbol, name))
    return tuple(results)


def search_options(query, selected=None, limit=8):
    """Dropdown options matching ticker, company name, or familiar alias.

    Popular matches return locally and immediately. Other queries use Yahoo
    Finance search, with a short timeout and a ten-minute response cache.
    """
    query = " ".join(str(query or "").split())[:80]
    limit = max(0, min(int(limit), 20))
    if not query or not limit:
        return local_search_options(query, selected, limit)
    local = local_search_options(query, limit=limit)
    if local:
        return local
    if len(query) < 2:
        return []
    seen = set()
    options = []
    for symbol, name in _remote_search(query.casefold(), int(time.time() // 600)):
        if symbol in seen:
            continue
        seen.add(symbol)
        options.append(_option(symbol, name))
        if len(options) == limit:
            break
    if not options and query == query.upper():
        symbol = _clean_symbol(query)
        if symbol:
            options.append(_option(symbol, _NAMES.get(symbol, "")))
    return options


def _nonnegative_count(value):
    if isinstance(value, bool):
        raise ValueError("Invalid analyst count")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Invalid analyst count")
    return int(number)


def _current_votes(summary):
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        raise ValueError("Current-month analyst consensus is unavailable")
    if "period" in summary.columns:
        current = summary.loc[summary["period"].astype(str).str.strip() == "0m"]
    else:
        current = summary.loc[summary.index.astype(str) == "0m"]
    if len(current) != 1:
        raise ValueError("Current-month analyst consensus is unavailable")
    row = current.iloc[0]
    counts = {key: _nonnegative_count(row[key]) for key in (
        "strongBuy", "buy", "hold", "sell", "strongSell"
    )}
    if sum(counts.values()) == 0:
        raise ValueError("Current-month analyst consensus has no ratings")
    return {
        "buy": counts["strongBuy"] + counts["buy"],
        "hold": counts["hold"],
        "sell": counts["sell"] + counts["strongSell"],
    }


def _positive_price(value):
    if isinstance(value, bool):
        raise ValueError("Invalid analyst price target")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Invalid analyst price target")
    return number


def _price_targets(targets):
    if not isinstance(targets, dict):
        raise ValueError("Analyst price targets are unavailable")
    prices = {key: _positive_price(targets[key]) for key in ("low", "mean", "high")}
    if prices["low"] > prices["mean"] or prices["mean"] > prices["high"]:
        raise ValueError("Analyst price targets are inconsistent")
    return prices


def get_analyst_snapshot(symbol):
    """Fetch today's opinion data separately from the historical prediction.

    ``checked_at`` is fetch time, not the date of each analyst opinion. Missing
    fields remain ``None``; the two provider requests fail independently.
    """
    symbol = _clean_symbol(symbol)
    if not symbol:
        raise ValueError("Choose a valid stock symbol")
    result = {
        "symbol": symbol,
        "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "period": None,
        "votes": None,
        "price_targets": None,
        "currency": None,
        "errors": {},
    }
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        result["errors"] = {
            "recommendations": "Current-month analyst consensus is unavailable",
            "price_targets": "Analyst price targets are unavailable",
        }
        return result
    try:
        result["votes"] = _current_votes(ticker.recommendations_summary)
        result["period"] = "0m"
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        result["errors"]["recommendations"] = str(exc) or "Current-month analyst consensus is unavailable"
    except Exception:
        result["errors"]["recommendations"] = "Current-month analyst consensus is unavailable"
    try:
        result["price_targets"] = _price_targets(ticker.analyst_price_targets)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        result["errors"]["price_targets"] = str(exc) or "Analyst price targets are unavailable"
    except Exception:
        result["errors"]["price_targets"] = "Analyst price targets are unavailable"
    try:
        currency = ticker.fast_info.currency
        if isinstance(currency, str) and _CURRENCY.fullmatch(currency.upper()):
            result["currency"] = currency.upper()
    except Exception:
        pass
    return result
