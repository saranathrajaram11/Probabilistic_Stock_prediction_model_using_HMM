"""
Ticker autocomplete: match symbols and company names as the user types.

Covers US / international listings and Indian NSE (.NS) / BSE (.BO) symbols.
Used by the Streamlit dashboard suggestion panel.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# US & international (no exchange suffix)
# ---------------------------------------------------------------------------
US_TICKERS: Dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. Class A",
    "GOOG": "Alphabet Inc. Class C",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc. Class B",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "UNH": "UnitedHealth Group Inc.",
    "XOM": "Exxon Mobil Corporation",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Company",
    "HD": "Home Depot Inc.",
    "CVX": "Chevron Corporation",
    "MRK": "Merck & Co. Inc.",
    "ABBV": "AbbVie Inc.",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "COST": "Costco Wholesale Corporation",
    "AVGO": "Broadcom Inc.",
    "LLY": "Eli Lilly and Company",
    "MCD": "McDonald's Corporation",
    "CSCO": "Cisco Systems Inc.",
    "DIS": "Walt Disney Company",
    "NFLX": "Netflix Inc.",
    "ADBE": "Adobe Inc.",
    "CRM": "Salesforce Inc.",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "ORCL": "Oracle Corporation",
    "BAC": "Bank of America Corporation",
    "WFC": "Wells Fargo & Company",
    "GS": "Goldman Sachs Group Inc.",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "BABA": "Alibaba Group Holding Ltd. (US ADR)",
    "TSM": "Taiwan Semiconductor Manufacturing",
    "NVO": "Novo Nordisk A/S",
    "SONY": "Sony Group Corporation",
    "BP": "BP p.l.c.",
    "SHEL": "Shell plc",
}

# ---------------------------------------------------------------------------
# India — NSE (yfinance suffix .NS)
# ---------------------------------------------------------------------------
NSE_TICKERS: Dict[str, str] = {
    # User-requested large caps & ETFs
    "RELIANCE.NS": "Reliance Industries Ltd (NSE)",
    "TCS.NS": "Tata Consultancy Services Ltd (NSE)",
    "INFY.NS": "Infosys Ltd (NSE)",
    "HDFCBANK.NS": "HDFC Bank Ltd (NSE)",
    "ICICIBANK.NS": "ICICI Bank Ltd (NSE)",
    "SBIN.NS": "State Bank of India (NSE)",
    "WIPRO.NS": "Wipro Ltd (NSE)",
    "ZOMATO.NS": "Zomato Ltd (NSE)",
    "ADANIENT.NS": "Adani Enterprises Ltd (NSE)",
    "TATAMOTORS.NS": "Tata Motors Ltd (NSE)",
    "BAJFINANCE.NS": "Bajaj Finance Ltd (NSE)",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd (NSE)",
    "AXISBANK.NS": "Axis Bank Ltd (NSE)",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd (NSE)",
    "NESTLEIND.NS": "Nestle India Ltd (NSE)",
    "GOLDBEES.NS": "Nippon India ETF Gold BeES (NSE)",
    "LIQUIDBEES.NS": "Nippon India ETF Liquid BeES (NSE)",
    "TATAGOLD.NS": "Tata Gold ETF (NSE)",
    # Additional popular NSE equities
    "ITC.NS": "ITC Ltd (NSE)",
    "LT.NS": "Larsen & Toubro Ltd (NSE)",
    "BHARTIARTL.NS": "Bharti Airtel Ltd (NSE)",
    "ASIANPAINT.NS": "Asian Paints Ltd (NSE)",
    "MARUTI.NS": "Maruti Suzuki India Ltd (NSE)",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd (NSE)",
    "HCLTECH.NS": "HCL Technologies Ltd (NSE)",
    "TITAN.NS": "Titan Company Ltd (NSE)",
    "ULTRACEMCO.NS": "UltraTech Cement Ltd (NSE)",
    "POWERGRID.NS": "Power Grid Corporation of India Ltd (NSE)",
    "NTPC.NS": "NTPC Ltd (NSE)",
    "ONGC.NS": "Oil & Natural Gas Corporation Ltd (NSE)",
    "COALINDIA.NS": "Coal India Ltd (NSE)",
    "ADANIPORTS.NS": "Adani Ports and SEZ Ltd (NSE)",
    "JSWSTEEL.NS": "JSW Steel Ltd (NSE)",
    "TATASTEEL.NS": "Tata Steel Ltd (NSE)",
    "INDIGO.NS": "InterGlobe Aviation Ltd / IndiGo (NSE)",
    "DMART.NS": "Avenue Supermarts Ltd / DMart (NSE)",
    "NYKAA.NS": "FSN E-Commerce Ventures Ltd / Nykaa (NSE)",
    "PAYTM.NS": "One 97 Communications Ltd / Paytm (NSE)",
    "BAJAJFINSV.NS": "Bajaj Finserv Ltd (NSE)",
    "BAJAJ-AUTO.NS": "Bajaj Auto Ltd (NSE)",
    "M&M.NS": "Mahindra & Mahindra Ltd (NSE)",
    "HEROMOTOCO.NS": "Hero MotoCorp Ltd (NSE)",
    "EICHERMOT.NS": "Eicher Motors Ltd (NSE)",
    "DIVISLAB.NS": "Divi's Laboratories Ltd (NSE)",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Ltd (NSE)",
    "CIPLA.NS": "Cipla Ltd (NSE)",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Ltd (NSE)",
    "TECHM.NS": "Tech Mahindra Ltd (NSE)",
    "LTIM.NS": "LTIMindtree Ltd (NSE)",
    "HDFCLIFE.NS": "HDFC Life Insurance Company Ltd (NSE)",
    "SBILIFE.NS": "SBI Life Insurance Company Ltd (NSE)",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance Company Ltd (NSE)",
    "TRENT.NS": "Trent Ltd (NSE)",
    "PIDILITIND.NS": "Pidilite Industries Ltd (NSE)",
    "GRASIM.NS": "Grasim Industries Ltd (NSE)",
    "ADANIGREEN.NS": "Adani Green Energy Ltd (NSE)",
    "ADANIPOWER.NS": "Adani Power Ltd (NSE)",
    "VEDL.NS": "Vedanta Ltd (NSE)",
    "HINDALCO.NS": "Hindalco Industries Ltd (NSE)",
    "IOC.NS": "Indian Oil Corporation Ltd (NSE)",
    "BPCL.NS": "Bharat Petroleum Corporation Ltd (NSE)",
    "GAIL.NS": "GAIL (India) Ltd (NSE)",
    "IRCTC.NS": "Indian Railway Catering & Tourism Corporation Ltd (NSE)",
    "HAL.NS": "Hindustan Aeronautics Ltd (NSE)",
    "BEL.NS": "Bharat Electronics Ltd (NSE)",
    "DLF.NS": "DLF Ltd (NSE)",
    "GODREJCP.NS": "Godrej Consumer Products Ltd (NSE)",
    "DABUR.NS": "Dabur India Ltd (NSE)",
    "BRITANNIA.NS": "Britannia Industries Ltd (NSE)",
    "COLPAL.NS": "Colgate-Palmolive (India) Ltd (NSE)",
    "MARICO.NS": "Marico Ltd (NSE)",
    "INDUSINDBK.NS": "IndusInd Bank Ltd (NSE)",
    "PNB.NS": "Punjab National Bank (NSE)",
    "BANKBARODA.NS": "Bank of Baroda (NSE)",
    "CANBK.NS": "Canara Bank (NSE)",
    "IDFCFIRSTB.NS": "IDFC First Bank Ltd (NSE)",
    "FEDERALBNK.NS": "Federal Bank Ltd (NSE)",
    "AUBANK.NS": "AU Small Finance Bank Ltd (NSE)",
    "POLICYBZR.NS": "PB Fintech Ltd / Policybazaar (NSE)",
    "NAUKRI.NS": "Info Edge (India) Ltd / Naukri (NSE)",
    "ETERNAL.NS": "Eternal Ltd (NSE)",
    "JIOFIN.NS": "Jio Financial Services Ltd (NSE)",
    # Popular NSE ETFs (BeES and others)
    "NIFTYBEES.NS": "Nippon India ETF Nifty 50 BeES (NSE)",
    "BANKBEES.NS": "Nippon India ETF Bank BeES (NSE)",
    "JUNIORBEES.NS": "Nippon India ETF Junior BeES (NSE)",
    "ITBEES.NS": "Nippon India ETF IT BeES (NSE)",
    "PSUBNKBEES.NS": "Nippon India ETF PSU Bank BeES (NSE)",
    "SILVERBEES.NS": "Nippon India ETF Silver BeES (NSE)",
    "MON100.NS": "Motilal Oswal Nifty 100 ETF (NSE)",
    "SETFNIF50.NS": "SBI ETF Nifty 50 (NSE)",
    "HDFCSILVER.NS": "HDFC Silver ETF (NSE)",
}

# ---------------------------------------------------------------------------
# India — BSE (yfinance suffix .BO)
# ---------------------------------------------------------------------------
BSE_TICKERS: Dict[str, str] = {
    "RELIANCE.BO": "Reliance Industries Ltd (BSE)",
    "TCS.BO": "Tata Consultancy Services Ltd (BSE)",
    "INFY.BO": "Infosys Ltd (BSE)",
    "HDFCBANK.BO": "HDFC Bank Ltd (BSE)",
    "ICICIBANK.BO": "ICICI Bank Ltd (BSE)",
    "SBIN.BO": "State Bank of India (BSE)",
    "ITC.BO": "ITC Ltd (BSE)",
    "LT.BO": "Larsen & Toubro Ltd (BSE)",
    "BHARTIARTL.BO": "Bharti Airtel Ltd (BSE)",
    "AXISBANK.BO": "Axis Bank Ltd (BSE)",
    "KOTAKBANK.BO": "Kotak Mahindra Bank Ltd (BSE)",
    "TATAMOTORS.BO": "Tata Motors Ltd (BSE)",
    "HINDUNILVR.BO": "Hindustan Unilever Ltd (BSE)",
    "BAJFINANCE.BO": "Bajaj Finance Ltd (BSE)",
    "WIPRO.BO": "Wipro Ltd (BSE)",
    "MARUTI.BO": "Maruti Suzuki India Ltd (BSE)",
    "ASIANPAINT.BO": "Asian Paints Ltd (BSE)",
    "NESTLEIND.BO": "Nestle India Ltd (BSE)",
    "M&M.BO": "Mahindra & Mahindra Ltd (BSE)",
    "POWERGRID.BO": "Power Grid Corporation of India Ltd (BSE)",
    "NTPC.BO": "NTPC Ltd (BSE)",
    "ONGC.BO": "Oil & Natural Gas Corporation Ltd (BSE)",
    "COALINDIA.BO": "Coal India Ltd (BSE)",
    "TATASTEEL.BO": "Tata Steel Ltd (BSE)",
    "ADANIENT.BO": "Adani Enterprises Ltd (BSE)",
    "ZOMATO.BO": "Zomato Ltd (BSE)",
    "GOLDBEES.BO": "Nippon India ETF Gold BeES (BSE)",
    "LIQUIDBEES.BO": "Nippon India ETF Liquid BeES (BSE)",
}


def _merge_ticker_maps() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for block in (US_TICKERS, NSE_TICKERS, BSE_TICKERS):
        for sym, name in block.items():
            merged[sym.upper()] = name
    return dict(sorted(merged.items()))


TICKER_NAMES: Dict[str, str] = _merge_ticker_maps()

# Mixed US + India defaults when the search box is empty
DEFAULT_HINTS = [
    "AAPL",
    "RELIANCE.NS",
    "TCS.NS",
    "NVDA",
    "HDFCBANK.NS",
    "GOLDBEES.NS",
    "INFY.NS",
    "MSFT",
]


def _base_symbol(sym: str) -> str:
    """RELIANCE.NS -> RELIANCE (for matching without exchange suffix)."""
    if "." in sym:
        return sym.rsplit(".", 1)[0]
    return sym


def _exchange_tag(sym: str) -> str:
    if sym.endswith(".NS"):
        return "NSE INDIA"
    if sym.endswith(".BO"):
        return "BSE INDIA"
    return "US INTL"


def _matches_query(sym: str, name: str, q: str) -> Tuple[int, str] | None:
    """
    Return (priority, sym) if this row matches query, else None.
    Lower priority number = ranked higher.
    """
    name_u = name.upper()
    base = _base_symbol(sym)
    exchange = _exchange_tag(sym)

    if sym == q or base == q:
        return (0, sym)
    if sym.startswith(q):
        return (1, sym)
    if base.startswith(q):
        return (2, sym)
    if q in sym or q in base:
        return (3, sym)
    if q in name_u or q in exchange:
        return (4, sym)
    # Allow "india", "nse", "bse" to surface Indian listings
    if q in ("INDIA", "NSE", "BSE", "INDIAN") and (sym.endswith(".NS") or sym.endswith(".BO")):
        return (5, sym)
    return None


def suggest_tickers(query: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Return (symbol, company_name) pairs matching the query across US, NSE, and BSE.

    Priority: exact symbol/base match, symbol prefix, base prefix, contains, name match.
    """
    q = query.strip().upper()
    if not q:
        return [(s, TICKER_NAMES[s]) for s in DEFAULT_HINTS if s in TICKER_NAMES][:limit]

    ranked: List[Tuple[int, str, str]] = []
    for sym, name in TICKER_NAMES.items():
        hit = _matches_query(sym, name, q)
        if hit is not None:
            priority, _ = hit
            ranked.append((priority, sym, name))

    ranked.sort(key=lambda row: (row[0], row[1]))

    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for _, sym, name in ranked:
        if sym in seen:
            continue
        seen.add(sym)
        out.append((sym, name))
        if len(out) >= limit:
            break
    return out
