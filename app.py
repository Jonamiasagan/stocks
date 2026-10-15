"""
AlphaEngine: Autonomous Stock Discovery, Research & Continuous Risk Monitor
==========================================================================
A production-grade, institutional buyside equity intelligence and portfolio risk platform.
Integrates:
- Tab 1: Autonomous Opportunity Finder (Fundamental Screener & Compounder Discovery)
- Tab 2: 5-Pillar Institutional Research Dossier (Deep Forensic & Adversarial AI Audit)
- Tab 3: Continuous Portfolio Watchdog (Thesis Health Tracker & AI CIO Risk Advisor)

Built with Streamlit, yfinance, and the modern Google GenAI SDK (gemini-2.5-flash).
"""

import os
import io
import json
import datetime
import concurrent.futures
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import streamlit as st
import yfinance as yf
import pypdf
from dotenv import load_dotenv

# Modern Google GenAI SDK
from google import genai
from google.genai import types

# Load local environment if present
load_dotenv()

# ==============================================================================
# 1. STREAMLIT APP CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="AlphaEngine | Autonomous Equity Intelligence & Risk Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional Dark Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Brand Bar */
    .brand-container {
        background: linear-gradient(135deg, #0B1120 0%, #162035 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 25px -4px rgba(0, 0, 0, 0.5);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.025em;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.28);
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .brand-desc {
        color: #94A3B8;
        font-size: 0.94rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* KPI Cards */
    .kpi-card {
        background: #111B2E;
        border: 1px solid #1E2E4A;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-intact {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-caution {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-breached {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Section Headers */
    .module-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 10px;
    }
    .module-tag {
        background: #1E293B;
        color: #38BDF8;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Modern Primary Action Buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.35rem !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WATCHDOG_FILE = "portfolio_watchdog.json"

# ==============================================================================
# 2. PERSISTENCE HELPERS (WATCHDOG JSON & STATE)
# ==============================================================================
def load_watchdog_portfolio() -> List[Dict[str, Any]]:
    """Load tracked watchlist from local JSON cache with default fallback."""
    if os.path.exists(WATCHDOG_FILE):
        try:
            with open(WATCHDOG_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass

    # Default institutional starter watchlist
    default_holdings = [
        {
            "ticker": "SUZLON.NS",
            "name": "Suzlon Energy Limited",
            "entryPrice": 38.5,
            "targetFairValue": 55.0,
            "currency": "INR",
            "invalidationRule": "Exit if Operating Margin drops below 10% or D/E rises above 0.25",
            "minOperatingMargin": 0.10,
            "maxDebtToEquity": 0.25,
            "status": "Thesis Intact",
            "statusBadge": "🟢",
            "lastAudited": "2026-10-15",
        },
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "entryPrice": 220.0,
            "targetFairValue": 275.0,
            "currency": "USD",
            "invalidationRule": "Exit if Services gross margin contracts or FCF conversion drops below 80%",
            "minOperatingMargin": 0.25,
            "maxDebtToEquity": 1.50,
            "status": "Thesis Intact",
            "statusBadge": "🟢",
            "lastAudited": "2026-10-15",
        },
        {
            "ticker": "TCS.NS",
            "name": "Tata Consultancy Services",
            "entryPrice": 4100.0,
            "targetFairValue": 4800.0,
            "currency": "INR",
            "invalidationRule": "Exit if BFSI vertical growth turns negative for two quarters or EBIT margin < 23%",
            "minOperatingMargin": 0.23,
            "maxDebtToEquity": 0.10,
            "status": "Thesis Intact",
            "statusBadge": "🟢",
            "lastAudited": "2026-10-15",
        },
    ]
    save_watchdog_portfolio(default_holdings)
    return default_holdings


def save_watchdog_portfolio(holdings: List[Dict[str, Any]]):
    """Save tracked watchlist to local JSON file."""
    try:
        with open(WATCHDOG_FILE, "w") as f:
            json.dump(holdings, f, indent=2)
    except Exception as e:
        st.error(f"Error saving portfolio watchdog: {e}")


# Initialize Session State
if "watchdog_list" not in st.session_state:
    st.session_state["watchdog_list"] = load_watchdog_portfolio()
if "target_ticker" not in st.session_state:
    st.session_state["target_ticker"] = "SUZLON.NS"
if "peer_ticker" not in st.session_state:
    st.session_state["peer_ticker"] = "INOXWIND.NS"
if "thesis_text" not in st.session_state:
    st.session_state["thesis_text"] = "Expanding wind turbine order book (>5GW), 30%+ operating margin trajectory, turning net-cash with aggressive multi-year Indian renewable capacity expansion."
if "audit_completed" not in st.session_state:
    st.session_state["audit_completed"] = False
if "target_snapshot" not in st.session_state:
    st.session_state["target_snapshot"] = None
if "peer_snapshot" not in st.session_state:
    st.session_state["peer_snapshot"] = None
if "module_outputs" not in st.session_state:
    st.session_state["module_outputs"] = {}
if "full_dossier_md" not in st.session_state:
    st.session_state["full_dossier_md"] = ""
if "discovery_results" not in st.session_state:
    st.session_state["discovery_results"] = None


# ==============================================================================
# 3. FORMATTING UTILITIES & FILE PARSING
# ==============================================================================
def format_currency_val(val: Optional[float], currency: str = "USD") -> str:
    """Format large numbers into institutional abbreviations (B, M, Cr, Lakh)."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "N/A"

    curr_sym = "₹" if currency == "INR" else ("$" if currency == "USD" else f"{currency} ")

    if currency == "INR":
        abs_v = abs(val)
        if abs_v >= 1e7:
            return f"{curr_sym}{val / 1e7:,.2f} Cr"
        elif abs_v >= 1e5:
            return f"{curr_sym}{val / 1e5:,.2f} L"
        else:
            return f"{curr_sym}{val:,.2f}"
    else:
        abs_v = abs(val)
        if abs_v >= 1e9:
            return f"{curr_sym}{val / 1e9:,.2f}B"
        elif abs_v >= 1e6:
            return f"{curr_sym}{val / 1e6:,.2f}M"
        elif abs_v >= 1e3:
            return f"{curr_sym}{val / 1e3:,.2f}K"
        else:
            return f"{curr_sym}{val:,.2f}"


def format_ratio_val(val: Optional[float], suffix: str = "x", decimal_places: int = 2) -> str:
    """Format valuation multiples and ratios."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        val = float(val)
        return f"{val:.{decimal_places}f}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def format_percent_val(val: Optional[float], decimal_places: int = 2) -> str:
    """Format percentages cleanly, handling both 0.15 and 15.0 format."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        val = float(val)
        if abs(val) <= 1.0 and val != 0.0:
            val = val * 100
        return f"{val:.{decimal_places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def extract_text_from_file(uploaded_file) -> str:
    """Extract raw text from uploaded .txt or .pdf files."""
    if uploaded_file is None:
        return ""
    try:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text_pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_pages.append(f"--- [Page {i+1}] ---\n{text}")
            return "\n\n".join(text_pages)
        else:
            return uploaded_file.getvalue().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Error parsing file: {str(e)}]"


# Peer Auto-Matcher Dictionary
PEER_MAP = {
    "SUZLON.NS": "INOXWIND.NS",
    "INOXWIND.NS": "SUZLON.NS",
    "TCS.NS": "INFY.NS",
    "INFY.NS": "TCS.NS",
    "HDFCBANK.NS": "ICICIBANK.NS",
    "ICICIBANK.NS": "HDFCBANK.NS",
    "TATAMOTORS.NS": "M&M.NS",
    "M&M.NS": "TATAMOTORS.NS",
    "RELIANCE.NS": "ONGC.NS",
    "EMAMIPAP.NS": "JKPAPER.NS",
    "JPPOWER.NS": "RPOWER.NS",
    "PMCFIN.BO": "GEOJITFSL.NS",
    "AAPL": "MSFT",
    "MSFT": "AAPL",
    "NVDA": "AMD",
    "AMD": "NVDA",
    "AMZN": "WMT",
    "GOOGL": "META",
    "META": "GOOGL",
}


# ==============================================================================
# 4. QUANTITATIVE EXTRACTION ENGINE (yfinance)
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_financial_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive institutional quantitative snapshot using yfinance.
    Implements statement inspection fallbacks for international tickers (.NS/.BO).
    """
    sym = ticker.strip().upper()
    if not sym:
        return {"error": "Empty ticker symbol"}

    try:
        t = yf.Ticker(sym)
        info = t.info or {}

        if not info or ("shortName" not in info and "regularMarketPrice" not in info and "previousClose" not in info):
            hist = t.history(period="5d")
            if hist.empty:
                return {
                    "ticker": sym,
                    "error": f"Ticker '{sym}' was not found or has no active trading data on Yahoo Finance.",
                }

        cfo = info.get("operatingCashflow")
        fcf = info.get("freeCashflow")
        total_debt = info.get("totalDebt")
        cash_and_equiv = info.get("totalCash") or info.get("cashAndCashEquivalents")

        # Statement deep-inspection fallback
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                if cfo is None:
                    for cfo_key in ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]:
                        if cfo_key in cf.index:
                            cfo = float(cf.loc[cfo_key].iloc[0])
                            break
                if fcf is None:
                    if "Free Cash Flow" in cf.index:
                        fcf = float(cf.loc["Free Cash Flow"].iloc[0])
                    elif cfo is not None and "Capital Expenditure" in cf.index:
                        capex = float(cf.loc["Capital Expenditure"].iloc[0])
                        fcf = cfo + capex if capex < 0 else cfo - capex
        except Exception:
            pass

        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                if total_debt is None:
                    for debt_key in ["Total Debt", "Total Liabilities Net Minority Interest", "Long Term Debt And Capital Lease Obligation"]:
                        if debt_key in bs.index:
                            total_debt = float(bs.loc[debt_key].iloc[0])
                            break
                if cash_and_equiv is None:
                    for cash_key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                        if cash_key in bs.index:
                            cash_and_equiv = float(bs.loc[cash_key].iloc[0])
                            break
        except Exception:
            pass

        # ROE & OPM Financial Statement Fallback (Crucial for Indian NSE/BSE tickers)
        roe_val = info.get("returnOnEquity")
        opm_val = info.get("operatingMargins")
        try:
            fin = t.financials
            bs = t.balance_sheet
            if roe_val is None and fin is not None and bs is not None and not fin.empty and not bs.empty:
                net_inc = None
                for k in ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"]:
                    if k in fin.index:
                        net_inc = float(fin.loc[k].iloc[0])
                        break
                eq = None
                for k in ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"]:
                    if k in bs.index:
                        eq = float(bs.loc[k].iloc[0])
                        break
                if net_inc is not None and eq is not None and eq != 0:
                    roe_val = net_inc / eq

            if opm_val is None and fin is not None and not fin.empty:
                ebit = None
                for k in ["Operating Income", "EBIT"]:
                    if k in fin.index:
                        ebit = float(fin.loc[k].iloc[0])
                        break
                rev = None
                for k in ["Total Revenue", "Operating Revenue"]:
                    if k in fin.index:
                        rev = float(fin.loc[k].iloc[0])
                        break
                if ebit is not None and rev is not None and rev != 0:
                    opm_val = ebit / rev
        except Exception:
            pass

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or info.get("open")
        )
        if price is None:
            hist = t.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        currency = info.get("currency", "USD")

        # Normalized D/E
        de_val = info.get("debtToEquity")
        if de_val is not None:
            normalized_de = de_val / 100.0 if (de_val > 5.0 and de_val < 10000) else de_val
        else:
            normalized_de = None

        return {
            "ticker": sym,
            "name": info.get("shortName") or info.get("longName") or sym,
            "currency": currency,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currentPrice": price,
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "operatingMargins": opm_val,
            "profitMargins": info.get("profitMargins"),
            "returnOnEquity": roe_val,
            "returnOnAssets": info.get("returnOnAssets"),
            "totalDebt": total_debt,
            "cashAndEquivalents": cash_and_equiv,
            "debtToEquity": normalized_de,
            "currentRatio": info.get("currentRatio"),
            "operatingCashflow": cfo,
            "freeCashflow": fcf,
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "sharesOutstanding": info.get("sharesOutstanding"),
        }
    except Exception as e:
        return {"ticker": sym, "error": str(e)}


# Parallel Batch Screener for Tab 1
@st.cache_data(ttl=600, show_spinner=False)
def fetch_universe_snapshots(tickers: Sequence[str]) -> List[Dict[str, Any]]:
    """Fetch raw financial snapshots in parallel across concurrent threads."""
    snapshots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_financial_snapshot, sym): sym for sym in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res and "error" not in res and res.get("currentPrice"):
                snapshots.append(res)
    return snapshots



def evaluate_hurdles_and_score(
    snapshots: List[Dict[str, Any]],
    min_roe: float,
    max_de: float,
    min_opm: float,
    require_fcf: bool,
    max_peg: float,
) -> List[Dict[str, Any]]:
    """Instant in-memory evaluation of cached snapshots against hurdle criteria."""
    results = []
    for snap in snapshots:
        roe = snap.get("returnOnEquity") or 0.0
        de = snap.get("debtToEquity") or 0.0
        opm = snap.get("operatingMargins") or 0.0
        fcf = snap.get("freeCashflow") or 0.0
        pe = snap.get("trailingPE")
        peg = snap.get("pegRatio")
        curr = snap.get("currency", "USD")

        # Normalize percentages
        roe_pct = roe * 100 if abs(roe) <= 1.0 and roe != 0 else roe
        opm_pct = opm * 100 if abs(opm) <= 1.0 and opm != 0 else opm

        # Filter Evaluations
        passed_roe = roe_pct >= min_roe
        passed_de = de <= max_de
        passed_opm = opm_pct >= min_opm
        passed_fcf = (fcf > 0) if require_fcf else True
        passed_peg = (peg <= max_peg) if (peg and peg > 0) else True

        all_passed = passed_roe and passed_de and passed_opm and passed_fcf and passed_peg

        # Composite Quality Score (0 to 100)
        score = 0
        if roe_pct >= 25: score += 25
        elif roe_pct >= 15: score += 15
        elif roe_pct >= 10: score += 8

        if fcf > 0: score += 20
        if de <= 0.2: score += 20
        elif de <= 0.5: score += 12

        if opm_pct >= 20: score += 20
        elif opm_pct >= 12: score += 12

        if (pe and pe <= 25) or (peg and peg <= 1.5): score += 15
        elif (pe and pe <= 35): score += 8

        results.append({
            "Ticker": snap.get("ticker"),
            "Company": snap.get("name")[:24],
            "Sector": snap.get("sector", "N/A"),
            "Price": f"{curr} {snap.get('currentPrice', 'N/A')}",
            "P/E": format_ratio_val(pe, ""),
            "PEG": format_ratio_val(peg, ""),
            "ROE %": f"{roe_pct:.1f}%",
            "D/E": format_ratio_val(de, ""),
            "FCF": format_currency_val(fcf, curr),
            "Quality Score": score,
            "Filter Status": "Passed All" if all_passed else "Failed Hurdle",
            "all_passed": all_passed,
            "raw_snapshot": snap,
        })

    results.sort(key=lambda x: (x["all_passed"], x["Quality Score"]), reverse=True)
    return results


# ==============================================================================
# 5. LLM RESEARCH & FORENSIC ORCHESTRATION PIPELINE
# ==============================================================================
SYSTEM_INSTRUCTION_AUDIT = """
You are an elite Buyside Chief Investment Officer (CIO) and Forensic Auditor at a premier institutional fund.
You produce rigorous, unhedged, data-dense fundamental research with forensic skepticism.
You cut through corporate PR and executive hyperbole.
Ground every statement in unit economics, capital allocation discipline, cash conversion reality, and margin-of-safety principles.
Never generate boilerplate text. Provide quantifiable conclusions and institutional-grade reasoning.
"""

def generate_llm_response(
    api_key: str,
    prompt: str,
    system_instruction: str = SYSTEM_INSTRUCTION_AUDIT,
    temperature: float = 0.2,
) -> str:
    """Execute Gemini prompt using modern google-genai Client."""
    try:
        client = genai.Client(api_key=api_key)
        for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=temperature,
                    ),
                )
                if response.text:
                    return response.text
            except Exception:
                continue
        return "[No response text generated by model]"
    except Exception as e:
        return f"### [Generation Error]\n\n`{str(e)}`\n\nPlease verify your API key and network connectivity."


def build_module_1_prompt(target: dict, peer: dict, thesis: str) -> str:
    return f"""
# TASK: MODULE 1 - BUSINESS DURABILITY & MOAT ARCHITECTURE AUDIT

Target Company: {target.get('name')} ({target.get('ticker')})
Sector: {target.get('sector')} | Industry: {target.get('industry')}
Market Cap: {target.get('currency')} {target.get('marketCap')} | Price: {target.get('currentPrice')}
Operating Margin: {target.get('operatingMargins')} | Net Margin: {target.get('profitMargins')}
Operating Cash Flow: {target.get('operatingCashflow')} | Free Cash Flow: {target.get('freeCashflow')}
Total Debt: {target.get('totalDebt')} | Cash: {target.get('cashAndEquivalents')} | D/E: {target.get('debtToEquity')}

Peer Benchmark: {peer.get('name')} ({peer.get('ticker')})
Peer OPM: {peer.get('operatingMargins')} | Peer D/E: {peer.get('debtToEquity')} | Peer FCF: {peer.get('freeCashflow')}

Initial Investor Thesis:
\"\"\"{thesis}\"\"\"

Conduct a forensic analysis covering:
1. **Unit Economics & Revenue Quality**: Dissect contract structures, recurring vs. lumpy/project-based exposure, customer concentration, and order backlog conversion rate.
2. **Economic Moat Durability**: Evaluate switching costs, network effects, cost advantages, scale economies vs. rival {peer.get('ticker')}. Is the moat widening or decaying?
3. **Capital Allocation & Accruals Audit**: Free Cash Flow conversion vs. reported Net Profit, CAPEX discipline (maintenance vs growth), and debt maturity profile.

OUTPUT FORMAT:
- Markdown comparison table:
  | Evaluation Pillar | {target.get('ticker')} Evaluation | {peer.get('ticker')} Benchmark | Moat Advantage / Divergence |
- Followed by analytical sections:
  - `### Unit Economics & Revenue Quality`
  - `### Moat Durability & Competitive Advantage`
  - `### Capital Allocation, Accruals & Balance Sheet Hygiene`
"""


def build_module_2_prompt(target: dict, earnings_text: str, thesis: str) -> str:
    transcript_context = earnings_text.strip() if earnings_text.strip() else "[No transcript provided. Base analysis on reported financials, industry dynamics, and consensus execution]."
    return f"""
# TASK: MODULE 2 - FORENSIC TRANSCRIPT & GROUND-TRUTH AUDIT

Target Company: {target.get('name')} ({target.get('ticker')})
Margins / Financials: OPM: {target.get('operatingMargins')} | PE: {target.get('trailingPE')} | FCF: {target.get('freeCashflow')}

Earnings Call Material / Management Excerpt:
\"\"\"{transcript_context[:45000]}\"\"\"

Base Thesis:
\"\"\"{thesis}\"\"\"

Perform a forensic transcript and executive narrative audit:
1. **Metric vs. Narrative Divergence**: Contrast management's optimism against raw quantitative realities (margin compression, inventory build-up, receivables stretch, or debt burden).
2. **Evasive Language Check**: Flag instances where leadership dodged concrete numbers, provided qualitative platitudes, or altered guidance metrics.
3. **Execution Drift**: Track lowered guidance, delayed milestone timelines, deferred capacity expansion, or working capital build-up.

OUTPUT FORMAT:
- Exactly 4 concise, high-impact forensic bullet points highlighting the most severe divergences/red flags.
- Follow each bullet with direct illustrative management quotes or transcript evidence in blockquotes with forensic commentary.
- Conclude with an `Executive Trust & Execution Score` (scale of 1 to 10 with institutional justification).
"""


def build_module_3_prompt(target: dict, peer: dict) -> str:
    return f"""
# TASK: MODULE 3 - 360-DEGREE SCENARIO MODELING (3 TO 5 YEARS)

Target Company: {target.get('name')} ({target.get('ticker')})
Current Price: {target.get('currency')} {target.get('currentPrice')}
Trailing P/E: {target.get('trailingPE')} | Forward P/E: {target.get('forwardPE')} | PEG: {target.get('pegRatio')}
EV/EBITDA: {target.get('enterpriseToEbitda')} | EV/Sales: {target.get('enterpriseToRevenue')}
Operating Margin: {target.get('operatingMargins')} | ROE: {target.get('returnOnEquity')}
Peer Multiples: {peer.get('name')} ({peer.get('ticker')}) P/E: {peer.get('trailingPE')} | EV/EBITDA: {peer.get('enterpriseToEbitda')}

Conduct a multi-scenario valuation over a 3 to 5-year investment horizon:
1. **Baseline Multiple Context**: Compare current multiples against historical sector medians and peer averages.
2. **Model 3 Scenarios**:
   - **Bull Case**: Top-decile execution, high revenue CAGR, operating leverage, multiple expansion.
   - **Base Case**: Consensus steady-state, stable margins, mean-reverting multiple.
   - **Bear Case**: Growth halves, 300+ bps margin compression, multiple de-rating.

OUTPUT FORMAT:
- Present a Markdown Scenario Matrix Table with EXACTLY these columns:
  | Scenario | Revenue CAGR | Operating Margin | Valuation Multiple | Implied Fair Value ({target.get('currency')}) | Expected Return (%) | Key Catalysts / Invalidation Risks |
- Follow with:
  - `### Valuation Sensitivity & Margin of Safety Analysis` (analyzing risk/reward asymmetry at current price {target.get('currency')} {target.get('currentPrice')}).
"""


def build_module_4_prompt(target: dict, peer: dict) -> str:
    return f"""
# TASK: MODULE 4 - COMPETITIVE FACE-OFF (TARGET VS. PEER)

Asset A (Target): {target.get('name')} ({target.get('ticker')})
- Market Cap: {target.get('currency')} {target.get('marketCap')} | Price: {target.get('currentPrice')}
- OPM: {target.get('operatingMargins')} | D/E: {target.get('debtToEquity')} | FCF: {target.get('freeCashflow')} | P/E: {target.get('trailingPE')}

Asset B (Peer Benchmark): {peer.get('name')} ({peer.get('ticker')})
- Market Cap: {peer.get('currency')} {peer.get('marketCap')} | Price: {peer.get('currentPrice')}
- OPM: {peer.get('operatingMargins')} | D/E: {peer.get('debtToEquity')} | FCF: {peer.get('freeCashflow')} | P/E: {peer.get('trailingPE')}

Conduct an adversarial head-to-head competitive assessment:
1. **Operational Divergence**: Capital efficiency, asset turnover, gross/operating margin durability.
2. **Balance Sheet Vulnerability**: Solvency buffer and credit risk in a downturn.
3. **Summary Decision Matrix**: Rank both companies explicitly across:
   (1) Pricing Power
   (2) Capital Efficiency
   (3) Margin of Safety

OUTPUT FORMAT:
- Head-to-head operational analysis.
- Summary Decision Matrix table:
  | Evaluation Pillar | {target.get('ticker')} Score (1-10) | {peer.get('ticker')} Score (1-10) | Institutional Winner | Rationale |
- Conclude with a definitive one-paragraph institutional allocation verdict.
"""


def build_module_5_prompt(target: dict, peer: dict, thesis: str) -> str:
    return f"""
# TASK: MODULE 5 - ADVERSARIAL RED-TEAM STRESS TEST (SHORT-SELLER AUDIT)

Target Company: {target.get('name')} ({target.get('ticker')})
Current Price: {target.get('currency')} {target.get('currentPrice')} | Debt: {target.get('totalDebt')} | Cash: {target.get('cashAndEquivalents')}
P/E: {target.get('trailingPE')} | Beta: {target.get('beta')}

Initial Investor Thesis:
\"\"\"{thesis}\"\"\"

Act as a forensic short-seller and Head of Risk Management preparing an adversarial short audit:
1. **Top 3 Structural Blind Spots**: Demolish the base thesis. Identify 3 fatal flaws or hidden risks the bull case ignores.
2. **Macro & Micro Sensitivity Stress Test**:
   - +100 bps rate hikes (debt service shock).
   - Foreign exchange / raw material price spikes.
   - Regulatory shift or subsidized rival entry.
3. **Quantifiable Invalidation Triggers (Non-Negotiable Portfolio Kill Switches)**:
   - Provide 3 explicit, non-negotiable exit thresholds with precise numerical targets (e.g., debtor days rising above X days, operating margin falling below Y%, order cancellation rate exceeding Z%).

OUTPUT FORMAT:
- `### Top 3 Structural Blind Spots (The Short Counter-Thesis)`
- `### Macro Stress-Test Matrix` (Markdown table with shock scenarios, financial impact, and solvency buffer)
- `### Non-Negotiable Invalidation Triggers (Portfolio Exit Rules)` (Numbered list with specific metric thresholds)
"""


def compile_full_dossier_markdown(target: dict, peer: dict, thesis: str, module_outputs: dict) -> str:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    t_curr = target.get("currency", "USD")

    return f"""# ALPHAENGINE INSTITUTIONAL RESEARCH DOSSIER
**Target Asset:** {target.get('name')} (`{target.get('ticker')}`) | **Peer Benchmark:** {peer.get('name')} (`{peer.get('ticker')}`)  
**Generated:** {now_str}  
**Classification:** Institutional Buyside Fundamental Research & Continuous Risk Audit  
**AI Brain:** Google Gemini 2.5 Flash via `google-genai` SDK  

---

## 1. EXECUTIVE SUMMARY & BASELINE
- **Target Price:** {t_curr} {target.get('currentPrice', 'N/A')}
- **Market Cap:** {format_currency_val(target.get('marketCap'), t_curr)}
- **Trailing P/E:** {format_ratio_val(target.get('trailingPE'))} | **Forward P/E:** {format_ratio_val(target.get('forwardPE'))}
- **Operating Margin:** {format_percent_val(target.get('operatingMargins'))} | **Net Margin:** {format_percent_val(target.get('profitMargins'))}
- **Debt-to-Equity:** {format_ratio_val(target.get('debtToEquity'))} | **Current Ratio:** {format_ratio_val(target.get('currentRatio'))}
- **Free Cash Flow:** {format_currency_val(target.get('freeCashflow'), t_curr)}

### Audited Investment Thesis:
> {thesis}

---

## 2. QUANTITATIVE BENCHMARK MATRIX
| Metric | {target.get('ticker')} | {peer.get('ticker')} |
| :--- | :--- | :--- |
| **Market Cap** | {format_currency_val(target.get('marketCap'), target.get('currency', 'USD'))} | {format_currency_val(peer.get('marketCap'), peer.get('currency', 'USD'))} |
| **Trailing P/E** | {format_ratio_val(target.get('trailingPE'))} | {format_ratio_val(peer.get('trailingPE'))} |
| **Forward P/E** | {format_ratio_val(target.get('forwardPE'))} | {format_ratio_val(peer.get('forwardPE'))} |
| **Operating Margin** | {format_percent_val(target.get('operatingMargins'))} | {format_percent_val(peer.get('operatingMargins'))} |
| **Return on Equity (ROE)** | {format_percent_val(target.get('returnOnEquity'))} | {format_percent_val(peer.get('returnOnEquity'))} |
| **Debt / Equity** | {format_ratio_val(target.get('debtToEquity'))} | {format_ratio_val(peer.get('debtToEquity'))} |
| **Free Cash Flow** | {format_currency_val(target.get('freeCashflow'), target.get('currency', 'USD'))} | {format_currency_val(peer.get('freeCashflow'), peer.get('currency', 'USD'))} |

---

## MODULE 1: BUSINESS DURABILITY & MOAT ARCHITECTURE
{module_outputs.get('module_1', 'N/A')}

---

## MODULE 2: FORENSIC TRANSCRIPT & GROUND-TRUTH AUDIT
{module_outputs.get('module_2', 'N/A')}

---

## MODULE 3: 360-DEGREE SCENARIO MODELING (3-5 YEARS)
{module_outputs.get('module_3', 'N/A')}

---

## MODULE 4: COMPETITIVE FACE-OFF ({target.get('ticker')} vs. {peer.get('ticker')})
{module_outputs.get('module_4', 'N/A')}

---

## MODULE 5: ADVERSARIAL RED-TEAM STRESS TEST (SHORT AUDIT)
{module_outputs.get('module_5', 'N/A')}

---
*Generated by AlphaEngine: Autonomous Stock Discovery, Research & Continuous Risk Monitor.*
"""


# ==============================================================================
# 6. SIDEBAR CONTROLS & AUTHENTICATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 14px;">
            <span style="font-size:2rem;">⚡</span>
            <div>
                <div style="font-weight:800; font-size:1.1rem; color:#F8FAFC; line-height:1.2;">ALPHAENGINE</div>
                <div style="font-size:0.75rem; color:#38BDF8; font-weight:600;">Autonomous Equity Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Authentication resolution (Silent background resolution, zero UI exposure)
    api_key_input = ""
    try:
        api_key_input = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    if not api_key_input:
        api_key_input = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    st.markdown("---")
    st.markdown("#### Global Navigation & Quick Stats")
    tracked_count = len(st.session_state["watchdog_list"])
    st.metric("Portfolio Watchdog Assets", f"{tracked_count} Tracked")
    if st.session_state["audit_completed"]:
        st.caption(f"Active Dossier: **{st.session_state['target_ticker']}**")

    st.markdown("---")
    if st.button("🔄 Reset Engine State", use_container_width=True):
        st.session_state["audit_completed"] = False
        st.session_state["target_snapshot"] = None
        st.session_state["peer_snapshot"] = None
        st.session_state["module_outputs"] = {}
        st.session_state["full_dossier_md"] = ""
        st.rerun()


# ==============================================================================
# 7. MAIN INTERFACE & TAB NAVIGATION
# ==============================================================================
st.markdown(
    """
    <div class="brand-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="brand-title">
                <span>⚡ AlphaEngine</span>
                <span class="brand-badge">Institutional CIO & Risk Suite</span>
            </div>
        </div>
        <div class="brand-desc">
            End-to-end autonomous equity intelligence: screen resilient compounders across global universes,
            execute 5-pillar forensic and adversarial deep audits, and continuously monitor held portfolios for thesis invalidation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Tab 1: Autonomous Opportunity Finder",
    "📑 Tab 2: 5-Pillar Research Dossier",
    "🛡️ Tab 3: Continuous Portfolio Watchdog",
])


# ==============================================================================
# TAB 1: 🔍 AUTONOMOUS OPPORTUNITY FINDER (DISCOVERY ENGINE)
# ==============================================================================
with tab1:
    st.markdown("### 🔍 Autonomous Opportunity Finder")
    st.caption("Filters market universes using multi-year fundamental hurdles to uncover high-durability compounders.")

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        universe_preset = st.selectbox(
            "Select Market Universe",
            options=[
                "Nifty Next 50 & High-Growth Compounders (NSE)",
                "Nifty 50 Core Leaders (NSE)",
                "US Mega-Cap Quality & Tech (NASDAQ / NYSE)",
                "Custom Ticker Universe",
            ],
            index=0,
        )
    with col_u2:
        scan_mode = st.radio(
            "Screening Depth",
            options=["Top 30 Tickers (Recommended)", "Top 15 Tickers (Fast)", "Full Universe"],
            index=0,
            horizontal=True,
        )

    # Expanded Preset Ticker Universes (Full institutional market universe)
    if universe_preset == "Nifty Next 50 & High-Growth Compounders (NSE)":
        full_tickers = [
            "VBL.NS", "HAL.NS", "BEL.NS", "TRENT.NS", "SUZLON.NS", "INOXWIND.NS", 
            "SIEMENS.NS", "ABB.NS", "POLYCAB.NS", "PERSISTENT.NS", "TATAELXSI.NS", 
            "DLF.NS", "CHOLAFIN.NS", "JIOFIN.NS", "DIVISLAB.NS", "HAVELLS.NS", 
            "INDIGO.NS", "LODHA.NS", "MOTHERSON.NS", "NAUKRI.NS", "PIDILITIND.NS", 
            "SRF.NS", "TATACONSUM.NS", "TORNTPHARM.NS", "VEDL.NS", "BOSCHLTD.NS", 
            "COLPAL.NS", "DABUR.NS", "GODREJCP.NS", "MARICO.NS", "AMBUJACEM.NS", 
            "BANKBARODA.NS", "BHEL.NS", "CANBK.NS", "CGPOWER.NS", "CUMMINSIND.NS", 
            "GAIL.NS", "IRFC.NS", "JSWENERGY.NS", "MAXHEALTH.NS", "NHPC.NS", 
            "PFC.NS", "PNB.NS", "RECLTD.NS", "RVNL.NS", "SAIL.NS", "SOLARINDS.NS", 
            "TVSMOTOR.NS", "UNIONBANK.NS", "ZOMATO.NS"
        ]
    elif universe_preset == "Nifty 50 Core Leaders (NSE)":
        full_tickers = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "BHARTIARTL.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "SBIN.NS", 
            "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", 
            "ULTRACEMCO.NS", "KOTAKBANK.NS", "M&M.NS", "AXISBANK.NS", "NTPC.NS", 
            "ONGC.NS", "POWERGRID.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", 
            "TATASTEEL.NS", "JSWSTEEL.NS", "NESTLEIND.NS", "GRASIM.NS", "TECHM.NS",
            "WIPRO.NS", "HCLTECH.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "EICHERMOT.NS",
            "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "BPCL.NS",
            "HEROMOTOCO.NS", "BRITANNIA.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS",
            "SHRIRAMFIN.NS", "TRENT.NS", "BEL.NS", "HAL.NS", "BAJAJ-AUTO.NS"
        ]
    elif universe_preset == "US Mega-Cap Quality & Tech (NASDAQ / NYSE)":
        full_tickers = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", 
            "COST", "CRM", "AMD", "ORCL", "NFLX", "ADBE", "QCOM", "TXN", 
            "INTU", "NOW", "AMAT", "BKNG", "LRCX", "V", "MA", "UNH", "LLY",
            "JNJ", "JPM", "PG", "HD", "WMT", "XOM", "CVX", "PEP", "KO", "ABBV"
        ]
    else:
        custom_input = st.text_area("Custom Tickers (comma-separated)", "SUZLON.NS, INOXWIND.NS, TCS.NS, INFY.NS, AAPL, MSFT, NVDA, RELIANCE.NS, HAL.NS, BEL.NS")
        full_tickers = [x.strip().upper() for x in custom_input.split(",") if x.strip()]

    if scan_mode == "Top 15 Tickers (Fast)":
        active_tickers = full_tickers[:15]
    elif scan_mode == "Top 30 Tickers (Recommended)":
        active_tickers = full_tickers[:30]
    else:
        active_tickers = full_tickers

    st.markdown("#### ⚙️ Fundamental Quality Hurdles (Adjusting Sliders Re-filters Instantly!)")
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    with h_col1:
        min_roe_val = st.slider("Min ROE / Capital Efficiency (%)", min_value=5.0, max_value=35.0, value=15.0, step=1.0)
    with h_col2:
        max_de_val = st.slider("Max Debt-to-Equity (D/E)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    with h_col3:
        min_opm_val = st.slider("Min Operating Margin (OPM %)", min_value=5.0, max_value=30.0, value=12.0, step=1.0)
    with h_col4:
        require_fcf_val = st.toggle("Require Positive FCF (FCF > 0)", value=True)
        max_peg_val = st.number_input("Max PEG Ratio", min_value=0.5, max_value=5.0, value=2.5, step=0.1)

    # Automatic screening: fully hands-free data loading
    status_col_left, status_col_right = st.columns([3, 1])
    with status_col_left:
        st.caption(f"⚡ Autonomously screening **{len(active_tickers)} assets** from *{universe_preset}* (cached for 10 min, sliders re-score in 0 ms).")
    with status_col_right:
        if st.button("🔄 Force Re-Fetch Latest Prices", help="Optional: Clear cache to query live prices right now from the exchange."):
            fetch_universe_snapshots.clear()
            st.rerun()

    active_tickers_tuple = tuple(active_tickers)
    raw_snaps = fetch_universe_snapshots(active_tickers_tuple)
    st.session_state["screened_snapshots"] = raw_snaps

    if raw_snaps:
        results = evaluate_hurdles_and_score(
            snapshots=raw_snaps,
            min_roe=min_roe_val,
            max_de=max_de_val,
            min_opm=min_opm_val,
            require_fcf=require_fcf_val,
            max_peg=max_peg_val,
        )

        passed_count = sum(1 for r in results if r["all_passed"])
        
        # Display Controls Bar
        filt_col1, filt_col2 = st.columns([2, 1])
        with filt_col1:
            st.markdown(f"**Live Screening Results:** **{passed_count} of {len(results)}** assets cleared all active hurdles.")
        with filt_col2:
            display_filter = st.radio("Display View", options=["Show All Screened", "Show Passed Hurdles Only"], horizontal=True)

        filtered_results = [r for r in results if r["all_passed"]] if display_filter == "Show Passed Hurdles Only" else results

        # Candidate Table
        display_df = pd.DataFrame([
            {
                "Ticker": r["Ticker"],
                "Company": r["Company"],
                "Sector": r["Sector"],
                "Price": r["Price"],
                "P/E": r["P/E"],
                "PEG": r["PEG"],
                "ROE": r["ROE %"],
                "D/E": r["D/E"],
                "FCF": r["FCF"],
                "Quality Score": r["Quality Score"],
                "Status": r["Filter Status"],
            }
            for r in filtered_results
        ])

        if not display_df.empty:
            st.dataframe(
                display_df.style.apply(
                    lambda row: ["background-color: rgba(16, 185, 129, 0.15); font-weight: 600;" if row["Status"] == "Passed All" else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
                height=420,
            )
        else:
            st.warning("No assets passed with the current strict filters. Try lowering Min ROE or raising Max D/E!")

        st.markdown("---")
        st.markdown("#### 🤝 One-Click Handshake to Deep Audit Engine")
        handshake_col1, handshake_col2 = st.columns([3, 2])
        with handshake_col1:
            discovered_symbols = [r["Ticker"] for r in filtered_results] if filtered_results else [r["Ticker"] for r in results]
            selected_discovery_ticker = st.selectbox(
                "Select Discovered Asset for Immediate Institutional Audit",
                options=discovered_symbols,
                index=0 if discovered_symbols else 0,
            )
        with handshake_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("➡️ Send to Deep Audit Engine (Tab 2)", type="primary", use_container_width=True):
                st.session_state["target_ticker"] = selected_discovery_ticker
                suggested_peer = PEER_MAP.get(selected_discovery_ticker, "INOXWIND.NS" if ".NS" in selected_discovery_ticker else "MSFT")
                st.session_state["peer_ticker"] = suggested_peer
                st.session_state["thesis_text"] = f"Discovered by AlphaEngine with high capital efficiency and resilient balance sheet. Auditing multi-year growth runway and moat durability vs {suggested_peer}."
                st.success(f"✅ Handshake successful! '{selected_discovery_ticker}' paired with '{suggested_peer}' and loaded into Tab 2.")


# ==============================================================================
# TAB 2: 📑 5-PILLAR INSTITUTIONAL RESEARCH DOSSIER (DEEP AUDIT)
# ==============================================================================
with tab2:
    st.markdown("### 📑 5-Pillar Institutional Research Dossier")
    st.caption("Deep forensic valuation, metric vs. narrative discrepancy audit, and adversarial red-team stress test.")

    # Input Controls Bar
    in_col1, in_col2 = st.columns(2)
    with in_col1:
        target_in = st.text_input(
            "Target Ticker",
            value=st.session_state["target_ticker"],
            help="Supports NSE/BSE (.NS, .BO) and global tickers (AAPL, NVDA, MSFT).",
        ).strip().upper()
        st.session_state["target_ticker"] = target_in

    with in_col2:
        peer_in = st.text_input(
            "Primary Sector Peer / Benchmark Ticker",
            value=st.session_state["peer_ticker"],
            help="Competitor for head-to-head benchmarking (e.g. INOXWIND.NS, MSFT, AMD).",
        ).strip().upper()
        st.session_state["peer_ticker"] = peer_in

    thesis_in = st.text_area(
        "Base Investment Thesis (To Be Audited)",
        value=st.session_state["thesis_text"],
        height=85,
    )
    st.session_state["thesis_text"] = thesis_in

    st.markdown("#### Earnings Material Input (Dual-Mode)")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        uploaded_doc = st.file_uploader(
            "Upload Concall Transcript / Report (.pdf, .txt)",
            type=["pdf", "txt"],
            key="tab2_uploader",
        )
    with t_col2:
        fallback_notes = st.text_area(
            "Or Paste Raw Concall Excerpts / Notes",
            height=95,
            placeholder="Paste earnings call Q&A highlights, management statements, or consensus notes here...",
        )

    run_audit_click = st.button("🚀 Run Institutional 5-Pillar Audit", type="primary", use_container_width=True)

    # Execute Audit Workflow
    if run_audit_click:
        if not api_key_input:
            st.error("❌ Authentication Error: GEMINI_API_KEY is not configured in `.streamlit/secrets.toml` or `.env`.")
            st.stop()
        if not target_in or not peer_in:
            st.error("❌ Input Error: Both Target Ticker and Peer Ticker are required.")
            st.stop()

        status_box = st.status("Executing Institutional Research Pipeline...", expanded=True)
        with status_box:
            st.write(f"📡 Fetching live financial snapshot for **{target_in}**...")
            t_snap = fetch_financial_snapshot(target_in)
            if "error" in t_snap:
                st.error(f"Error fetching {target_in}: {t_snap['error']}")
                status_box.update(label="Audit Failed: Data extraction error", state="error")
                st.stop()

            st.write(f"📡 Fetching peer benchmark metrics for **{peer_in}**...")
            p_snap = fetch_financial_snapshot(peer_in)
            if "error" in p_snap:
                st.error(f"Error fetching {peer_in}: {p_snap['error']}")
                status_box.update(label="Audit Failed: Peer extraction error", state="error")
                st.stop()

            # Prepare transcript
            doc_text = extract_text_from_file(uploaded_doc) if uploaded_doc else fallback_notes

            # Run 5 Chained Modules
            module_res = {}
            st.write("🏰 [1/5] Synthesizing Business Durability & Moat Architecture...")
            m1_p = build_module_1_prompt(t_snap, p_snap, thesis_in)
            module_res["module_1"] = generate_llm_response(api_key_input, m1_p)

            st.write("🔎 [2/5] Running Forensic Transcript & Evasive Language Audit...")
            m2_p = build_module_2_prompt(t_snap, doc_text, thesis_in)
            module_res["module_2"] = generate_llm_response(api_key_input, m2_p)

            st.write("📈 [3/5] Computing 360-Degree Scenario Valuation (Bull / Base / Bear)...")
            m3_p = build_module_3_prompt(t_snap, p_snap)
            module_res["module_3"] = generate_llm_response(api_key_input, m3_p)

            st.write("⚔️ [4/5] Executing Head-to-Head Competitive Face-Off...")
            m4_p = build_module_4_prompt(t_snap, p_snap)
            module_res["module_4"] = generate_llm_response(api_key_input, m4_p)

            st.write("🛡️ [5/5] Deploying Adversarial Red-Team Stress Test (Short Audit)...")
            m5_p = build_module_5_prompt(t_snap, p_snap, thesis_in)
            module_res["module_5"] = generate_llm_response(api_key_input, m5_p)

            # Compile Full Markdown Report
            dossier_text = compile_full_dossier_markdown(t_snap, p_snap, thesis_in, module_res)

            # Save state
            st.session_state["target_snapshot"] = t_snap
            st.session_state["peer_snapshot"] = p_snap
            st.session_state["module_outputs"] = module_res
            st.session_state["full_dossier_md"] = dossier_text
            st.session_state["audit_completed"] = True
            status_box.update(label="✅ Institutional Audit Complete & Dossier Generated!", state="complete", expanded=False)

    # Render Completed Audit
    if st.session_state["audit_completed"] and st.session_state["target_snapshot"]:
        t_data = st.session_state["target_snapshot"]
        p_data = st.session_state["peer_snapshot"]
        m_outs = st.session_state["module_outputs"]
        t_curr = t_data.get("currency", "USD")
        p_curr = p_data.get("currency", "USD")

        st.markdown("---")
        # Quantitative Baseline KPI Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Current Price</div>
                    <div class="kpi-value">{t_curr} {t_data.get('currentPrice', 'N/A')}</div>
                    <div class="kpi-sub">{t_data.get('ticker')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Market Cap</div>
                    <div class="kpi-value">{format_currency_val(t_data.get('marketCap'), t_curr)}</div>
                    <div class="kpi-sub">Peer: {format_currency_val(p_data.get('marketCap'), p_curr)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Trailing P/E</div>
                    <div class="kpi-value">{format_ratio_val(t_data.get('trailingPE'))}</div>
                    <div class="kpi-sub">Peer: {format_ratio_val(p_data.get('trailingPE'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Operating Margin</div>
                    <div class="kpi-value">{format_percent_val(t_data.get('operatingMargins'))}</div>
                    <div class="kpi-sub">Peer: {format_percent_val(p_data.get('operatingMargins'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Free Cash Flow</div>
                    <div class="kpi-value">{format_currency_val(t_data.get('freeCashflow'), t_curr)}</div>
                    <div class="kpi-sub">D/E: {format_ratio_val(t_data.get('debtToEquity'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Export & Watchdog Actions Toolbar
        tb_col1, tb_col2 = st.columns([1, 1])
        with tb_col1:
            st.download_button(
                label="📥 Download Full Dossier (.md)",
                data=st.session_state["full_dossier_md"],
                file_name=f"AlphaEngine_Audit_{t_data.get('ticker')}_{datetime.datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with tb_col2:
            if st.button("🛡️ Add Audited Asset to Portfolio Watchdog (Tab 3)", type="primary", use_container_width=True):
                # Add to watchdog portfolio if not already present
                existing = next((item for item in st.session_state["watchdog_list"] if item["ticker"] == t_data.get("ticker")), None)
                if not existing:
                    new_item = {
                        "ticker": t_data.get("ticker"),
                        "name": t_data.get("name"),
                        "entryPrice": t_data.get("currentPrice") or 0.0,
                        "targetFairValue": (t_data.get("currentPrice") or 0.0) * 1.25,
                        "currency": t_curr,
                        "invalidationRule": "Exit if Operating Margin drops by >250 bps or D/E exceeds 0.5",
                        "minOperatingMargin": (t_data.get("operatingMargins") or 0.12) * 0.8,
                        "maxDebtToEquity": 0.50,
                        "status": "Thesis Intact",
                        "statusBadge": "🟢",
                        "lastAudited": datetime.datetime.now().strftime("%Y-%m-%d"),
                    }
                    st.session_state["watchdog_list"].append(new_item)
                    save_watchdog_portfolio(st.session_state["watchdog_list"])
                    st.success(f"Added {t_data.get('ticker')} to Portfolio Watchdog!")
                else:
                    st.info(f"{t_data.get('ticker')} is already tracked in Portfolio Watchdog.")

        # Side-by-Side Quantitative Matrix Expander
        with st.expander("🔍 Quantitative Fundamental Comparison Matrix", expanded=True):
            matrix_df = pd.DataFrame([
                {"Metric": "Market Capitalization", t_data.get('ticker'): format_currency_val(t_data.get('marketCap'), t_curr), p_data.get('ticker'): format_currency_val(p_data.get('marketCap'), p_curr)},
                {"Metric": "Enterprise Value (EV)", t_data.get('ticker'): format_currency_val(t_data.get('enterpriseValue'), t_curr), p_data.get('ticker'): format_currency_val(p_data.get('enterpriseValue'), p_curr)},
                {"Metric": "Trailing P/E", t_data.get('ticker'): format_ratio_val(t_data.get('trailingPE')), p_data.get('ticker'): format_ratio_val(p_data.get('trailingPE'))},
                {"Metric": "Forward P/E", t_data.get('ticker'): format_ratio_val(t_data.get('forwardPE')), p_data.get('ticker'): format_ratio_val(p_data.get('forwardPE'))},
                {"Metric": "PEG Ratio", t_data.get('ticker'): format_ratio_val(t_data.get('pegRatio'), ''), p_data.get('ticker'): format_ratio_val(p_data.get('pegRatio'), '')},
                {"Metric": "EV / EBITDA", t_data.get('ticker'): format_ratio_val(t_data.get('enterpriseToEbitda')), p_data.get('ticker'): format_ratio_val(p_data.get('enterpriseToEbitda'))},
                {"Metric": "Operating Margin (OPM)", t_data.get('ticker'): format_percent_val(t_data.get('operatingMargins')), p_data.get('ticker'): format_percent_val(p_data.get('operatingMargins'))},
                {"Metric": "Return on Equity (ROE)", t_data.get('ticker'): format_percent_val(t_data.get('returnOnEquity')), p_data.get('ticker'): format_percent_val(p_data.get('returnOnEquity'))},
                {"Metric": "Debt-to-Equity (D/E)", t_data.get('ticker'): format_ratio_val(t_data.get('debtToEquity')), p_data.get('ticker'): format_ratio_val(p_data.get('debtToEquity'))},
                {"Metric": "Free Cash Flow", t_data.get('ticker'): format_currency_val(t_data.get('freeCashflow'), t_curr), p_data.get('ticker'): format_currency_val(p_data.get('freeCashflow'), p_curr)},
            ])
            st.dataframe(matrix_df.set_index("Metric"), use_container_width=True)

        # 5 Qualitative Forensic Modules
        st.markdown("### 🏛️ 5-Pillar Buyside Forensic Modules")

        with st.expander("🏰 Module 1: Business Durability & Moat Architecture", expanded=True):
            st.markdown('<div class="module-header"><span class="module-tag">PILLAR 01</span><span>Unit Economics, Moat Durability & Capital Allocation</span></div>', unsafe_allow_html=True)
            st.markdown(m_outs.get("module_1", "Output pending..."))

        with st.expander("🔎 Module 2: Forensic Transcript & Ground-Truth Audit", expanded=True):
            st.markdown('<div class="module-header"><span class="module-tag">PILLAR 02</span><span>Metric vs. Narrative Divergence & Evasive Language Detection</span></div>', unsafe_allow_html=True)
            st.markdown(m_outs.get("module_2", "Output pending..."))

        with st.expander("📈 Module 3: 360-Degree Scenario Modeling (3-5 Years)", expanded=True):
            st.markdown('<div class="module-header"><span class="module-tag">PILLAR 03</span><span>Bull / Base / Bear Scenario Matrix Table & Valuation Skew</span></div>', unsafe_allow_html=True)
            st.markdown(m_outs.get("module_3", "Output pending..."))

        with st.expander("⚔️ Module 4: Competitive Face-Off (Target vs. Peer)", expanded=True):
            st.markdown('<div class="module-header"><span class="module-tag">PILLAR 04</span><span>Operational Divergence & Summary Decision Matrix</span></div>', unsafe_allow_html=True)
            st.markdown(m_outs.get("module_4", "Output pending..."))

        with st.expander("🛡️ Module 5: Adversarial Red-Team Stress Test (Short Audit)", expanded=True):
            st.markdown('<div class="module-header"><span class="module-tag">PILLAR 05</span><span>Top 3 Blind Spots, Macro Sensitivity & Invalidation Kill Switches</span></div>', unsafe_allow_html=True)
            st.markdown(m_outs.get("module_5", "Output pending..."))


# ==============================================================================
# TAB 3: 🛡️ CONTINUOUS MONITOR & THESIS TRACKER (PORTFOLIO WATCHDOG)
# ==============================================================================
with tab3:
    st.markdown("### 🛡️ Continuous Monitor & Thesis Tracker (Groww & Demat Watchdog)")
    st.caption("Tracks your personal Groww and Demat holdings against thesis invalidation triggers, monitors live P&L, and alerts on fundamental deterioration.")

    watchdog_items = st.session_state["watchdog_list"]

    # Calculate Total Portfolio Summary Metrics
    total_invested_inr = 0.0
    total_current_inr = 0.0
    intact_count = 0
    caution_count = 0
    breached_count = 0

    for it in watchdog_items:
        qty = float(it.get("quantity", 1.0))
        entry_p = float(it.get("entryPrice", 0.0))
        curr_p = float(it.get("currentPrice", entry_p))
        c_sym = it.get("currency", "USD")

        # Rough currency conversion multiplier for aggregated summary if mixed
        fx = 84.0 if c_sym == "USD" else 1.0
        total_invested_inr += (entry_p * qty * fx)
        total_current_inr += (curr_p * qty * fx)

        badge = it.get("statusBadge", "🟢")
        if badge == "🟢":
            intact_count += 1
        elif badge == "🟡":
            caution_count += 1
        elif badge == "🔴":
            breached_count += 1

    total_pnl_inr = total_current_inr - total_invested_inr
    total_pnl_pct = ((total_pnl_inr / total_invested_inr) * 100) if total_invested_inr > 0 else 0.0

    # Top Portfolio Performance Summary Bar
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Invested</div>
                <div class="kpi-value">₹{total_invested_inr:,.2f}</div>
                <div class="kpi-sub">{len(watchdog_items)} Holdings Tracked</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Current Value</div>
                <div class="kpi-value">₹{total_current_inr:,.2f}</div>
                <div class="kpi-sub">Real-Time Market Valuation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w_col3:
        pnl_color = "#34D399" if total_pnl_inr >= 0 else "#F87171"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Unrealized P&L</div>
                <div class="kpi-value" style="color: {pnl_color};">₹{total_pnl_inr:+,.2f}</div>
                <div class="kpi-sub" style="color: {pnl_color}; font-weight: 600;">{total_pnl_pct:+.2f}% Overall Return</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with w_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Health & Risk Status</div>
                <div class="kpi-value" style="font-size: 1.15rem; font-family: 'Inter', sans-serif;">
                    🟢 {intact_count} &nbsp; 🟡 {caution_count} &nbsp; 🔴 {breached_count}
                </div>
                <div class="kpi-sub">Intact | Caution | Breached</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Groww File Ingestion & Refresh Action Bar
    st.markdown("---")
    act_col1, act_col2 = st.columns([2, 1])
    with act_col1:
        st.markdown("#### 📥 Import Holdings from Groww (CSV / Excel)")
        groww_file = st.file_uploader(
            "Upload Groww Holdings or P&L Report (.csv, .xlsx)",
            type=["csv", "xlsx"],
            key="groww_uploader",
            help="Download your holdings report from Groww App -> Profile -> Reports -> Holdings / P&L.",
        )
        if groww_file is not None:
            try:
                if groww_file.name.endswith(".csv"):
                    gdf = pd.read_csv(groww_file)
                else:
                    gdf = pd.read_excel(groww_file)

                # Identify relevant columns
                sym_col = next((c for c in gdf.columns if any(k in c.lower() for k in ["symbol", "ticker", "instrument", "stock name", "company"])), None)
                price_col = next((c for c in gdf.columns if any(k in c.lower() for k in ["avg", "buy", "cost", "price"])), None)
                qty_col = next((c for c in gdf.columns if any(k in c.lower() for k in ["qty", "quantity", "shares"])), None)

                if sym_col:
                    added_count = 0
                    for _, row in gdf.iterrows():
                        raw_sym = str(row[sym_col]).strip().upper()
                        if not raw_sym or raw_sym == "NAN":
                            continue
                        # Normalize Indian ticker symbol
                        if not raw_sym.endswith(".NS") and not raw_sym.endswith(".BO") and "." not in raw_sym:
                            norm_ticker = f"{raw_sym}.NS"
                        else:
                            norm_ticker = raw_sym

                        qty_val = float(row[qty_col]) if (qty_col and pd.notna(row[qty_col])) else 1.0
                        price_val = float(row[price_col]) if (price_col and pd.notna(row[price_col])) else 0.0

                        # Check if already in list
                        existing = next((it for it in watchdog_items if it["ticker"] == norm_ticker), None)
                        if not existing:
                            watchdog_items.append({
                                "ticker": norm_ticker,
                                "name": raw_sym,
                                "quantity": qty_val,
                                "entryPrice": price_val,
                                "targetFairValue": price_val * 1.25 if price_val > 0 else 0.0,
                                "currency": "INR",
                                "invalidationRule": "Exit if Operating Margin drops below 10% or D/E > 0.5",
                                "minOperatingMargin": 0.10,
                                "maxDebtToEquity": 0.50,
                                "status": "Thesis Intact",
                                "statusBadge": "🟢",
                                "lastAudited": datetime.datetime.now().strftime("%Y-%m-%d"),
                            })
                            added_count += 1

                    if added_count > 0:
                        save_watchdog_portfolio(watchdog_items)
                        st.success(f"✅ Successfully imported {added_count} holdings from your Groww report!")
                        st.rerun()
                    else:
                        st.info("All holdings from the uploaded report are already present in your watchdog.")
            except Exception as e:
                st.error(f"Error parsing Groww file: {e}")

    with act_col2:
        st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)
        rescan_clicked = st.button("🔄 Re-Scan Live Fundamental Health", type="primary", use_container_width=True)

    if rescan_clicked:
        with st.spinner("Re-evaluating live prices, operating margins, and solvency triggers for your holdings..."):
            for item in watchdog_items:
                live_snap = fetch_financial_snapshot(item["ticker"])
                if "error" not in live_snap and live_snap.get("currentPrice"):
                    curr_p = live_snap.get("currentPrice")
                    opm = live_snap.get("operatingMargins") or 0.0
                    de = live_snap.get("debtToEquity") or 0.0
                    fcf = live_snap.get("freeCashflow") or 0.0

                    min_opm = item.get("minOperatingMargin", 0.10)
                    max_de = item.get("maxDebtToEquity", 0.50)
                    target_fv = item.get("targetFairValue", curr_p * 1.25)

                    # Automated Rule Checks
                    if (opm < min_opm) or (de > max_de) or (fcf < 0):
                        item["status"] = "Thesis Breached"
                        item["statusBadge"] = "🔴"
                    elif curr_p > target_fv * 1.15:
                        item["status"] = "Valuation Stretched"
                        item["statusBadge"] = "🟡"
                    else:
                        item["status"] = "Thesis Intact"
                        item["statusBadge"] = "🟢"

                    item["currentPrice"] = curr_p
                    item["lastAudited"] = datetime.datetime.now().strftime("%Y-%m-%d")

            save_watchdog_portfolio(watchdog_items)
            st.success("✅ Portfolio health status updated against live market metrics!")

    # Active Holdings Table
    st.markdown("#### 📋 Tracked Portfolio Holdings")
    portfolio_rows = []
    for it in watchdog_items:
        c_sym = "₹" if it.get("currency") == "INR" else ("$" if it.get("currency") == "USD" else it.get("currency", "₹"))
        entry_p = float(it.get("entryPrice", 0.0))
        curr_p = float(it.get("currentPrice", entry_p))
        qty = float(it.get("quantity", 1.0))
        invested = entry_p * qty
        cur_val = curr_p * qty
        unrealized_pnl = ((curr_p - entry_p) / entry_p * 100) if entry_p > 0 else 0.0

        portfolio_rows.append({
            "Status": f"{it.get('statusBadge', '🟢')} {it.get('status', 'Thesis Intact')}",
            "Ticker": it.get("ticker"),
            "Shares": f"{qty:,.0f}" if qty.is_integer() else f"{qty:,.2f}",
            "Avg Buy Price": f"{c_sym} {entry_p:,.2f}",
            "Current Price": f"{c_sym} {curr_p:,.2f}",
            "Invested Value": f"{c_sym} {invested:,.2f}",
            "Current Value": f"{c_sym} {cur_val:,.2f}",
            "P&L (%)": f"{unrealized_pnl:+.1f}%",
            "Target Fair Value": f"{c_sym} {it.get('targetFairValue', 0.0):,.2f}",
            "Exit Invalidation Trigger": it.get("invalidationRule", "N/A"),
        })

    if portfolio_rows:
        port_df = pd.DataFrame(portfolio_rows)
        st.dataframe(port_df.set_index("Status"), use_container_width=True)
    else:
        st.info("No holdings tracked yet. Upload your Groww report or add assets below.")

    # One-Click Audit Handshake from Groww Holdings
    st.markdown("---")
    st.markdown("#### 🤝 One-Click Handshake to Deep Forensic Audit (Tab 2)")
    hs_col1, hs_col2 = st.columns([3, 2])
    with hs_col1:
        tracked_symbols = [it["ticker"] for it in watchdog_items]
        selected_holding = st.selectbox(
            "Select any of your held stocks to run a full 5-Pillar Buyside Audit",
            options=tracked_symbols,
            index=0 if tracked_symbols else 0,
        )
    with hs_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Audit This Stock in Tab 2", type="primary", use_container_width=True):
            st.session_state["target_ticker"] = selected_holding
            suggested_peer = PEER_MAP.get(selected_holding, "INOXWIND.NS" if ".NS" in selected_holding else "MSFT")
            st.session_state["peer_ticker"] = suggested_peer

            # Look up entry price from holding
            h_entry = next((it.get("entryPrice", 0.0) for it in watchdog_items if it["ticker"] == selected_holding), 0.0)
            st.session_state["thesis_text"] = f"Held in Groww portfolio with average buy price of ₹{h_entry:,.2f}. Auditing operating margin trajectory, moat durability, and fair value headroom vs {suggested_peer}."
            st.success(f"✅ Loaded {selected_holding} paired with {suggested_peer} into Tab 2! Head over to '📑 Tab 2: 5-Pillar Research Dossier' to run or view the audit.")

    st.markdown("---")
    st.markdown("#### 🧠 Conversational AI Chief Investment Officer (CIO Advisory)")
    st.caption("Interact with an AI CIO & Head of Risk grounded in your live portfolio holdings and invalidation thresholds.")

    cio_query = st.text_input(
        "Ask the AI Chief Investment Officer",
        placeholder="e.g. Which of my holdings has the highest downside risk if interest rates rise 100 bps? Should I trim Suzlon?",
    )

    quick_q1, quick_q2, quick_q3 = st.columns(3)
    preset_q = ""
    with quick_q1:
        if st.button("💥 Downside Stress Analysis", use_container_width=True):
            preset_q = "Evaluate the downside risk and solvency sensitivity across my current portfolio holdings."
    with quick_q2:
        if st.button("⚖️ Capital Allocation Check", use_container_width=True):
            preset_q = "Which of my holdings has the highest margin of safety vs. most stretched valuation?"
    with quick_q3:
        if st.button("🚨 Thesis Invalidation Warning", use_container_width=True):
            preset_q = "Are any of my holdings showing early signs of thesis breach or operating cash flow strain?"

    active_cio_prompt = preset_q if preset_q else cio_query

    if active_cio_prompt:
        if not api_key_input:
            st.warning("⚠️ GEMINI_API_KEY is not configured in `.streamlit/secrets.toml` or `.env`.")
        else:
            with st.spinner("CIO analyzing portfolio positions, capital efficiency, and invalidation rules..."):
                portfolio_context = json.dumps(watchdog_items, indent=2)
                cio_system = """
                You are the Chief Investment Officer (CIO) and Head of Risk Management at an institutional asset management firm.
                You provide sharp, decisive, risk-first advice grounded in the user's active holdings, entry prices, fair value targets, and exit rules.
                Avoid generic platitudes. Directly assess risk/reward asymmetry and capital allocation.
                """
                cio_full_prompt = f"""
                Portfolio Holdings & Risk Rules:
                \"\"\"{portfolio_context}\"\"\"

                Investor Question:
                \"\"\"{active_cio_prompt}\"\"\"

                Provide a structured institutional CIO memo answering the question with actionable advice.
                """
                cio_reply = generate_llm_response(
                    api_key=api_key_input,
                    prompt=cio_full_prompt,
                    system_instruction=cio_system,
                    temperature=0.4,
                )
                st.markdown("##### 🏛️ Institutional CIO Memo")
                st.markdown(cio_reply)

    # Manage Watchdog Positions (Add / Delete)
    st.markdown("---")
    with st.expander("⚙️ Manage Watchdog Holdings (Add / Remove)"):
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("**Add Position Manually**")
            new_sym = st.text_input("Ticker Symbol (e.g. SUZLON.NS, RELIANCE.NS, WAAREERTL.BO)").strip().upper()
            new_qty = st.number_input("Shares / Quantity", min_value=1.0, value=100.0, step=1.0)
            new_entry = st.number_input("Average Buy Price (₹ or $)", min_value=0.0, value=40.0, step=1.0)
            new_target = st.number_input("Target Fair Value (₹ or $)", min_value=0.0, value=60.0, step=1.0)
            new_rule = st.text_input("Invalidation Rule", value="Exit if Operating Margin drops below 12% or D/E > 0.4")
            if st.button("➕ Add Holding to Watchdog"):
                if new_sym:
                    snap_new = fetch_financial_snapshot(new_sym)
                    curr_price = snap_new.get("currentPrice") or new_entry
                    detected_curr = snap_new.get("currency", "INR" if (".NS" in new_sym or ".BO" in new_sym) else "USD")
                    st.session_state["watchdog_list"].append({
                        "ticker": new_sym,
                        "name": snap_new.get("name", new_sym),
                        "quantity": new_qty,
                        "entryPrice": new_entry,
                        "currentPrice": curr_price,
                        "targetFairValue": new_target,
                        "currency": detected_curr,
                        "invalidationRule": new_rule,
                        "minOperatingMargin": 0.12,
                        "maxDebtToEquity": 0.40,
                        "status": "Thesis Intact",
                        "statusBadge": "🟢",
                        "lastAudited": datetime.datetime.now().strftime("%Y-%m-%d"),
                    })
                    save_watchdog_portfolio(st.session_state["watchdog_list"])
                    st.success(f"Added {new_sym} ({new_qty:,.0f} shares at ₹{new_entry}) to Watchdog!")
                    st.rerun()

        with m_col2:
            st.markdown("**Remove Position**")
            tracked_syms = [item["ticker"] for item in st.session_state["watchdog_list"]]
            if tracked_syms:
                remove_choice = st.selectbox("Select Asset to Remove", options=tracked_syms)
                if st.button("🗑️ Remove Asset"):
                    st.session_state["watchdog_list"] = [it for it in st.session_state["watchdog_list"] if it["ticker"] != remove_choice]
                    save_watchdog_portfolio(st.session_state["watchdog_list"])
                    st.success(f"Removed {remove_choice} from watchdog.")
                    st.rerun()
            else:
                st.caption("No assets to remove.")
