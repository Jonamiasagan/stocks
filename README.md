# AlphaEngine: Autonomous Stock Discovery, Research & Continuous Risk Monitor

A production-grade, institutional buyside equity intelligence and portfolio risk platform built with **Streamlit**, **yfinance**, and the modern Google GenAI SDK (**`google-genai`** with **`gemini-2.5-flash`**).

---

## 🏛️ Platform Architecture

### Tab 1: 🔍 Autonomous Opportunity Finder (Discovery Engine)
- **Multi-Universe Screening**: Presets for Nifty Next 50 (India Mid/Growth), Nifty 50 Core Leaders, US Mega-Cap Quality & Tech, or custom ticker lists.
- **Fundamental Quality Hurdles**: Configurable sliders for Min ROE/ROCE (>=15%), Max Debt-to-Equity (<=0.5), Min Operating Margins (>=12%), Positive Free Cash Flow (FCF > 0), and Max PEG Ratio (<=2.5).
- **Parallel Batch Screening**: Concurrent multi-threaded financial data extraction via `ThreadPoolExecutor` with progress indicators.
- **Candidate Quality Scoring**: Computes a 0–100 composite fundamental quality score.
- **One-Click Handshake**: Automatically loads any discovered asset and a matched sector competitor directly into the Deep Audit Engine (Tab 2).

### Tab 2: 📑 5-Pillar Institutional Research Dossier (Deep Audit)
- **Quantitative Baseline**: Side-by-side comparison matrix of Target vs. Primary Sector Competitor (Market Cap, EV, P/E, PEG, EV/EBITDA, OPM, ROE, D/E, FCF).
- **Dual-Mode Transcript Ingestion**: Upload `.pdf` / `.txt` earnings call transcripts or paste raw concall notes.
- **Pillar 1: Business Durability & Moat Architecture**: Unit economics, recurring vs. project revenue, switching costs, and capital allocation discipline (FCF conversion vs. reported PAT).
- **Pillar 2: Forensic Transcript & Ground-Truth Audit**: Flags metric vs. narrative divergence, evasive management statements, and guidance drift, accompanied by direct quotes and an Executive Trust Score (1–10).
- **Pillar 3: 360-Degree Scenario Modeling (3–5 Years)**: Bull, Base, and Bear cases in an institutional Markdown Scenario Matrix Table with asymmetry skew analysis.
- **Pillar 4: Competitive Face-Off**: Head-to-head operational resilience, balance sheet comparison, and Summary Decision Matrix (Pricing Power, Capital Efficiency, Margin of Safety).
- **Pillar 5: Adversarial Red-Team Stress Test**: Forensic short-seller counter-thesis, macro shock sensitivity matrix (+100 bps rate hikes, FX, input inflation), and 3 non-negotiable quantitative exit kill switches.
- **Export & Handshake**: Single-click `"Download Full Dossier (.md)"` and `"Add to Portfolio Watchdog (Tab 3)"`.

### Tab 3: 🛡️ Continuous Monitor & Thesis Tracker (Portfolio Watchdog)
- **Active Watchlist & Portfolio Table**: Tracks held positions with entry prices, live market prices, unrealized P&L %, target fair values, and custom invalidation triggers.
- **Automated Health Check Monitor**: One-click live re-evaluation against fundamental thresholds with instant status badges:
  - 🟢 **Thesis Intact**: Operational metrics healthy; trading within margin of safety.
  - 🟡 **Valuation Stretched**: Multiples or prices exceed target fair values.
  - 🔴 **Thesis Breached**: Invalidation rule hit (margins collapsed, debt spiked, or FCF turned negative).
- **Conversational AI CIO (Temperature 0.4)**: Real-time interactive Chief Investment Officer and Head of Risk advisory widget answering portfolio queries with institutional buyside reasoning.
- **Persistence**: Persists holdings automatically to local `portfolio_watchdog.json` across sessions.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
git clone <your-repo>
cd Stocks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Automatic Authentication Without Repeated Prompts
To make the application run **automatically without ever prompting you for an API key**, save your key in `.streamlit/secrets.toml`:

```bash
mkdir -p .streamlit
cat <<EOF > .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
EOF
```

*(You can also use `.env` with `GEMINI_API_KEY="your_key"`, or enter it in the app's sidebar).*

### 4. Run the Engine
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your code to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: AlphaEngine autonomous discovery and risk monitor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**, choose your repository and branch (`main`), and set the main file to `app.py`.
4. Under **Advanced Settings > Secrets**, enter:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
5. Click **Deploy**. Streamlit Cloud will build the app using `requirements.txt` and run it securely.
