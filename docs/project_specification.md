# Project Specification: Real-Time Intraday Volatility Regime Classifier (NIFTY 50)

## 1. Introduction & Objectives
Retail intraday traders in India take heavy losses from sudden volatility spikes, and existing risk tools are too mathematical to interpret in real time. This project builds an **application** that tells a trader, live, which volatility regime the NIFTY 50 is currently in, **Quiet**, **Normal** or **Chaotic**, and alerts them when the market turns Chaotic.

**Problem framing: regime classification (nowcasting), not price or volatility forecasting.**
*   The system answers *"What state is the market in right now?"*, not *"What will volatility be tomorrow?"*.
*   A regime is a latent (hidden) state. It is only known with confidence **in hindsight**, once enough subsequent bars have been seen. In real time, a regime change is ambiguous. One large bar could be noise or the start of a Chaotic phase.
*   The ML task is therefore: **using only information available up to bar *t*, recover the regime that a hindsight labeler assigns to bar *t*, as accurately and with as little detection delay as possible.**
*   Out of core scope: the probability that the *next* bar is Chaotic. That is forecasting, and lives in `06_Experimental_Lab/` if time permits.

**Deliverables**
1.  A labeled historical dataset of NIFTY 50 intraday regimes.
2.  A trained, benchmarked real-time regime classifier.
3.  A Streamlit "Risk Radar" dashboard (Green/Yellow/Red) running on live market data.
4.  Automated alerts via a Telegram bot (free) when a Chaotic regime is detected. Twilio SMS is optional.
5.  Project report and final review presentation.

## 2. Requirements Specification
*   **Target Asset:** NIFTY 50 Index (`^NSEI`).
*   **Primary Bar Frequency:** 15 minutes (25 bars per session). 5-minute and 1-hour bars are used for robustness checks.
*   **Supporting Series:** India VIX (intraday, via Angel One), USD/INR and Brent crude (daily, via yfinance).
*   **Compute Environment:**
    *   **Local machine:** data ingestion (`scripts/fetch_data.py`) and later the live app. Broker APIs are unreliable from Colab's shared cloud IPs.
    *   **Google Colab:** research notebooks (labeling, training; GPU for LSTM/GRU), reading the parquet archive from Google Drive (`MyDrive/vrdet/data/raw/`).
*   **Programming Language:** Python 3.10+
*   **Primary Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `hmmlearn`, `arch`, `pytorch` or `tensorflow`, `streamlit`, `smartapi-python`, `pyotp`, `python-telegram-bot` (or plain HTTP calls to the Bot API).
*   **Data Sources:**
    | Source | Used for | Notes |
    |---|---|---|
    | **Angel One SmartAPI** (primary) | NIFTY 50 (token `99926000`) and India VIX (token `99926017`) candles, 1m to 1d; live data for the app | Free with an account. Historical requests are capped per call (e.g., 200 days for 15m, 100 days for 5m, 30 days for 1m), so downloads are chunked and rate-limited (~3 req/s). Login needs API key, client code, PIN and TOTP. |
    | yfinance (secondary) | USD/INR (`USDINR=X`), Brent (`BZ=F`) daily; cross-check of NIFTY daily | Intraday history on yfinance is short (1h ≈ 730 days, 15m ≈ 60 days), so it is not used for primary intraday data. |
*   **Credentials** (API key, client code, PIN, TOTP secret, Telegram bot token) are kept in a local, gitignored `.env` file (template: `.env.example`), never in notebooks or git.
*   **Out of scope:** index-level bid-ask spread, order flow and tick volume. NIFTY is an index, not a traded instrument, so index candles carry no meaningful volume. NIFTY futures volume/OI from Angel One is an optional later addition.

## 3. Methodology

### 3.1 Data Ingestion
*   Download (Angel One in date-chunked requests), convert timestamps to IST, restrict to NSE session hours (09:15–15:30), drop holidays and partial or duplicate bars, and align supporting series to NIFTY timestamps using forward-fill of the last known value only.
*   Daily data (2008 onwards) is also downloaded for long-history research: calibrating regime definitions across real crises (2008, 2020, 2022) that the 2-year hourly window does not contain.

### 3.2 Feature Engineering (strictly causal: bar *t* uses data ≤ *t* only)
*   **Returns:** log return, absolute return, signed return over the last k bars.
*   **Volatility estimators:** rolling realized volatility (several windows), Garman-Klass, Parkinson, Rogers-Satchell, ATR, intrabar range / open.
*   **Volatility dynamics:** ratio of short to long volatility (expansion or contraction), volatility z-score versus trailing distribution.
*   **Trend context:** SMA/EMA distance, MACD, drawdown from rolling high.
*   **Implied volatility:** India VIX level, change, rate of change, VIX versus realized-volatility spread.
*   **Cross-asset:** USD/INR and Brent returns and volatility.
*   **Calendar / seasonality:** time-of-day (sin/cos), day-of-week, opening/closing-hour flags, overnight gap, scheduled event flags (see `macro_and_alternative_data.md`).
*   **Lags:** lagged copies of key features for tree models; 3D sequence tensors for LSTM/GRU.

### 3.3 Regime Labeling (the "ground truth")
Three alternative labelers, each mapping every bar to Quiet / Normal / Chaotic:
*   **Pipeline A – HMM** (Gaussian HMM, 3 states, on returns and volatility features). Labels come from **smoothed / Viterbi** decoding, which uses the full sequence and gives the hindsight-best estimate.
*   **Pipeline B – GMM** on volatility features, with optional temporal smoothing (e.g., a minimum regime duration) so labels do not flicker bar to bar.
*   **Pipeline C – GARCH** conditional volatility, binned by percentile thresholds (e.g., <30th = Quiet, >90th = Chaotic), fitted on the training span.

Rules:
*   States are **ordered by mean volatility** after every fit, so the integer code means the same regime across refits and pipelines.
*   Labeler **parameters are fitted on the training period only**, then frozen and applied to validation/test periods. Hindsight decoding within a period is allowed because it defines the *target*. Only the classifier's *inputs* must be causal.
*   **Sanity check:** known stress events (COVID crash Mar 2020, Jun 2022 sell-off, 4 Jun 2024 election result, budget days) must come out Chaotic; low-activity holiday-season weeks should come out Quiet.
*   Agreement between pipelines (Cohen's kappa) is reported. One pipeline is chosen as the primary target; the others are used for robustness checks.

**Implemented configuration and findings** (15m bars, labelers fitted on 2015–2021):
*   HMM observations are per-bar `[z, log range_ds]` (deseasonalised return and high-low range), standardised. Fitted with `hmmlearn` `GaussianHMM` (diagonal covariance, best of 4 seeds), with a sticky Dirichlet prior on self-transitions (pseudo-count = 0.2 × training length) and a minimum regime length of 4 bars (1 hour). Hindsight labels = Viterbi path. Live labels = forward-filtered probabilities (Rabiner, 1989), implemented in `vrdet.labels.forward_filter` because `hmmlearn` only exposes smoothed (future-using) probabilities; the app needs only the saved parameters (`models/hmm_labeler.json`), not `hmmlearn`.
*   Rolling-volatility observations were rejected: their hindsight labels match the live filter 96% of the time (circular target) and label 22% of bars Chaotic.

    | Labeler | Quiet / Normal / Chaotic share | Chaotic episodes | Chaotic share on shock sessions (mean / min) | Live vs hindsight agreement |
    |---|---|---|---|---|
    | **HMM (primary)** | 51% / 43% / 6.1% | 277 (median 2.25 h) | 0.89 / 0.48 | 0.86 |
    | GMM | 70% / 28% / 1.9% | 59 | 0.53 / 0.00 | 0.75 |
    | GARCH (30th/94th pct) | 32% / 63% / 4.8% | 150 | 0.68 / 0.00 | 0.98 |
*   Cohen's kappa: HMM–GMM 0.53, HMM–GARCH 0.51, GMM–GARCH 0.30.
*   **Data validation** (`vrdet.quality`, run in `scripts/fetch_data.py` and before feature building): every intraday session is checked against the official daily candles; valid = not flat, ≥ 13 bars, and reproduces at least 2 of the daily high / low / close within 1% (official closes are computed differently from the last trade, and official India VIX lows occasionally contain a momentary opening-quote dip). Results:
    *   NIFTY 15m: 99.6–100% valid every year; only 2 invalid sessions (24 Feb 2021 NSE trading halt, 21 Oct 2025 Muhurat session), both dropped.
    *   India VIX 15m: Dec 2015 – Sep 2016 corrupt in the vendor archive (flat, scaled by 1/100, then integer-rounded); 99–100% valid from 2017. Invalid sessions are set to missing, so VIX-based features exist from 22 Sep 2016.
    *   Since 5 Aug 2026 the vendor's 15:15 NIFTY candle is usually a single print at the official close (zero range); zero-range candles are treated as missing range information, not as "quiet".
    *   A few sessions have missing candles (e.g. 2017-07-10 morning); returns spanning missing candles are set to missing rather than treated as 15-minute moves.
    *   The fixes changed the HMM labels negligibly (Chaotic share 6.1%, 276 vs 277 episodes, same shock-day results).

### 3.4 Real-Time Classifiers
Supervised models mapping causal features at bar *t* to the hindsight label of bar *t*:
*   **Tree-based:** Random Forest, AdaBoost, XGBoost, LightGBM.
*   **Sequence models:** LSTM, GRU on rolling windows.
*   **Ensembles:** soft/hard voting, stacking, blending of the best base models.
*   Class imbalance is handled with class weights and threshold tuning on the Chaotic probability, not by blind oversampling.

### 3.5 Validation Protocol
*   **Walk-forward (expanding-window) splits only.** No random shuffling.
*   Purge/embargo a gap of several bars between train and test so rolling-window features do not overlap.
*   The final hold-out is the most recent period, untouched until the end.

### 3.6 Application
*   **Streamlit dashboard:** current regime as a Green/Yellow/Red indicator, Chaotic probability, recent price chart shaded by regime, model confidence, last-updated time.
*   **Alerting:** fires on transition into Chaotic (with hysteresis/cooldown so it does not spam), via a Telegram bot. Users subscribe by messaging the bot.
*   **Inference loop:** on each new bar, fetch → features → classify → update dashboard → alert if needed.

## 4. Project Structure
| Path | Content |
|---|---|
| `scripts/fetch_data.py` | Local data ingestion (Angel One + yfinance) → `data/raw/*.parquet` |
| `src/vrdet/` | Shared package: `data.py` (loading), `features.py` (causal features, VIX cleaning), `labels.py` (HMM/GMM/GARCH labelers, live filter, event checks) |
| `main.ipynb` | Stage 01: load and inspect raw data |
| `notebooks/02_features_and_labels.ipynb` | Stages 02–03: features, labelers, sanity checks, label comparison |
| `notebooks/04_*.ipynb` | Baselines and classifiers (one notebook per model family) |
| `notebooks/05_ensemble_lab.ipynb` | Voting, stacking, blending |
| `notebooks/06_experimental_lab.ipynb` | Next-bar Chaotic probability, alternative data, etc. |
| `app/` | Streamlit dashboard, inference loop, Telegram alerts |
| `models/` | Small fitted artefacts committed to git (e.g. `hmm_labeler.json`) |

Notebooks import `src/vrdet`. In Colab, the first cell clones this (public) GitHub repo and pulls the raw data from a Drive share link; data and credentials are never committed.

## 5. Evaluation Metrics
**Classification quality** (against hindsight labels, on walk-forward test folds):
*   **Macro-F1:** primary metric (classes are imbalanced; Normal dominates).
*   **Chaotic recall:** missing a Chaotic regime is the costliest error.
*   **Chaotic precision:** too many false alarms and users ignore the alerts.
*   **Confusion matrix:** especially Quiet↔Chaotic confusions.
*   **ROC-AUC / PR-AUC (one-vs-rest):** PR-AUC is more informative for the rare Chaotic class.

**Application quality** (what matters to a trader):
*   **Detection delay:** number of bars between the start of a Chaotic episode (per hindsight labels) and the first live Chaotic alert.
*   **False alarms per week:** alerts not followed by a labeled Chaotic episode.
*   **Episode-level recall:** fraction of Chaotic episodes caught at all.
*   **Inference latency:** time from bar close to dashboard update.

## 6. Benchmarking
ML models must beat all of these on macro-F1 and detection delay to justify their complexity:
1.  **Majority class:** always "Normal".
2.  **Persistence (reference only):** current regime = previous bar's hindsight label. Not deployable, since hindsight labels need future bars, but it shows how much of the score comes from regimes simply persisting.
3.  **Fixed threshold:** realized volatility above its trailing 90th percentile = Chaotic, below 30th = Quiet.
4.  **GARCH(1,1) threshold:** conditional volatility binned by fixed percentiles.
5.  **Causal HMM filter (key benchmark):** the same HMM used for labeling, but decoded with the **forward (filtered) probabilities only**, which is exactly what HMM itself can do in real time. If ML cannot beat this, the HMM filter alone should power the app.

## 7. Decisions & Plan
**Decided**
*   Data: Angel One SmartAPI for NIFTY / India VIX intraday; 15-minute primary bars.
*   Task: real-time classification of the current regime; next-bar probability is experimental only.
*   Alerts: Telegram bot (free); Twilio optional.
*   Primary labeler: HMM (see 3.3 findings). Labeler training span 2015–2021.

**Still open**
*   Dashboard hosting (local vs Streamlit Community Cloud).

**Work order** (deadline 2 Nov, flexible; full-time effort)
1.  Data ingestion and Drive archive (Angel One + yfinance), data-quality checks.
2.  Causal features and exploratory analysis (intraday volatility seasonality, VIX relationship).
3.  HMM labeling (hindsight vs filtered), event sanity checks; then GMM and GARCH pipelines.
4.  Baselines, then LightGBM / XGBoost with walk-forward validation.
5.  Streamlit Risk Radar on live Angel One data and Telegram alerts: minimum viable application.
6.  Remaining model families (RF, AdaBoost, LSTM, GRU) and ensembles.
7.  Report and final review deck.

## 8. Risks & Criticism
*   **Circularity:** if labels come from realized volatility and the classifier is fed realized volatility over the same window, it will trivially relearn the labeling rule. The design avoids this by labeling with *hindsight decoding* and requiring the classifier to work *causally*. The causal HMM filter benchmark measures whether ML adds anything.
*   **Few Chaotic episodes:** even with years of 15-minute bars, the number of distinct Chaotic *episodes* is small (dozens), and consecutive bars are highly correlated. Report episode-level metrics, use strong regularization, and do not trust bar-level scores alone.
*   **API dependence:** Angel One rate limits, session expiry and occasional missing candles must be handled (chunking, retries, gap checks). The archive on Drive means research never depends on re-downloading.
*   **Label instability:** HMM/GMM state assignments can change between refits. Order states by volatility and check stability across refits.
*   **Regime flicker:** raw per-bar labels can oscillate. Use minimum-duration smoothing for labels and hysteresis for alerts.
*   **Data quality:** yfinance intraday data can have gaps, delayed bars and timezone quirks. Validate every download before appending to the archive.
*   **Not investment advice:** the app reports a risk state. It does not recommend trades.
