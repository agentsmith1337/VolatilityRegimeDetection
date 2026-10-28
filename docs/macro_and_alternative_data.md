# Macroeconomic & Alternative Data Points for Volatility Prediction

When forecasting volatility regimes, market microstructure (OHLCV, technical indicators) provides the *symptoms*, but macroeconomic events, policy changes, and corporate actions provide the *catalysts*. Incorporating these exogenous data points can dramatically improve a model's ability to foresee sudden shifts from "Quiet/Normal" to "Chaotic".

## 1. Reserve Bank of India (RBI) Announcements
*   **Monetary Policy Committee (MPC) Meetings:** The RBI typically meets bi-monthly. The days leading up to, and especially the day of, the rate decision (Repo Rate, Reverse Repo, CRR, SLR) exhibit heavily altered volatility dynamics.
*   **Unscheduled Interventions:** Sudden liquidity tightening/easing measures or forex interventions by the RBI.
*   **Feature Engineering:** Create binary features (e.g., `is_rbi_mpc_day`, `is_rbi_mpc_week`) or ordinal/continuous features tracking "days until next scheduled RBI announcement".

## 2. Corporate Earnings & Filings (For NIFTY 50 Constituents)
*   Because the NIFTY 50 is a market-cap-weighted index, earnings reports from major heavyweights (e.g., HDFC Bank, Reliance Industries, ICICI Bank, Infosys, TCS) can single-handedly drive index-level volatility.
*   **Feature Engineering:** 
    *   Track the earnings calendar of the top 10-15 NIFTY constituents by weight.
    *   `num_heavyweight_earnings_this_week`: Count of major earnings releases.
    *   `earnings_surprise_momentum`: If a major heavyweight vastly misses/beats estimates, it creates sector-wide contagion that impacts the broader index.

## 3. Global Macroeconomic Indicators
Indian markets do not operate in a vacuum; they are highly sensitive to global liquidity, foreign capital flows, and international sentiment.
*   **US Federal Reserve (FOMC):** Interest rate decisions by the Fed directly dictate Foreign Institutional Investor (FII) flows into emerging markets like India.
*   **US Non-Farm Payrolls (NFP) & CPI:** Major US data releases affect global bond yields (e.g., US 10-Year Treasury) and global equity risk appetite.
*   **USD/INR Exchange Rate:** Sharp, sudden depreciations in the Rupee often correlate with FII selling and higher NIFTY volatility.
*   **Crude Oil Prices (Brent/WTI):** Since India is a massive net importer of crude oil, oil price shocks almost instantly trigger "Chaotic" risk-off regimes in Indian equities.

## 4. Institutional Flows (FII & DII Data)
*   **Foreign & Domestic Institutional Investors:** Net buy/sell figures in the cash and derivatives market (published daily by NSE).
*   **Feature Engineering:** A sudden regime shift where FIIs transition from steady buying to aggressive net selling is a strong leading indicator of impending market stress. Use rolling 3-day or 5-day sums/averages of FII net flows.

## 5. Derivative Market Sentiment
*   **India VIX:** The implied volatility index for NIFTY options. Using the *rate of change* or moving average crossovers of the India VIX is highly predictive of future realized volatility.
*   **Put-Call Ratio (PCR):** An extreme PCR (too high = fearful/hedged, too low = overly greedy) indicates an overextended market prone to sharp, chaotic mean reversions.

## 6. How to Integrate into Training & Inference
*   **Event Flags:** The simplest method is adding boolean columns (0 or 1) for `is_event_day`.
*   **Decay Features:** Instead of a binary flag, use an exponential decay feature (e.g., spikes to 1.0 on the event day and decays to 0 over the next 3-5 days) to model the lingering psychological effects of a macro shock.
*   **Validating Clustering:** You can use these dates to validate your unsupervised clustering models. If your HMM correctly assigns "Chaotic" labels to historical RBI surprise rate hike days or COVID-19 lockdown announcement days, it proves the clustering logic is sound.
