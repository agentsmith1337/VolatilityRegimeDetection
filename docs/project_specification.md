# Project Specification: Volatility Regime Prediction in Indian Markets

## 1. Introduction & Objectives
The overarching goal of this project is to classify and forecast the market volatility regime (Chaotic, Normal, Quiet) of the NIFTY 50 index. The pipeline will ingest market data, compute volatility-centric features, cluster historical data into distinct regimes using unsupervised learning/econometrics, and subsequently train predictive machine learning models to forecast future volatility states.

## 2. Requirements Specification
*   **Target Asset:** NIFTY 50 Index (and optionally heavy-weight constituents).
*   **Compute Environment:** Google Colab (utilizing GPU resources for Deep Learning & intensive ensemble models).
*   **Format:** Jupyter Notebooks (`.ipynb`) organized sequentially by pipeline stage.
*   **Programming Language:** Python 3.10+
*   **Primary Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `hmmlearn`, `arch` (for GARCH), `tensorflow`/`keras` or `pytorch` (for LSTM/GRU).
*   **Data Sources:** `yfinance` (fallback), NSE website, or specialized data vendors (e.g., Zerodha Kite Connect, TrueData) for granular tick/order flow data.

## 3. Standard Operating Procedure (SOP) & Project Structure
The project will be organized into modular, independent notebooks/folders:

1.  **`01_Data_Ingestion/`**: Fetching NIFTY 50 data (OHLCV, tick, order flow, macro events).
2.  **`02_Feature_Engineering/`**: 
    *   *Realized Volatility & Historical Volatility*
    *   *Moving Averages (SMA, EMA, MACD)*
    *   *Garman-Klass Volatility Estimator*
    *   *Bid-Ask Spread, Tick Volume, Order Flow* (Subject to data vendor availability).
3.  **`03_Regime_Clustering/`**: Creating the target labels (Quiet, Normal, Chaotic).
    *   **Pipeline A:** Hidden Markov Models (HMM)
    *   **Pipeline B:** Gaussian Mixture Models (GMM)
    *   **Pipeline C:** GARCH (Binning conditional variance forecasts into discrete regimes)
4.  **`04_Predictive_Modeling/`**: Supervised classification models mapping historical features to future regime labels.
    *   *Tree-Based:* Random Forest, XGBoost, AdaBoost, LightGBM (using lagged/windowed features).
    *   *Deep Learning:* LSTM, GRU (utilizing 3D tensors of sequence data).
5.  **`05_Ensemble_Lab/`**: Combining predictions of the best base models (Soft/Hard Voting, Stacking, Blending) to improve robustness.
6.  **`06_Experimental_Lab/`**: Sandbox for testing new ideas, alternate datasets, or reinforcement learning approaches.

## 4. Expected Results & Specifications
*   **Objective:** Successfully forecast the next period's (day/hour) volatility regime to aid in quantitative strategies, options pricing, or risk management.
*   **Evaluation Metrics:**
    *   **Macro / Weighted F1-Score:** The primary metric. Accuracy can be highly misleading in financial data due to class imbalance (e.g., the market is "Normal" 70% of the time). F1 balances precision and recall.
    *   **Recall for "Chaotic" Class:** High recall here is vital. Missing a "Chaotic" regime prediction (False Negative) is financially much more disastrous than misclassifying a "Quiet" regime.
    *   **Confusion Matrix:** Essential for visualizing severe misclassifications (e.g., predicting "Quiet" when the reality is "Chaotic").
    *   **ROC-AUC (One-vs-Rest):** To measure the model's ability to distinguish between regimes at various probability thresholds.

## 5. Benchmarking
*   **Naive Baseline (Persistence):** Predicting that tomorrow's regime will be exactly the same as today's. 
*   **Majority Class Baseline:** Always predicting the most frequent class (e.g., "Normal").
*   **Econometric Baseline:** Standard GARCH(1,1) forecasted volatility converted into regimes via fixed statistical thresholds (e.g., > 90th percentile = Chaotic).
*   *Requirement:* The Machine Learning models and ensembles must significantly outperform these baselines in terms of F1-Score to justify their complexity.

## 6. Feedback & Constructive Criticism
*   **Data Availability Constraints:** `yfinance` primarily provides daily (and sometimes minute-level) OHLCV data. It **does not** provide bid-ask spreads, tick volume, or accurate order flow data for Indian markets. To engineer those specific features, you will likely need a subscription to a vendor like TrueData, Global Datafeeds, or a broker API like Zerodha/Upstox.
*   **GARCH as a Clusterer:** GARCH is inherently a variance forecasting model, not a clustering model like HMM/GMM. However, using historical GARCH conditional variances and applying K-Means or percentile thresholding to create your regime labels is a very valid approach.
*   **Beware of Look-Ahead Bias (Data Leakage):** When labeling regimes using HMM/GMM, ensure the clustering is applied in an *expanding window* or rolling basis. If you fit an HMM on the *entire* 10-year dataset to generate labels, and then train an XGBoost model to predict those labels, you are providing the ML model with future information. The labels must be generated using only data available up to time *t*.
*   **Sequential Data for Trees:** Tree models (RF, XGBoost) do not inherently understand sequences like LSTMs do. You must explicitly engineer lagged features (e.g., `volatility_t-1`, `volatility_t-2`) for them to recognize temporal patterns.
