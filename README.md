# 📈 dynamic-hedge-pairs-trading

A **quantitative pairs trading strategy** leveraging **dynamic hedge ratios** and **spreads** estimated via **Kalman filtering** 🧮 and **cointegration techniques** 🔗.  

Designed for **statistical arbitrage** and **mean reversion** in equities 📊.

---

## 🛠️ Workflow Overview

~~~text
🚀 Start: Candidate Universe of Securities
│
├── 🧹 Step 1: Pre-filter for Liquidity & Data Quality
│       ↳ Remove illiquid or missing-price series
│
├── 🔍 Step 2: Initial Cointegration Screening
│       ├── Option A: Engle–Granger test (static β, recent window)
│       └── Option B: Johansen test (multi-asset, recent window)
│
├── ✅ Step 3: Keep Pairs Passing Stationarity Test on Static Spread
│       ↳ Reduce universe to plausible relationships
│
├── 📏 Step 4: Apply Kalman Filter to Estimate βₜ (Dynamic Hedge Ratio)
│       ↳ Model: P1ₜ = αₜ + βₜ · P2ₜ + εₜ
│
├── 📉 Step 5: Test Kalman Filter Residuals εₜ for Stationarity
│       ├── 📊 ADF test
│       ├── 📊 KPSS test (confirm mean reversion)
│       └── 📉 Hurst exponent check (H < 0.5)
│
├── 🏁 Step 6: Keep Pairs with Mean-Reverting Dynamic Spread
│
└── 📦 Output: Trading Universe for Dynamic β Pairs Trading
~~~

---

## ✨ Key Features

- 📌 **Dynamic Hedge Ratios** estimated with a Kalman filter for adaptive trading.  
- 🔗 **Cointegration-based pre-selection** (Engle–Granger/Johansen) to ensure long-term equilibrium relationships.  
- 📉 **Multi-test validation** of spread mean reversion: ADF, KPSS, and Hurst exponent (H < 0.5).  
- 🧠 **Backtest-ready** structure designed for statistical arbitrage workflows and live deployment.  

---

## 📚 References

- Engle, R. F., & Granger, C. W. J. (1987). *Cointegration and Error Correction: Representation, Estimation, and Testing*.  
- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models*.  
- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*.  
