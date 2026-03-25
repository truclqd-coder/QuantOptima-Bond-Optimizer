# ⚖️ QuantOptima: AI-Enhanced Bond Portfolio Optimizer

**QuantOptima** is a sophisticated fixed-income decisioning tool that utilizes the **Black-Litterman Model** to overcome the limitations of traditional Mean-Variance Optimization (MVO). By blending **Market Equilibrium** with **Subjective Investor Views**, this application provides a stabilized, institution-grade approach to bond portfolio construction.

## 🚀 Strategic Value Proposition
In traditional Asset Liability Management (ALM), standard optimization often leads to "corner solutions"—extreme, undiversified bets based on minor input errors. **QuantOptima** addresses this by:
- **Anchoring to Market Priors:** Starting with a neutral, market-cap-weighted equilibrium.
- **Bayesian Updating:** Incorporating specific forecasts (Views) weighted by confidence levels.
- **Explainable Asset Allocation:** Clearly visualizing how active views "tilt" the portfolio away from the benchmark.

## 🧠 The Mathematical Engine
The core of this application is a **Bayesian Inference** framework. It treats the market's expected returns as a "Prior" distribution and updates it into a "Posterior" distribution once new evidence (User Views) is introduced.

### 1. Reverse Optimization (The Prior)
We calculate the Market Implied Returns ($\Pi$) using the current market capitalization and risk-aversion coefficient ($\delta$):
$$\Pi = \delta \Sigma w_{mkt}$$

### 2. The Black-Litterman Formula
The model blends these priors with user views ($Q$) and uncertainty ($\Omega$):
$$E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]$$

## 🛠️ Technical Stack
- **Language:** Python 3.11
- **Optimization:** [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/) (Black-Litterman & Efficient Frontier modules)
- **Visualization:** Plotly (Interactive Financial Charts)
- **Dashboard:** Streamlit Cloud
- **Data Handling:** Pandas & NumPy

## 📊 How to Use
1. **Analyze the Prior:** View the initial market-cap-weighted bond portfolio.
2. **Inject Views:** Use the sidebar to set your expected returns for specific sectors (e.g., High Yield or Emerging Markets).
3. **Set Confidence:** Adjust the "Uncertainty" slider to determine how much the model should trust your forecast versus the market consensus.
4. **Optimize:** Generate the new "Max Sharpe" portfolio weights and compare them against the neutral benchmark.

## 📈 Future AI Integration
The next phase of this project involves replacing manual user "Views" with:
- **LSTM Time-Series Forecasts:** Using deep learning to predict yield curve shifts.
- **NLP Sentiment Analysis:** Scrapping Fed meeting minutes to automate "Hawkish/Dovish" tilts.

---
*Developed by Truc Nguyen, MSF, MScs candidate in AI - Data Analytics specialization, Product Support Specialist & Knowledge Lead at Moody's Analytics, Inc.*
