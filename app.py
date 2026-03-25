import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | Bond Optimizer", layout="wide")

# --- 2. DATA GENERATION ---
@st.cache_data
def get_bond_market_data():
    tickers = ["US_TREASURY_10Y", "CORP_BOND_AAA", "HIGH_YIELD_BB", "MUNI_BOND", "EM_SOVEREIGN"]
    market_caps = {
        "US_TREASURY_10Y": 550e6, "CORP_BOND_AAA": 250e6, 
        "HIGH_YIELD_BB": 100e6, "MUNI_BOND": 60e6, "EM_SOVEREIGN": 40e6
    }
    np.random.seed(42)
    # Generate more stable synthetic returns
    returns = np.random.normal(0.0002, 0.01, (500, 5))
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    return prices, market_caps

prices_df, mkt_caps = get_bond_market_data()

# --- 3. SIDEBAR ---
st.sidebar.header("🔍 Step 1: Investor Views")
views = {}
for ticker in ["HIGH_YIELD_BB", "EM_SOVEREIGN"]:
    views[ticker] = st.sidebar.slider(f"{ticker} Expected Return (%)", 0.0, 15.0, 5.0) / 100

run_opt = st.sidebar.button("🚀 Run AI Optimization")

# --- 4. MAIN PAGE ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")

if not run_opt:
    st.info("👈 Adjust your views in the sidebar and click **'Run AI Optimization'** to generate the portfolio.")
    # Show the "Neutral" starting point while waiting
    st.subheader("Current Market Equilibrium (Benchmark)")
    mkt_total = sum(mkt_caps.values())
    mkt_weights = {k: v/mkt_total for k, v in mkt_caps.items()}
    st.bar_chart(pd.Series(mkt_weights))
else:
    try:
        # A. Risk Model
        S = risk_models.sample_cov(prices_df)
        delta = 2.5
        prior_returns = black_litterman.market_implied_prior_returns(mkt_caps, delta, S)

        # B. BL Model
        bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=views)
        bl_rets = bl.bl_returns()

        # C. Optimization
        ef = EfficientFrontier(bl_rets, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # D. Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimized Portfolio Weights")
            fig = go.Figure([go.Bar(x=list(cleaned_weights.keys()), y=list(cleaned_weights.values()))])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Blended Expected Returns")
            st.write(pd.DataFrame({"Market Implied": prior_returns, "BL Blended": bl_rets}).style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
        st.warning("Try adjusting the sliders to more conservative values.")
