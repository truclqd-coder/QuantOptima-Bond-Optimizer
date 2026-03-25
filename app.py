import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. SETTINGS ---
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
    returns = np.random.normal(0.0002, 0.01, (500, 5))
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    return prices, market_caps

prices_df, mkt_caps = get_bond_market_data()

# --- 3. SIDEBAR: THE INPUTS ---
st.sidebar.header("🕹️ Portfolio Controls")
st.sidebar.markdown("Modify market assumptions to rebalance the portfolio.")

views = {}
for ticker in ["HIGH_YIELD_BB", "EM_SOVEREIGN"]:
    views[ticker] = st.sidebar.slider(
        f"{ticker} Forecast (%)", 
        0.0, 15.0, 5.0, step=0.5,
        key=f"s_{ticker}"
    ) / 100

st.sidebar.divider()
tau = st.sidebar.select_slider(
    "Bayesian Confidence (τ)", 
    options=[0.01, 0.05, 0.1], 
    value=0.05
)

# --- 4. MAIN DASHBOARD ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")
st.markdown("---")

try:
    # --- MATH ENGINE ---
    S = risk_models.sample_cov(prices_df)
    delta = 2.5
    prior_returns = black_litterman.market_implied_prior_returns(mkt_caps, delta, S)

    bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=views, tau=tau)
    bl_rets = bl.bl_returns()

    ef = EfficientFrontier(bl_rets, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Performance Metrics
    ret, vol, sharpe = ef.portfolio_performance()

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Asset Allocation: Equilibrium vs. Optimized")
        mkt_total = sum(mkt_caps.values())
        mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
        opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=mkt_w, name="Market Equilibrium", marker_color='#CBD5E0'))
        fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=opt_w, name="Post-View Optimization", marker_color='#2B6CB0'))
        
        fig.update_layout(barmode='group', height=350, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Expected Return Profile")
        rets_df = pd.DataFrame({
            "Market Prior": prior_returns, 
            "Blended Posterior": bl_rets
        })
        st.dataframe(rets_df.style.format("{:.2%}"), use_container_width=True)

    # NEW: KEY PERFORMANCE INDICATORS (KPIs)
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Expected Annual Return", f"{ret:.2%}")
    k2.metric("Portfolio Volatility", f"{vol:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

except Exception as e:
    st.error(f"Solver Error: {e}")
