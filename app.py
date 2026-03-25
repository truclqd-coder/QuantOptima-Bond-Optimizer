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
    returns = np.random.normal(0.0002, 0.01, (500, 5))
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    return prices, market_caps

prices_df, mkt_caps = get_bond_market_data()

# --- 3. SIDEBAR: STEP 1 (INPUTS) ---
st.sidebar.header("🔍 Step 1: Investor Views")
st.sidebar.markdown("Specify your active tilts for the bond market.")

views = {}
for ticker in ["HIGH_YIELD_BB", "EM_SOVEREIGN"]:
    views[ticker] = st.sidebar.slider(f"{ticker} Expected Return (%)", 0.0, 15.0, 5.0) / 100

st.sidebar.divider()
st.sidebar.header("🛡️ Risk Controls")
tau = st.sidebar.select_slider("View Confidence (Tau)", options=[0.01, 0.05, 0.1], value=0.05, 
                               help="Lower Tau means you trust the Market more. Higher Tau means you trust your AI views more.")

run_opt = st.sidebar.button("🚀 Run AI Optimization")

# --- 4. MAIN PAGE ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")

if not run_opt:
    st.info("👈 **Step 1:** Adjust your views in the sidebar and click **'Run AI Optimization'**.")
    st.subheader("Current Market Equilibrium (Neutral Benchmark)")
    mkt_total = sum(mkt_caps.values())
    mkt_weights = pd.Series({k: v/mkt_total for k, v in mkt_caps.items()})
    st.bar_chart(mkt_weights)
else:
    try:
        # A. Math Engine
        S = risk_models.sample_cov(prices_df)
        delta = 2.5
        prior_returns = black_litterman.market_implied_prior_returns(mkt_caps, delta, S)

        # B. BL Model (Blending the Prior + Views)
        bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=views, tau=tau)
        bl_rets = bl.bl_returns()

        # C. Optimization (Efficient Frontier)
        ef = EfficientFrontier(bl_rets, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # --- STEP 2: DISPLAY RESULTS ---
        st.header("🎯 Step 2: Optimized Results")
        st.markdown("The model has blended your views with market equilibrium to find the **Max Sharpe Ratio**.")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Portfolio Weight Rebalancing")
            # Compare Benchmark vs BL
            mkt_total = sum(mkt_caps.values())
            mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
            opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=mkt_w, name="Market Benchmark", marker_color='lightgrey'))
            fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=opt_w, name="AI Optimized", marker_color='#1f77b4'))
            fig.update_layout(barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Blended Returns")
            rets_df = pd.DataFrame({"Market Implied": prior_returns, "BL Posterior": bl_rets})
            st.write(rets_df.style.format("{:.2%}"), use_container_width=True)
            st.success("Optimization Successful!")

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
