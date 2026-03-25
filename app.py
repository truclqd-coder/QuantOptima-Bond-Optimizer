import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | Bond Optimizer", layout="wide")

# --- 2. DATA GENERATION ---
@st.cache_data
def get_bond_market_data():
    tickers = ["MUNI_BOND", "US_TREASURY_10Y", "CORP_BOND_AAA", "HIGH_YIELD_BB", "EM_SOVEREIGN"]
    durations = [3, 10, 12, 15, 20] 
    market_caps = {
        "MUNI_BOND": 60e6, "US_TREASURY_10Y": 550e6, 
        "CORP_BOND_AAA": 250e6, "HIGH_YIELD_BB": 100e6, "EM_SOVEREIGN": 40e6
    }
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, (1000, 5)) 
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    return prices, market_caps, durations

prices_df, mkt_caps, durations = get_bond_market_data()

# --- 3. SIDEBAR: CATEGORIZED MARKET EXPECTATIONS ---
st.sidebar.header("📈 Market Expectations")
st.sidebar.markdown("Define your conviction and return forecasts to update the model.")

views = {}

with st.sidebar:
    # --- CATEGORY 1: MODEL CONVICTION ---
    st.subheader("🎯 Model Conviction")
    tau = st.select_slider(
        "Bayesian Confidence (τ)", 
        options=[0.01, 0.05, 0.1], 
        value=0.05,
        help="**Bayesian Confidence (τ):** Defines how much you trust your views vs. market history.\n\n"
             "* **0.01 (Low):** Anchors heavily to market history.\n"
             "* **0.10 (High):** Aggressively tilts toward your forecasts."
    )
    
    st.divider()
    
    # --- CATEGORY 2: CORE MARKET FORECASTS (High Quality/Low Risk) ---
    st.subheader("🏦 Core Market Forecasts")
    core_tickers = ["MUNI_BOND", "US_TREASURY_10Y", "CORP_BOND_AAA"]
    for ticker in core_tickers:
        if ticker == "US_TREASURY_10Y":
            h_text = "**US Treasury 10Y:** The global risk-free benchmark. Highly sensitive to inflation and Fed policy."
        elif ticker == "MUNI_BOND":
            h_text = "**Muni Bond:** Tax-exempt debt issued by local governments. Generally lower volatility."
        else:
            h_text = "**Corp Bond AAA:** Highest-quality corporate debt with minimal default risk."
            
        views[ticker] = st.slider(f"{ticker} (%)", 0.0, 15.0, 4.0, step=0.5, key=f"s_{ticker}", help=h_text) / 100

    st.divider()

    # --- CATEGORY 3: TACTICAL RISK OUTLOOK (Higher Yield/Risk) ---
    st.subheader("🔥 Tactical Risk Outlook")
    tactical_tickers = ["HIGH_YIELD_BB", "EM_SOVEREIGN"]
    for ticker in tactical_tickers:
        if ticker == "HIGH_YIELD_BB":
            h_text = "**High Yield BB:** 'Crossover' corporate debt. Higher income with moderate default risk."
        else:
            h_text = "**EM Sovereign:** Government bonds from Emerging Market nations. Higher political and currency risk."
            
        views[ticker] = st.slider(f"{ticker} (%)", 0.0, 15.0, 7.0, step=0.5, key=f"s_{ticker}", help=h_text) / 100

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
    try:
        weights = ef.max_sharpe(risk_free_rate=0.0)
    except:
        weights = ef.min_volatility()
        st.warning("⚠️ Max Sharpe failed to converge. Showing Minimum Volatility portfolio.")

    cleaned_weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.0)

    # --- ROW 1: YIELD CURVE ---
    st.header("📈 Implied Yield Curve Shift")
    curve_df = pd.DataFrame({
        "Duration": durations,
        "Prior": prior_returns.values * 100,
        "Posterior": bl_rets.values * 100
    }).sort_values("Duration")

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Prior"], mode='lines+markers', name="Market Prior", line=dict(color='#CBD5E0', dash='dash')))
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Posterior"], mode='lines+markers', name="Blended Posterior", line=dict(color='#3182CE', width=4)))
    fig_curve.update_layout(height=400, xaxis_title="Duration (Years)", yaxis_title="Exp. Return (%)", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig_curve, use_container_width=True)

    # --- ROW 2: ALLOCATION & DATA ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Asset Allocation: Equilibrium vs. Optimized")
        mkt_total = sum(mkt_caps.values())
        mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
        opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=mkt_w, name="Market Equilibrium", marker_color='#E2E8F0'))
        fig_bar.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=opt_w, name="Post-View Optimization", marker_color='#3182CE'))
        fig_bar.update_layout(barmode='group', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Return Profile (%)")
        st.dataframe(pd.DataFrame({"Prior": prior_returns, "Posterior": bl_rets}).style.format("{:.2%}"), use_container_width=True)

    # --- ROW 3: KPI METRICS ---
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Exp. Annual Return", f"{ret:.2%}")
    k2.metric("Portfolio Volatility", f"{vol:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

except Exception as e:
    st.error(f"🚨 Model Error: {e}")
