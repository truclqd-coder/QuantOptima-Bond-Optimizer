import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | Generative Bayesian Optimizer", layout="wide")

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

# --- 3. SIDEBAR: GENERATIVE BAYESIAN CALIBRATION ---
st.sidebar.header("🤖 Generative Bayesian Model")
st.sidebar.markdown("Adjust return expectations and market assumptions to update the Generative Bayesian model.")

views = {}

with st.sidebar:
    # --- CATEGORY 1: MODEL CONVICTION ---
    st.subheader("🧠 Bayesian Conviction")
    tau = st.select_slider(
        "Confidence (τ)", 
        options=[0.01, 0.05, 0.1], 
        value=0.05,
        help="**Bayesian Confidence (τ):** A hyperparameter representing the certainty of the input views relative to the market prior. A higher value gives more weight to the user's specific expectations in the final posterior distribution."
    )
    
    st.divider()
    
    # --- CATEGORY 2: STABLE ASSET EXPECTATIONS ---
    st.subheader("🏦 Stable Asset Expectations")
    
    views["MUNI_BOND"] = st.slider(
        "MUNI_BOND (%)", 0.0, 15.0, 4.0, step=0.5, key="s_muni",
        help="**Municipal Bonds:** Debt securities issued by state or local governments. They generally offer lower yields due to their tax-exempt status and are a cornerstone for lower-volatility, income-focused fixed income portfolios."
    ) / 100
    
    views["US_TREASURY_10Y"] = st.slider(
        "US_TREASURY_10Y (%)", 0.0, 15.0, 4.0, step=0.5, key="s_ust",
        help="**10-Year US Treasury:** The global benchmark for 'risk-free' returns. Its yield reflects market expectations for long-term inflation and economic growth, serving as the primary anchor for the entire yield curve."
    ) / 100
    
    views["CORP_BOND_AAA"] = st.slider(
        "CORP_BOND_AAA (%)", 0.0, 15.0, 5.0, step=0.5, key="s_aaa",
        help="**AAA Corporate Bonds:** The highest tier of private-sector debt. These bonds represent companies with the strongest capacity to meet financial commitments, offering a small yield premium (spread) over government debt."
    ) / 100

    st.divider()

    # --- CATEGORY 3: HIGH-YIELD EXPECTATIONS ---
    st.subheader("🔥 High-Yield Expectations")
    
    views["HIGH_YIELD_BB"] = st.slider(
        "HIGH_YIELD_BB (%)", 0.0, 15.0, 7.0, step=0.5, key="s_hy",
        help="**High Yield (BB):** 'Crossover' debt that sits just below investment grade. These bonds offer higher income potential to compensate for increased default risk and higher sensitivity to corporate credit cycles."
    ) / 100
    
    views["EM_SOVEREIGN"] = st.slider(
        "EM_SOVEREIGN (%)", 0.0, 15.0, 8.0, step=0.5, key="s_em",
        help="**Emerging Market Sovereign:** Debt issued by governments of developing nations. These assets provide significant yield premiums but are sensitive to geopolitical stability, currency fluctuations, and global liquidity."
    ) / 100

# --- 4. MAIN DASHBOARD ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")
st.info("🧬 **AI Engine Status:** Generative Bayesian Inference active. Updating portfolio distribution based on return expectations.")
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
        st.warning("⚠️ Optimization failed to converge on Max Sharpe. Showing Minimum Volatility weights.")

    cleaned_weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.0)

    # --- ROW 1: YIELD CURVE ---
    st.header("📈 Generative Term Structure Update")
    curve_df = pd.DataFrame({
        "Duration": durations,
        "Market Prior": prior_returns.values * 100,
        "Posterior (AI)": bl_rets.values * 100
    }).sort_values("Duration")

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Market Prior"], mode='lines+markers', name="Market Equilibrium (Prior)", line=dict(color='#CBD5E0', dash='dash')))
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Posterior (AI)"], mode='lines+markers', name="Generated Posterior", line=dict(color='#3182CE', width=4)))
    fig_curve.update_layout(height=400, xaxis_title="Duration (Years)", yaxis_title="Exp. Return (%)", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig_curve, use_container_width=True)

    # --- ROW 2: ALLOCATION & DATA ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Asset Allocation Comparison")
        mkt_total = sum(mkt_caps.values())
        mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
        opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=mkt_w, name="Market Equilibrium", marker_color='#E2E8F0'))
        fig_bar.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=opt_w, name="Optimized Posterior Weights", marker_color='#3182CE'))
        fig_bar.update_layout(barmode='group', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Return Analysis (%)")
        st.dataframe(pd.DataFrame({"Neutral Prior": prior_returns, "Posterior (AI)": bl_rets}).style.format("{:.2%}"), use_container_width=True)

    # --- ROW 3: KPI METRICS ---
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Exp. Annual Return", f"{ret:.2%}")
    k2.metric("Portfolio Risk", f"{vol:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

except Exception as e:
    st.error(f"🚨 Model Error: {e}")
