import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | AI Model Optimizer", layout="wide")

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

# --- 3. SIDEBAR: AI MODEL CALIBRATION ---
st.sidebar.header("🤖 AI Model Calibration")
st.sidebar.markdown("Adjust return expectations and market assumptions to update the Bayesian Inference AI model.")

views = {}

with st.sidebar:
    st.subheader("🧠 Bayesian Conviction")
    tau = st.select_slider(
        "Confidence (τ)", 
        options=[0.01, 0.05, 0.1], 
        value=0.05,
        help="""**Bayesian Confidence (τ):** Adjusts the 'Weight of Evidence' between historical data and your custom views. 
        \n\n**Examples:**
        \n* **0.01 (Low):** The AI relies heavily on historical market equilibrium, treating your views as minor adjustments.
        \n* **0.10 (High):** The AI treats your views as strong predictive signals, significantly shifting the portfolio allocation toward your expectations."""
    )
    
    st.divider()
    
    st.subheader("🏦 Stable Asset Expectations")
    views["MUNI_BOND"] = st.slider("MUNI_BOND (%)", 0.0, 15.0, 4.0, step=0.5, key="s_muni", help="**Municipal Bonds:** Debt issued by state or local governments. Offers tax-advantaged stability.") / 100
    views["US_TREASURY_10Y"] = st.slider("US_TREASURY_10Y (%)", 0.0, 15.0, 4.0, step=0.5, key="s_ust", help="**10-Year US Treasury:** The global benchmark for risk-free assets.") / 100
    views["CORP_BOND_AAA"] = st.slider("CORP_BOND_AAA (%)", 0.0, 15.0, 5.0, step=0.5, key="s_aaa", help="**AAA Corporate Bonds:** Prime-grade debt with the lowest risk of default.") / 100

    st.divider()

    st.subheader("🔥 High-Yield Expectations")
    views["HIGH_YIELD_BB"] = st.slider("HIGH_YIELD_BB (%)", 0.0, 15.0, 7.0, step=0.5, key="s_hy", help="**High Yield (BB):** Sensitive to corporate earnings; higher income potential in exchange for credit risk.") / 100
    views["EM_SOVEREIGN"] = st.slider("EM_SOVEREIGN (%)", 0.0, 15.0, 8.0, step=0.5, key="s_em", help="**Emerging Market Sovereign:** Growth play exposed to geopolitical risks and currency fluctuations.") / 100

# --- 4. MAIN DASHBOARD ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")
st.info("🧬 **AI Engine Status:** Bayesian Inference active. Synthesizing historical priors with user-defined calibration.")
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
        st.warning("⚠️ Optimization Warning: Displaying Minimum Volatility weights.")

    cleaned_weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.0)

    # --- ROW 1: YIELD CURVE OPTIMIZATION ---
    st.header("📈 Yield Curve Optimization")
    curve_df = pd.DataFrame({
        "Duration": durations,
        "Market History": prior_returns.values * 100,
        "AI Optimized": bl_rets.values * 100
    }).sort_values("Duration")

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Market History"], mode='lines+markers', name="Market Equilibrium (Prior)", line=dict(color='#CBD5E0', dash='dash')))
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["AI Optimized"], mode='lines+markers', name="AI-Optimized Yield Curve", line=dict(color='#3182CE', width=4)))
    fig_curve.update_layout(height=400, xaxis_title="Duration (Years)", yaxis_title="Expected Return (%)", legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    st.plotly_chart(fig_curve, use_container_width=True)

    # --- ROW 2: ALLOCATION & DATA ---
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Equilibrium vs. AI-Optimized Weights")
        mkt_total = sum(mkt_caps.values())
        mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
        opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
        
        fig_bar = go.Figure()
        # Market Equilibrium Trace
        fig_bar.add_trace(go.Bar(
            x=list(cleaned_weights.keys()), 
            y=mkt_w, 
            name="Market Equilibrium (Prior)", 
            marker_color='#E2E8F0',
            text=[f"{val:.1%}" for val in mkt_w],
            textposition='auto'
        ))
        # AI-Optimized Trace
        fig_bar.add_trace(go.Bar(
            x=list(cleaned_weights.keys()), 
            y=opt_w, 
            name="AI-Optimized Weights (Posterior)", 
            marker_color='#3182CE',
            text=[f"{val:.1%}" for val in opt_w],
            textposition='auto'
        ))
        fig_bar.update_layout(
            barmode='group', 
            height=400, 
            legend=dict(orientation="h", y=1.1),
            yaxis_tickformat=".0%" # Format Y-axis as percentage
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Expected Return Comparison")
        st.dataframe(pd.DataFrame({"Market Equilibrium": prior_returns, "AI-Optimized": bl_rets}).style.format("{:.2%}"), use_container_width=True)

    # --- ROW 3: KPI METRICS ---
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Exp. Annual Return", f"{ret:.2%}")
    k2.metric("Portfolio Volatility", f"{vol:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

except Exception as e:
    st.error(f"🚨 Model Error: {e}")
