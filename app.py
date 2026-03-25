import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | Bond Optimizer", layout="wide")

# --- 2. DATA GENERATION (Fixed Income Universe) ---
@st.cache_data
def get_bond_market_data():
    # Assets ordered by typical Duration/Risk profile (Short to Long)
    tickers = ["MUNI_BOND", "US_TREASURY_10Y", "CORP_BOND_AAA", "HIGH_YIELD_BB", "EM_SOVEREIGN"]
    durations = [3, 10, 12, 15, 20] 
    
    market_caps = {
        "MUNI_BOND": 60e6, 
        "US_TREASURY_10Y": 550e6, 
        "CORP_BOND_AAA": 250e6, 
        "HIGH_YIELD_BB": 100e6, 
        "EM_SOVEREIGN": 40e6
    }
    
    # Synthetic price history for covariance calculation
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, (1000, 5)) 
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    return prices, market_caps, durations

prices_df, mkt_caps, durations = get_bond_market_data()

# --- 3. SIDEBAR: MARKET EXPECTATIONS & FORECASTS ---
st.sidebar.header("📈 Market Expectations & Forecasts")
st.sidebar.markdown("Adjust sectoral return assumptions to tilt the equilibrium portfolio.")

with st.sidebar:
    # Bayesian Confidence Tooltip
    tau = st.select_slider(
        "Bayesian Confidence (τ)", 
        options=[0.01, 0.05, 0.1], 
        value=0.05,
        help="**Bayesian Confidence (τ):** Represents how much you trust your personal views vs. the market.\n\n"
             "* **0.01 (Low):** Anchors heavily to market history.\n"
             "* **0.10 (High):** Aggressively tilts toward your forecasts."
    )
    
    st.divider()
    
    views = {}
    for ticker in prices_df.columns:
        # Custom Tooltips for each Bond Sector
        if ticker == "HIGH_YIELD_BB":
            h_text = "**High Yield BB:** 'Crossover' corporate debt rated BB (just below investment grade). Offers higher income with moderate default risk."
        elif ticker == "EM_SOVEREIGN":
            h_text = "**EM Sovereign:** Government bonds from Emerging Market nations. Higher political/currency risk with significant yield premiums."
        elif ticker == "US_TREASURY_10Y":
            h_text = "**US Treasury 10Y:** The global risk-free benchmark. Highly sensitive to inflation and Fed policy."
        elif ticker == "MUNI_BOND":
            h_text = "**Muni Bond:** Tax-exempt debt issued by local governments. Generally lower volatility and shorter duration."
        else:
            h_text = f"Expected annual return for {ticker}."

        # Slider with help tooltip
        views[ticker] = st.slider(
            f"{ticker} Forecast (%)", 
            0.0, 15.0, 5.0, 
            step=0.5, 
            key=f"s_{ticker}",
            help=h_text
        ) / 100

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
        # Set risk_free_rate to 0 for stable simulation
        weights = ef.max_sharpe(risk_free_rate=0.0)
    except:
        weights = ef.min_volatility()
        st.warning("⚠️ Optimization failed to converge on Max Sharpe. Showing Minimum Volatility portfolio.")

    cleaned_weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=0.0)

    # --- ROW 1: YIELD CURVE VISUALIZATION ---
    st.header("📈 Implied Yield Curve Shift")
    st.markdown("Visualizing the Bayesian transition from the **Market Prior** to the **Blended Posterior**.")
    
    curve_df = pd.DataFrame({
        "Duration": durations,
        "Market Prior": prior_returns.values * 100,
        "Blended Posterior": bl_rets.values * 100
    }).sort_values("Duration")

    fig_curve = go.Figure()
    # Prior Curve (Dashed)
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Market Prior"], 
                                  mode='lines+markers', name="Market Prior Curve", 
                                  line=dict(color='#CBD5E0', dash='dash')))
    # Posterior Curve (Solid)
    fig_curve.add_trace(go.Scatter(x=curve_df["Duration"], y=curve_df["Blended Posterior"], 
                                  mode='lines+markers', name="Blended Posterior Curve", 
                                  line=dict(color='#3182CE', width=4)))
    
    fig_curve.update_layout(height=400, xaxis_title="Duration (Years)", yaxis_title="Exp. Return (%)",
                           legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
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
        st.dataframe(pd.DataFrame({
            "Market Prior": prior_returns, 
            "Blended Posterior": bl_rets
        }).style.format("{:.2%}"), use_container_width=True)

    # --- ROW 3: KPI METRICS ---
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Exp. Annual Return", f"{ret:.2%}")
    k2.metric("Portfolio Volatility", f"{vol:.2%}")
    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")

except Exception as e:
    st.error(f"🚨 Model Error: {e}")
