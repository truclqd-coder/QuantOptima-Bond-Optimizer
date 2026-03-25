import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. SETTINGS ---
st.set_page_config(page_title="QuantOptima | Bond Optimizer", layout="wide")

# --- 2. DATA GENERATION ---
# We cache the price data (the history), but NOT the optimization math.
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

# --- 3. SIDEBAR: INVESTOR VIEWS ---
st.sidebar.header("🔍 Market Outlook")
st.sidebar.markdown("Adjust the expected annual returns to see how the portfolio rebalances in real-time.")

# These sliders will now trigger an instant refresh of the math below
views = {}
for ticker in ["HIGH_YIELD_BB", "EM_SOVEREIGN"]:
    # Using a unique key and default value to ensure reactivity
    views[ticker] = st.sidebar.slider(
        f"{ticker} Forecast (%)", 
        min_value=0.0, 
        max_value=15.0, 
        value=5.0,
        step=0.5,
        key=f"slider_{ticker}"
    ) / 100

st.sidebar.divider()
st.sidebar.header("🛡️ Risk Parameters")
tau = st.sidebar.select_slider(
    "Model Confidence", 
    options=[0.01, 0.05, 0.1], 
    value=0.05,
    help="Higher values give more weight to your forecasts vs. market history."
)

# --- 4. MAIN DASHBOARD ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")
st.markdown("---")

try:
    # --- THE MATHEMATICAL ENGINE (Runs on every slider move) ---
    S = risk_models.sample_cov(prices_df)
    delta = 2.5
    prior_returns = black_litterman.market_implied_prior_returns(mkt_caps, delta, S)

    # Calculate Black-Litterman Posterior Returns
    bl = BlackLittermanModel(S, pi=prior_returns, absolute_views=views, tau=tau)
    bl_rets = bl.bl_returns()

    # Optimize for Max Sharpe Ratio
    ef = EfficientFrontier(bl_rets, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Active Portfolio Weights")
        mkt_total = sum(mkt_caps.values())
        mkt_w = [mkt_caps[t]/mkt_total for t in cleaned_weights.keys()]
        opt_w = [cleaned_weights[t] for t in cleaned_weights.keys()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=mkt_w, name="Market Benchmark", marker_color='#E2E8F0'))
        fig.add_trace(go.Bar(x=list(cleaned_weights.keys()), y=opt_w, name="Optimized Allocation", marker_color='#2B6CB0'))
        
        fig.update_layout(
            barmode='group', 
            height=400, 
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Projected Returns (%)")
        # We create a dataframe and style it to ensure it refreshes visually
        rets_df = pd.DataFrame({
            "Market Consensus": prior_returns, 
            "Your AI View": bl_rets
        })
        st.dataframe(rets_df.style.format("{:.2%}"), use_container_width=True)

    st.info("💡 **Observation:** As you increase the forecast for a sector, the model mathematically 'tilts' the allocation toward that bond while maintaining a diversified risk profile.")

except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.warning("Please ensure your forecasts are within reasonable market bounds (0% - 15%).")
