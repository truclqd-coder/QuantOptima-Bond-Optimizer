import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import black_litterman, risk_models, BlackLittermanModel, EfficientFrontier

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="QuantOptima | Bond Optimizer", layout="wide")

# Custom CSS to align with a professional financial portfolio look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA GENERATION (Synthetic Bond Universe) ---
@st.cache_data
def get_bond_market_data():
    tickers = ["US_TREASURY_10Y", "CORP_BOND_AAA", "HIGH_YIELD_BB", "MUNI_BOND", "EM_SOVEREIGN"]
    
    # Market Capitalization (Approximate weights for Equilibrium)
    market_caps = {
        "US_TREASURY_10Y": 550e6,
        "CORP_BOND_AAA": 250e6,
        "HIGH_YIELD_BB": 100e6,
        "MUNI_BOND": 60e6,
        "EM_SOVEREIGN": 40e6
    }
    
    # Generate 2 years of synthetic daily returns
    np.random.seed(42)
    mu = [0.0001, 0.00012, 0.00025, 0.00009, 0.0003]
    std = [0.005, 0.007, 0.015, 0.006, 0.02]
    corr = np.array([
        [1.0, 0.8, 0.2, 0.5, 0.1],
        [0.8, 1.0, 0.4, 0.4, 0.2],
        [0.2, 0.4, 1.0, 0.1, 0.6],
        [0.5, 0.4, 0.1, 1.0, 0.1],
        [0.1, 0.2, 0.6, 0.1, 1.0]
    ])
    cov = np.outer(std, std) * corr
    returns = np.random.multivariate_normal(mu, cov, size=500)
    prices = pd.DataFrame(100 * (1 + returns).cumprod(axis=0), columns=tickers)
    
    return prices, market_caps

prices_df, mkt_caps = get_bond_market_data()

# --- 3. SIDEBAR: USER VIEWS (The "AI" Input) ---
st.sidebar.header("📊 Investor Views & Confidence")
st.sidebar.markdown("Define your active tilts relative to the market consensus.")

views = {}
confidences = {}

for ticker in ["HIGH_YIELD_BB", "EM_SOVEREIGN"]:
    st.sidebar.subheader(f"{ticker}")
    views[ticker] = st.sidebar.slider(f"Expected Annual Return", 0.0, 15.0, 6.0, key=f"v_{ticker}") / 100
    conf = st.sidebar.select_slider(f"Confidence Level", options=["Low", "Medium", "High"], value="Medium", key=f"c_{ticker}")
    conf_map = {"Low": 0.1, "Medium": 0.05, "High": 0.01}
