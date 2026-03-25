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
st.sidebar.header("🤖 Generative Bayesian Model Calibration")
st.sidebar.markdown("Adjust return expectations and market assumptions to update the Generative Bayesian model.")

views = {}

with st.sidebar:
    # --- CATEGORY 1: MODEL CONVICTION ---
    st.subheader("🧠 Bayesian Conviction")
    tau = st.select_slider(
        "Confidence (τ)", 
        options=[0.01, 0.05, 0.1], 
        value=0.05,
        help="**Bayesian Confidence (τ):** A hyperparameter representing the certainty of the input expectations relative to the market equilibrium. A higher value shifts the final results more aggressively toward your specified views."
    )
    
    st.divider()
    
    # --- CATEGORY 2: STABLE ASSET EXPECTATIONS ---
    st.subheader("🏦 Stable Asset Expectations")
    
    views["MUNI_BOND"] = st.slider(
        "MUNI_BOND (%)", 0.0, 15.0, 4.0, step=0.5, key="s_muni",
        help="**Municipal Bonds:** Debt issued by state or local governments. These securities typically offer lower volatility and tax-advantaged status, serving as a core defensive component of fixed-income portfolios."
    ) / 100
    
    views["US_TREASURY_10Y"] = st.slider(
        "US_TREASURY_10Y (%)", 0.0, 15.0, 4.0, step=0.5, key="s_ust",
        help="**10-Year US Treasury:** The global benchmark for risk-free assets. Its yield reflects the market outlook on inflation and growth, acting as the fundamental anchor for credit pricing."
    ) / 100
    
    views["CORP_BOND_AAA"] = st.slider(
        "CORP_BOND_AAA (%)", 0.0, 15.0, 5.0, step=0.5, key="s_aaa",
        help="**AAA Corporate Bonds:** High-grade private debt with the lowest risk of default. These offer a modest yield premium (credit spread) over government benchmarks to compensate for corporate-specific risks."
    ) / 100

    st.divider()

    # --- CATEGORY 3: HIGH-YIELD EXPECTATIONS ---
    st.subheader("🔥 High-Yield Expectations")
    
    views["HIGH_YIELD_BB"] = st.slider(
        "HIGH
