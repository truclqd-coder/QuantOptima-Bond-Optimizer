# --- 4. MAIN DASHBOARD ---
st.title("⚖️ QuantOptima: Black-Litterman Bond Optimizer")

# --- NEW: GLOSSARY POP-OVER ---
with st.popover("📖 Glossary & Key Terms"):
    st.markdown("""
    ### 🧠 Bayesian Confidence ($\tau$)
    In the Black-Litterman model, this represents the **uncertainty of the prior**. 
    * **Low Tau (0.01):** You trust the market equilibrium more than your own views. The results will stay close to the benchmark.
    * **High Tau (0.10):** You have high conviction in your forecasts. The model will "tilt" the portfolio more aggressively toward your views.

    ### 🌍 EM Sovereign Forecast
    This is your expected annual return for **Emerging Market Government Bonds** (e.g., Brazil, Mexico, Indonesia). 
    * These typically offer higher yields but come with higher **political and currency risk**. 
    * In your yield curve, these are at the "Long End" (20-year duration proxy).

    ### 📉 High Yield BB Forecast
    This represents the "Crossover" segment of corporate debt. These are companies rated **BB**, just below "Investment Grade."
    * **Why BB?** They are often called "Rising Stars." They offer a "sweet spot" of higher income (spread) without the extreme default risk of lower-rated "junk" bonds.
    """)

st.markdown("---")
