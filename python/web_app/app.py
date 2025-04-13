import streamlit as st
from pages import landing, comparison

# Page setup
st.set_page_config(
    page_title="Loan Default Prediction Models",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Landing Page", "Model Comparison"])

# Page routing
if page == "Landing Page":
    landing.show()
elif page == "Model Comparison":
    comparison.show()
