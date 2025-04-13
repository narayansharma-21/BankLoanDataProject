import streamlit as st

def show():
    st.title("Loan Default Prediction with Neural Networks and Bayesian Models")

    st.markdown("""
    This project explores the use of two contrasting modeling approaches — **Multilayer Perceptrons (MLPs)** and **Bayesian Logistic Regression** — to predict whether a borrower will default on a loan.

    We trained and compared **four neural network models** with different characteristics:
    - `loan_fnn_model.pth`: A well-tuned baseline MLP with feature engineering
    - `loan_fnn_model_mix.pth`: A hybrid MLP architecture mixing deeper and shallow layers
    - `loan_fnn_model_single_layer.pth`: A simple one-layer MLP
    - `loan_fnn_model_no_feature_engineering.pth`: MLP trained directly on raw features

    These models were trained on a cleaned credit dataset containing borrower and loan attributes like credit score, loan intent, loan amount, and more.

    In contrast, we also implemented a **Bayesian Logistic Regression** model using Metropolis-Hastings sampling to explore uncertainty and interpretability in loan prediction.

    Use the sidebar to navigate and compare the models interactively.
    """)
