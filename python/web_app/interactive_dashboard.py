# interactive_dashboard.py

import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from torch import nn

# ==== Load Dataset ====
@st.cache_data
def load_data():
    df = pd.read_csv("loan_data_cleaned.csv")  # update to your filename
    return df

# ==== Model Class (MLP) ====
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        for hidden in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ==== Load Model ====
def load_model(path, input_dim, hidden_layers):
    model = MLP(input_dim, hidden_layers)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ==== Run App ====
def main():
    st.title("🔍 Loan Default Model Explorer")

    df = load_data()
    feature_columns = [col for col in df.columns if col != "loan_status"]
    X = df[feature_columns].copy()
    y = df["loan_status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_options = {
        "Mix (Best)": ("loan_fnn_model_mix.pth", [128, 64, 32]),
        "No Feature Engineering": ("loan_fnn_model_no_feature_engineering.pth", [64, 32]),
        "Single Layer": ("loan_fnn_model_no_single_layer.pth", [500]),
        "Initial Model": ("loan_fnn_model_.pth", [64, 32])
    }

    model_choice = st.selectbox("Choose a Model", list(model_options.keys()))
    x_feature = st.selectbox("X-Axis Feature", feature_columns)
    y_feature = st.selectbox("Y-Axis Feature", feature_columns, index=1)

    model_path, layers = model_options[model_choice]
    model = load_model(model_path, X.shape[1], layers)

    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()
        predicted_labels = (preds > 0.5).astype(int)

    # Prepare DataFrame for Plotting
    plot_df = df[[x_feature, y_feature]].copy()
    plot_df["Prediction"] = predicted_labels
    plot_df["Prediction Label"] = plot_df["Prediction"].map({1: "Repaid", 0: "Defaulted"})

    # Plot
    fig = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color="Prediction Label",
        title=f"Scatter Plot of {model_choice} Predictions",
        opacity=0.7,
        color_discrete_map={"Repaid": "green", "Defaulted": "red"},
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
