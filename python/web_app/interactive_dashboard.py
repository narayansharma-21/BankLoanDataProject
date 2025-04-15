import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import nn

# ==== Load Dataset ====
@st.cache_data
def load_data():
    df = pd.read_csv("../../loan_data.csv")  # update to your filename
    return df

# ==== Model Class (MLP) ====
class LoanNN(nn.Module):
    def __init__(self, input_size):
        super(LoanNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class LoanNN_2(nn.Module):  # Used for Initial Model
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

class LoanNN_3(nn.Module):  # Used for Single Layer
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x



# ==== Load Model ====
def load_model(path, model_class, input_dim):
    model = model_class(input_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def show_landing_page():
    st.title("📌 Loan Default Prediction Project")
    st.markdown("""
    Welcome to the **Loan Default Prediction Explorer**!  
    This app lets you visualize how different neural network architectures predict loan repayment outcomes based on a rich dataset of loan applications.

    ### 🔍 Project Highlights:
    - **Dataset**: Loan application data with financial, employment, and credit history.
    - **Models**: Three neural network architectures trained on standardized inputs.
    - **Goal**: Classify loans as likely to be **repaid** or **defaulted**.

    ---
    👉 Navigate to the **Interactive Dashboard** tab to explore predictions!
    """)


def dashboard_main():
    st.title("📊 Interactive Loan Default Dashboard")

    df = load_data()
    categorical_features = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_features = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    model_options = {
        "Mix (Best)": ("../models/loan_fnn_model_mix.pth", LoanNN_2),
        "Initial Model": ("../models/loan_fnn_model.pth", LoanNN),
        "Single Layer": ("../models/loan_fnn_model_single_layer.pth", LoanNN_3)
    }

    feature_columns = categorical_features + numerical_features
    X = df[feature_columns].values

    model_choice = st.selectbox("Choose a Model", list(model_options.keys()))
    x_feature = st.selectbox("X-Axis Feature", feature_columns)
    y_feature = st.selectbox("Y-Axis Feature", feature_columns, index=1)

    model_path, model_class = model_options[model_choice]
    model = load_model(model_path, model_class, X.shape[1])

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
        predicted_labels = (preds > 0.5).astype(int)

    plot_df = df[[x_feature, y_feature]].copy()
    plot_df["Prediction"] = predicted_labels
    plot_df["Prediction Label"] = plot_df["Prediction"].map({1: "Repaid", 0: "Defaulted"})

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

    # 🔍 Interpretation Blurb
    st.markdown("""
    **🧠 How to interpret this plot:**

    Each point represents a loan applicant. The X and Y axes correspond to the features you selected above.
    
    - **Green points** are applicants the model predicts will **repay** their loans.
    - **Red points** are those predicted to **default**.
    
    Clusters or separations in the plot may reveal how certain features influence the model's decision-making.
    
    Try different combinations of features to uncover patterns in model predictions!

    **NOTE: The web app will crash if the X and Y axis are the same label!**
    """)



def main():
    tab1, tab2 = st.tabs(["🏠 Landing Page", "📊 Interactive Dashboard"])

    with tab1:
        show_landing_page()

    with tab2:
        dashboard_main()


if __name__ == "__main__":
    main()