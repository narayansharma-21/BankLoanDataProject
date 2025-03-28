# Loan Prediction Project

## Overview
This project aims to predict whether a loan will be repaid successfully (loan_status = 1) or if the applicant will default (loan_status = 0) using a feed-forward neural network (MLP). The dataset consists of various applicant and loan-related features. Additionally, an R-based Bayesian statistics model is included to predict loan amounts.

## Setup Instructions

### Python Environment
#### Prerequisites:
- Python 3.8+
- Virtual environment tool (venv or conda)

#### Steps:
1. Clone the repository:
   ```sh
   git clone <repo_url>
   ```
2. Create and activate a virtual environment:
   ```sh
   python3 -m venv .venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset Description
The dataset includes the following features:
- `person_age`: Applicant's age
- `person_gender`: Gender (categorical)
- `person_education`: Education level (categorical)
- `person_income`: Annual income in USD
- `person_emp_exp`: Employment experience in years
- `person_home_ownership`: Type of home ownership (categorical)
- `loan_amnt`: Loan amount requested (target variable for R model)
- `loan_intent`: Purpose of the loan (categorical)
- `loan_int_rate`: Interest rate on the loan
- `loan_percent_income`: Loan amount as a percentage of income
- `cb_person_cred_hist_length`: Credit history length in years
- `credit_score`: Applicant's credit score
- `previous_loan_defaults_on_file`: Whether the applicant defaulted before (Yes/No)
- `loan_status`: Target variable for MLP model (1 = repaid, 0 = defaulted)


## Contributors
- **Narayan Sharma, Joshua Yu**

For any questions, reach out via [GitHub](https://github.com/narayansharma-21) or [LinkedIn](https://www.linkedin.com/in/ns324/).

