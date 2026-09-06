import streamlit as st
import predict

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

st.title("💰 Loan Approval Prediction App")
st.markdown("Enter the applicant's details below to predict whether their loan will be **Approved** or **Rejected**.")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=9600000, step=100000)
    loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, value=29900000, step=100000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=12, step=1)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=778, step=1)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=2400000, step=100000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=17600000, step=100000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=22700000, step=100000)
    bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=8000000, step=100000)

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    # Prepare the feature dictionary
    features_dict = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': float(income_annum),
        'loan_amount': float(loan_amount),
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': float(residential_assets_value),
        'commercial_assets_value': float(commercial_assets_value),
        'luxury_assets_value': float(luxury_assets_value),
        'bank_asset_value': float(bank_asset_value)
    }

    # Display a loading spinner
    with st.spinner("Analyzing profile..."):
        status, confidence = predict.predict_single_instance(features_dict)

    if status:
        st.subheader("Prediction Result")
        if status == "Approved":
            st.success(f"✅ **Loan Approved**")
        else:
            st.error(f"❌ **Loan Rejected**")
        
        if confidence is not None:
            st.info(f"Model Confidence: **{confidence:.2%}**")
