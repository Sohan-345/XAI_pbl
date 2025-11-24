import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # <-- Import the os module

# --- 1. LOAD MODELS AND ARTIFACTS ---

# Define the path to the models folder
MODEL_DIR = "Saved_models"

@st.cache_resource
def load_models():
    """
    Loads all pickled models and preprocessing artifacts.
    Uses @st.cache_resource for efficiency.
    """
    try:
        # Update paths to look inside the MODEL_DIR
        log_reg = joblib.load(os.path.join(MODEL_DIR, 'logistic_model.pkl'))
        svm = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
        column_info = joblib.load(os.path.join(MODEL_DIR, 'column_info.pkl'))
    except Exception as e:
        st.error(f"Error loading model or column_info files: {e}")
        return None, None, None, None

    # Try to load the primary scaler name, fallback to the ' (1)' version
    try:
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        try:
            # Update fallback path
            scaler_path_fallback = os.path.join(MODEL_DIR, 'scaler (1).pkl')
            scaler = joblib.load(scaler_path_fallback)
            st.warning("Loaded 'scaler (1).pkl'. Consider renaming it to 'scaler.pkl' for clarity.")
        except FileNotFoundError:
            st.error("Could not find 'scaler.pkl' or 'scaler (1).pkl' inside the 'Saved_models' folder.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None, None, None, None

    return log_reg, svm, scaler, column_info

# Load all assets
log_reg_model, svm_model, scaler, column_info = load_models()

# Extract column info (if loaded successfully)
if column_info:
    BINARY_COLS = column_info.get('binary_cols', [])
    MULTI_COLS = column_info.get('multi_cols', [])
    NUMERIC_COLS = column_info.get('numeric_cols', [])
    ALL_FEATURES_ORDER = column_info.get('all_features_order', [])
    
    # Remove 'Churn' if it's in the binary columns list, as it's the target
    if 'Churn' in BINARY_COLS:
        BINARY_COLS.remove('Churn')
else:
    # Set to empty lists if loading fails
    BINARY_COLS, MULTI_COLS, NUMERIC_COLS, ALL_FEATURES_ORDER = [], [], [], []


# --- 2. USER INPUT SIDEBAR ---

st.sidebar.title("Enter Customer Information")

def get_user_input():
    """
    Creates sidebar widgets to get user input for all model features.
    Returns a dictionary of inputs.
    """
    # Based on the notebook's data columns
    input_data = {}

    st.sidebar.header("Customer Demographics")
    input_data['gender'] = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    input_data['SeniorCitizen'] = st.sidebar.selectbox('Senior Citizen', [0, 1], help="0 = No, 1 = Yes")
    input_data['Partner'] = st.sidebar.selectbox('Has Partner?', ['No', 'Yes'])
    input_data['Dependents'] = st.sidebar.selectbox('Has Dependents?', ['No', 'Yes'])

    st.sidebar.header("Tenure and Charges")
    input_data['tenure'] = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    input_data['MonthlyCharges'] = st.sidebar.number_input('Monthly Charges ($)', min_value=0.0, max_value=150.0, value=50.0, step=0.01)
    input_data['TotalCharges'] = st.sidebar.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=500.0, step=0.01)

    st.sidebar.header("Services")
    input_data['PhoneService'] = st.sidebar.selectbox('Phone Service?', ['No', 'Yes'])
    
    # Options derived from the notebook's 'No phone service' value
    input_data['MultipleLines'] = st.sidebar.selectbox('Multiple Lines?', ['No', 'Yes', 'No phone service'])
    
    # Options derived from the notebook's 'No' (no internet) value
    input_data['InternetService'] = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    
    # Options for internet-dependent services
    inet_service_options = ['No', 'Yes', 'No internet service']
    input_data['OnlineSecurity'] = st.sidebar.selectbox('Online Security', inet_service_options)
    input_data['OnlineBackup'] = st.sidebar.selectbox('Online Backup', inet_service_options)
    input_data['DeviceProtection'] = st.sidebar.selectbox('Device Protection', inet_service_options)
    input_data['TechSupport'] = st.sidebar.selectbox('Tech Support', inet_service_options)
    input_data['StreamingTV'] = st.sidebar.selectbox('Streaming TV', inet_service_options)
    input_data['StreamingMovies'] = st.sidebar.selectbox('Streaming Movies', inet_service_options)

    st.sidebar.header("Contract and Payment")
    input_data['Contract'] = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    input_data['PaperlessBilling'] = st.sidebar.selectbox('Paperless Billing?', ['No', 'Yes'])
    input_data['PaymentMethod'] = st.sidebar.selectbox('Payment Method', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    
    predict_button = st.sidebar.button("Predict Churn")
    
    return input_data, predict_button

# Get inputs
user_input, predict_clicked = get_user_input()


# --- 3. PREDICTION AND DISPLAY LOGIC ---

st.title("Telco Customer Churn Predictor")
st.write("This app predicts the probability of a customer churning using your trained Logistic Regression and SVM models.")

if predict_clicked and all([log_reg_model, svm_model, scaler, column_info]):
    
    # --- 3A. Preprocessing ---
    # Create a single-row DataFrame from the user's input
    input_df = pd.DataFrame([user_input])

    # 1. Encode Binary Columns (using map, as LabelEncoders weren't saved)
    #    This assumes 'No'=0, 'Yes'=1 and 'Female'=0, 'Male'=1
    #    This MUST match the logic from your notebook.
    
    # Map for 'gender'
    gender_map = {'Female': 0, 'Male': 1}
    input_df['gender'] = input_df['gender'].map(gender_map)
    
    # Map for standard 'Yes'/'No' columns
    yes_no_map = {'No': 0, 'Yes': 1}
    binary_cols_to_map = [col for col in BINARY_COLS if col in input_df.columns and col != 'gender']
    
    for col in binary_cols_to_map:
        input_df[col] = input_df[col].map(yes_no_map)

    # 2. One-Hot Encode Multi-Class Columns
    #    pd.get_dummies will handle the categorical columns
    #    drop_first=True matches the notebook
    try:
        input_df_processed = pd.get_dummies(input_df, columns=MULTI_COLS, drop_first=True)
    except Exception as e:
        st.error(f"Error during one-hot encoding: {e}")
        st.stop()

    # 3. Align Columns to Match Model's Training Data
    #    This is the most critical step.
    #    It reorders columns and adds any missing dummy columns (with value 0).
    try:
        input_df_aligned = input_df_processed.reindex(columns=ALL_FEATURES_ORDER, fill_value=0)
    except Exception as e:
        st.error(f"Error aligning columns: {e}")
        st.dataframe(input_df_processed.head())
        st.write("Expected columns:", ALL_FEATURES_ORDER)
        st.stop()

    # 4. Scale Numeric Features
    #    Apply the loaded scaler to the numeric columns identified during training.
    #    The `NUMERIC_COLS` list from your notebook includes all numeric and encoded binary columns.
    try:
        input_df_aligned[NUMERIC_COLS] = scaler.transform(input_df_aligned[NUMERIC_COLS])
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.write("Columns being scaled:", NUMERIC_COLS)
        st.dataframe(input_df_aligned[NUMERIC_COLS].head())
        st.stop()

    # --- 3B. Prediction ---
    try:
        # Predict probabilities (we want the probability of class 1, i.e., 'Churn')
        pred_log_reg_proba = log_reg_model.predict_proba(input_df_aligned)[0][1]
        pred_svm_proba = svm_model.predict_proba(input_df_aligned)[0][1]

        # Get binary prediction (0 or 1)
        pred_log_reg_label = log_reg_model.predict(input_df_aligned)[0]
        pred_svm_label = svm_model.predict(input_df_aligned)[0]

        # --- 3C. Display Results ---
        st.subheader("Model Predictions")
        st.write("Based on the information provided, the customer's churn risk is:")

        col1, col2 = st.columns(2)

        with col1:
            st.info("Logistic Regression")
            st.metric(label="Churn Probability", value=f"{pred_log_reg_proba:.2%}")
            churn_status_lr = "Likely to Churn" if pred_log_reg_label == 1 else "Unlikely to Churn"
            st.write(f"**Prediction:** {churn_status_lr}")

        with col2:
            st.info("Support Vector Machine (SVM)")
            st.metric(label="Churn Probability", value=f"{pred_svm_proba:.2%}")
            churn_status_svm = "Likely to Churn" if pred_svm_label == 1 else "Unlikely to Churn"
            st.write(f"**Prediction:** {churn_status_svm}")
        
        # --- 3D. Debugging/Inspectation Info ---
        with st.expander("Show Processing Details"):
            st.subheader("Raw User Input (as a DataFrame)")
            st.dataframe(input_df)
            st.subheader("Fully Processed Data (Sent to Model)")
            st.dataframe(input_df_aligned)
            st.write("**Columns Scaled:**", NUMERIC_COLS)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Data sent to model (first 5 columns):")
        st.dataframe(input_df_aligned.iloc[:, :5].head())

elif predict_clicked:
    st.error("Models are not loaded. Please check the file paths and ensure all .pkl files are in the same directory as app.py.")
else:
    st.info("Please fill in the customer details in the sidebar and click 'Predict Churn'.")

