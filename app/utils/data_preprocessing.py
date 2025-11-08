import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_dataset(upload_folder="uploads"):
    """
    Handles dataset upload and loading in Streamlit.
    """
    uploaded_file = st.file_uploader("Upload Dataset (CSV format):", type=["csv"])
    if uploaded_file:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(file_path)
        st.success("Dataset uploaded successfully!")
        return df
    return None

def preprocess_data(data, input_columns, target_columns, apriori_columns=None):
    """
    Prepares the dataset by handling missing values, scaling inputs, and filtering useful features.
    """
    if data is None:
        return None, None, None, None

    # Ensure target column does not have NaN values
    known_targets = ~data[target_columns[0]].isna()
    inputs_train = data.loc[known_targets, input_columns]
    targets_train = data.loc[known_targets, target_columns]
    inputs_infer = data.loc[~known_targets, input_columns]
    
    # Scale Inputs & Targets
    scaler_inputs = StandardScaler().fit(inputs_train + 1e-8)
    scaler_targets = StandardScaler().fit(targets_train + 1e-8)

    
    inputs_train_scaled = scaler_inputs.transform(inputs_train)
    targets_train_scaled = scaler_targets.transform(targets_train)
    inputs_infer_scaled = scaler_inputs.transform(inputs_infer)
    
    # Handle Apriori Data
    if apriori_columns:
        apriori_train = data.loc[known_targets, apriori_columns]
        apriori_infer = data.loc[~known_targets, apriori_columns]
        scaler_apriori = StandardScaler().fit(apriori_train)
        apriori_train_scaled = scaler_apriori.transform(apriori_train)
        apriori_infer_scaled = scaler_apriori.transform(apriori_infer)
    else:
        apriori_train_scaled, apriori_infer_scaled = None, None
    
    return (inputs_train_scaled, targets_train_scaled, inputs_infer_scaled,
            apriori_train_scaled, apriori_infer_scaled, scaler_inputs, scaler_targets)
