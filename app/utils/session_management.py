import json
import streamlit as st
import pandas as pd

def restore_session(session_data):
    """
    Restores session state from a saved JSON file.
    """
    try:
        input_columns = session_data.get("input_columns", [])
        target_columns = session_data.get("target_columns", [])
        apriori_columns = session_data.get("apriori_columns", [])
        weights_targets = session_data.get("weights_targets", [])
        weights_apriori = session_data.get("weights_apriori", [])
        thresholds_targets = session_data.get("thresholds_targets", [])
        thresholds_apriori = session_data.get("thresholds_apriori", [])
        curiosity = session_data.get("curiosity", 0.0)
        result_df = pd.DataFrame(session_data.get("results", []))

        return {
            "input_columns": input_columns,
            "target_columns": target_columns,
            "apriori_columns": apriori_columns,
            "weights_targets": weights_targets,
            "weights_apriori": weights_apriori,
            "thresholds_targets": thresholds_targets,
            "thresholds_apriori": thresholds_apriori,
            "curiosity": curiosity,
            "results": result_df
        }
    except Exception as e:
        st.error(f"Failed to restore session: {str(e)}")
        return None

def save_session(input_columns, target_columns, apriori_columns, weights_targets,
                  weights_apriori, thresholds_targets, thresholds_apriori, curiosity, result_df):
    """
    Saves session state to a JSON file for later retrieval.
    """
    session_data = {
        "input_columns": input_columns,
        "target_columns": target_columns,
        "apriori_columns": apriori_columns,
        "weights_targets": weights_targets,
        "weights_apriori": weights_apriori,
        "thresholds_targets": thresholds_targets,
        "thresholds_apriori": thresholds_apriori,
        "curiosity": curiosity,
        "results": result_df.to_dict(orient="records")
    }
    session_json = json.dumps(session_data, indent=4)
    st.download_button(
        label="Download Session as JSON",
        data=session_json,
        file_name="session.json",
        mime="application/json"
    )
