"""
WEBSLAMD-style Experiment Preprocessor

Handles data validation, cleaning, and encoding before ML training.
"""
import numpy as np
import pandas as pd
import logging


class ExperimentPreprocessor:
    """
    Preprocesses experiment data following WEBSLAMD patterns:
    1. Filter A-priori with thresholds
    2. Filter missing inputs
    3. Validate experiment requirements
    4. Encode categorical features
    """
    
    # Model-specific minimum sample requirements
    MODEL_MIN_SAMPLES = {
        'random_forest': 2,
        'lolopy': 2,
        'gaussian_process': 1,
        'gp': 1,
        'tuned_gp': 4,
        'tuned_rf': 4,
        'pinn': 3,
        'maml': 5,
        'dkl': 3,
        'protonet': 3,
        'reptile': 3,
    }
    
    @classmethod
    def preprocess(cls, data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
        """
        Main preprocessing entry point.
        
        Args:
            data: Raw DataFrame with features and targets
            config: Experiment configuration containing:
                - model: Model name
                - input_columns: List of feature column names
                - target_columns: List of target configs with 'name', 'weight', 'optimization', 'threshold'
                - apriori_columns: Optional A-priori info configs
                - curiosity: Exploration factor
        
        Returns:
            (processed_data, validation_result) where validation_result contains
            any errors or warnings
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'dropped_columns': [],
            'encoded_columns': [],
            'filtered_rows': 0
        }
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Step 1: Filter A-priori with thresholds
        df, filtered_count = cls._filter_apriori_thresholds(df, config)
        result['filtered_rows'] = filtered_count
        
        # Step 2: Filter missing inputs
        df, dropped_cols = cls._filter_missing_inputs(df, config.get('input_columns', []))
        result['dropped_columns'] = dropped_cols
        if dropped_cols:
            result['warnings'].append(f"Dropped columns with NaN values: {dropped_cols}")
        
        # Step 3: Validate experiment
        errors = cls._validate_experiment(df, config)
        if errors:
            result['valid'] = False
            result['errors'] = errors
            return df, result
        
        # Step 4: Encode categoricals
        df, encoded_cols = cls._encode_categoricals(df, config.get('input_columns', []))
        result['encoded_columns'] = encoded_cols
        if encoded_cols:
            result['warnings'].append(f"Encoded categorical columns: {encoded_cols}")
        
        logging.info(f"✅ Preprocessing complete. Valid={result['valid']}, Warnings={len(result['warnings'])}")
        return df, result
    
    @classmethod
    def _filter_apriori_thresholds(cls, df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, int]:
        """
        Filter rows based on A-priori threshold constraints.
        Only filters unlabelled rows (where targets are NaN).
        
        WEBSLAMD logic:
        - For 'max' targets: drop rows where value < threshold
        - For 'min' targets: drop rows where value > threshold
        """
        apriori_configs = config.get('apriori_columns', [])
        target_columns = [t['name'] for t in config.get('target_columns', [])]
        
        if not apriori_configs:
            return df, 0
        
        original_len = len(df)
        
        for apriori in apriori_configs:
            col_name = apriori.get('name')
            threshold = apriori.get('threshold')
            max_or_min = apriori.get('optimization', 'max')
            
            if col_name not in df.columns or threshold is None:
                continue
            
            try:
                threshold = float(threshold)
            except (ValueError, TypeError):
                continue
            
            # Only filter unlabelled rows (targets are NaN)
            if target_columns:
                nodata_mask = df[target_columns].isna().all(axis=1)
            else:
                nodata_mask = pd.Series([True] * len(df), index=df.index)
            
            if max_or_min.lower() == 'max':
                # For maximization: drop rows below threshold
                drop_mask = (df[col_name] < threshold) & nodata_mask
            else:
                # For minimization: drop rows above threshold
                drop_mask = (df[col_name] > threshold) & nodata_mask
            
            df = df[~drop_mask].reset_index(drop=True)
            
        filtered_count = original_len - len(df)
        if filtered_count > 0:
            logging.info(f"📊 Filtered {filtered_count} rows by A-priori thresholds")
        
        return df, filtered_count
    
    @classmethod
    def _filter_missing_inputs(cls, df: pd.DataFrame, input_columns: list) -> tuple[pd.DataFrame, list]:
        """
        Drop input columns that have any NaN values.
        WEBSLAMD drops entire columns if they have missing values.
        """
        dropped_columns = []
        
        for col in input_columns.copy():
            if col in df.columns and df[col].isna().any():
                dropped_columns.append(col)
        
        # Don't actually drop from DataFrame, just track them
        # The ML engine should handle this by excluding these columns
        
        return df, dropped_columns
    
    @classmethod
    def _validate_experiment(cls, df: pd.DataFrame, config: dict) -> list:
        """
        Validate experiment configuration against WEBSLAMD requirements.
        Returns list of error messages (empty if valid).
        
        WEBSLAMD checks per-target:
        - If ALL data is labelled for a target → Error (use as A-priori instead)
        - If not enough labelled samples → Error (model-specific minimum)
        """
        errors = []
        
        model = config.get('model', '').lower()
        target_configs = config.get('target_columns', [])
        target_columns = [t['name'] for t in target_configs]
        input_columns = config.get('input_columns', [])
        
        # Check if model is valid
        if not model:
            errors.append("No model specified")
            return errors
        
        # Check for targets
        if not target_columns:
            errors.append("No target columns specified")
            return errors
        
        # Check for features
        if not input_columns:
            errors.append("No input features specified")
            return errors
        
        # Get minimum required samples for this model
        min_samples = cls.MODEL_MIN_SAMPLES.get(model, 1)
        
        # --- WEBSLAMD: Per-target validation ---
        for target in target_columns:
            if target not in df.columns:
                errors.append(f"Target column '{target}' not found in data")
                continue
            
            # Count how many values are labelled for this target
            labelled_count = df[target].notna().sum()
            total_count = len(df)
            
            # WEBSLAMD: If ALL data is labelled for this target → Error
            # This column should be used as A-priori, not as target
            if labelled_count == total_count:
                errors.append(
                    f"All data is already labelled for target '{target}'. "
                    f"This column has no empty values to predict. "
                    f"It can only be used as A-priori information, not as a target."
                )
            
            # WEBSLAMD: Model-specific minimum samples
            elif labelled_count < min_samples:
                errors.append(
                    f"Not enough labelled values for target '{target}'. "
                    f"'{model}' requires at least {min_samples} labelled values, "
                    f"but only {labelled_count} was/were found."
                )
        
        # Check target configurations
        for tc in target_configs:
            opt = tc.get('optimization', 'max')
            if opt not in ['max', 'min']:
                errors.append(f"Invalid optimization direction '{opt}' for target '{tc.get('name')}'")
        
        return errors
    
    @classmethod
    def _encode_categoricals(cls, df: pd.DataFrame, input_columns: list) -> tuple[pd.DataFrame, list]:
        """
        Encode non-numeric feature columns using factorize().
        WEBSLAMD converts categorical strings to numeric codes.
        """
        encoded_columns = []
        
        for col in input_columns:
            if col not in df.columns:
                continue
            
            # Check if column is non-numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Factorize: convert categories to integer codes
                df[col], _ = df[col].factorize()
                encoded_columns.append(col)
        
        return df, encoded_columns
    
    @classmethod
    def validate_only(cls, data: pd.DataFrame, config: dict) -> dict:
        """
        Quick validation without modifying data.
        Returns validation result dictionary.
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        errors = cls._validate_experiment(data, config)
        if errors:
            result['valid'] = False
            result['errors'] = errors
        
        return result
