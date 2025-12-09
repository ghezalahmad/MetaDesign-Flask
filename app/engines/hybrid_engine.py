
import pandas as pd
import numpy as np
import logging
from app.engines.ml_engine import MLEngine
from app.engines.llm_agent import LLMAgent
from app.engines.semantic_matcher import SemanticMatcher
from app.utils.settings_manager import SettingsManager
from app.utils.trajectory_tracker import TrajectoryTracker

class HybridEngine:
    """
    Orchestrates the Active Learning process.
    
    Modes:
    - ML_MODE: Uses ML surrogate models only
    - LLM_AGENT_MODE: Uses LLM only (no ML training)
    - HYBRID_MODE: Combines ML + LLM signals
    """
    
    @classmethod
    def run_experiment(cls, data, config):
        """
        Main entry point for running an experiment based on the current mode.
        """
        mode = SettingsManager.get_setting("active_learning_mode", "ML_MODE")
        logging.info(f"Running Experiment in {mode}")
        
        input_columns = config.get('input_columns')
        target_columns_config = config.get('target_columns')
        target_names = [t['name'] for t in target_columns_config]

        # ============================================================
        # ML_MODE: Pure ML - train surrogate, calculate utility
        # ============================================================
        if mode == "ML_MODE":
            result_df = MLEngine.run_experiment(
                data, 
                config.get('model'), 
                input_columns, 
                target_columns_config,
                curiosity=float(config.get('curiosity', 0.5))
            )
            # Record trajectory for ML mode
            cls._record_trajectory(result_df, input_columns, mode)
            return result_df

        # ============================================================
        # LLM_AGENT_MODE: Pure LLM - no ML training
        # ============================================================
        if mode == "LLM_AGENT_MODE":
            return cls._run_llm_only_mode(data, config, input_columns, target_names)

        # ============================================================
        # HYBRID_MODE: ML + LLM combined
        # ============================================================
        if mode == "HYBRID_MODE":
            return cls._run_hybrid_mode(data, config, input_columns, target_names)

        # Fallback to ML mode
        return MLEngine.run_experiment(data, config.get('model'), input_columns, target_columns_config)

    @classmethod
    def _run_llm_only_mode(cls, data, config, input_columns, target_names):
        """
        LLM-only mode: Uses LLM to propose experiments, no ML training.
        Follows LLM-AL paper approach.
        """
        logging.info("🧠 Running in LLM-ONLY mode (no ML surrogate)")
        
        # Get LLM agent and semantic matcher
        agent = cls._get_llm_agent()
        matcher = cls._get_semantic_matcher()
        
        # Get target config for optimization direction
        target_config = config.get('target_columns', [])
        
        # Prepare labeled history for LLM context (few-shot examples)
        labeled_data = data.dropna(subset=target_names)
        history_df = labeled_data[input_columns + target_names]
        
        # Prepare candidate space (unlabeled data)
        unlabeled_mask = data[target_names].isna().any(axis=1)
        
        if (~unlabeled_mask).sum() == len(data):
            logging.warning("No unlabeled candidates left!")
            return data.copy()
        
        # Build context and get LLM proposal with optimization direction
        context = f"Optimize {', '.join(target_names)}"
        params_config = {col: "continuous" for col in input_columns}
        prompt_style = SettingsManager.get_setting("prompt_style", "parameter-format")
        llm_strategy = SettingsManager.get_setting("llm_strategy", "balanced")
        
        # Pass target_config and strategy (LLM-AL paper improvements)
        llm_proposal = agent.propose_next_experiment(
            context, history_df, params_config, prompt_style, 
            target_config=target_config,
            strategy=llm_strategy
        )
        logging.info(f"🤖 LLM Proposal: {llm_proposal[:200]}...")
        
        # Match LLM proposal to actual candidate row
        candidate_data = data[unlabeled_mask].copy()
        best_match_row, score = matcher.find_best_match(llm_proposal, candidate_data, input_columns)
        
        # Build result DataFrame with ALL data (labeled + unlabeled) for TSNE
        result_df = data.copy()
        result_df['Utility'] = 0.0
        result_df['Selected for Testing'] = False
        result_df['Predicted'] = np.nan  # No predictions in LLM-only mode
        result_df['Uncertainty'] = np.nan
        result_df['is_train_data'] = ~unlabeled_mask  # Labeled = True, Unlabeled = False
        
        if best_match_row is not None:
            match_idx = best_match_row.name
            result_df.loc[match_idx, 'Utility'] = 1.0
            result_df.loc[match_idx, 'Selected for Testing'] = True
            logging.info(f"✅ LLM Selected Candidate Index: {match_idx} (Semantic Score: {score:.3f})")
        
        # Record trajectory for LLM mode
        cls._record_trajectory(result_df, input_columns, "LLM_AGENT_MODE")
        
        return result_df

    @classmethod
    def _run_hybrid_mode(cls, data, config, input_columns, target_names):
        """
        Hybrid mode: Combines ML predictions + LLM proposals.
        """
        logging.info("🔀 Running in HYBRID mode (ML + LLM)")
        
        # 1. Run ML to get predictions and base utility
        ml_results_df = MLEngine.run_experiment(
            data, 
            config.get('model'), 
            input_columns, 
            config.get('target_columns'),
            curiosity=float(config.get('curiosity', 0.5))
        )
        
        # 2. Get LLM proposal with optimization direction
        agent = cls._get_llm_agent()
        target_config = config.get('target_columns', [])
        
        labeled_data = data.dropna(subset=target_names)
        history_df = labeled_data[input_columns + target_names]
        context = f"Optimize {', '.join(target_names)}"
        params_config = {col: "continuous" for col in input_columns}
        prompt_style = SettingsManager.get_setting("prompt_style", "parameter-format")
        
        llm_proposal_text = agent.propose_next_experiment(
            context, history_df, params_config, prompt_style,
            target_config=target_config
        )
        logging.info(f"🤖 LLM Proposal: {llm_proposal_text[:200]}...")
        
        # 3. Hybrid Fusion: w_llm * semantic + w_ml * ucb
        weights = SettingsManager.get_setting("hybrid_weights", {"w_llm": 0.5, "w_ml": 0.5})
        w_llm = weights.get("w_llm", 0.5)
        w_ml = weights.get("w_ml", 0.5)
        
        # Normalize ML Utility (0-1)
        scaler = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ml_utility = scaler(ml_results_df['Utility'])
        
        # Semantic Scoring using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        candidate_texts = []
        for idx, row in ml_results_df.iterrows():
            desc = ", ".join([f"{col}: {row[col]}" for col in input_columns])
            candidate_texts.append(desc)
            
        vectorizer = TfidfVectorizer(stop_words='english')
        text_corpus = [llm_proposal_text] + candidate_texts
        vectors = vectorizer.fit_transform(text_corpus)
        
        query_vec = vectors[0:1]
        doc_vecs = vectors[1:]
        
        semantic_scores = cosine_similarity(query_vec, doc_vecs).flatten()
        
        # Fuse scores
        final_scores = (w_ml * ml_utility) + (w_llm * semantic_scores)
        
        ml_results_df['ML_Utility'] = ml_utility
        ml_results_df['Semantic_Score'] = semantic_scores
        ml_results_df['Utility'] = final_scores
        
        logging.info(f"✅ Hybrid fusion complete. w_ml={w_ml}, w_llm={w_llm}")
        
        # Record trajectory for Hybrid mode
        cls._record_trajectory(ml_results_df, input_columns, "HYBRID_MODE")
        
        return ml_results_df

    @classmethod
    def _get_llm_agent(cls):
        """Helper to create LLM agent from settings."""
        llm_provider = SettingsManager.get_setting("llm_provider", "ollama")
        ollama_model = SettingsManager.get_setting("ollama_model", "mistral:latest")
        mistral_api_key = SettingsManager.get_setting("mistral_api_key", "")
        
        return LLMAgent(
            provider=llm_provider,
            model=ollama_model,
            api_key=mistral_api_key
        )

    @classmethod
    def _get_semantic_matcher(cls):
        """Helper to create semantic matcher from settings."""
        cohere_key = SettingsManager.get_setting("cohere_api_key", "")
        return SemanticMatcher(api_key=cohere_key)

    @classmethod
    def _record_trajectory(cls, result_df, input_columns, mode):
        """
        Record the selected point to the trajectory tracker.
        
        Finds the point with highest utility (selected for testing) and
        adds it to the trajectory history.
        """
        try:
            # Find the selected point (highest utility)
            if 'Selected for Testing' in result_df.columns:
                selected = result_df[result_df['Selected for Testing'] == True]
                if not selected.empty:
                    selected_row = selected.iloc[0]
                else:
                    # Fallback: use highest utility
                    selected_row = result_df.loc[result_df['Utility'].idxmax()]
            else:
                # Fallback: use highest utility
                selected_row = result_df.loc[result_df['Utility'].idxmax()]
            
            iteration = TrajectoryTracker.get_iteration_count() + 1
            TrajectoryTracker.add_point(
                iteration=iteration,
                selected_row=selected_row,
                feature_columns=input_columns,
                mode=mode
            )
        except Exception as e:
            logging.warning(f"⚠️ Could not record trajectory point: {e}")

