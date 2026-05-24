
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
            acquisition = config.get('acquisition_function', 'webslamd')
            batch_size = int(config.get('batch_size', 1))
            result_df = MLEngine.run_experiment(
                data, 
                config.get('model'), 
                input_columns, 
                target_columns_config,
                curiosity=float(config.get('curiosity', 0.5)),
                apriori_config=config.get('apriori_columns'),
                acquisition_function=acquisition,
                batch_size=batch_size
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
        batch_size = int(config.get('batch_size', 1))
        return MLEngine.run_experiment(data, config.get('model'), input_columns, target_columns_config, batch_size=batch_size)

    @classmethod
    def _run_llm_only_mode(cls, data, config, input_columns, target_names):
        """
        LLM-only mode: Uses LLM to propose experiments, no ML training.
        Follows LLM-AL paper approach.
        """
        logging.info("🧠 Running in LLM-ONLY mode (no ML surrogate)")
        print("\n" + "="*60)
        print("🧠 LLM-ONLY MODE - Starting experiment")
        print("="*60)

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

        # Build the prompt explicitly so we can capture it for the trace
        system_prompt = agent._build_system_prompt(prompt_style, llm_strategy)
        user_prompt = agent._build_user_prompt(
            context, history_df, params_config, prompt_style,
            target_config=target_config, strategy=llm_strategy
        )
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

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
        result_df['Uncertainty'] = 0.1  # LLM doesn't provide uncertainty
        result_df['is_train_data'] = ~unlabeled_mask  # Labeled = True, Unlabeled = False

        # Initialize prediction columns
        for col in target_names:
            if col not in result_df.columns or result_df[col].isna().all():
                result_df[f"LLM_Predicted_{col}"] = np.nan

        match_idx = None
        predictions = {}
        if best_match_row is not None:
            match_idx = best_match_row.name
            result_df.loc[match_idx, 'Utility'] = 1.0
            result_df.loc[match_idx, 'Selected for Testing'] = True
            logging.info(f"✅ LLM Selected Candidate Index: {match_idx} (Semantic Score: {score:.3f})")
            print(f"✅ LLM Selected Candidate Index: {match_idx} (Score: {score:.3f})")
            print(f"🔮 Calling LLM to predict target values...")

            # Ask LLM to predict values for the selected sample
            try:
                predictions = agent.predict_values(
                    sample_row=best_match_row,
                    input_columns=input_columns,
                    target_columns=target_names,
                    history_df=history_df,
                    target_config=target_config
                )

                # Populate the predicted values for the selected sample
                for col, value in predictions.items():
                    result_df.loc[match_idx, col] = value
                    logging.info(f"🔮 LLM Prediction for {col}: {value}")
                    print(f"🔮 LLM Prediction for {col}: {value}")

            except Exception as e:
                logging.warning(f"LLM prediction failed: {e}")
                print(f"❌ LLM prediction FAILED: {e}")
                import traceback
                traceback.print_exc()

        # ── LLM Trace: captured for UI display ──────────────────────────
        provider = SettingsManager.get_setting("llm_provider", "ollama")
        model_name = SettingsManager.get_setting("ollama_model", "mistral:latest")

        # Capture actual matched candidate values so UI can show proposed vs actual
        matched_candidate_values = None
        if best_match_row is not None:
            try:
                matched_candidate_values = {
                    col: (round(float(best_match_row[col]), 5)
                          if isinstance(best_match_row[col], (int, float)) and best_match_row[col] == best_match_row[col]
                          else str(best_match_row[col]))
                    for col in input_columns if col in best_match_row.index
                }
            except Exception:
                matched_candidate_values = None

        result_df.attrs['llm_trace'] = {
            'mode': 'LLM_AGENT_MODE',
            'provider': provider,
            'model': model_name,
            'strategy': llm_strategy,
            'prompt': full_prompt,
            'raw_response': llm_proposal,
            'matched_index': int(match_idx) if match_idx is not None else None,
            'matched_candidate_values': matched_candidate_values,
            'semantic_score': round(float(score), 4) if score else None,
            'predictions': {k: (round(float(v), 4) if v is not None and v == v else None)
                            for k, v in predictions.items()},
            'n_labeled': int(len(labeled_data)),
            'n_candidates': int(unlabeled_mask.sum()),
        }
        # ────────────────────────────────────────────────────────────────

        # Record trajectory for LLM mode
        cls._record_trajectory(result_df, input_columns, "LLM_AGENT_MODE")

        return result_df

    @classmethod
    def _run_hybrid_mode(cls, data, config, input_columns, target_names):
        """
        Hybrid mode: Combines ML predictions + LLM proposals.
        """
        logging.info("🔀 Running in HYBRID mode (ML + LLM)")

        batch_size = int(config.get('batch_size', 1))

        # 1. Run ML to get predictions and base utility (batch_size=1, we'll apply batch after fusion)
        ml_results_df = MLEngine.run_experiment(
            data,
            config.get('model'),
            input_columns,
            config.get('target_columns'),
            curiosity=float(config.get('curiosity', 0.5)),
            batch_size=1  # We apply batch selection after fusion
        )

        # 2. Get LLM proposal with optimization direction
        agent = cls._get_llm_agent()
        target_config = config.get('target_columns', [])
        llm_strategy = SettingsManager.get_setting("llm_strategy", "balanced")

        labeled_data = data.dropna(subset=target_names)
        history_df = labeled_data[input_columns + target_names]
        context = f"Optimize {', '.join(target_names)}"
        params_config = {col: "continuous" for col in input_columns}
        prompt_style = SettingsManager.get_setting("prompt_style", "parameter-format")

        # Build prompt explicitly for trace capture
        system_prompt = agent._build_system_prompt(prompt_style, llm_strategy)
        user_prompt = agent._build_user_prompt(
            context, history_df, params_config, prompt_style,
            target_config=target_config, strategy=llm_strategy
        )
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

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

        # 4. Apply batch selection after fusion
        from app.utils.batch_selector import select_batch
        ml_results_df = select_batch(
            ml_results_df,
            n_samples=batch_size,
            input_columns=input_columns,
            diversity_weight=0.3
        )
        logging.info(f"✅ Batch selection: {batch_size} samples selected")

        # ── LLM Trace: captured for UI display ──────────────────────────
        # Find the top-scored candidate for reporting
        top_idx = ml_results_df['Utility'].idxmax() if not ml_results_df.empty else None
        top_semantic = float(ml_results_df.loc[top_idx, 'Semantic_Score']) if top_idx is not None else None
        top_ml_util = float(ml_results_df.loc[top_idx, 'ML_Utility']) if top_idx is not None else None

        provider = SettingsManager.get_setting("llm_provider", "ollama")
        model_name = SettingsManager.get_setting("ollama_model", "mistral:latest")
        ml_results_df.attrs['llm_trace'] = {
            'mode': 'HYBRID_MODE',
            'provider': provider,
            'model': model_name,
            'strategy': llm_strategy,
            'prompt': full_prompt,
            'raw_response': llm_proposal_text,
            'matched_index': int(top_idx) if top_idx is not None else None,
            'semantic_score': round(top_semantic, 4) if top_semantic is not None else None,
            'ml_utility': round(top_ml_util, 4) if top_ml_util is not None else None,
            'w_llm': w_llm,
            'w_ml': w_ml,
            'n_labeled': int(len(labeled_data)),
            'n_candidates': int(len(ml_results_df)),
        }
        # ────────────────────────────────────────────────────────────────

        # Record trajectory for Hybrid mode
        cls._record_trajectory(ml_results_df, input_columns, "HYBRID_MODE")

        return ml_results_df

    @classmethod
    def _get_llm_agent(cls):
        """Helper to create LLM agent from settings."""
        llm_provider = SettingsManager.get_setting("llm_provider", "ollama")
        ollama_model = SettingsManager.get_setting("ollama_model", "mistral:latest")
        # Use get_api_key: checks MISTRAL_API_KEY env var first, then settings file
        mistral_api_key = SettingsManager.get_api_key("mistral_api_key", "MISTRAL_API_KEY")
        
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

