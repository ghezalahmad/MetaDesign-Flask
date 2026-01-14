
import logging
import json
import requests
import pandas as pd

class LLMAgent:
    """
    LLM Agent supporting both Ollama (local) and Mistral Cloud API.
    
    Implements LLM-AL framework from paper:
    "Training-Free Active Learning Framework in Materials Science with Large Language Models"
    
    Provider can be:
    - 'ollama': Uses local Ollama server (no API key required)
    - 'mistral_cloud': Uses Mistral AI Cloud API (requires API key)
    """
    def __init__(self, provider='ollama', model='mistral:latest', api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.iteration_count = 0  # Track iteration count
        
        # Ollama endpoints
        self.ollama_chat_url = "http://localhost:11434/api/chat"
        
        # Mistral Cloud endpoint
        self.mistral_cloud_url = "https://api.mistral.ai/v1/chat/completions"

    def propose_next_experiment(self, context, history_df, parameters_config, 
                                  prompt_style="parameter-format", target_config=None,
                                  strategy="balanced"):
        """
        Generates a proposal for the next experiment using the configured provider.
        
        Args:
            context: Optimization context string
            history_df: DataFrame of previous experiments
            parameters_config: Dict of parameter names and types
            prompt_style: "parameter-format" or "report-format"
            target_config: List of target column configurations with max/min direction
            strategy: "explore", "exploit", or "balanced"
        """
        self.iteration_count += 1
        logging.info(f"🔄 LLM-AL Iteration {self.iteration_count} (Strategy: {strategy})")
        
        if self.provider == 'ollama':
            return self._call_ollama(context, history_df, parameters_config, prompt_style, target_config, strategy)
        else:
            return self._call_mistral_cloud(context, history_df, parameters_config, prompt_style, target_config, strategy)

    def _call_ollama(self, context, history_df, parameters_config, prompt_style, target_config, strategy):
        """Call local Ollama server with temperature=0 for consistency."""
        if not self._is_ollama_running():
            logging.warning("Ollama is not running. Start it with 'ollama serve'")
            return "OLLAMA_NOT_RUNNING: Start Ollama with 'ollama serve'"

        system_prompt = self._build_system_prompt(prompt_style, strategy)
        user_prompt = self._build_user_prompt(context, history_df, parameters_config, prompt_style, target_config, strategy)
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0,  # Paper: temperature=0 for reproducibility
                "seed": 42  # Fixed seed for consistency
            }
        }
        
        try:
            response = requests.post(self.ollama_chat_url, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')
        except requests.exceptions.ConnectionError:
            logging.error("Cannot connect to Ollama")
            return "OLLAMA_CONNECTION_ERROR: Cannot connect to Ollama"
        except Exception as e:
            logging.error(f"Ollama request failed: {e}")
            return f"OLLAMA_ERROR: {str(e)}"

    def _call_mistral_cloud(self, context, history_df, parameters_config, prompt_style, target_config, strategy):
        """Call Mistral Cloud API with temperature=0."""
        if not self.api_key:
            logging.warning("No Mistral API Key provided")
            return "NO_API_KEY_PROVIDED: Please enter your Mistral API key"

        system_prompt = self._build_system_prompt(prompt_style, strategy)
        user_prompt = self._build_user_prompt(context, history_df, parameters_config, prompt_style, target_config, strategy)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0  # Paper: temperature=0 for reproducibility
        }
        
        try:
            response = requests.post(self.mistral_cloud_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Mistral Cloud request failed: {e}")
            return f"MISTRAL_CLOUD_ERROR: {str(e)}"

    def _is_ollama_running(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _build_system_prompt(self, style, strategy="balanced"):
        """Build system prompt with strategy-specific guidance."""
        
        # Strategy-specific instructions
        if strategy == "explore":
            strategy_text = """
🔍 EXPLORATION MODE - YOU MUST EXPLORE:
- Choose parameter values VERY DIFFERENT from all previous experiments
- Look for UNEXPLORED REGIONS of the parameter space
- DO NOT propose values similar to existing observations
- Maximize DIVERSITY - pick values far from existing data clusters
- Your goal is to DISCOVER unknown regions, not optimize known ones
"""
        elif strategy == "exploit":
            strategy_text = """
📈 EXPLOITATION MODE - YOU MUST EXPLOIT:
- Focus on parameters similar to the BEST performing experiments
- Look at experiments with highest target values and propose NEARBY variations
- Make SMALL adjustments to improve on the best configurations
- Stay close to promising regions in the parameter space
- Your goal is to OPTIMIZE within known good regions
"""
        else:
            strategy_text = """
⚖️ BALANCED MODE: Balance exploration and exploitation based on your judgment.
"""
        
        base = f"""You are an expert materials scientist and optimization specialist.
Your task is to guide an active learning experiment.

{strategy_text}

IMPORTANT: Your output MUST be ONLY a valid JSON object with the parameter values.
No explanation, just the JSON."""
        
        return base

    def _build_user_prompt(self, context, history_df, parameters_config, style, target_config=None, strategy="balanced"):
        """Build user prompt with optimization direction and strategy emphasis."""
        
        # Build parameter description with types/ranges
        params_desc = "\n".join([f"  - {p}: {t}" for p, t in parameters_config.items()])
        
        # Build optimization direction if target_config provided
        if target_config:
            objectives = []
            for target in target_config:
                name = target.get('name', 'Unknown')
                direction = target.get('max_or_min', 'max')
                direction_word = "MAXIMIZE" if direction == 'max' else "MINIMIZE"
                objectives.append(f"  - {direction_word}: {name}")
            objectives_str = "\n".join(objectives)
        else:
            objectives_str = f"  - {context}"
        
        # Format history as few-shot examples
        if len(history_df) > 0:
            history_sample = history_df.tail(10)
            history_str = history_sample.to_string(index=False)
            n_observed = len(history_df)
        else:
            history_str = "No previous experiments."
            n_observed = 0
        
        # Add strategy-specific instruction
        if strategy == "explore":
            strategy_reminder = f"\n⚠️ EXPLORE: You have {n_observed} observations. Propose values VERY DIFFERENT from these!\n"
        elif strategy == "exploit":
            strategy_reminder = "\n⚠️ EXPLOIT: Find the BEST results below and propose SIMILAR values with small changes.\n"
        else:
            strategy_reminder = ""
        
        prompt = f"""=== ACTIVE LEARNING ITERATION {self.iteration_count} ==={strategy_reminder}
OPTIMIZATION OBJECTIVES:
{objectives_str}

PARAMETER SPACE:
{params_desc}

OBSERVED EXPERIMENTS ({n_observed} total):
{history_str}

Propose the NEXT experiment. Respond with ONLY valid JSON."""
        
        return prompt
    
    def get_iteration_count(self):
        """Return current iteration count."""
        return self.iteration_count
    
    def reset_iteration_count(self):
        """Reset iteration counter for new experiment run."""
        self.iteration_count = 0

    def predict_values(self, sample_row, input_columns, target_columns, history_df, target_config=None):
        """
        Ask the LLM to predict target values for a given sample.
        
        Args:
            sample_row: Series or dict with input feature values
            input_columns: List of input feature column names
            target_columns: List of target column names
            history_df: DataFrame of historical experiments (with both inputs and targets)
            target_config: Optional list of target configuration dicts
            
        Returns:
            dict: Predicted values for each target column
        """
        logging.info(f"🔮 LLM predicting values for sample...")
        
        # Build the prompt for value prediction - handle both Series and dict
        sample_parts = []
        for col in input_columns:
            try:
                if hasattr(sample_row, 'loc'):
                    value = sample_row.loc[col] if col in sample_row.index else sample_row.get(col, 0)
                elif hasattr(sample_row, 'get'):
                    value = sample_row.get(col, 0)
                else:
                    value = sample_row[col] if col in sample_row else 0
                
                # Format based on type
                if isinstance(value, float):
                    sample_parts.append(f"{col}={value:.4g}")
                else:
                    sample_parts.append(f"{col}={value}")
            except Exception as e:
                logging.warning(f"Could not format column {col}: {e}")
                sample_parts.append(f"{col}=?")
        
        sample_desc = ", ".join(sample_parts)
        logging.info(f"🔮 Sample description: {sample_desc[:200]}...")
        
        # Format history for context
        if len(history_df) > 0:
            history_sample = history_df.tail(15)  # Show more history for prediction
            history_str = history_sample.to_string(index=False)
        else:
            history_str = "No previous experiments available."
        
        # Build optimization direction context
        if target_config:
            target_desc = []
            for target in target_config:
                name = target.get('name', 'Unknown')
                direction = target.get('optimization', 'max')
                target_desc.append(f"  - {name} (goal: {direction.upper()})")
            targets_str = "\n".join(target_desc)
        else:
            targets_str = "\n".join([f"  - {col}" for col in target_columns])
        
        system_prompt = """You are an expert materials scientist. Based on the provided experimental history 
and your domain knowledge, estimate the expected target property values for a new sample.

IMPORTANT: Your output MUST be ONLY a valid JSON object with the target names as keys and predicted numeric values.
Example: {"compressive_strength": 45.2, "cost": 12.5}
No explanation, just the JSON."""

        user_prompt = f"""=== PROPERTY PREDICTION REQUEST ===

TARGET PROPERTIES TO PREDICT:
{targets_str}

SAMPLE TO PREDICT:
{sample_desc}

EXPERIMENTAL HISTORY ({len(history_df)} experiments):
{history_str}

Based on the patterns in the experimental history and your materials science knowledge,
predict the expected values for each target property for this sample.

Respond with ONLY valid JSON containing the predicted values."""

        logging.info(f"🔮 Sending prediction request to LLM...")
        print(f"🔮 Targets to predict: {target_columns}")
        
        # Call the LLM
        if self.provider == 'ollama':
            response = self._call_llm_raw(system_prompt, user_prompt, self.ollama_chat_url, is_ollama=True)
        else:
            response = self._call_llm_raw(system_prompt, user_prompt, self.mistral_cloud_url, is_ollama=False)
        
        logging.info(f"🔮 LLM Response: {response[:300]}...")
        print(f"🔮 LLM Raw Response: {response}")
        
        # Parse the response
        predictions = self._parse_prediction_response(response, target_columns)
        print(f"🔮 Parsed predictions: {predictions}")
        return predictions

    def _call_llm_raw(self, system_prompt, user_prompt, url, is_ollama=True):
        """Call LLM with custom prompts."""
        if is_ollama:
            if not self._is_ollama_running():
                return "{}"
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1}  # Slight temperature for prediction
            }
            try:
                response = requests.post(url, json=data, timeout=180)
                response.raise_for_status()
                result = response.json()
                return result.get('message', {}).get('content', '{}')
            except Exception as e:
                logging.error(f"LLM prediction call failed: {e}")
                return "{}"
        else:
            if not self.api_key:
                return "{}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": "mistral-large-latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1
            }
            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
            except Exception as e:
                logging.error(f"LLM prediction call failed: {e}")
                return "{}"

    def _parse_prediction_response(self, response, target_columns):
        """Parse LLM prediction response into dict of values."""
        predictions = {}
        
        def normalize_name(name):
            """Normalize a column name for comparison by removing special chars."""
            import re
            # Remove everything except alphanumeric, then lowercase
            normalized = re.sub(r'[^a-zA-Z0-9]', '', str(name).lower())
            return normalized
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                print(f"🔮 Parsed JSON: {parsed}")
                
                for col in target_columns:
                    col_normalized = normalize_name(col)
                    print(f"🔮 Looking for match for: '{col}' (normalized: '{col_normalized}')")
                    
                    # Try exact match first
                    if col in parsed:
                        predictions[col] = float(parsed[col])
                        print(f"🔮 Exact match found!")
                        continue
                    
                    # Try normalized matching
                    for key, value in parsed.items():
                        key_normalized = normalize_name(key)
                        print(f"🔮   Comparing with: '{key}' (normalized: '{key_normalized}')")
                        
                        # Check if normalized versions match or overlap
                        if key_normalized == col_normalized:
                            predictions[col] = float(value)
                            print(f"🔮   ✓ Normalized match!")
                            break
                        # Check if key is contained in column or vice versa
                        elif key_normalized in col_normalized or col_normalized in key_normalized:
                            predictions[col] = float(value)
                            print(f"🔮   ✓ Partial match!")
                            break
                        # Check first significant word/number overlap
                        elif len(set(key_normalized.split()) & set(col_normalized.split())) > 0:
                            predictions[col] = float(value)
                            print(f"🔮   ✓ Word overlap match!")
                            break
                            
        except Exception as e:
            logging.warning(f"Failed to parse LLM prediction response: {e}")
            print(f"❌ Parsing error: {e}")
        
        # Fill missing predictions with NaN
        import numpy as np
        for col in target_columns:
            if col not in predictions:
                predictions[col] = np.nan
        
        return predictions
