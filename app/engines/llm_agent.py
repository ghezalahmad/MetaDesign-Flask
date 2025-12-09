
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
