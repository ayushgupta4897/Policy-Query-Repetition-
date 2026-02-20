"""
Configuration for PQR experiments.
Copy this file to config.py and fill in your API keys.

    cp config.example.py config.py
"""

# ============================================================
# API Keys — fill in your own
# ============================================================
OPENAI_API_KEY = "sk-..."
DEEPINFRA_API_KEY = "your-deepinfra-key"
GEMINI_API_KEY = "your-gemini-key"

# ============================================================
# Model Configurations
# ============================================================
MODELS = {
    # OpenAI models
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "description": "GPT-4o - Latest multimodal"
    },
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
        "description": "GPT-4.1 - Latest GPT-4 series"
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "description": "GPT-3.5 Turbo - Fast baseline"
    },
    "o3-mini": {
        "provider": "openai",
        "model_id": "o3-mini",
        "description": "o3-mini - Reasoning model"
    },
    
    # DeepInfra models (OpenAI-compatible API)
    "llama-3.3-70b": {
        "provider": "deepinfra",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Llama 3.3 70B - Meta's flagship"
    },
    "qwen-2.5-72b": {
        "provider": "deepinfra",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Qwen 2.5 72B - Alibaba's flagship"
    },
    "deepseek-r1": {
        "provider": "deepinfra",
        "model_id": "deepseek-ai/DeepSeek-R1",
        "description": "DeepSeek R1 - Reasoning model (OSS)"
    },
    
    # Gemini models
    "gemini-2.0-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.0-flash",
        "description": "Gemini 2.0 Flash - Google's latest"
    },
}

# Experiment parameters
DEFAULT_SAMPLES_PER_CASE = 3
TEMPERATURE = 0.7
MAX_TOKENS = 1024
RESULTS_DIR = "results"

# Select which models to run
ACTIVE_MODELS = [
    "gpt-4o",
    "gpt-4.1",
    "gpt-3.5-turbo",
    # "o3-mini",
    "llama-3.3-70b",
    "qwen-2.5-72b",
    # "deepseek-r1",
    "gemini-2.0-flash",
]
