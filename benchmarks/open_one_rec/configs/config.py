# Imported from: https://https://github.com/Kuaishou-OneRec/OpenOneRec
# Original commit: ae668a6a30bd71bde7a3f459fe0ac958182ce1d2
# Date imported: 2026‑01‑09
# License: Apache 2.0 License (inherited from source)
"""
Label Prediction Task Configuration

This is a classification task for predicting user engagement with video content.
Uses logprobs-based classification with AUC and wuAUC metrics.
"""

# Label Pred Task Configuration
LABEL_PRED_CONFIG = {
    "name": "label_pred",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 346190,
    "sample_size": 346190,
    "description": "Predict user engagement with video content (yes/no classification)",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    "generation_config": {
        "max_new_tokens": 1,
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "do_sample": True,
        "num_return_sequences": 1,
        "return_logprobs": True,  # Need to return logprobs for probability extraction
        "logprobs": 10000,  # Return top-10 logprobs to ensure "是" and "否" are included
        "target_tokens": [
            "是",
            "否",
        ],  # Target tokens for logprobs extraction (classification)
        "max_new_thinking_tokens": 1000,
    },
    "evaluation_config": {
        "metrics": ["auc"],
    },
    "task_type": "logprobs_classification",  # Special task type
}

"""
Item Understand Task Configuration
"""

# Item Understand Task Configuration
ITEM_UNDERSTAND_CONFIG = {
    "name": "item_understand",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 500,
    "sample_size": 500,
    "description": "Video SID to Caption generation task",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": False,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    # Generation parameter configuration
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 128,
        "temperature": 0.01,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "do_sample": False,
        "num_return_thinking_sequences": 1,
        "max_new_thinking_tokens": 1000,
    },
    "evaluation_config": {
        "metrics": [
            "macro_wip_double_weighted_f1",
            "micro_wip_double_weighted_f1",
        ],
        "bertscore_model_type": "bert-base-chinese",
        "bertscore_num_layers": 9,
        "bertscore_lang": "zh",
        # WIP (Weighted Information Points) evaluation config
        "wip_enabled": True,  # Whether to enable WIP evaluation
        "wip_judge_model": "gemini",  # Judge LLM type: gemini/deepseek/claude
        "wip_max_workers": 1,  # Concurrent workers for LLM calls
        "wip_core_threshold": 5,  # Core threshold for importance score (1-5)
        "wip_max_samples": 500,  # Max samples to evaluate (None for all)
    },
}

"""
Recommendation Reason Task Configuration
"""

# Recommendation Reason Task Configuration
REC_REASON_CONFIG = {
    "name": "rec_reason",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 470,
    "sample_size": 470,
    "description": "Recommendation reason inference",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": {
        "enable_thinking": True,  # Enable thinking mode for apply_chat_template
        "custom_chat_template": "qwen3_soft_switch.jinja2",  # Custom jinja2 template (file in v1_0 directory)
    },
    "generation_config": {
        "num_return_sequences": 1,
        "max_new_tokens": 2000,
        "temperature": 0.01,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "do_sample": False,
        "num_return_thinking_sequences": 1,
        "max_new_thinking_tokens": 10000,
    },
    "evaluation_config": {
        "metrics": ["avg_score"],
        # LLM multi-dimensional evaluation config
        "llm_eval_enabled": True,  # Whether to enable LLM evaluation
        "llm_judge_model": "gemini",  # Judge LLM type: gemini/deepseek/claude
        "llm_max_workers": 1,  # Concurrent workers for LLM calls
        "llm_max_samples": 470,  # Max samples to evaluate (None for all)
    },
}

"""
Recommendation Task Configurations

This module contains configurations for all recommendation tasks including:
- label_cond: Predict next video given specified consumption behavior
- video: Next video prediction
- product: Predict next clicked product
- ad: Predict next clicked advertisement
"""

# Common prompt config for recommendation tasks
RECOMMENDATION_PROMPT_CONFIG = {
    "enable_thinking": False,
    "custom_chat_template": "qwen3_soft_switch.jinja2",
}

# Common generation config for recommendation tasks
RECOMMENDATION_GENERATION_CONFIG = {
    "num_return_sequences": 128,
    "max_new_tokens": 3,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 50,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "prompt_token": "<|sid_begin|>",  # Token to append for two-stage generation
    "max_new_thinking_tokens": 1000,
    "num_return_thinking_sequences": 8,  # Number of thinking candidates to generate in stage 1
    "num_beams": 16,
}

# Common evaluation config for recommendation tasks
RECOMMENDATION_EVALUATION_CONFIG = {
    "metrics": ["pass@k", "position1_pass@k", "recall@k"],
    "k_values": [1, 32],
    "select_k": "first_k",  # Strategy for selecting k predictions: 'first_k' or 'random_k'
    # PID-based evaluation settings
    "evaluation_mode": "both",  # Evaluation mode: 'sid', 'pid', or 'both'
    "sid_to_pid_strategy": "most_popular_after_downsampling",  # Strategy for SID->PID conversion: 'most_popular_originally', 'most_popular_after_downsampling', or 'random'
}

# Label Cond Task Configuration
LABEL_COND_CONFIG = {
    "name": "label_cond",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 34891,
    "sample_size": 34891,
    "description": "Predict next video given specified consumption behavior",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# SID USER Doc Task Configuration
VIDEO_CONFIG = {
    "name": "video",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 38781,
    "sample_size": 38781,
    "description": "Next video prediction",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Product Task Configuration
PRODUCT_CONFIG = {
    "name": "product",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 27910,
    "sample_size": 27910,
    "description": "Predict next clicked product",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Ad Task Configuration
AD_CONFIG = {
    "name": "ad",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 27677,
    "sample_size": 27677,
    "description": "Predict next clicked advertisement",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Interactive Task Configuration
INTERACTIVE_CONFIG = {
    "name": "interactive",
    "source": "Kuaishou Internal",
    "splits": ["test"],
    "size": 1000,
    "sample_size": 1000,
    "description": "Predict next interacted video",
    "data_fields": {
        "messages_field": "messages",
        "metadata_field": "metadata",
    },
    "prompt_config": RECOMMENDATION_PROMPT_CONFIG.copy(),
    "generation_config": RECOMMENDATION_GENERATION_CONFIG.copy(),
    "evaluation_config": RECOMMENDATION_EVALUATION_CONFIG.copy(),
}

# Task configuration mapping
RECOMMENDATION_TASK_CONFIGS = {
    "label_cond": LABEL_COND_CONFIG,
    "video": VIDEO_CONFIG,
    "product": PRODUCT_CONFIG,
    "ad": AD_CONFIG,
    "interactive": INTERACTIVE_CONFIG,
}
